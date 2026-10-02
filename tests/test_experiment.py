from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brace_f1.calibration import MonotonePlattCalibrator
from brace_f1.experiment import (
    HORIZONS_S,
    PRIMARY_BAYESIAN_METHOD,
    PRIMARY_DETERMINISTIC_METHOD,
    REGISTERED_METHODS,
    ExperimentData,
    ExperimentPaths,
    ScaledSideHazard,
    _canonical_json_hash,
    _code_content_hash,
    _fit_equal_cluster_residual,
    _model_metadata,
    _parser,
    _run_identity,
    _sha256,
    _validate_frozen_config,
    aggregate_completed_heldout_folds,
    apply_horizon_calibrators,
    apply_synthetic_proposal_delays,
    attach_partition_once,
    build_hazard_training_rows,
    calibration_threshold_grid,
    compact_forecast_batch,
    create_threshold_freeze_seal,
    evaluate_heldout_operating_points,
    evaluate_probability_performance,
    fit_fold_models,
    fit_horizon_calibrators,
    hazard_segment_bins,
    horizon_monotonicity_diagnostics,
    join_exposure_to_split,
    join_frames_targets,
    l_star_from_contributions,
    load_experiment_data,
    paired_l_star_sensitivity_table,
    reliability_bin_source_table,
    reliability_cluster_source_table,
    run_outer_fold,
    score_partition_methods,
    select_calibration_operating_points,
    select_ridge_alpha_inner_loco,
    synthetic_delay_l_star_table,
    validate_resume_manifest,
    validate_threshold_freeze_seal,
)
from brace_f1.forecast import ForecastProbabilities
from brace_f1.geometry import CircuitCorridor
from brace_f1.io import DataValidationError
from brace_f1.residual import ResidualTrainingSet

KEYS = ["circuit", "car_id", "frame_index", "time_seconds"]


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "circuit": ["A", "A", "B", "B"],
            "car_id": ["a", "a", "b", "b"],
            "source_session_id": ["sa", "sa", "sb", "sb"],
            "source_revision": ["r", "r", "r", "r"],
            "source_unit_sha256": ["a" * 64, "a" * 64, "b" * 64, "b" * 64],
            "frame_index": [0, 1, 0, 1],
            "time_seconds": [0.0, 0.05, 0.0, 0.05],
            "input_valid_causal": [True] * 4,
            "in_corridor": [True] * 4,
            "hard_break": [True, False, True, False],
        }
    )


def _target_rows() -> pd.DataFrame:
    rows = _frame_rows()[KEYS].copy()
    rows["next_qualifying_event_id"] = pd.Series([pd.NA, "e1", pd.NA, "e2"], dtype="string")
    rows["time_to_next_excursion_seconds"] = [np.nan, 0.2, np.nan, 0.4]
    for suffix in ("0p25", "0p50", "1p00", "1p50"):
        rows[f"outcome_evaluable_{suffix}s"] = True
    return rows


def _split_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_test_circuit": ["B", "B"],
            "circuit": ["A", "B"],
            "source_session_id": ["sa", "sb"],
            "car_id": ["a", "b"],
            "source_revision": ["r", "r"],
            "source_unit_sha256": ["a" * 64, "b" * 64],
            "partition": ["fit", "test"],
        }
    )


def test_four_key_join_is_one_to_one_and_rejects_duplicate_or_misaligned_truth() -> None:
    joined = join_frames_targets(_frame_rows(), _target_rows())
    assert len(joined) == 4
    assert joined.loc[1, "next_qualifying_event_id"] == "e1"

    duplicated = pd.concat([_target_rows(), _target_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="one-to-one"):
        join_frames_targets(_frame_rows(), duplicated)

    missing = _target_rows().iloc[:-1]
    with pytest.raises(DataValidationError, match="same four-key rows"):
        join_frames_targets(_frame_rows(), missing)


def test_partition_attachment_keeps_whole_cars_and_rejects_heldout_leakage() -> None:
    attached = attach_partition_once(_frame_rows(), _split_rows(), fold_test_circuit="B")
    assert attached.groupby(["circuit", "car_id"])["partition"].nunique().max() == 1
    assert set(attached.loc[attached["partition"] == "test", "circuit"]) == {"B"}

    leaking = _split_rows().copy()
    leaking.loc[1, "partition"] = "fit"
    with pytest.raises(DataValidationError, match="held-out circuit"):
        attach_partition_once(_frame_rows(), leaking, fold_test_circuit="B")

    split_car = pd.concat(
        [_split_rows(), _split_rows().iloc[[0]].assign(partition="calibration")],
        ignore_index=True,
    )
    with pytest.raises(DataValidationError, match="multiple assignments"):
        attach_partition_once(_frame_rows(), split_car, fold_test_circuit="B")


def test_exact_bulk_tail_threshold_grid_uses_only_supplied_calibration_scores() -> None:
    scores = np.arange(101, dtype=float) / 100.0
    grid = calibration_threshold_grid(
        scores,
        bulk_quantiles=3,
        bulk_quantile_range=(0.0, 0.50),
        upper_tail_quantiles=3,
        upper_tail_survival_exponents=(1.0, 2.0),
        include_endpoints=(0.0, 1.0),
    )
    # q = [0, .25, .50] and [1-10^-1, 1-10^-1.5, 1-10^-2], plus endpoints.
    np.testing.assert_allclose(
        grid,
        [0.0, 0.25, 0.50, 0.90, 1.0 - 10.0**-1.5, 0.99, 1.0],
        atol=1e-12,
    )


def test_hazard_adapter_floors_half_horizon_motion_and_wraps_circularly() -> None:
    bins = hazard_segment_bins(
        arclength_m=np.asarray([95.0, 5.0]),
        longitudinal_speed_mps=np.asarray([20.0, -50.0]),
        horizons_s=(0.5, 1.5),
        track_length_m=100.0,
        segment_length_m=25.0,
    )
    # 95 + 20*.5/2 = 100 -> wrap to bin 0; 95 + 20*1.5/2 = 110 -> bin 0.
    # Negative speed is clipped to zero, leaving arclength 5 in bin 0.
    np.testing.assert_array_equal(bins, [[0, 0], [0, 0]])


def test_hazard_one_step_labels_use_only_fit_sources_and_next_onset_side() -> None:
    rows = []
    for circuit, partition, onset_side in (
        ("A", "fit", "left"),
        ("B", "fit", "right"),
        ("C", "calibration", "left"),
        ("D", "test", "right"),
    ):
        for frame in range(3):
            rows.append(
                {
                    "circuit": circuit,
                    "car_id": f"car-{circuit}",
                    "frame_index": frame,
                    "time_seconds": frame * 0.05,
                    "continuous_segment_id": 0,
                    "partition": partition,
                    "input_valid_causal": True,
                    "in_corridor": frame < 2,
                    "qualifying_event_onset_projected": frame == 2,
                    "next_excursion_side": onset_side,
                    "body_speed_longitudinal_mps": 10.0,
                    "body_speed_lateral_mps": 0.0,
                    "yaw_rate_radps": 0.0,
                    "body_acceleration_longitudinal_mps2": 0.0,
                    "body_acceleration_lateral_mps2": 0.0,
                    "heading_error_rad": 0.0,
                    "track_offset_m": 0.0,
                    "track_curvature_per_m": 0.0,
                }
            )
    training = build_hazard_training_rows(pd.DataFrame(rows))
    assert set(training.row_keys["circuit"]) == {"A", "B"}
    assert training.labels.tolist() == ["no_exit", "left", "no_exit", "right"]
    assert training.X.shape == (4, 9)


def test_hazard_onset_side_comes_from_pre_onset_source_not_projected_row() -> None:
    rows = []
    for frame, next_side in enumerate(("left", pd.NA, "right")):
        rows.append(
            {
                "circuit": "A",
                "car_id": "car-A",
                "frame_index": frame,
                "time_seconds": frame * 0.05,
                "continuous_segment_id": 0,
                "partition": "fit",
                "input_valid_causal": True,
                "in_corridor": frame < 2,
                "qualifying_event_onset_projected": frame == 1,
                "next_excursion_side": next_side,
                "body_speed_longitudinal_mps": 10.0,
                "body_speed_lateral_mps": 0.0,
                "yaw_rate_radps": 0.0,
                "body_acceleration_longitudinal_mps2": 0.0,
                "body_acceleration_lateral_mps2": 0.0,
                "heading_error_rad": 0.0,
                "track_offset_m": 0.0,
                "track_curvature_per_m": 0.0,
            }
        )

    training = build_hazard_training_rows(pd.DataFrame(rows))

    assert training.labels.tolist() == ["left", "no_exit"]


def test_inner_loco_ridge_records_each_alpha_and_never_validates_on_its_training_circuit() -> None:
    rng = np.random.default_rng(91)
    circuits = np.repeat(["A", "B", "C"], 12)
    cars = np.repeat(["a1", "a2", "b1", "b2", "c1", "c2"], 6)
    X = rng.normal(size=(36, 2))
    coefficient = np.asarray([[0.8, -0.3], [0.2, 0.5]])
    Y = X @ coefficient + rng.normal(scale=0.05, size=(36, 2))
    pairs = ResidualTrainingSet(
        X=X,
        Y=Y,
        circuits=circuits,
        cars=cars,
        feature_names=("x1", "x2"),
        target_names=("y1", "y2"),
        row_keys=pd.DataFrame({"row": np.arange(36)}),
    )
    selection = select_ridge_alpha_inner_loco(pairs, alpha_grid=(0.01, 0.1, 1.0))
    assert set(selection.mean_gaussian_nll_by_alpha) == {0.01, 0.1, 1.0}
    assert set(selection.fold_scores["validation_circuit"]) == {"A", "B", "C"}
    assert len(selection.fold_scores) == 9
    assert selection.selected_alpha == min(
        selection.mean_gaussian_nll_by_alpha,
        key=lambda alpha: (selection.mean_gaussian_nll_by_alpha[alpha], alpha),
    )
    assert (selection.fold_scores["fit_circuit_count"] == 2).all()


def test_equal_cluster_ridge_fit_is_invariant_to_duplicate_rows_within_one_car() -> None:
    rows: list[tuple[str, str, float, float, float, float]] = []
    grid = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    for circuit_index, circuit in enumerate(("A", "B", "C")):
        for car_index in range(2):
            car = f"{circuit.lower()}{car_index}"
            for value in grid:
                x1 = float(value)
                x2 = float(value * value - 2.0)
                y1 = (0.5 + 0.15 * car_index) * x1 + 0.1 * circuit_index
                y2 = -0.25 * x2 + 0.2 * car_index - 0.05 * circuit_index
                rows.append((circuit, car, x1, x2, y1, y2))

    def make_pairs(
        source: list[tuple[str, str, float, float, float, float]],
    ) -> ResidualTrainingSet:
        array = np.asarray([row[2:] for row in source], dtype=np.float64)
        return ResidualTrainingSet(
            X=array[:, :2],
            Y=array[:, 2:],
            circuits=np.asarray([row[0] for row in source]),
            cars=np.asarray([row[1] for row in source]),
            feature_names=("x1", "x2"),
            target_names=("y1", "y2"),
            row_keys=pd.DataFrame({"row": np.arange(len(source))}),
        )

    duplicated_car = [row for row in rows if row[0] == "A" and row[1] == "a0"]
    original_pairs = make_pairs(rows)
    duplicated_pairs = make_pairs([*rows, *duplicated_car, *duplicated_car])
    original = _fit_equal_cluster_residual(
        original_pairs.X,
        original_pairs.Y,
        circuits=original_pairs.circuits,
        cars=original_pairs.cars,
        ridge_alpha=1.0,
    )
    duplicated = _fit_equal_cluster_residual(
        duplicated_pairs.X,
        duplicated_pairs.Y,
        circuits=duplicated_pairs.circuits,
        cars=duplicated_pairs.cars,
        ridge_alpha=1.0,
    )

    np.testing.assert_allclose(duplicated.feature_center, original.feature_center, atol=0.0)
    np.testing.assert_allclose(duplicated.feature_scale, original.feature_scale, atol=0.0)
    np.testing.assert_allclose(duplicated.coefficients, original.coefficients, atol=1e-12)
    np.testing.assert_allclose(
        duplicated.process_covariance,
        original.process_covariance,
        rtol=0.0,
        atol=1e-12,
    )


def _fit_feature_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for circuit_index, (circuit, partition) in enumerate(
        (("A", "fit"), ("B", "fit"), ("C", "fit"), ("D", "calibration"), ("E", "test"))
    ):
        for frame in range(8):
            poison = 10_000.0 if partition != "fit" else 0.0
            rows.append(
                {
                    "circuit": circuit,
                    "car_id": f"car-{circuit}",
                    "frame_index": frame,
                    "time_seconds": frame * 0.05,
                    "continuous_segment_id": 0,
                    "partition": partition,
                    "input_valid_causal": True,
                    "in_corridor": frame != 7,
                    "qualifying_event_onset_projected": frame == 7,
                    "next_excursion_side": "left" if circuit_index % 2 == 0 else "right",
                    "body_speed_longitudinal_mps": 10.0 + circuit_index + frame * 0.02 + poison,
                    "body_speed_lateral_mps": 0.01 * frame,
                    "yaw_rate_radps": 0.001 * frame,
                    "body_acceleration_longitudinal_mps2": 0.4,
                    "body_acceleration_lateral_mps2": 0.02,
                    "heading_error_rad": 0.001 * frame,
                    "track_offset_m": 0.1 * circuit_index,
                    "track_curvature_per_m": 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_outer_fold_fit_uses_fit_pairs_only_and_primary_comparator_is_exact_twin() -> None:
    table = _fit_feature_table()
    fitted = fit_fold_models(table, seed=77, alpha_grid=(0.01, 0.1))
    assert fitted.bayesian.coefficient_draws.shape[0] == 256
    assert fitted.bayesian.fit_row_count == 18  # 3 circuits * six in-corridor transitions
    assert fitted.bayesian.feature_center[0] < 100.0
    np.testing.assert_allclose(
        fitted.posterior_mean_twin.coefficients,
        fitted.bayesian.coefficient_draws.mean(axis=0),
        atol=0.0,
    )
    np.testing.assert_array_equal(
        fitted.posterior_mean_twin.feature_center,
        fitted.bayesian.feature_center,
    )
    assert fitted.models[PRIMARY_BAYESIAN_METHOD] is fitted.bayesian
    assert fitted.models[PRIMARY_DETERMINISTIC_METHOD] is fitted.posterior_mean_twin
    assert set(fitted.ridge_selection.fold_scores["validation_circuit"]) == {"A", "B", "C"}
    assert fitted.calibration_tuned_dynamics is not None
    metadata = _model_metadata(fitted)["calibration_mse_tuned_dynamics"]
    assert metadata["selected_alpha"] in {0.01, 0.1}
    assert set(metadata["validation_mse_by_alpha"]) == {"0.01", "0.1"}
    assert metadata["content_hash"] == fitted.calibration_tuned_dynamics.content_hash


def _forecast() -> ForecastProbabilities:
    outcome = np.zeros((2, 2, 3, 5), dtype=float)
    outcome[0, 0, 0, 4] = 0.2
    outcome[0, 1, 0, [4, 0, 1]] = [0.2, 0.3, 0.1]
    outcome[0, 1, 1, 2] = 0.3
    outcome[1, 0, 1, 3] = 0.4
    outcome[1, 1, 1, [2, 3, 4]] = [0.2, 0.4, 0.2]
    no_exit = 1.0 - outcome.sum(axis=(2, 3))
    return ForecastProbabilities(
        horizons_s=np.asarray([0.5, 1.5]),
        no_exit_probability=no_exit,
        outcome_probability=outcome,
        side_order=("left", "right", "unknown"),
        segment_bin_count=5,
        first_crossing_time_s=np.full((2, 1), np.nan),
        first_crossing_side_index=np.full((2, 1), -1, dtype=np.int8),
        first_crossing_bin=np.full((2, 1), -1, dtype=np.int64),
        seed=1,
        model_hash="m",
        config_hash="c",
    )


def test_compact_batch_keeps_totals_and_primary_decision_but_never_dense_or_truth() -> None:
    keys = pd.DataFrame(
        {
            "circuit": ["A", "A"],
            "source_session_id": ["s", "s"],
            "car_id": ["c", "c"],
            "frame_index": [1, 2],
            "time_seconds": [0.05, 0.10],
            "input_valid_causal": [True, True],
            "in_corridor": [True, True],
            "hard_break": [False, False],
        }
    )
    compact = compact_forecast_batch(keys, _forecast(), method="brace", primary_horizon_s=1.5)
    np.testing.assert_allclose(compact["raw_exit_probability_0p50s"], [0.2, 0.4])
    np.testing.assert_allclose(compact["raw_exit_probability_1p50s"], [0.9, 0.8])
    np.testing.assert_allclose(compact["raw_primary_neighborhood_probability"], [0.6, 0.8])
    assert compact["predicted_side"].tolist() == ["left", "right"]
    assert compact["predicted_segment_bin"].tolist() == [0, 3]
    assert not any("outcome_probability" in column for column in compact)
    assert not any(column.startswith(("next_", "time_to_next_")) for column in compact)


def _score_features() -> pd.DataFrame:
    rows = []
    for partition, car, y in (("calibration", "cal", 0.0), ("test", "test", 1.0)):
        for frame in range(3):
            rows.append(
                {
                    "fold_test_circuit": "A",
                    "partition": partition,
                    "circuit": "A",
                    "source_session_id": f"session-{car}",
                    "car_id": car,
                    "frame_index": frame,
                    "time_seconds": frame * 0.05,
                    "continuous_segment_id": 0,
                    "input_valid_causal": True,
                    "in_corridor": True,
                    "hard_break": frame == 0,
                    "map_x_m": 7.5,
                    "map_y_m": y,
                    "yaw_rad": 0.0,
                    "yaw_rate_radps": 0.0,
                    "body_speed_longitudinal_mps": 8.0,
                    "body_speed_lateral_mps": 0.0,
                    "body_acceleration_longitudinal_mps2": 0.0,
                    "body_acceleration_lateral_mps2": 0.0,
                    "heading_error_rad": 0.0,
                    "track_offset_m": 0.0,
                    "track_curvature_per_m": 0.0,
                    "centerline_arclength_wrapped_m": 0.0,
                    "track_length_m": 60.0,
                }
            )
    return pd.DataFrame(rows)


def _square_ring() -> CircuitCorridor:
    return CircuitCorridor.from_arrays(
        centerline=np.asarray([[-7.5, -7.5], [7.5, -7.5], [7.5, 7.5], [-7.5, 7.5]]),
        boundary_a=np.asarray([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]),
        boundary_b=np.asarray([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]),
    )


def test_batch_scoring_is_invariant_and_emits_only_calibration_and_test_rows() -> None:
    features = _score_features()
    by_one = score_partition_methods(
        features,
        corridors={"A": _square_ring()},
        models={},
        methods=("constant_velocity",),
        outer_fold="A",
        base_seed=2027,
        batch_size=1,
    )
    by_four = score_partition_methods(
        features,
        corridors={"A": _square_ring()},
        models={},
        methods=("constant_velocity",),
        outer_fold="A",
        base_seed=2027,
        batch_size=4,
    )
    pd.testing.assert_frame_equal(by_one, by_four)
    assert set(by_one["partition"]) == {"calibration", "test"}
    assert len(by_one) == len(features)


def test_primary_method_names_fix_bayesian_vs_posterior_mean_twin_comparison() -> None:
    assert PRIMARY_BAYESIAN_METHOD == "brace_bayesian"
    assert PRIMARY_DETERMINISTIC_METHOD == "posterior_mean_twin"


def test_calibrators_fit_only_calibration_evaluable_rows_and_apply_without_truth_columns() -> None:
    score = pd.DataFrame(
        {
            **{key: _target_rows()[key] for key in KEYS},
            "partition": ["calibration", "calibration", "test", "test"],
            "method": ["brace"] * 4,
            "raw_exit_probability_0p50s": [0.1, 0.8, 0.2, 0.9],
            "raw_primary_neighborhood_probability": [0.05, 0.6, 0.1, 0.7],
        }
    )
    target = _target_rows().copy()
    calibrators = fit_horizon_calibrators(score, target, horizons_s=(0.5,))
    assert set(calibrators) == {("brace", 0.5)}

    transformed = apply_horizon_calibrators(score, calibrators, horizons_s=(0.5,))
    assert "exit_probability_0p50s" in transformed
    assert "proposal_score" in transformed
    assert not set(target.columns).difference(KEYS).intersection(transformed.columns)

    poisoned = score.copy()
    poisoned.loc[poisoned["partition"] == "test", "raw_exit_probability_0p50s"] = 0.0
    poisoned_calibrators = fit_horizon_calibrators(poisoned, target, horizons_s=(0.5,))
    assert poisoned_calibrators[("brace", 0.5)] == calibrators[("brace", 0.5)]

    missing_truth = target.drop(index=1).copy()
    with pytest.raises(DataValidationError, match="every score row"):
        fit_horizon_calibrators(score, missing_truth, horizons_s=(0.5,))


def test_compact_calibration_keeps_exact_zero_exit_mass_at_zero() -> None:
    score = pd.DataFrame(
        {
            **{key: _target_rows().loc[:1, key] for key in KEYS},
            "partition": ["calibration", "test"],
            "method": ["brace", "brace"],
            "raw_exit_probability_0p50s": [0.0, 0.2],
            "raw_primary_neighborhood_probability": [0.0, 0.1],
        }
    )
    calibrators = {
        ("brace", 0.5): MonotonePlattCalibrator(slope=0.0, intercept=0.0)
    }

    transformed = apply_horizon_calibrators(score, calibrators, horizons_s=(0.5,))

    np.testing.assert_allclose(transformed["exit_probability_0p50s"], [0.0, 0.5])
    np.testing.assert_allclose(transformed["proposal_score"], [0.0, 0.25])


def test_reliability_source_table_persists_every_equal_width_bin() -> None:
    target = _target_rows().copy()
    score = pd.DataFrame(
        {
            **{key: target[key] for key in KEYS},
            "source_session_id": ["sa", "sa", "sb", "sb"],
            "fold_test_circuit": ["B"] * 4,
            "partition": ["test"] * 4,
            "method": [PRIMARY_BAYESIAN_METHOD] * 4,
            "input_valid_causal": [True] * 4,
            "in_corridor": [True] * 4,
            "exit_probability_0p50s": [0.05, 0.25, 0.65, 0.95],
        }
    )
    reference = pd.DataFrame(
        {
            "fold_test_circuit": ["B"],
            "method": [PRIMARY_BAYESIAN_METHOD],
            "horizon_s": [0.5],
            "reference_prevalence": [0.5],
        }
    )
    metrics = evaluate_probability_performance(score, target, reference, horizons_s=(0.5,))
    reliability = reliability_bin_source_table(score, target, horizons_s=(0.5,))

    assert len(metrics) == 3  # two circuits plus the pooled diagnostic
    assert "average_precision_stepwise" in metrics.columns
    assert (metrics["declared_rate_hz"] == 20.0).all()
    np.testing.assert_allclose(metrics["evaluable_exposure_hours"], metrics["n"] / 72000.0)
    pooled_metric = metrics.loc[metrics["scope"] == "pooled"].iloc[0]
    assert pooled_metric["average_precision_stepwise"] == pytest.approx(5.0 / 6.0)
    assert len(reliability) == 30
    assert {
        "bin_index",
        "bin_left",
        "bin_right",
        "count",
        "mean_probability",
        "observed_frequency",
    }.issubset(reliability.columns)
    pooled = reliability.loc[reliability["scope"] == "pooled"]
    assert pooled["count"].sum() == 4
    assert pooled["bin_index"].tolist() == list(range(10))
    cluster = reliability_cluster_source_table(score, target, horizons_s=(0.5,))
    assert {
        "source_session_id",
        "car_id",
        "count",
        "probability_sum",
        "event_count",
    }.issubset(cluster.columns)
    assert cluster["count"].sum() == 4


def test_exposure_is_joined_by_complete_unit_and_partition_not_counted_from_frames() -> None:
    timing = pd.DataFrame(
        {
            "circuit": ["A", "B"],
            "source_session_id": ["sa", "sb"],
            "car_id": ["a", "b"],
            "source_revision": ["r", "r"],
            "source_unit_sha256": ["a" * 64, "b" * 64],
            "buffered_exposure_seconds": [3600.0, 1800.0],
        }
    )
    joined = join_exposure_to_split(timing, _split_rows(), fold_test_circuit="B")
    assert joined.set_index("partition")["exposure_hours"].to_dict() == {
        "fit": 1.0,
        "test": 0.5,
    }

    duplicate = pd.concat([timing, timing.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="one row per car-session"):
        join_exposure_to_split(duplicate, _split_rows(), fold_test_circuit="B")


def test_operating_thresholds_use_calibration_only_for_every_lead_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(0.0, 4.05, 0.05)
    scores = np.zeros(times.size)
    scores[(times >= 0.50) & (times <= 0.90)] = 0.90
    scores[(times >= 2.20) & (times <= 2.40)] = 0.70
    in_corridor = np.ones(times.size, dtype=bool)
    in_corridor[(times >= 1.00) & (times < 1.10)] = False
    calibration = pd.DataFrame(
        {
            "fold_test_circuit": "Held",
            "partition": "calibration",
            "method": PRIMARY_BAYESIAN_METHOD,
            "circuit": "Cal",
            "source_session_id": "s-cal",
            "car_id": "car-cal",
            "frame_index": np.arange(times.size),
            "time_seconds": times,
            "input_valid_causal": True,
            "in_corridor": in_corridor,
            "hard_break": False,
            "proposal_score": scores,
            "predicted_side": "left",
            "predicted_segment_bin": 3,
            "n_segment_bins": 10,
        }
    )
    # Test rows carry an extreme score but must be irrelevant to threshold selection.
    test_rows = calibration.iloc[:3].copy()
    test_rows["partition"] = "test"
    test_rows["circuit"] = "Held"
    test_rows["source_session_id"] = "s-test"
    test_rows["car_id"] = "car-test"
    test_rows["proposal_score"] = 1.0
    score_table = pd.concat([calibration, test_rows], ignore_index=True)
    events = pd.DataFrame(
        {
            "candidate_event_id": ["e-cal", "e-test"],
            "circuit": ["Cal", "Held"],
            "car_id": ["car-cal", "car-test"],
            "start_time_seconds": [1.0, 0.2],
            "side_at_onset": ["left", "right"],
            "segment_bin_25m_at_onset": [3, 9],
            "qualified": [True, True],
        }
    )
    exposure = pd.DataFrame(
        {
            "fold_test_circuit": ["Held", "Held"],
            "partition": ["calibration", "test"],
            "circuit": ["Cal", "Held"],
            "source_session_id": ["s-cal", "s-test"],
            "car_id": ["car-cal", "car-test"],
            "exposure_hours": [2.0, 100.0],
        }
    )
    import brace_f1.experiment as experiment_module

    original_state_machine = experiment_module.run_proposal_state_machine
    calls = 0

    def counted_state_machine(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_state_machine(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "run_proposal_state_machine", counted_state_machine)
    selected = select_calibration_operating_points(
        score_table,
        events,
        exposure,
        outer_fold="Held",
        required_leads_s=(0.25, 0.5),
        false_budgets_per_hour=(1.0, 2.0),
    )
    assert len(selected) == 4
    assert set(selected["threshold_source_partition"]) == {"calibration"}
    assert set(selected["required_lead_s"]) == {0.25, 0.5}
    assert set(selected["false_budget_per_hour"]) == {1.0, 2.0}
    expected_threshold_count = calibration_threshold_grid(scores).size
    assert calls == expected_threshold_count

    low_exposure = exposure.copy()
    low_exposure.loc[low_exposure["partition"] == "calibration", "exposure_hours"] = 0.5
    low_capacity = select_calibration_operating_points(
        score_table,
        events,
        low_exposure,
        outer_fold="Held",
        required_leads_s=(0.25,),
        false_budgets_per_hour=(1.0, 2.0),
    ).set_index("false_budget_per_hour")
    assert low_capacity.loc[1.0, "threshold"] == "NOT_ESTIMABLE"
    assert low_capacity.loc[1.0, "operating_point_status"] == "not_estimable_insufficient_exposure"
    assert low_capacity.loc[1.0, "calibration_false_count_capacity"] == pytest.approx(0.5)
    assert low_capacity.loc[2.0, "operating_point_status"] == "estimable"


def test_heldout_evaluation_requires_frozen_calibration_thresholds_and_emits_car_clusters() -> None:
    times = np.arange(0.0, 2.05, 0.05)
    score_values = np.zeros(times.size)
    score_values[(times >= 0.50) & (times <= 0.80)] = 0.9
    in_corridor = np.ones(times.size, dtype=bool)
    in_corridor[(times >= 1.0) & (times < 1.1)] = False
    scores = pd.DataFrame(
        {
            "fold_test_circuit": "Held",
            "partition": "test",
            "method": PRIMARY_BAYESIAN_METHOD,
            "circuit": "Held",
            "source_session_id": "s-test",
            "car_id": "car-test",
            "frame_index": np.arange(times.size),
            "time_seconds": times,
            "input_valid_causal": True,
            "in_corridor": in_corridor,
            "hard_break": False,
            "proposal_score": score_values,
            "predicted_side": "left",
            "predicted_segment_bin": 3,
            "n_segment_bins": 10,
        }
    )
    events = pd.DataFrame(
        {
            "candidate_event_id": ["e-test"],
            "circuit": ["Held"],
            "car_id": ["car-test"],
            "start_time_seconds": [1.0],
            "side_at_onset": ["left"],
            "segment_bin_25m_at_onset": [3],
            "qualified": [True],
        }
    )
    exposure = pd.DataFrame(
        {
            "fold_test_circuit": ["Held"],
            "partition": ["test"],
            "circuit": ["Held"],
            "source_session_id": ["s-test"],
            "car_id": ["car-test"],
            "exposure_hours": [1.0],
        }
    )
    thresholds = pd.DataFrame(
        {
            "fold_test_circuit": ["Held"],
            "method": [PRIMARY_BAYESIAN_METHOD],
            "required_lead_s": [0.25],
            "false_budget_per_hour": [2.0],
            "threshold": [0.8],
            "threshold_source_partition": ["calibration"],
        }
    )
    evaluated = evaluate_heldout_operating_points(
        scores,
        events,
        exposure,
        thresholds,
        outer_fold="Held",
    )
    assert evaluated.metrics.iloc[0]["localized_event_recall"] == pytest.approx(1.0)
    assert evaluated.car_contributions.iloc[0]["localized_event_hits"] == 1
    assert evaluated.car_contributions.iloc[0]["qualified_events"] == 1
    assert evaluated.car_contributions.iloc[0]["exposure_hours"] == pytest.approx(1.0)
    assert {
        "correct_side",
        "exact_bin",
        "exact_segment",
        "within_segment_tolerance",
        "lead_seconds",
        "resolution_time_seconds",
        "resolution_type",
    }.issubset(evaluated.proposal_labels.columns)
    assert {
        "correct_side_event_recall",
        "within_segment_tolerance_event_recall",
        "exact_bin_event_recall",
        "abstention_rate",
        "false_proposals_per_hour_upper_95",
    }.issubset(evaluated.metrics.columns)
    assert not evaluated.delay_metrics.empty
    assert set(evaluated.delay_metrics["synthetic_delay_ms"]) == {0, 40, 80, 160}
    assert not evaluated.delay_car_contributions.empty
    metric = evaluated.metrics.iloc[0]
    reason_counts = json.loads(metric["abstention_reason_counts"])
    assert sum(reason_counts.values()) == metric["abstention_rows"]
    assert metric["least_favorable_false_proposals"] == (
        metric["false_proposals"] + metric["unresolved_proposals"]
    )
    contribution = evaluated.car_contributions.iloc[0]
    assert contribution["least_favorable_false_proposals"] == (
        contribution["false_proposals"] + contribution["unresolved_proposals"]
    )

    delayed = apply_synthetic_proposal_delays(evaluated.proposal_labels, delays_ms=(0, 40, 80, 160))
    assert set(delayed["synthetic_delay_ms"]) == {0, 40, 80, 160}
    original_lead = float(evaluated.proposal_labels.iloc[0]["lead_seconds"])
    delayed_160 = delayed.loc[delayed["synthetic_delay_ms"] == 160].iloc[0]
    assert delayed_160["effective_lead_seconds"] == pytest.approx(original_lead - 0.160)
    assert delayed_160["models_refit"] == False  # noqa: E712

    wrong_side_scores = scores.assign(predicted_side="right")
    wrong_side = evaluate_heldout_operating_points(
        wrong_side_scores,
        events,
        exposure,
        thresholds,
        outer_fold="Held",
    ).proposal_labels.iloc[0]
    assert bool(wrong_side["exact_bin"])
    assert not bool(wrong_side["exact_segment"])
    assert not bool(wrong_side["localized_hit"])

    stale = thresholds.assign(threshold_source_partition="test")
    with pytest.raises(DataValidationError, match="calibration-frozen"):
        evaluate_heldout_operating_points(
            scores,
            events,
            exposure,
            stale,
            outer_fold="Held",
        )


def test_unresolved_proposals_are_censored_primary_and_false_in_sensitivity() -> None:
    times = np.arange(0.0, 0.25, 0.05)
    scores = pd.DataFrame(
        {
            "fold_test_circuit": "Held",
            "partition": "test",
            "method": PRIMARY_BAYESIAN_METHOD,
            "circuit": "Held",
            "source_session_id": "s-test",
            "car_id": "car-test",
            "frame_index": np.arange(times.size),
            "time_seconds": times,
            "input_valid_causal": True,
            "in_corridor": True,
            "hard_break": False,
            "proposal_score": [0.0, 0.0, 0.9, 0.9, 0.9],
            "predicted_side": "left",
            "predicted_segment_bin": 0,
            "n_segment_bins": 4,
        }
    )
    events = pd.DataFrame(
        columns=[
            "candidate_event_id",
            "circuit",
            "car_id",
            "start_time_seconds",
            "side_at_onset",
            "segment_bin_25m_at_onset",
            "qualified",
        ]
    )
    exposure = pd.DataFrame(
        {
            "fold_test_circuit": ["Held"],
            "partition": ["test"],
            "circuit": ["Held"],
            "source_session_id": ["s-test"],
            "car_id": ["car-test"],
            "exposure_hours": [1.0],
        }
    )
    thresholds = pd.DataFrame(
        {
            "fold_test_circuit": ["Held"],
            "method": [PRIMARY_BAYESIAN_METHOD],
            "required_lead_s": [0.25],
            "false_budget_per_hour": [2.0],
            "threshold": [0.8],
            "threshold_source_partition": ["calibration"],
        }
    )
    result = evaluate_heldout_operating_points(
        scores, events, exposure, thresholds, outer_fold="Held"
    )
    row = result.metrics.iloc[0]
    assert row["unresolved_proposals"] == 1
    assert row["false_proposals"] == 0
    assert row["least_favorable_false_proposals"] == 1
    assert result.car_contributions.iloc[0]["least_favorable_false_proposals"] == 1

    no_proposals = evaluate_heldout_operating_points(
        scores.assign(proposal_score=0.0),
        events,
        exposure,
        thresholds,
        outer_fold="Held",
    )
    assert no_proposals.proposal_labels.empty
    assert {
        "correct_side",
        "exact_bin",
        "within_segment_tolerance",
        "lead_seconds",
        "resolution_time_seconds",
    }.issubset(no_proposals.proposal_labels.columns)


def test_l_star_uses_pooled_counts_at_registered_budget_and_returns_zero_when_none_pass() -> None:
    contributions = pd.DataFrame(
        {
            "circuit": ["A"] * 4,
            "source_session_id": ["s"] * 4,
            "car_id": ["car"] * 4,
            "method": ["brace"] * 4,
            "required_lead_s": [0.25, 0.5, 1.0, 1.5],
            "false_budget_per_hour": [2.0] * 4,
            "localized_event_hits": [6, 6, 5, 4],
            "qualified_events": [10] * 4,
            "false_proposals": [2, 2, 2, 2],
            "exposure_hours": [2.0] * 4,
            "operating_point_status": ["estimable"] * 4,
        }
    )
    assert l_star_from_contributions(contributions, method="brace") == pytest.approx(1.0)
    contributions["localized_event_hits"] = 4
    assert l_star_from_contributions(contributions, method="brace") == pytest.approx(0.0)


def test_delay_l_star_table_reports_primary_paired_delta_without_refitting() -> None:
    rows = []
    for delay, brace_max, twin_max in ((0, 1.0, 0.5), (160, 0.5, 0.25)):
        for method, maximum in (
            (PRIMARY_BAYESIAN_METHOD, brace_max),
            (PRIMARY_DETERMINISTIC_METHOD, twin_max),
        ):
            for lead in (0.25, 0.5, 1.0, 1.5):
                rows.append(
                    {
                        "circuit": "A",
                        "source_session_id": "s",
                        "car_id": "car",
                        "synthetic_delay_ms": delay,
                        "method": method,
                        "required_lead_s": lead,
                        "false_budget_per_hour": 2.0,
                        "localized_event_hits": int(lead <= maximum),
                        "qualified_events": 1,
                        "false_proposals": 0,
                        "least_favorable_false_proposals": 0,
                        "exposure_hours": 1.0,
                        "operating_point_status": "estimable",
                    }
                )
    sensitivity = synthetic_delay_l_star_table(pd.DataFrame(rows))
    assert sensitivity.set_index("synthetic_delay_ms")["delta_l_star_s"].to_dict() == {
        0: 0.5,
        160: 0.25,
    }
    assert "least_favorable_delta_l_star_s" in sensitivity


def test_l_star_and_horizon_coherence_sensitivities_are_explicit() -> None:
    rows: list[dict[str, object]] = []
    for method in (PRIMARY_BAYESIAN_METHOD, PRIMARY_DETERMINISTIC_METHOD):
        for lead in HORIZONS_S:
            rows.append(
                {
                    "circuit": "A",
                    "source_session_id": "s",
                    "car_id": "car",
                    "method": method,
                    "required_lead_s": lead,
                    "false_budget_per_hour": 2.0,
                    "localized_event_hits": int(lead <= 1.0),
                    "qualified_events": 1,
                    "false_proposals": 0,
                    "least_favorable_false_proposals": (
                        3 if method == PRIMARY_BAYESIAN_METHOD else 0
                    ),
                    "exposure_hours": 1.0,
                    "operating_point_status": "estimable",
                }
            )
    sensitivity = paired_l_star_sensitivity_table(pd.DataFrame(rows))
    by_analysis = sensitivity.set_index("analysis")
    assert by_analysis.loc["primary_unresolved_censored", "l_star_brace_s"] == 1.0
    assert by_analysis.loc["least_favorable_unresolved_counted_as_false", "l_star_brace_s"] == 0.0

    score = pd.DataFrame(
        {
            "partition": ["test", "test"],
            "circuit": ["A", "A"],
            "method": ["m", "m"],
            "exit_probability_0p25s": [0.1, 0.2],
            "exit_probability_0p50s": [0.2, 0.1],
            "exit_probability_1p00s": [0.3, 0.3],
            "exit_probability_1p50s": [0.4, 0.25],
        }
    )
    diagnostic = horizon_monotonicity_diagnostics(score)
    pooled = diagnostic.loc[diagnostic["scope"] == "pooled"].iloc[0]
    assert pooled["violating_row_count"] == 1
    assert pooled["adjacent_pair_violation_count"] == 2
    assert pooled["coherence_correction_applied"] == False  # noqa: E712


def test_only_the_canonical_frozen_study_config_is_accepted() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "study.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_frozen_config(config)
    config["threshold_tie_break"] = "lower_threshold"
    with pytest.raises(DataValidationError, match="canonical frozen study config"):
        _validate_frozen_config(config)


def test_resume_manifest_refuses_any_stale_content_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    run_identity = {"data": "d", "config": "c", "code": "k", "fold": "A"}
    manifest.write_text(
        json.dumps(
            {
                "input_content_hash": "abc",
                "run_identity": run_identity,
                "model_identity_hash": "model-a",
                "artifacts": {str(artifact): digest},
            }
        ),
        encoding="utf-8",
    )
    validate_resume_manifest(
        manifest,
        expected_input_content_hash="abc",
        expected_run_identity=run_identity,
        expected_model_identity_hash="model-a",
    )

    with pytest.raises(DataValidationError, match="run identity"):
        validate_resume_manifest(
            manifest,
            expected_input_content_hash="abc",
            expected_run_identity={**run_identity, "fold": "B"},
        )
    with pytest.raises(DataValidationError, match="model identity"):
        validate_resume_manifest(
            manifest,
            expected_input_content_hash="abc",
            expected_model_identity_hash="model-b",
        )

    artifact.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="stale resume artifact"):
        validate_resume_manifest(manifest, expected_input_content_hash="abc")
    with pytest.raises(DataValidationError, match="input content hash"):
        validate_resume_manifest(manifest, expected_input_content_hash="different")


def _pcd(points: list[tuple[float, float]]) -> str:
    body = "\n".join(f"{x} {y} 0" for x, y in points)
    return (
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA ascii\n{body}\n"
    )


def test_loader_validates_all_tables_pcd_hashes_and_four_key_accounting(tmp_path: Path) -> None:
    frames = _frame_rows()
    targets = _target_rows()
    events = pd.DataFrame(
        {
            "candidate_event_id": ["e1", "e2"],
            "circuit": ["A", "B"],
            "car_id": ["a", "b"],
            "start_time_seconds": [0.2, 0.4],
            "side_at_onset": ["left", "right"],
            "segment_bin_25m_at_onset": [0, 0],
            "qualified": [True, True],
        }
    )
    timing = pd.DataFrame(
        {
            "circuit": ["A", "B"],
            "source_session_id": ["sa", "sb"],
            "car_id": ["a", "b"],
            "source_revision": ["r", "r"],
            "source_unit_sha256": ["a" * 64, "b" * 64],
            "buffered_exposure_seconds": [1.0, 1.0],
        }
    )
    paths = {
        "frames": tmp_path / "frames.parquet",
        "targets": tmp_path / "targets.parquet",
        "events": tmp_path / "events.parquet",
        "timing": tmp_path / "timing.csv",
        "splits": tmp_path / "splits.csv",
        "source_manifest": tmp_path / "files.csv",
    }
    frames.to_parquet(paths["frames"], index=False)
    targets.to_parquet(paths["targets"], index=False)
    events.to_parquet(paths["events"], index=False)
    timing.to_csv(paths["timing"], index=False)
    fold_a = _split_rows().assign(
        fold_test_circuit="A",
        partition=["test", "fit"],
    )
    pd.concat([fold_a, _split_rows()], ignore_index=True).to_csv(paths["splits"], index=False)

    manifest_rows: list[dict[str, object]] = []
    center = [(-7.0, -7.0), (7.0, -7.0), (7.0, 7.0), (-7.0, 7.0)]
    inner = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
    outer = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]
    for circuit in ("A", "B"):
        for filename, points in (
            ("center_line.pcd", center),
            ("inner_boundary.pcd", inner),
            ("outer_boundary.pcd", outer),
        ):
            path = tmp_path / f"{circuit}-{filename}"
            path.write_text(_pcd(points), encoding="ascii")
            manifest_rows.append(
                {
                    "circuit": circuit,
                    "role": "boundary",
                    "filename": filename,
                    "local_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(paths["source_manifest"], index=False)

    build_manifest = tmp_path / "deepracing-build.json"
    build_manifest.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in paths.items()
                    if name in {"frames", "targets", "events", "timing", "splits"}
                ],
                "input_manifest": {
                    "path": str(paths["source_manifest"]),
                    "bytes": paths["source_manifest"].stat().st_size,
                    "sha256": hashlib.sha256(paths["source_manifest"].read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    paths["build_manifest"] = build_manifest

    loaded = load_experiment_data(ExperimentPaths(**paths), base_dir=tmp_path)
    assert len(loaded.frame_targets) == 4
    assert set(loaded.corridors) == {"A", "B"}
    assert len(loaded.input_content_hash) == 64
    assert loaded.build_manifest_hash == hashlib.sha256(build_manifest.read_bytes()).hexdigest()

    targets.iloc[:-1].to_parquet(paths["targets"], index=False)
    with pytest.raises(DataValidationError, match="differs from build manifest"):
        load_experiment_data(ExperimentPaths(**paths), base_dir=tmp_path)


def test_hazard_scaling_arrays_are_defensively_copied_and_read_only() -> None:
    design = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    model = ScaledSideHazard.fit(
        design,
        np.asarray(["no_exit", "left", "right", "no_exit", "left", "right"]),
        feature_names=("x", "y"),
    )
    assert not model.feature_center.flags.writeable
    assert not model.feature_scale.flags.writeable
    with pytest.raises(ValueError):
        model.feature_center[0] = 99.0


def test_fold_cli_declares_authoritative_build_manifest_and_resumable_stage() -> None:
    args = _parser().parse_args(["run-fold", "--fold", "Bahrain", "--resume"])
    assert args.build_manifest == Path("data/manifests/deepracing-build.json")
    assert args.fold == "Bahrain"
    assert args.resume is True


def _registered_synthetic_data() -> tuple[ExperimentData, dict[str, object]]:
    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "study.json").read_text(encoding="utf-8")
    )
    circuits = tuple(config["cohort"]["circuits"])
    counts = dict(zip(circuits, (15, 15, 14, 14), strict=True))
    frame_rows: list[dict[str, object]] = []
    for circuit_index, circuit in enumerate(circuits):
        for car_index in range(counts[circuit]):
            frame_rows.append(
                {
                    "circuit": circuit,
                    "source_session_id": f"session-{circuit}",
                    "car_id": f"car-{circuit_index}-{car_index}",
                    "source_revision": "r1",
                    "source_unit_sha256": f"{circuit_index + 1:x}" * 64,
                    "frame_index": 0,
                    "time_seconds": 0.0,
                    "input_valid_causal": True,
                    "in_corridor": True,
                    "hard_break": True,
                }
            )
    frames = pd.DataFrame(frame_rows)
    targets = frames.loc[:, KEYS].copy()
    targets["next_qualifying_event_id"] = pd.Series([pd.NA] * len(targets), dtype="string")
    targets["time_to_next_excursion_seconds"] = np.nan
    for suffix in ("0p25", "0p50", "1p00", "1p50"):
        targets[f"outcome_evaluable_{suffix}s"] = True
    timing = frames.loc[
        :,
        [
            "circuit",
            "source_session_id",
            "car_id",
            "source_revision",
            "source_unit_sha256",
        ],
    ].copy()
    timing["buffered_exposure_seconds"] = 3600.0
    split_rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(circuits):
        calibration_circuit = circuits[(fold_index + 1) % len(circuits)]
        for row in timing.itertuples(index=False):
            partition = (
                "test"
                if row.circuit == fold
                else "calibration"
                if row.circuit == calibration_circuit
                else "fit"
            )
            split_rows.append(
                {
                    "fold_test_circuit": fold,
                    **row._asdict(),
                    "partition": partition,
                }
            )
    splits = pd.DataFrame(split_rows)
    events = pd.DataFrame(
        columns=[
            "candidate_event_id",
            "circuit",
            "car_id",
            "start_time_seconds",
            "side_at_onset",
            "segment_bin_25m_at_onset",
            "qualified",
        ]
    )
    data = ExperimentData(
        frames=frames,
        targets=targets,
        frame_targets=frames.merge(targets, on=KEYS, validate="one_to_one"),
        events=events,
        timing=timing,
        splits=splits,
        corridors={circuit: _square_ring() for circuit in circuits},
        input_content_hash="synthetic-data-hash",
        source_manifest_hash="synthetic-source-manifest-hash",
        build_manifest_hash="synthetic-build-manifest-hash",
    )
    return data, config


def _write_synthetic_sealed_folds(
    data: ExperimentData,
    config: dict[str, object],
    root: Path,
    *,
    circuits: tuple[str, ...] | None = None,
) -> None:
    selected_circuits = tuple(data.corridors) if circuits is None else circuits
    budgets = (0.5, 1.0, 2.0, 5.0, 10.0)
    for circuit in selected_circuits:
        fold_dir = root / f"fold={circuit}"
        fold_dir.mkdir(parents=True)
        test_units = data.splits.loc[
            (data.splits["fold_test_circuit"] == circuit) & (data.splits["partition"] == "test")
        ]
        frame_keys = data.frames.merge(
            test_units.loc[
                :,
                [
                    "circuit",
                    "source_session_id",
                    "car_id",
                    "source_revision",
                    "source_unit_sha256",
                ],
            ],
            on=[
                "circuit",
                "source_session_id",
                "car_id",
                "source_revision",
                "source_unit_sha256",
            ],
            how="inner",
            validate="many_to_one",
        )
        score_rows: list[pd.DataFrame] = []
        for method in REGISTERED_METHODS:
            score = frame_keys.loc[
                :,
                [
                    "circuit",
                    "source_session_id",
                    "car_id",
                    "frame_index",
                    "time_seconds",
                    "input_valid_causal",
                    "in_corridor",
                    "hard_break",
                ],
            ].copy()
            score["fold_test_circuit"] = circuit
            score["partition"] = "test"
            score["method"] = method
            score["proposal_score"] = 0.0
            score["predicted_side"] = "left"
            score["predicted_segment_bin"] = 0
            score["n_segment_bins"] = 4
            for horizon in HORIZONS_S:
                suffix = f"{horizon:.2f}".replace(".", "p")
                score[f"exit_probability_{suffix}s"] = 0.1
            score_rows.append(score)
        compact_scores = pd.concat(score_rows, ignore_index=True)
        compact_path = fold_dir / "compact-calibrated-scores.parquet"
        compact_scores.to_parquet(compact_path, index=False)
        threshold_rows = [
            {
                "fold_test_circuit": circuit,
                "method": method,
                "required_lead_s": lead,
                "false_budget_per_hour": budget,
                "threshold": 1.0,
                "calibration_localized_event_recall": 0.0,
                "calibration_false_proposals_per_hour": 0.0,
                "calibration_false_proposals": 0,
                "calibration_qualified_events": 0,
                "calibration_proposal_count": 0,
                "calibration_exposure_hours": 2.0,
                "calibration_false_count_capacity": budget * 2.0,
                "minimum_false_count_capacity": 1.0,
                "operating_point_status": "estimable",
                "estimability_reason": "",
                "threshold_candidate_count": 1,
                "threshold_source_partition": "calibration",
            }
            for method in REGISTERED_METHODS
            for lead in HORIZONS_S
            for budget in budgets
        ]
        threshold_path = fold_dir / "calibration-thresholds.csv"
        pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)
        calibrator_path = fold_dir / "calibrators.json"
        calibrator_path.write_text(
            json.dumps(
                {
                    "fit_partition": "calibration",
                    "calibrators": [],
                    "reference_prevalence": [
                        {
                            "fold_test_circuit": circuit,
                            "method": method,
                            "horizon_s": horizon,
                            "reference_prevalence": 0.1,
                            "calibration_evaluable_rows": 1,
                            "calibration_events": 0,
                        }
                        for method in REGISTERED_METHODS
                        for horizon in HORIZONS_S
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata_path = fold_dir / "model-metadata.json"
        metadata_path.write_text("{}\n", encoding="utf-8")
        frozen_paths = [metadata_path, compact_path, calibrator_path, threshold_path]
        for method in REGISTERED_METHODS:
            raw_path = fold_dir / f"raw-scores-{method}.parquet"
            compact_scores.loc[compact_scores["method"] == method].to_parquet(raw_path, index=False)
            frozen_paths.append(raw_path)
        run_identity = _run_identity(data, config, outer_fold=circuit)
        manifest = {
            "schema_version": 1,
            "fold_test_circuit": circuit,
            "input_content_hash": _canonical_json_hash(run_identity),
            "data_content_hash": data.input_content_hash,
            "source_manifest_hash": data.source_manifest_hash,
            "build_manifest_hash": data.build_manifest_hash,
            "config_content_hash": _canonical_json_hash(config),
            "code_content_hash": _code_content_hash(),
            "run_identity": run_identity,
            "model_identity_hash": f"model-{circuit}",
            "method_identities": {method: f"{method}-{circuit}" for method in REGISTERED_METHODS},
            "status": "thresholds_complete",
            "completed_methods": list(REGISTERED_METHODS),
            "artifacts": {str(path.resolve()): _sha256(path) for path in frozen_paths},
        }
        (fold_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_threshold_seal_heldout_transition_and_pooled_aggregation_are_immutable(
    tmp_path: Path,
) -> None:
    data, config = _registered_synthetic_data()
    _write_synthetic_sealed_folds(data, config, tmp_path)
    seal_path = create_threshold_freeze_seal(data, config=config, output_dir=tmp_path)
    validate_threshold_freeze_seal(data, config=config, output_dir=tmp_path)
    assert seal_path.is_file()

    first_fold = str(config["cohort"]["circuits"][0])
    first_dir = tmp_path / f"fold={first_fold}"
    frozen_names = [
        "model-metadata.json",
        "compact-calibrated-scores.parquet",
        "calibrators.json",
        "calibration-thresholds.csv",
        *(f"raw-scores-{method}.parquet" for method in REGISTERED_METHODS),
    ]
    before = {name: _sha256(first_dir / name) for name in frozen_names}
    resumed_thresholds = run_outer_fold(
        data,
        outer_fold=first_fold,
        config=config,
        output_dir=tmp_path,
        resume=True,
        evaluate_heldout=False,
    )
    assert resumed_thresholds.status == "thresholds_complete"
    assert before == {name: _sha256(first_dir / name) for name in frozen_names}
    (first_dir / "heldout-operating-metrics.csv").write_text("interrupted\n", encoding="utf-8")
    first_result = run_outer_fold(
        data,
        outer_fold=first_fold,
        config=config,
        output_dir=tmp_path,
        resume=True,
        evaluate_heldout=True,
    )
    assert first_result.status == "heldout_complete"
    assert before == {name: _sha256(first_dir / name) for name in frozen_names}
    first_manifest = json.loads(first_result.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["heldout_consumed_sealed_artifacts"] is True
    assert first_manifest["models_refit_for_heldout"] is False

    for circuit in config["cohort"]["circuits"][1:]:
        run_outer_fold(
            data,
            outer_fold=str(circuit),
            config=config,
            output_dir=tmp_path,
            resume=True,
            evaluate_heldout=True,
        )
    pooled = aggregate_completed_heldout_folds(data, config=config, output_dir=tmp_path)
    assert pooled.status == "pooled_heldout_complete"
    pooled_manifest = json.loads(pooled.manifest_path.read_text(encoding="utf-8"))
    assert pooled_manifest["threshold_freeze_seal"]["sha256"] == _sha256(seal_path)
    contributions = pd.read_parquet(
        pooled.manifest_path.parent / "pooled-heldout-car-contributions.parquet"
    )
    assert (
        contributions[["circuit", "source_session_id", "car_id"]].drop_duplicates().shape[0] == 58
    )
    runtime = json.loads(
        (pooled.manifest_path.parent / "runtime.jsonl").read_text(encoding="utf-8")
    )
    assert runtime["elapsed_seconds"] > 0.0


def test_threshold_seal_fails_closed_on_missing_fold_and_bad_capacity(
    tmp_path: Path,
) -> None:
    data, config = _registered_synthetic_data()
    circuits = tuple(str(value) for value in config["cohort"]["circuits"])
    _write_synthetic_sealed_folds(data, config, tmp_path, circuits=circuits[:-1])
    with pytest.raises(DataValidationError, match="manifest"):
        create_threshold_freeze_seal(data, config=config, output_dir=tmp_path)

    _write_synthetic_sealed_folds(data, config, tmp_path, circuits=(circuits[-1],))
    threshold_path = tmp_path / f"fold={circuits[0]}" / "calibration-thresholds.csv"
    thresholds = pd.read_csv(threshold_path)
    thresholds.loc[0, "calibration_false_count_capacity"] = 999.0
    thresholds.to_csv(threshold_path, index=False)
    manifest_path = threshold_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][str(threshold_path.resolve())] = _sha256(threshold_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(DataValidationError, match="capacity accounting"):
        create_threshold_freeze_seal(data, config=config, output_dir=tmp_path)
