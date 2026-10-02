from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brace_f1.bootstrap import (
    main,
    paired_car_session_bootstrap,
    paired_two_stage_circuit_car_bootstrap,
    reliability_bin_cluster_bootstrap,
    within_circuit_car_weights,
)
from brace_f1.experiment import (
    HORIZONS_S,
    PRIMARY_BAYESIAN_METHOD,
    PRIMARY_DETERMINISTIC_METHOD,
    REGISTERED_METHODS,
    _canonical_json_hash,
    _code_content_hash,
)
from brace_f1.io import DataValidationError


def _contributions(*, identical: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for circuit in ("A", "B"):
        for car_index in range(3):
            for method in (PRIMARY_BAYESIAN_METHOD, PRIMARY_DETERMINISTIC_METHOD):
                for lead in (0.25, 0.5, 1.0, 1.5):
                    if identical:
                        hit = int(lead <= 0.5 and car_index < 2)
                    elif method == PRIMARY_BAYESIAN_METHOD:
                        hit = int(lead <= 1.0 and car_index < 2)
                    else:
                        hit = int(lead <= 0.5 and car_index < 2)
                    rows.append(
                        {
                            "fold_test_circuit": circuit,
                            "circuit": circuit,
                            "source_session_id": f"session-{circuit}",
                            "car_id": f"car-{car_index}",
                            "method": method,
                            "required_lead_s": lead,
                            "false_budget_per_hour": 2.0,
                            "localized_event_hits": hit,
                            "qualified_events": 1,
                            "false_proposals": 0,
                            "unresolved_proposals": 0,
                            "least_favorable_false_proposals": 0,
                            "exposure_hours": 1.0,
                            "fixed_model_and_threshold": True,
                            "operating_point_status": "estimable",
                        }
                    )
    return pd.DataFrame(rows)


def test_within_circuit_resampling_returns_duplicate_cluster_integer_weights() -> None:
    units = (
        _contributions()
        .loc[:, ["circuit", "source_session_id", "car_id"]]
        .drop_duplicates(ignore_index=True)
    )
    weights = within_circuit_car_weights(units, rng=np.random.default_rng(4))
    assert weights.shape == (6,)
    assert np.issubdtype(weights.dtype, np.integer)
    assert weights[:3].sum() == 3
    assert weights[3:].sum() == 3
    assert np.any(weights > 1)

    non_range_index = units.copy()
    non_range_index.index = np.arange(20, 20 + len(non_range_index)) * 3
    shifted = within_circuit_car_weights(non_range_index, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(shifted, weights)


def test_paired_bootstrap_reuses_identical_car_draws_across_methods() -> None:
    result = paired_car_session_bootstrap(
        _contributions(identical=True),
        n_resamples=200,
        seed=17,
    )
    np.testing.assert_array_equal(
        result.delta_l_star_draws,
        result.l_star_brace_draws - result.l_star_comparator_draws,
    )
    np.testing.assert_array_equal(
        result.l_star_brace_draws,
        result.l_star_comparator_draws,
    )
    np.testing.assert_array_equal(result.delta_l_star_draws, 0.0)
    assert result.delta_l_star == pytest.approx(0.0)
    assert result.percentile_95 == (0.0, 0.0)
    assert result.cluster_level == "car_session_within_circuit"


def test_paired_bootstrap_estimates_delta_l_star_and_percentile_interval() -> None:
    result = paired_car_session_bootstrap(
        _contributions(),
        n_resamples=300,
        seed=23,
    )
    assert result.l_star_brace == pytest.approx(1.0)
    assert result.l_star_comparator == pytest.approx(0.5)
    assert result.delta_l_star == pytest.approx(0.5)
    assert result.l_star_brace_draws.shape == (300,)
    assert result.l_star_comparator_draws.shape == (300,)
    assert result.delta_l_star_draws.shape == (300,)
    np.testing.assert_array_equal(
        result.delta_l_star_draws,
        result.l_star_brace_draws - result.l_star_comparator_draws,
    )
    assert (
        result.l_star_brace_percentile_95[0]
        <= result.l_star_brace
        <= (result.l_star_brace_percentile_95[1])
    )
    assert (
        result.l_star_comparator_percentile_95[0]
        <= result.l_star_comparator
        <= (result.l_star_comparator_percentile_95[1])
    )
    assert result.percentile_95[0] <= result.delta_l_star <= result.percentile_95[1]
    assert result.models_and_thresholds_refit is False

    least_favorable = _contributions()
    least_favorable.loc[
        least_favorable["method"] == PRIMARY_BAYESIAN_METHOD,
        ["unresolved_proposals", "least_favorable_false_proposals"],
    ] = 3
    sensitivity = paired_car_session_bootstrap(
        least_favorable,
        n_resamples=50,
        seed=23,
        false_count_column="least_favorable_false_proposals",
    )
    assert sensitivity.l_star_brace == 0.0


def test_two_stage_circuit_car_sensitivity_is_paired_and_not_frame_iid() -> None:
    result = paired_two_stage_circuit_car_bootstrap(
        _contributions(identical=True),
        n_resamples=150,
        seed=31,
    )
    np.testing.assert_array_equal(result.delta_l_star_draws, 0.0)
    assert result.cluster_level == "circuit_then_car_session"

    duplicate = pd.concat([_contributions(), _contributions().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="one contribution per car-session"):
        paired_car_session_bootstrap(duplicate, n_resamples=10, seed=1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table.assign(fixed_model_and_threshold="False"), "boolean"),
        (lambda table: table.assign(localized_event_hits=0.5), "integer counts"),
        (lambda table: table.assign(false_proposals=-1), "non-negative"),
        (
            lambda table: table.assign(localized_event_hits=table["qualified_events"] + 1),
            "cannot exceed",
        ),
        (lambda table: table.assign(exposure_hours=np.inf), "finite"),
    ],
)
def test_bootstrap_rejects_invalid_fixed_cluster_contributions(mutation, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        paired_car_session_bootstrap(mutation(_contributions()), n_resamples=10, seed=1)


def test_bootstrap_rejects_absent_or_nearby_budget_and_method() -> None:
    with pytest.raises(DataValidationError, match="requested methods and budget"):
        paired_car_session_bootstrap(
            _contributions(), n_resamples=10, seed=1, brace_method="missing"
        )

    near_budget = pd.concat(
        [
            _contributions(),
            _contributions().assign(false_budget_per_hour=2.0 + 5e-13),
        ],
        ignore_index=True,
    )
    with pytest.raises(DataValidationError, match="one source row"):
        paired_car_session_bootstrap(near_budget, n_resamples=10, seed=1)

    near_lead = pd.concat(
        [
            _contributions(),
            _contributions()
            .loc[lambda table: table["required_lead_s"] == 0.25]
            .assign(required_lead_s=0.25 + 5e-13),
        ],
        ignore_index=True,
    )
    with pytest.raises(DataValidationError, match="one source row"):
        paired_car_session_bootstrap(near_lead, n_resamples=10, seed=1)


@pytest.mark.parametrize(("column", "value"), [("circuit", np.nan), ("car_id", " ")])
def test_bootstrap_rejects_null_or_blank_cluster_keys(column: str, value) -> None:
    invalid = _contributions()
    invalid.loc[0, column] = value
    with pytest.raises(DataValidationError, match="cluster keys"):
        paired_car_session_bootstrap(invalid, n_resamples=10, seed=1)


def test_bootstrap_rejects_rows_not_from_their_declared_heldout_circuit() -> None:
    invalid = _contributions()
    invalid.loc[0, "fold_test_circuit"] = "B"
    with pytest.raises(DataValidationError, match="held-out circuit"):
        paired_car_session_bootstrap(invalid, n_resamples=10, seed=1)


def test_two_stage_bootstrap_requires_multiple_circuits() -> None:
    one_circuit = _contributions().loc[lambda table: table["circuit"] == "A"]
    with pytest.raises(DataValidationError, match="at least two circuits"):
        paired_two_stage_circuit_car_bootstrap(one_circuit, n_resamples=10, seed=1)
    with pytest.raises(DataValidationError, match="requested methods and budget"):
        paired_car_session_bootstrap(
            _contributions().assign(false_budget_per_hour=2.0 + 1e-9),
            n_resamples=10,
            seed=1,
            false_budget_per_hour=2.0,
        )


def _reliability_contributions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for circuit in ("A", "B"):
        for car_index in range(2):
            for method in ("m1", "m2"):
                for bin_index in range(2):
                    count = 0 if bin_index == 1 else 2
                    rows.append(
                        {
                            "fold_test_circuit": circuit,
                            "circuit": circuit,
                            "source_session_id": f"s-{circuit}",
                            "car_id": f"car-{car_index}",
                            "method": method,
                            "horizon_s": 0.5,
                            "bin_index": bin_index,
                            "count": count,
                            "probability_sum": count * 0.2,
                            "event_count": count // 2,
                        }
                    )
    return pd.DataFrame(rows)


def test_reliability_bootstrap_uses_paired_fixed_bin_car_clusters() -> None:
    source = _reliability_contributions()
    first = reliability_bin_cluster_bootstrap(
        source,
        n_resamples=100,
        seed=9,
        horizons_s=(0.5,),
        n_bins=2,
        expected_methods=("m1", "m2"),
    )
    second = reliability_bin_cluster_bootstrap(
        source,
        n_resamples=100,
        seed=9,
        horizons_s=(0.5,),
        n_bins=2,
        expected_methods=("m1", "m2"),
    )
    pd.testing.assert_frame_equal(first.summary, second.summary)
    nonempty = first.summary.loc[first.summary["bin_index"] == 0]
    np.testing.assert_allclose(nonempty["mean_probability"], 0.2)
    np.testing.assert_allclose(nonempty["observed_frequency"], 0.5)
    empty = first.summary.loc[first.summary["bin_index"] == 1]
    assert empty["mean_probability"].isna().all()
    assert empty["observed_frequency_lower_95"].isna().all()
    assert (first.summary["circuit_count"] == 2).all()
    assert not first.summary["transport_uncertainty_included"].any()


def test_reliability_bootstrap_rejects_inconsistent_support_and_bin_mass() -> None:
    inconsistent_support = _reliability_contributions()
    selector = (
        (inconsistent_support["circuit"] == "A")
        & (inconsistent_support["car_id"] == "car-0")
        & (inconsistent_support["method"] == "m2")
        & (inconsistent_support["bin_index"] == 0)
    )
    inconsistent_support.loc[selector, "count"] = 3
    with pytest.raises(DataValidationError, match="frame and event support"):
        reliability_bin_cluster_bootstrap(
            inconsistent_support,
            n_resamples=5,
            seed=9,
            horizons_s=(0.5,),
            n_bins=2,
            expected_methods=("m1", "m2"),
        )

    invalid_bin_mass = _reliability_contributions()
    invalid_bin_mass.loc[invalid_bin_mass["bin_index"] == 0, "probability_sum"] = 2.0
    with pytest.raises(DataValidationError, match="assigned bin bounds"):
        reliability_bin_cluster_bootstrap(
            invalid_bin_mass,
            n_resamples=5,
            seed=9,
            horizons_s=(0.5,),
            n_bins=2,
            expected_methods=("m1", "m2"),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"false_budget_per_hour": -1.0}, "false budget"),
        ({"false_budget_per_hour": np.nan}, "false budget"),
        ({"minimum_recall": 1.1}, "minimum recall"),
        ({"lead_grid_s": (0.25, np.nan)}, "lead grid"),
    ],
)
def test_bootstrap_validates_registered_operating_parameters(kwargs, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        paired_car_session_bootstrap(_contributions(), n_resamples=10, seed=1, **kwargs)


def test_least_favorable_bootstrap_requires_unresolved_proposal_counts() -> None:
    missing = _contributions().drop(columns="unresolved_proposals")
    with pytest.raises(DataValidationError, match="unresolved_proposals"):
        paired_car_session_bootstrap(
            missing,
            n_resamples=10,
            seed=1,
            false_count_column="least_favorable_false_proposals",
        )


@pytest.mark.parametrize(
    ("unresolved", "least_favorable", "message"),
    [
        (1.0, 0.0, "false_proposals \\+ unresolved_proposals"),
        (-1.0, -1.0, "non-negative integer"),
        (0.5, 0.5, "non-negative integer"),
    ],
)
def test_least_favorable_bootstrap_validates_rowwise_count_identity(
    unresolved: float,
    least_favorable: float,
    message: str,
) -> None:
    invalid = _contributions()
    invalid[["unresolved_proposals", "least_favorable_false_proposals"]] = invalid[
        ["unresolved_proposals", "least_favorable_false_proposals"]
    ].astype(float)
    invalid.loc[0, "unresolved_proposals"] = unresolved
    invalid.loc[0, "least_favorable_false_proposals"] = least_favorable
    with pytest.raises(DataValidationError, match=message):
        paired_car_session_bootstrap(
            invalid,
            n_resamples=10,
            seed=1,
            false_count_column="least_favorable_false_proposals",
        )


def test_least_favorable_bootstrap_rejects_boolean_counts() -> None:
    invalid = _contributions()
    invalid["unresolved_proposals"] = invalid["unresolved_proposals"].astype(object)
    invalid.loc[0, "unresolved_proposals"] = True
    invalid.loc[0, "least_favorable_false_proposals"] = 1
    with pytest.raises(DataValidationError, match="non-negative integer"):
        paired_car_session_bootstrap(
            invalid,
            n_resamples=10,
            seed=1,
            false_count_column="least_favorable_false_proposals",
        )


def test_bootstrap_cli_persists_draws_and_content_hashed_manifest(tmp_path) -> None:
    contribution_path = tmp_path / "contributions.csv"
    output_path = tmp_path / "bootstrap-bundle"
    _contributions().to_csv(contribution_path, index=False)

    assert (
        main(
            [
                str(contribution_path),
                "--resamples",
                "10",
                "--seed",
                "4",
                "--output",
                str(output_path),
                "--allow-noncanonical-synthetic",
            ]
        )
        == 0
    )
    summary_path = output_path / "summary.json"
    draws_path = output_path / "delta-l-star-draws.npy"
    brace_draws_path = output_path / "brace-l-star-draws.npy"
    comparator_draws_path = output_path / "comparator-l-star-draws.npy"
    manifest_path = output_path / "manifest.json"
    runtime_path = output_path / "runtime.jsonl"
    sensitivity_draws_path = output_path / "least-favorable-delta-l-star-draws.npy"
    sensitivity_brace_draws_path = output_path / "least-favorable-brace-l-star-draws.npy"
    sensitivity_comparator_draws_path = output_path / "least-favorable-comparator-l-star-draws.npy"
    assert output_path.is_dir()
    assert summary_path.is_file()
    assert draws_path.is_file()
    assert brace_draws_path.is_file()
    assert comparator_draws_path.is_file()
    assert manifest_path.is_file()
    assert runtime_path.is_file()
    assert sensitivity_draws_path.is_file()
    assert sensitivity_brace_draws_path.is_file()
    assert sensitivity_comparator_draws_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "l_star_brace_percentile_95",
        "l_star_comparator_percentile_95",
        "delta_l_star_percentile_95",
    ):
        assert len(summary[key]) == 2
        assert len(summary["least_favorable_unresolved_counted_as_false"][key]) == 2
    np.testing.assert_array_equal(
        np.load(draws_path),
        np.load(brace_draws_path) - np.load(comparator_draws_path),
    )
    np.testing.assert_array_equal(
        np.load(sensitivity_draws_path),
        np.load(sensitivity_brace_draws_path) - np.load(sensitivity_comparator_draws_path),
    )
    payload = pd.read_json(manifest_path, typ="series")
    assert payload["input"]["path"] == str(contribution_path.resolve())
    assert set(payload["artifacts"]) == {
        str(summary_path.resolve()),
        str(draws_path.resolve()),
        str(brace_draws_path.resolve()),
        str(comparator_draws_path.resolve()),
        str(runtime_path.resolve()),
        str(sensitivity_draws_path.resolve()),
        str(sensitivity_brace_draws_path.resolve()),
        str(sensitivity_comparator_draws_path.resolve()),
    }
    for artifact_path, expected_hash in payload["artifacts"].items():
        actual_hash = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
    assert payload["estimand"]["brace_method"] == PRIMARY_BAYESIAN_METHOD
    assert payload["estimand"]["comparator_method"] == PRIMARY_DETERMINISTIC_METHOD
    assert payload["estimand"]["false_budget_per_hour"] == 2.0
    assert payload["estimand"]["minimum_recall"] == 0.5
    assert payload["estimand"]["lead_grid_s"] == [0.25, 0.5, 1.0, 1.5]
    assert payload["sample"]["circuit_count"] == 2
    assert payload["sample"]["car_session_count"] == 6
    assert payload["sample"]["circuits"] == ["A", "B"]
    assert payload["noncanonical_synthetic"] is True
    assert payload["interval"]["construction"] == "percentile_95"
    assert len(payload["code_content_hash"]) == 64


def test_bootstrap_cli_requires_artifact_output_path() -> None:
    with pytest.raises(DataValidationError, match="requires --output"):
        main(["missing.csv"])


def _canonical_bootstrap_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "study.json").read_text(encoding="utf-8")
    )
    circuits = tuple(config["cohort"]["circuits"])
    counts = dict(zip(circuits, (15, 15, 14, 14), strict=True))
    contribution_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    for circuit_index, circuit in enumerate(circuits):
        for car_index in range(counts[circuit]):
            unit = {
                "fold_test_circuit": circuit,
                "circuit": circuit,
                "source_session_id": f"session-{circuit}",
                "car_id": f"car-{circuit_index}-{car_index}",
            }
            for method in (PRIMARY_BAYESIAN_METHOD, PRIMARY_DETERMINISTIC_METHOD):
                for lead in HORIZONS_S:
                    contribution_rows.append(
                        {
                            **unit,
                            "method": method,
                            "required_lead_s": lead,
                            "false_budget_per_hour": 2.0,
                            "localized_event_hits": int(lead <= 1.0),
                            "qualified_events": 1,
                            "false_proposals": 0,
                            "unresolved_proposals": 0,
                            "least_favorable_false_proposals": 0,
                            "exposure_hours": 1.0,
                            "fixed_model_and_threshold": True,
                            "operating_point_status": "estimable",
                        }
                    )
            for method in REGISTERED_METHODS:
                for horizon in HORIZONS_S:
                    for bin_index in range(10):
                        reliability_rows.append(
                            {
                                **unit,
                                "method": method,
                                "horizon_s": horizon,
                                "bin_index": bin_index,
                                "count": int(bin_index == 1),
                                "probability_sum": 0.15 if bin_index == 1 else 0.0,
                                "event_count": 0,
                            }
                        )
    contribution_path = tmp_path / "pooled-heldout-car-contributions.parquet"
    reliability_path = tmp_path / "pooled-heldout-reliability-cluster-contributions.parquet"
    pd.DataFrame(contribution_rows).to_parquet(contribution_path, index=False)
    pd.DataFrame(reliability_rows).to_parquet(reliability_path, index=False)
    artifacts = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (contribution_path, reliability_path)
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "pooled_heldout_complete",
                "config_content_hash": _canonical_json_hash(config),
                "code_content_hash": _code_content_hash(),
                "artifacts": artifacts,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return contribution_path, manifest_path


def test_production_bootstrap_authenticates_manifest_and_writes_atomic_bundle(
    tmp_path: Path,
) -> None:
    contribution_path, manifest_path = _canonical_bootstrap_inputs(tmp_path)
    output_dir = tmp_path / "production-bootstrap"
    assert (
        main(
            [
                str(contribution_path),
                "--output",
                str(output_dir),
                "--experiment-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["noncanonical_synthetic"] is False
    assert payload["experiment_manifest"]["path"] == str(manifest_path.resolve())
    assert payload["reliability_bootstrap"]["fixed_equal_width_bins"] == 10
    assert (output_dir / "reliability-bin-intervals.csv").is_file()
    assert (output_dir / "least-favorable-delta-l-star-draws.npy").is_file()
    assert (output_dir / "least-favorable-brace-l-star-draws.npy").is_file()
    assert (output_dir / "least-favorable-comparator-l-star-draws.npy").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "l_star_brace_percentile_95" in summary
    assert "l_star_comparator_percentile_95" in summary
    assert "delta_l_star_percentile_95" in summary
    sensitivity = summary["least_favorable_unresolved_counted_as_false"]
    assert "l_star_brace_percentile_95" in sensitivity
    assert "l_star_comparator_percentile_95" in sensitivity
    assert "delta_l_star_percentile_95" in sensitivity
    for artifact_path, expected_hash in payload["artifacts"].items():
        assert hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest() == expected_hash

    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale["code_content_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="current analysis code"):
        main(
            [
                str(contribution_path),
                "--output",
                str(tmp_path / "must-not-exist"),
                "--experiment-manifest",
                str(manifest_path),
            ]
        )


@pytest.mark.parametrize(
    "override",
    [
        ("--resamples", "9999"),
        ("--seed", "17"),
        ("--false-budget", "1.0"),
        ("--minimum-recall", "0.6"),
        ("--lead-grid", "0.25", "0.5", "1.0"),
        ("--brace-method", "noncanonical_bayes"),
        ("--comparator-method", "noncanonical_twin"),
    ],
)
def test_production_bootstrap_rejects_noncanonical_analysis_overrides(
    tmp_path: Path,
    override: tuple[str, ...],
) -> None:
    contribution_path, manifest_path = _canonical_bootstrap_inputs(tmp_path)
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(
        DataValidationError,
        match="canonical bootstrap.*--allow-noncanonical-synthetic",
    ):
        main(
            [
                str(contribution_path),
                "--output",
                str(output_dir),
                "--experiment-manifest",
                str(manifest_path),
                *override,
            ]
        )
    assert not output_dir.exists()


def test_noncanonical_escape_hatch_records_custom_analysis_contract(tmp_path: Path) -> None:
    contributions = _contributions().copy()
    contributions["false_budget_per_hour"] = 3.0
    contributions["required_lead_s"] = contributions["required_lead_s"].replace(
        {0.25: 0.2, 0.5: 0.4, 1.0: 0.8, 1.5: 1.2}
    )
    contributions["method"] = contributions["method"].replace(
        {
            PRIMARY_BAYESIAN_METHOD: "noncanonical_bayes",
            PRIMARY_DETERMINISTIC_METHOD: "noncanonical_twin",
        }
    )
    contribution_path = tmp_path / "custom-contributions.csv"
    output_dir = tmp_path / "custom-bootstrap"
    contributions.to_csv(contribution_path, index=False)

    assert (
        main(
            [
                str(contribution_path),
                "--resamples",
                "3",
                "--seed",
                "17",
                "--false-budget",
                "3.0",
                "--minimum-recall",
                "0.4",
                "--lead-grid",
                "0.2",
                "0.4",
                "0.8",
                "1.2",
                "--brace-method",
                "noncanonical_bayes",
                "--comparator-method",
                "noncanonical_twin",
                "--output",
                str(output_dir),
                "--allow-noncanonical-synthetic",
            ]
        )
        == 0
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["noncanonical_synthetic"] is True
    assert manifest["analysis"]["n_resamples"] == 3
    assert manifest["analysis"]["seed"] == 17
    assert manifest["estimand"] == {
        "brace_method": "noncanonical_bayes",
        "comparator_method": "noncanonical_twin",
        "false_budget_per_hour": 3.0,
        "lead_grid_s": [0.2, 0.4, 0.8, 1.2],
        "least_favorable_false_count_column": "least_favorable_false_proposals",
        "minimum_recall": 0.4,
    }


def test_bootstrap_output_bundle_is_absent_after_staged_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contribution_path = tmp_path / "contributions.csv"
    _contributions().to_csv(contribution_path, index=False)
    output_dir = tmp_path / "failed-bootstrap"

    original_save = np.save
    save_calls = 0

    def fail_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 5:
            raise OSError("simulated write failure")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(np, "save", fail_save)
    with pytest.raises(OSError, match="simulated write failure"):
        main(
            [
                str(contribution_path),
                "--resamples",
                "2",
                "--output",
                str(output_dir),
                "--allow-noncanonical-synthetic",
            ]
        )
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".failed-bootstrap-staging-*"))
    assert save_calls == 5
