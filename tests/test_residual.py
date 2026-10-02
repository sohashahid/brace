import numpy as np
import pandas as pd
import pytest

from brace_f1.io import DataValidationError
from brace_f1.residual import (
    RESIDUAL_FEATURE_COLUMNS,
    RESIDUAL_TARGET_NAMES,
    BayesianResidualDynamics,
    DeterministicResidualDynamics,
    body_frame_ctra_midpoint_velocity_step,
    build_residual_training_pairs,
    cluster_bayesian_bootstrap_weights,
)


def _feature_rows() -> pd.DataFrame:
    # The first segment follows the body-frame kinematic mean exactly.
    return pd.DataFrame(
        {
            "circuit": ["A", "A", "A", "A", "A"],
            "car_id": ["c1"] * 5,
            "frame_index": [0, 1, 2, 3, 4],
            "time_seconds": [0.00, 0.05, 0.10, 0.15, 0.20],
            "continuous_segment_id": [0, 0, 0, 1, 1],
            "input_valid_causal": [True, True, True, True, True],
            "in_corridor": [True, True, True, True, True],
            "yaw_rad": [0.0, 0.005, 0.01, 1.0, 1.1],
            "yaw_rate_radps": [0.1, 0.1, 0.1, 0.0, 2.0],
            "body_speed_longitudinal_mps": [
                10.0,
                10.049875,
                10.0994987515625,
                30.0,
                99.0,
            ],
            "body_speed_lateral_mps": [
                0.0,
                -0.050125,
                -0.1004987484375,
                0.0,
                99.0,
            ],
            "track_heading_rad": [0.0] * 5,
            "heading_error_rad": [0.0, 0.005, 0.01, 1.0, 1.1],
            "track_offset_m": [0.0, 0.0, 0.0, 1.0, 2.0],
            "track_curvature_per_m": [0.01] * 5,
            "body_acceleration_longitudinal_mps2": [1.0] * 5,
            "body_acceleration_lateral_mps2": [0.0] * 5,
        }
    )


def test_residual_pairs_use_only_consecutive_valid_rows_inside_hard_break_segment() -> None:
    pairs = build_residual_training_pairs(_feature_rows())
    assert pairs.X.shape == (3, len(RESIDUAL_FEATURE_COLUMNS))
    assert pairs.Y.shape == (3, len(RESIDUAL_TARGET_NAMES))
    assert pairs.row_keys["source_frame_index"].tolist() == [0, 1, 3]
    assert pairs.row_keys["target_frame_index"].tolist() == [1, 2, 4]
    # There is no 2 -> 3 pair across the reset even though the times are adjacent.
    crosses_reset = (pairs.row_keys["source_frame_index"] == 2) & (
        pairs.row_keys["target_frame_index"] == 3
    )
    assert not crosses_reset.any()
    np.testing.assert_allclose(pairs.Y[:2], 0.0, atol=1e-12)
    assert pairs.Y[2, 0] == pytest.approx(68.95)


def test_body_frame_midpoint_velocity_step_uses_midpoint_state() -> None:
    next_long, next_lateral, midpoint_long, midpoint_lateral = (
        body_frame_ctra_midpoint_velocity_step(
            np.asarray([10.0]),
            np.asarray([0.0]),
            np.asarray([0.1]),
            np.asarray([1.0]),
            np.asarray([0.0]),
            0.05,
        )
    )

    np.testing.assert_allclose(midpoint_long, [10.025])
    np.testing.assert_allclose(midpoint_lateral, [-0.025])
    np.testing.assert_allclose(next_long, [10.049875])
    np.testing.assert_allclose(next_lateral, [-0.050125])


def test_residual_pairs_do_not_depend_on_outcome_target_columns() -> None:
    clean = _feature_rows()
    poisoned = clean.assign(
        next_excursion_side="left",
        time_to_next_excursion_seconds=-999.0,
        qualifying_event_id_at_source="leak",
    )
    a = build_residual_training_pairs(clean)
    b = build_residual_training_pairs(poisoned)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.Y, b.Y)
    pd.testing.assert_frame_equal(a.row_keys, b.row_keys)


def test_residual_pairs_require_both_transition_rows_inside_corridor() -> None:
    features = _feature_rows()
    features.loc[1, "in_corridor"] = False
    pairs = build_residual_training_pairs(features)
    assert pairs.row_keys[["source_frame_index", "target_frame_index"]].values.tolist() == [[3, 4]]


def test_two_stage_cluster_bootstrap_weights_are_reproducible_and_frame_count_neutral() -> None:
    circuits = np.asarray(["A", "A", "A", "A", "B", "B"])
    cars = np.asarray(["a1", "a2", "a2", "a2", "b1", "b1"])
    weights = cluster_bayesian_bootstrap_weights(circuits, cars, n_draws=7, seed=42)
    repeated = cluster_bayesian_bootstrap_weights(circuits, cars, n_draws=7, seed=42)
    np.testing.assert_array_equal(weights, repeated)
    assert weights.shape == (7, 6)
    np.testing.assert_allclose(weights.sum(axis=1), 3.0)
    # All rows of a car share its draw weight divided by that car's row count.
    np.testing.assert_allclose(weights[:, 1], weights[:, 2])
    np.testing.assert_allclose(weights[:, 2], weights[:, 3])
    np.testing.assert_allclose(weights[:, 4], weights[:, 5])
    assert np.all(weights > 0.0)


def test_bayesian_ridge_posterior_mean_equals_deterministic_twin() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(24, 3))
    coefficient = np.asarray([[1.0, -0.5], [0.2, 0.7], [-0.1, 0.3]])
    Y = X @ coefficient + np.asarray([0.25, -0.75])
    circuits = np.repeat(["A", "B"], 12)
    cars = np.repeat(["a1", "a2", "b1", "b2"], 6)
    model = BayesianResidualDynamics.fit(
        X,
        Y,
        circuits=circuits,
        cars=cars,
        feature_names=("x1", "x2", "x3"),
        target_names=("y1", "y2"),
        n_draws=32,
        ridge_alpha=1e-6,
        seed=2027,
    )
    twin = model.deterministic_twin()
    points = X[:5]
    np.testing.assert_allclose(model.predict_mean(points), twin.predict(points), atol=1e-12)
    np.testing.assert_allclose(twin.coefficients, model.coefficient_draws.mean(axis=0))
    assert (
        model.content_hash
        == BayesianResidualDynamics.fit(
            X,
            Y,
            circuits=circuits,
            cars=cars,
            feature_names=("x1", "x2", "x3"),
            target_names=("y1", "y2"),
            n_draws=32,
            ridge_alpha=1e-6,
            seed=2027,
        ).content_hash
    )
    assert model.fit_algorithm == "per_car_normalized_sufficient_statistics_v1"
    assert model.fit_row_count == 24
    assert model.cluster_count == 4


def test_sufficient_statistic_fit_is_invariant_to_within_car_row_replication() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    Y = np.column_stack((X[:, 0], -2.0 * X[:, 0]))
    circuits = np.asarray(["A", "A", "B", "B"])
    cars = np.asarray(["a", "a", "b", "b"])
    kwargs = dict(
        feature_names=("x",),
        target_names=("u", "v"),
        n_draws=12,
        ridge_alpha=0.5,
        seed=37,
    )
    original = BayesianResidualDynamics.fit(X, Y, circuits=circuits, cars=cars, **kwargs)
    replicated = BayesianResidualDynamics.fit(
        np.repeat(X, 5, axis=0),
        np.repeat(Y, 5, axis=0),
        circuits=np.repeat(circuits, 5),
        cars=np.repeat(cars, 5),
        **kwargs,
    )
    np.testing.assert_allclose(original.coefficient_draws, replicated.coefficient_draws, atol=1e-12)
    np.testing.assert_allclose(
        original.process_covariance_draws,
        replicated.process_covariance_draws,
        atol=1e-12,
    )


def test_residual_particle_draws_are_seeded_finite_and_shape_checked() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    Y = np.column_stack((2.0 * X[:, 0], -X[:, 0]))
    model = BayesianResidualDynamics.fit(
        X,
        Y,
        circuits=np.asarray(["A", "A", "B", "B"]),
        cars=np.asarray(["a", "a", "b", "b"]),
        feature_names=("x",),
        target_names=("u", "v"),
        n_draws=8,
        ridge_alpha=0.1,
        seed=11,
    )
    samples = model.sample_residuals(X[:2], n_particles=5, seed=99)
    assert samples.shape == (2, 5, 2)
    assert np.isfinite(samples).all()
    np.testing.assert_array_equal(samples, model.sample_residuals(X[:2], n_particles=5, seed=99))
    assert not np.array_equal(samples, model.sample_residuals(X[:2], n_particles=5, seed=100))
    with pytest.raises(DataValidationError, match="feature count"):
        model.predict_mean(np.ones((2, 2)))


def test_deterministic_ridge_validates_weights_and_recovers_linear_signal() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    Y = 3.0 * X + 2.0
    model = DeterministicResidualDynamics.fit(
        X,
        Y,
        feature_names=("x",),
        target_names=("y",),
        ridge_alpha=0.0,
    )
    np.testing.assert_allclose(model.predict(np.asarray([[4.0]])), [[14.0]], atol=1e-12)
    with pytest.raises(DataValidationError, match="sample weights"):
        DeterministicResidualDynamics.fit(
            X,
            Y,
            feature_names=("x",),
            target_names=("y",),
            sample_weight=np.asarray([1.0, -1.0, 1.0, 1.0]),
        )


def test_residual_models_bind_fit_only_robust_scaling_and_accept_raw_designs() -> None:
    X = np.asarray([[0.0], [2.0], [4.0]])
    Y = 3.0 * X + 2.0
    deterministic = DeterministicResidualDynamics.fit(
        X,
        Y,
        feature_names=("x",),
        target_names=("y",),
        ridge_alpha=0.0,
    )
    bayesian = BayesianResidualDynamics.fit(
        X,
        Y,
        circuits=np.asarray(["A", "A", "B"]),
        cars=np.asarray(["a", "a", "b"]),
        feature_names=("x",),
        target_names=("y",),
        n_draws=2,
        ridge_alpha=0.0,
        seed=12,
    )

    np.testing.assert_allclose(deterministic.feature_center, [2.0])
    np.testing.assert_allclose(deterministic.feature_scale, [2.0])
    np.testing.assert_allclose(deterministic.transform_design([[0.0], [4.0]]), [[-1.0], [1.0]])
    np.testing.assert_allclose(deterministic.predict([[4.0]]), [[14.0]], atol=1e-10)
    np.testing.assert_allclose(bayesian.feature_center, [2.0])
    np.testing.assert_allclose(bayesian.feature_scale, [2.0])


def test_hashed_model_arrays_are_immutable() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    Y = np.column_stack((X[:, 0], -X[:, 0]))
    deterministic = DeterministicResidualDynamics.fit(
        X,
        Y,
        feature_names=("x",),
        target_names=("u", "v"),
    )
    bayesian = BayesianResidualDynamics.fit(
        X,
        Y,
        circuits=np.asarray(["A", "A", "B", "B"]),
        cars=np.asarray(["a", "a", "b", "b"]),
        feature_names=("x",),
        target_names=("u", "v"),
        n_draws=2,
    )

    for array in (
        deterministic.coefficients,
        deterministic.process_covariance,
        deterministic.feature_center,
        deterministic.feature_scale,
        bayesian.coefficient_draws,
        bayesian.process_covariance_draws,
        bayesian.feature_center,
        bayesian.feature_scale,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 999.0
