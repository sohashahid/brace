import hashlib

import numpy as np
import pandas as pd
import pytest

from brace_f1.forecast import (
    SIDE_ORDER,
    ForecastBatch,
    ForecastProbabilities,
    canonical_forecast_row_ids,
    canonical_forecast_row_seed,
    circular_bin_distance,
    circular_neighborhood_mask,
    forecast_ctra_particles,
    iter_forecast_ctra_batches,
    planar_states_from_features,
)
from brace_f1.geometry import CircuitCorridor
from brace_f1.io import DataValidationError
from brace_f1.residual import (
    RESIDUAL_FEATURE_COLUMNS,
    RESIDUAL_TARGET_NAMES,
    BayesianResidualDynamics,
    DeterministicResidualDynamics,
    residual_design_from_features,
)


def _square_ring() -> CircuitCorridor:
    return CircuitCorridor.from_arrays(
        centerline=np.asarray([[-7.5, -7.5], [7.5, -7.5], [7.5, 7.5], [-7.5, 7.5]]),
        boundary_a=np.asarray([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]),
        boundary_b=np.asarray([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]),
    )


def test_circular_bin_helpers_wrap_across_start_finish() -> None:
    np.testing.assert_array_equal(
        circular_bin_distance(np.asarray([0, 1, 8, 9]), center_bin=9, n_bins=10),
        [1, 2, 1, 0],
    )


def test_canonical_forecast_row_ids_encode_seed_fold_method_and_frame() -> None:
    ids = canonical_forecast_row_ids(
        base_seed=91,
        outer_fold="Britain",
        method="brace_bayesian",
        circuits=np.asarray(["Bahrain", "Jeddah"]),
        cars=np.asarray(["car_1", "car_2"]),
        frame_indices=np.asarray([12, 34]),
    )
    repeated = canonical_forecast_row_ids(
        base_seed=91,
        outer_fold="Britain",
        method="brace_bayesian",
        circuits=np.asarray(["Bahrain", "Jeddah"]),
        cars=np.asarray(["car_1", "car_2"]),
        frame_indices=np.asarray([12, 34]),
    )

    np.testing.assert_array_equal(ids, repeated)
    assert ids.shape == (2,)
    assert len(set(ids.tolist())) == 2
    assert all(value.startswith("brace-row-seed-v1:") for value in ids)
    expected_seed = int.from_bytes(
        hashlib.sha256(str(ids[0]).encode()).digest()[:8], byteorder="little", signed=False
    )
    assert canonical_forecast_row_seed(str(ids[0])) == expected_seed
    np.testing.assert_array_equal(
        circular_neighborhood_mask(np.arange(10), center_bin=9, tolerance_bins=1, n_bins=10),
        [True, False, False, False, False, False, False, False, True, True],
    )


def test_particle_forecast_finds_first_inside_to_outside_crossing_and_joint_bin() -> None:
    # Row 0 points radially out through the outer loop; row 1 travels along the ring.
    state = np.asarray(
        [
            [7.5, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7.5, 0.0, 2.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0],
        ]
    )
    result = forecast_ctra_particles(
        state,
        _square_ring(),
        horizons_s=(0.25, 0.50, 1.00, 1.50),
        dt_s=0.05,
        n_particles=16,
        seed=17,
        segment_length_m=25.0,
    )
    assert isinstance(result, ForecastProbabilities)
    assert SIDE_ORDER == ("left", "right", "unknown")
    assert result.no_exit_probability.shape == (2, 4)
    assert result.outcome_probability.shape == (2, 4, 3, 3)
    np.testing.assert_allclose(result.no_exit_probability[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.no_exit_probability[1], 1.0)
    right_index = SIDE_ORDER.index("right")
    assert result.outcome_probability[0, 1, right_index, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(result.first_crossing_time_s[0], 0.25, atol=2e-8)
    assert np.isnan(result.first_crossing_time_s[1]).all()
    np.testing.assert_array_equal(result.first_crossing_side_index[0], right_index)
    np.testing.assert_array_equal(result.first_crossing_bin[0], 0)


def test_particle_forecast_detects_tunneling_through_inner_hole() -> None:
    state = np.asarray([[-7.5, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    result = forecast_ctra_particles(
        state,
        _square_ring(),
        horizons_s=(0.05,),
        dt_s=0.05,
        n_particles=4,
        seed=3,
    )

    np.testing.assert_allclose(result.no_exit_probability, 0.0)
    np.testing.assert_allclose(result.first_crossing_time_s, 1.0 / 120.0, atol=1e-8)


def test_registered_residual_design_is_recomputed_from_evolving_particle_state() -> None:
    coefficients = np.zeros((len(RESIDUAL_FEATURE_COLUMNS) + 1, len(RESIDUAL_TARGET_NAMES)))
    coefficients[1, 0] = 1.0  # delta v_long = current v_long
    model = DeterministicResidualDynamics(
        RESIDUAL_FEATURE_COLUMNS,
        RESIDUAL_TARGET_NAMES,
        coefficients,
        np.zeros((len(RESIDUAL_TARGET_NAMES), len(RESIDUAL_TARGET_NAMES))),
        0.0,
        np.zeros(len(RESIDUAL_FEATURE_COLUMNS)),
        np.ones(len(RESIDUAL_FEATURE_COLUMNS)),
        "dynamic-state-test",
    )
    state = np.asarray([[7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    initial_design = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]])

    result = forecast_ctra_particles(
        state,
        _square_ring(),
        residual_model=model,
        residual_features=initial_design,
        horizons_s=(0.50,),
        dt_s=0.05,
        n_particles=2,
        seed=4,
    )

    # A stale residual adds only 1 m/s per step and crosses near 0.5 s.  The
    # registered state-conditioned model doubles speed and exits before 0.3 s.
    assert np.nanmax(result.first_crossing_time_s) < 0.30


def test_recursive_residual_forecast_applies_model_bound_scaling_each_step() -> None:
    coefficients = np.zeros((len(RESIDUAL_FEATURE_COLUMNS) + 1, len(RESIDUAL_TARGET_NAMES)))
    coefficients[1, 0] = 1.0  # delta v_long = scaled current v_long
    scale = np.ones(len(RESIDUAL_FEATURE_COLUMNS))
    scale[0] = 2.0
    model = DeterministicResidualDynamics(
        RESIDUAL_FEATURE_COLUMNS,
        RESIDUAL_TARGET_NAMES,
        coefficients,
        np.zeros((len(RESIDUAL_TARGET_NAMES), len(RESIDUAL_TARGET_NAMES))),
        0.0,
        np.zeros(len(RESIDUAL_FEATURE_COLUMNS)),
        scale,
        "scaled-dynamic-state-test",
    )
    state = np.asarray([[7.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    initial_design = np.asarray([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]])

    result = forecast_ctra_particles(
        state,
        _square_ring(),
        residual_model=model,
        residual_features=initial_design,
        horizons_s=(0.50,),
        dt_s=0.05,
        n_particles=2,
        seed=8,
    )

    # With the bound scale, v_next=1.5*v and the outer boundary is reached
    # after 0.30 s; silently mixing raw units would double v and exit sooner.
    assert 0.30 < np.nanmin(result.first_crossing_time_s) < 0.35


def test_particle_forecast_uses_full_geometry_only_for_new_crossings() -> None:
    base = _square_ring()

    class CountingCorridor:
        track_length_m = base.track_length_m

        def __init__(self) -> None:
            self.contains_point_counts: list[int] = []
            self.segment_point_counts: list[int] = []
            self.locate_point_counts: list[int] = []

        def contains_planar(self, points: np.ndarray) -> np.ndarray:
            self.contains_point_counts.append(len(points))
            return base.contains_planar(points)

        def locate(self, points: np.ndarray):
            self.locate_point_counts.append(len(points))
            return base.locate(points)

        def first_exit_along_segments(self, start: np.ndarray, end: np.ndarray):
            self.segment_point_counts.append(len(start))
            return base.first_exit_along_segments(start, end)

    corridor = CountingCorridor()
    state = np.asarray([[7.5, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    forecast_ctra_particles(
        state,
        corridor,
        n_particles=8,
        seed=3,
        dt_s=0.05,  # type: ignore[arg-type]
    )
    assert corridor.contains_point_counts[0] == 1
    assert corridor.contains_point_counts[1:] == [8] * 30
    assert corridor.segment_point_counts == [8] * 6
    assert corridor.locate_point_counts == [8]


def test_probability_mass_is_exactly_partitioned_at_every_horizon() -> None:
    state = np.asarray(
        [
            [7.5, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-7.5, 0.0, 8.0, 0.0, np.pi, 0.0, 0.0, 0.0],
        ]
    )
    result = forecast_ctra_particles(state, _square_ring(), n_particles=31, seed=5, dt_s=0.05)
    mass = result.no_exit_probability + result.outcome_probability.sum(axis=(2, 3))
    np.testing.assert_allclose(mass, 1.0, atol=1e-15)
    assert np.all(result.no_exit_probability >= 0.0)
    assert np.all(result.outcome_probability >= 0.0)
    assert len(result.config_hash) == 64


def test_bayesian_particle_forecast_is_reproducible_from_fixed_seed() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 2))
    Y = rng.normal(scale=0.02, size=(12, 5))
    model = BayesianResidualDynamics.fit(
        X,
        Y,
        circuits=np.repeat(["A", "B"], 6),
        cars=np.repeat(["a1", "a2", "b1", "b2"], 3),
        feature_names=("f1", "f2"),
        target_names=(
            "delta_v_longitudinal_mps",
            "delta_v_lateral_mps",
            "delta_yaw_rate_radps",
            "delta_acceleration_longitudinal_mps2",
            "delta_acceleration_lateral_mps2",
        ),
        n_draws=8,
        ridge_alpha=1.0,
        seed=12,
    )
    state = np.asarray([[7.5, 0.0, 9.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    kwargs = dict(
        initial_states=state,
        corridor=_square_ring(),
        residual_model=model,
        residual_features=np.asarray([[0.2, -0.3]]),
        n_particles=23,
        dt_s=0.05,
        row_ids=canonical_forecast_row_ids(
            base_seed=91,
            outer_fold="A",
            method="brace_bayesian",
            circuits=np.asarray(["A"]),
            cars=np.asarray(["a1"]),
            frame_indices=np.asarray([10]),
        ),
    )
    a = forecast_ctra_particles(**kwargs, seed=91)
    b = forecast_ctra_particles(**kwargs, seed=91)
    np.testing.assert_array_equal(a.no_exit_probability, b.no_exit_probability)
    np.testing.assert_array_equal(a.outcome_probability, b.outcome_probability)
    np.testing.assert_array_equal(a.first_crossing_time_s, b.first_crossing_time_s)
    c = forecast_ctra_particles(
        **{
            **kwargs,
            "row_ids": canonical_forecast_row_ids(
                base_seed=92,
                outer_fold="A",
                method="brace_bayesian",
                circuits=np.asarray(["A"]),
                cars=np.asarray(["a1"]),
                frame_indices=np.asarray([10]),
            ),
        },
        seed=92,
    )
    assert a.config_hash != c.config_hash
    assert a.model_hash == model.content_hash


def test_forecast_rejects_invalid_horizons_state_and_residual_shape() -> None:
    with pytest.raises(DataValidationError, match="shape .*8"):
        forecast_ctra_particles(np.zeros((1, 7)), _square_ring())
    with pytest.raises(DataValidationError, match="integer multiples"):
        forecast_ctra_particles(np.zeros((1, 8)), _square_ring(), horizons_s=(0.26,), dt_s=0.05)
    model = BayesianResidualDynamics.fit(
        np.ones((4, 1)),
        np.zeros((4, 5)),
        circuits=np.asarray(["A", "A", "B", "B"]),
        cars=np.asarray(["a", "a", "b", "b"]),
        feature_names=("f",),
        target_names=(
            "delta_v_longitudinal_mps",
            "delta_v_lateral_mps",
            "delta_yaw_rate_radps",
            "delta_acceleration_longitudinal_mps2",
            "delta_acceleration_lateral_mps2",
        ),
        n_draws=2,
    )
    with pytest.raises(DataValidationError, match="residual_features"):
        forecast_ctra_particles(
            np.zeros((1, 8)),
            _square_ring(),
            residual_model=model,
            residual_features=np.zeros((2, 1)),
            row_ids=canonical_forecast_row_ids(
                base_seed=20270927,
                outer_fold="A",
                method="brace_bayesian",
                circuits=np.asarray(["A"]),
                cars=np.asarray(["a"]),
                frame_indices=np.asarray([0]),
            ),
        )
    with pytest.raises(DataValidationError, match="canonical row_ids"):
        forecast_ctra_particles(
            np.zeros((1, 8)),
            _square_ring(),
            residual_model=model,
            residual_features=np.zeros((1, 1)),
            n_particles=2,
        )


def test_feature_table_to_model_tensors_has_registered_column_order() -> None:
    table = pd.DataFrame(
        {
            "map_x_m": [1.0, 2.0],
            "map_y_m": [3.0, 4.0],
            "body_speed_longitudinal_mps": [5.0, 6.0],
            "body_speed_lateral_mps": [0.5, 0.6],
            "yaw_rad": [0.1, 0.2],
            "yaw_rate_radps": [0.01, 0.02],
            "body_acceleration_longitudinal_mps2": [1.0, 2.0],
            "body_acceleration_lateral_mps2": [0.1, 0.2],
            "heading_error_rad": [-0.1, -0.2],
            "track_offset_m": [1.5, 1.6],
            "track_curvature_per_m": [0.001, 0.002],
        }
    )
    states = planar_states_from_features(table)
    np.testing.assert_allclose(states[0], [1.0, 3.0, 5.0, 0.5, 0.1, 0.01, 1.0, 0.1])
    design = residual_design_from_features(table, transition_dt_s=0.05)
    np.testing.assert_allclose(design[0], [5.0, 0.5, 0.01, 1.0, 0.1, -0.1, 1.5, 0.001, 0.05])


def test_batched_bayesian_forecast_matches_full_call_for_stable_row_ids() -> None:
    rng = np.random.default_rng(14)
    X = rng.normal(size=(16, 2))
    Y = rng.normal(scale=0.1, size=(16, 5))
    model = BayesianResidualDynamics.fit(
        X,
        Y,
        circuits=np.repeat(["A", "B"], 8),
        cars=np.repeat(["a1", "a2", "b1", "b2"], 4),
        feature_names=("f1", "f2"),
        target_names=(
            "delta_v_longitudinal_mps",
            "delta_v_lateral_mps",
            "delta_yaw_rate_radps",
            "delta_acceleration_longitudinal_mps2",
            "delta_acceleration_lateral_mps2",
        ),
        n_draws=6,
        ridge_alpha=1.0,
        seed=18,
    )
    states = np.asarray(
        [[7.5, float(y), 8.0 + i, 0.0, 0.0, 0.0, 0.0, 0.0] for i, y in enumerate([-2, -1, 0, 1, 2])]
    )
    design = np.column_stack((np.linspace(-1, 1, 5), np.linspace(1, -1, 5)))
    row_ids = canonical_forecast_row_ids(
        base_seed=221,
        outer_fold="B",
        method="brace_bayesian",
        circuits=np.repeat("A", 5),
        cars=np.repeat("score-car", 5),
        frame_indices=np.arange(5),
    )
    full = forecast_ctra_particles(
        states,
        _square_ring(),
        residual_model=model,
        residual_features=design,
        n_particles=13,
        seed=221,
        row_ids=row_ids,
    )
    batches = list(
        iter_forecast_ctra_batches(
            states,
            _square_ring(),
            residual_model=model,
            residual_features=design,
            n_particles=13,
            seed=221,
            row_ids=row_ids,
            batch_size=2,
        )
    )
    assert all(isinstance(batch, ForecastBatch) for batch in batches)
    np.testing.assert_array_equal(
        np.concatenate([batch.row_indices for batch in batches]), np.arange(5)
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.forecast.no_exit_probability for batch in batches]),
        full.no_exit_probability,
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.forecast.outcome_probability for batch in batches]),
        full.outcome_probability,
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.forecast.first_crossing_time_s for batch in batches]),
        full.first_crossing_time_s,
    )
