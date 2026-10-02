import numpy as np
import pandas as pd

from brace_f1.forecast import (
    canonical_forecast_row_ids,
    forecast_ctra_particles,
    planar_states_from_features,
)
from brace_f1.geometry import CircuitCorridor
from brace_f1.residual import (
    BayesianResidualDynamics,
    build_residual_training_pairs,
    residual_design_from_features,
)


def test_synthetic_transitions_fit_and_forecast_joint_exit_distribution() -> None:
    rows: list[dict[str, object]] = []
    for circuit_index, circuit in enumerate(("A", "B")):
        for car_index, car in enumerate(("c1", "c2")):
            for frame in range(6):
                rows.append(
                    {
                        "circuit": circuit,
                        "car_id": car,
                        "frame_index": frame,
                        "time_seconds": frame * 0.05,
                        "continuous_segment_id": 0,
                        "input_valid_causal": True,
                        "in_corridor": True,
                        "map_x_m": 7.5 + 0.02 * circuit_index,
                        "map_y_m": float(car_index),
                        "yaw_rad": 0.0,
                        "yaw_rate_radps": 0.0,
                        "body_speed_longitudinal_mps": 8.0 + 0.05 * frame,
                        "body_speed_lateral_mps": 0.0,
                        "body_acceleration_longitudinal_mps2": 1.0,
                        "body_acceleration_lateral_mps2": 0.0,
                        "track_heading_rad": 0.0,
                        "heading_error_rad": 0.0,
                        "track_offset_m": 0.0,
                        "track_curvature_per_m": 0.0,
                    }
                )
    features = pd.DataFrame(rows)
    pairs = build_residual_training_pairs(features)
    model = BayesianResidualDynamics.fit(
        pairs.X,
        pairs.Y,
        circuits=pairs.circuits,
        cars=pairs.cars,
        feature_names=pairs.feature_names,
        target_names=pairs.target_names,
        n_draws=8,
        ridge_alpha=1.0,
        seed=2027,
    )
    score_rows = features.groupby(["circuit", "car_id"], sort=True).tail(1)
    states = planar_states_from_features(score_rows)
    design = residual_design_from_features(score_rows, transition_dt_s=0.05)
    corridor = CircuitCorridor.from_arrays(
        centerline=np.asarray([[-7.5, -7.5], [7.5, -7.5], [7.5, 7.5], [-7.5, 7.5]]),
        boundary_a=np.asarray([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]),
        boundary_b=np.asarray([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]),
    )
    result = forecast_ctra_particles(
        states,
        corridor,
        residual_model=model,
        residual_features=design,
        n_particles=16,
        seed=71,
        row_ids=canonical_forecast_row_ids(
            base_seed=71,
            outer_fold="synthetic",
            method="brace_bayesian",
            circuits=score_rows["circuit"].to_numpy(),
            cars=score_rows["car_id"].to_numpy(),
            frame_indices=score_rows["frame_index"].to_numpy(),
        ),
    )
    assert result.no_exit_probability.shape == (4, 4)
    assert result.outcome_probability.shape == (4, 4, 3, 3)
    np.testing.assert_allclose(
        result.no_exit_probability + result.outcome_probability.sum(axis=(2, 3)),
        1.0,
        atol=1e-15,
    )
