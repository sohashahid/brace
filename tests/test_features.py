import numpy as np
import pandas as pd
import pytest

from brace_f1.features import (
    FEATURE_COLUMNS,
    FitOnlyRobustScaler,
    TrackReference,
    derive_causal_features,
    yaw_from_quaternions_xyzw,
)
from brace_f1.io import DataValidationError


def _straight_track() -> TrackReference:
    return TrackReference(
        np.asarray(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
                [20.0, 10.0],
                [0.0, 10.0],
            ]
        )
    )


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "circuit": ["Test"] * 5,
            "car_id": ["car_1"] * 5,
            "frame_index": np.arange(5),
            "time_seconds": [0.00, 0.05, 0.10, 0.15, 0.20],
            "continuous_segment_id": [0, 0, 0, 1, 1],
            "hard_break": [True, False, False, True, False],
            "input_valid_causal": [False, True, True, False, True],
            "in_corridor": [True, True, True, True, True],
            "map_x_m": [1.0, 2.0, 3.0, 4.0, 5.0],
            "map_y_m": [2.0, 2.0, 2.0, -1.0, -1.0],
            "velocity_x_mps": [10.0, 10.0, 10.0, 5.0, 5.0],
            "velocity_y_mps": [0.0, 0.0, 0.0, 0.0, 0.0],
            "body_acceleration_longitudinal_mps2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "body_acceleration_lateral_mps2": [0.0, 0.1, 0.2, 0.3, 0.4],
            "forward_x_map": np.cos([0.0, 0.1, 0.2, 1.0, 1.1]),
            "forward_y_map": np.sin([0.0, 0.1, 0.2, 1.0, 1.1]),
            "centerline_segment_index_pcd": [0, 0, 0, 0, 0],
            "future_event_side": ["left", "left", "right", "right", "left"],
            "time_to_next_excursion_seconds": [1.0, 0.95, 0.90, 0.85, 0.80],
        }
    )


def test_quaternion_yaw_uses_xyzw_order_and_wraps_to_pi() -> None:
    half = np.sqrt(0.5)
    yaw = yaw_from_quaternions_xyzw(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, half, half],
                [0.0, 0.0, -half, half],
            ]
        )
    )
    np.testing.assert_allclose(yaw, [0.0, np.pi / 2.0, -np.pi / 2.0], atol=1e-12)


def test_features_are_causal_reset_derivatives_and_ignore_future_columns() -> None:
    frames = _frames()
    reference = _straight_track()
    got = derive_causal_features(frames, {"Test": reference})

    np.testing.assert_allclose(got["yaw_rad"], [0.0, 0.1, 0.2, 1.0, 1.1])
    np.testing.assert_allclose(got["yaw_rate_radps"], [0.0, 2.0, 2.0, 0.0, 2.0])
    np.testing.assert_allclose(got["track_offset_m"], [2.0, 2.0, 2.0, -1.0, -1.0])
    np.testing.assert_allclose(got["track_heading_rad"], 0.0)
    np.testing.assert_allclose(got["track_curvature_per_m"], 0.0)
    np.testing.assert_allclose(
        got["body_speed_longitudinal_mps"],
        [10.0, 10.0 * np.cos(0.1), 10.0 * np.cos(0.2), 5.0 * np.cos(1.0), 5.0 * np.cos(1.1)],
        atol=1e-12,
    )
    assert set(FEATURE_COLUMNS).issubset(got.columns)
    assert got["in_corridor"].tolist() == [True] * 5
    assert "future_event_side" not in got.columns
    assert "time_to_next_excursion_seconds" not in got.columns

    poisoned = frames.copy()
    poisoned["future_event_side"] = "unknown"
    poisoned["time_to_next_excursion_seconds"] = -999.0
    repeated = derive_causal_features(poisoned, {"Test": reference})
    pd.testing.assert_frame_equal(got, repeated)


def test_prefix_features_are_identical_to_full_stream_prefix() -> None:
    frames = _frames()
    full = derive_causal_features(frames, {"Test": _straight_track()})
    prefix = derive_causal_features(frames.iloc[:3], {"Test": _straight_track()})
    pd.testing.assert_frame_equal(full.iloc[:3].reset_index(drop=True), prefix)


def test_hard_break_resets_yaw_rate_even_if_segment_identifier_is_unchanged() -> None:
    frames = _frames()
    frames["continuous_segment_id"] = 0
    frames["hard_break"] = [True, False, True, False, False]

    got = derive_causal_features(frames, {"Test": _straight_track()})

    assert got.loc[2, "yaw_rate_radps"] == pytest.approx(0.0)


def test_track_reference_reports_signed_offset_and_corner_curvature() -> None:
    reference = _straight_track()
    offset, heading, curvature = reference.sample(
        points_xy=np.asarray([[5.0, 3.0], [20.0, 5.0]]),
        segment_indices=np.asarray([0, 2]),
    )
    np.testing.assert_allclose(offset, [3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(heading, [0.0, np.pi / 2.0], atol=1e-12)
    assert curvature[0] == pytest.approx(0.0)
    assert curvature[1] > 0.0


def test_fit_only_robust_scaler_excludes_calibration_and_test_rows() -> None:
    table = pd.DataFrame(
        {
            "partition": ["fit", "fit", "fit", "calibration", "test"],
            "a": [0.0, 2.0, 4.0, 1_000.0, -1_000.0],
            "b": [10.0, 10.0, 10.0, 99.0, -99.0],
        }
    )
    scaler = FitOnlyRobustScaler.fit_from_partition(table, ["a", "b"])
    np.testing.assert_allclose(scaler.center, [2.0, 10.0])
    np.testing.assert_allclose(scaler.scale, [2.0, 1.0])
    transformed = scaler.transform(table)
    np.testing.assert_allclose(transformed.loc[:2, "a"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(transformed.loc[:2, "b"], 0.0)
    assert transformed.loc[3, "a"] == pytest.approx(499.0)
    assert (
        scaler.content_hash
        == FitOnlyRobustScaler.fit_from_partition(table, ["a", "b"]).content_hash
    )
    assert not scaler.center.flags.writeable
    assert not scaler.scale.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        scaler.center[0] = 999.0


def test_feature_builder_rejects_missing_or_nonmonotone_causal_inputs() -> None:
    with pytest.raises(DataValidationError, match="missing columns"):
        derive_causal_features(
            _frames().drop(columns="velocity_x_mps"), {"Test": _straight_track()}
        )
    nonmonotone = _frames()
    nonmonotone.loc[2, "time_seconds"] = 0.01
    with pytest.raises(DataValidationError, match="strictly increasing"):
        derive_causal_features(nonmonotone, {"Test": _straight_track()})
