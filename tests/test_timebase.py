from __future__ import annotations

import numpy as np

from brace_f1.timebase import causal_resample, reconstruct_time_axis


def _lap_fields(
    current: list[float],
    laps: list[int],
    *,
    last: list[float] | None = None,
    total: list[float] | None = None,
    result: list[int] | None = None,
) -> dict[str, np.ndarray]:
    n = len(current)
    return {
        "current_lap_times": np.asarray(current, dtype=float),
        "lap_numbers": np.asarray(laps, dtype=int),
        "last_lap_times": np.asarray(last if last is not None else [np.inf] * n, dtype=float),
        "total_distances": np.asarray(total if total is not None else np.arange(n), dtype=float),
        "result_status": np.asarray(result if result is not None else [2] * n, dtype=int),
        "pit_status": np.zeros(n, dtype=int),
    }


def test_reconstruct_time_axis_unwraps_lap_boundary_monotonically() -> None:
    lap = _lap_fields(
        [0.00, 0.01, 0.02, 0.00, 0.01],
        [1, 1, 1, 2, 2],
        last=[np.inf, np.inf, np.inf, 0.03, 0.03],
    )
    positions = np.column_stack((np.arange(5) * 0.1, np.zeros(5), np.zeros(5)))

    axis = reconstruct_time_axis(lap, positions)

    np.testing.assert_allclose(axis.time_seconds, [0.00, 0.01, 0.02, 0.03, 0.04])
    assert axis.lap_boundary.tolist() == [False, False, False, True, False]
    assert axis.hard_break.tolist() == [True, False, False, False, False]
    assert axis.diagnostics.native_frame_timestamps_available is False
    assert axis.diagnostics.timestamp_method == "reconstructed_from_lap_clock"


def test_reconstruct_time_axis_marks_clock_reset_and_continues_monotonically() -> None:
    lap = _lap_fields([0.00, 0.04, 0.08, 0.00, 0.04], [1, 1, 1, 1, 1])
    positions = np.column_stack((np.arange(5) * 0.1, np.zeros(5), np.zeros(5)))

    axis = reconstruct_time_axis(lap, positions)

    assert axis.clock_reset.tolist() == [False, False, False, True, False]
    assert axis.hard_break.tolist() == [True, False, False, True, False]
    assert np.all(np.diff(axis.time_seconds) >= 0)
    assert axis.diagnostics.clock_reset_count == 1


def test_reconstruct_time_axis_marks_physical_and_status_breaks() -> None:
    lap = _lap_fields(
        [0.00, 0.01, 0.02, 0.03],
        [1, 1, 1, 1],
        total=[0.0, 0.1, -10.0, -9.9],
        result=[2, 2, 3, 3],
    )
    positions = np.asarray([[0, 0, 0], [0.1, 0, 0], [10, 0, 0], [10.1, 0, 0]])

    axis = reconstruct_time_axis(lap, positions)

    assert axis.hard_break.tolist() == [True, False, True, False]
    assert axis.diagnostics.position_jump_count == 1
    assert axis.diagnostics.distance_reset_count == 1
    assert axis.diagnostics.status_transition_count == 1


def test_causal_resample_never_uses_a_future_source_frame() -> None:
    lap = _lap_fields([0.00, 0.03, 0.06, 0.09, 0.12], [1, 1, 1, 1, 1])
    positions = np.column_stack((np.arange(5) * 0.1, np.zeros(5), np.zeros(5)))
    axis = reconstruct_time_axis(lap, positions)

    sampled = causal_resample(
        axis,
        {"marker": np.asarray([10, 11, 12, 13, 14])},
        frequency_hz=20.0,
    )

    np.testing.assert_allclose(sampled.target_time_seconds, [0.00, 0.05, 0.10])
    assert sampled.source_indices.tolist() == [0, 1, 3]
    assert sampled.arrays["marker"].tolist() == [10, 11, 13]
    assert np.all(axis.time_seconds[sampled.source_indices] <= sampled.target_time_seconds)
    np.testing.assert_allclose(sampled.source_age_seconds, [0.00, 0.02, 0.01])


def test_causal_resample_does_not_carry_values_across_hard_break() -> None:
    lap = _lap_fields([0.00, 0.04, 0.00, 0.04], [1, 1, 1, 1])
    positions = np.column_stack((np.arange(4) * 0.1, np.zeros(4), np.zeros(4)))
    axis = reconstruct_time_axis(lap, positions, clock_reset_tolerance_seconds=0.01)

    sampled = causal_resample(axis, {"marker": np.asarray([1, 2, 100, 101])}, frequency_hz=25)

    assert sampled.source_indices.tolist() == [0, 1, 2, 3]
    assert sampled.segment_id.tolist() == [0, 0, 1, 1]
    assert sampled.arrays["marker"].tolist() == [1, 2, 100, 101]


def test_cadence_estimator_excludes_increments_at_or_above_0p05_seconds() -> None:
    lap = _lap_fields([0.00, 0.06, 0.12, 0.13, 0.14], [1, 1, 1, 1, 1])
    positions = np.column_stack((np.arange(5) * 0.1, np.zeros(5), np.zeros(5)))

    axis = reconstruct_time_axis(lap, positions)

    assert np.isclose(axis.diagnostics.median_source_dt_seconds, 0.01)
    assert axis.data_gap.tolist() == [False, True, True, False, False]
