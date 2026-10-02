from __future__ import annotations

import numpy as np

from brace_f1.exposure import compute_interval_exposure
from brace_f1.timebase import TimeAxis, TimingDiagnostics


def _axis(times: np.ndarray, artifact_indices: tuple[int, ...] = ()) -> TimeAxis:
    n = len(times)
    hard_break = np.zeros(n, dtype=bool)
    hard_break[0] = True
    lap_boundary = np.zeros(n, dtype=bool)
    lap_boundary[list(artifact_indices)] = True
    zeros = np.zeros(n, dtype=bool)
    diagnostics = TimingDiagnostics(
        timestamp_method="reconstructed_from_lap_clock",
        native_frame_timestamps_available=False,
        sample_count=n,
        median_source_dt_seconds=0.1,
        estimated_source_hz=10.0,
        lap_boundary_count=len(artifact_indices),
        clock_reset_count=0,
        data_gap_count=0,
        position_jump_count=0,
        distance_reset_count=0,
        status_transition_count=0,
        hard_break_count=0,
    )
    return TimeAxis(
        time_seconds=times,
        hard_break=hard_break,
        lap_boundary=lap_boundary,
        clock_reset=zeros.copy(),
        data_gap=zeros.copy(),
        artifact_proximity_seed=lap_boundary.copy(),
        diagnostics=diagnostics,
    )


def test_exposure_reports_prebuffer_and_segment_edge_buffered_time() -> None:
    times = np.arange(21, dtype=float) * 0.1

    exposure = compute_interval_exposure(
        _axis(times),
        active=np.ones(21, dtype=bool),
        pit_status=np.zeros(21, dtype=int),
        segment_edge_buffer_seconds=0.5,
        artifact_buffer_seconds=0.25,
    )

    assert np.isclose(exposure.prebuffer_seconds, 2.0)
    assert np.isclose(exposure.buffered_seconds, 1.0)
    assert exposure.prebuffer_interval_count == 20
    assert exposure.buffered_interval_count == 10


def test_exposure_excludes_intervals_overlapping_artifact_buffer() -> None:
    times = np.arange(21, dtype=float) * 0.1

    exposure = compute_interval_exposure(
        _axis(times, artifact_indices=(10,)),
        active=np.ones(21, dtype=bool),
        pit_status=np.zeros(21, dtype=int),
        segment_edge_buffer_seconds=0.0,
        artifact_buffer_seconds=0.25,
    )

    assert np.isclose(exposure.prebuffer_seconds, 2.0)
    assert np.isclose(exposure.buffered_seconds, 1.4)


def test_exposure_excludes_pit_and_hard_break_intervals() -> None:
    times = np.arange(11, dtype=float) * 0.1
    axis = _axis(times)
    axis.hard_break[7] = True
    pit_status = np.zeros(11, dtype=int)
    pit_status[3] = 1

    exposure = compute_interval_exposure(
        axis,
        active=np.ones(11, dtype=bool),
        pit_status=pit_status,
        segment_edge_buffer_seconds=0.0,
        artifact_buffer_seconds=0.0,
    )

    assert np.isclose(exposure.prebuffer_seconds, 0.7)
    assert exposure.prebuffer_interval_count == 7


def test_pit_intervals_do_not_create_additional_segment_edge_buffers() -> None:
    times = np.arange(31, dtype=float) * 0.1
    pit_status = np.zeros(31, dtype=int)
    pit_status[15] = 1

    exposure = compute_interval_exposure(
        _axis(times),
        active=np.ones(31, dtype=bool),
        pit_status=pit_status,
        segment_edge_buffer_seconds=0.5,
        artifact_buffer_seconds=0.0,
    )

    assert np.isclose(exposure.prebuffer_seconds, 2.8)
    assert np.isclose(exposure.buffered_seconds, 1.8)


def test_hard_break_does_not_create_a_half_second_edge_buffer() -> None:
    times = np.arange(31, dtype=float) * 0.1
    axis = _axis(times)
    axis.hard_break[15] = True

    exposure = compute_interval_exposure(
        axis,
        active=np.ones(31, dtype=bool),
        pit_status=np.zeros(31, dtype=int),
        segment_edge_buffer_seconds=0.5,
        artifact_buffer_seconds=0.0,
    )

    assert np.isclose(exposure.prebuffer_seconds, 2.9)
    assert np.isclose(exposure.buffered_seconds, 1.9)


def test_zero_duration_clock_jitter_does_not_create_an_edge_buffer() -> None:
    times = np.arange(31, dtype=float) * 0.1
    times[16] = times[15]

    exposure = compute_interval_exposure(
        _axis(times),
        active=np.ones(31, dtype=bool),
        pit_status=np.zeros(31, dtype=int),
        segment_edge_buffer_seconds=0.5,
        artifact_buffer_seconds=0.0,
    )

    assert np.isclose(exposure.prebuffer_seconds, 3.0)
    assert np.isclose(exposure.buffered_seconds, 2.0)


def test_artifact_buffer_uses_registered_physical_reset_seed() -> None:
    times = np.arange(21, dtype=float) * 0.1
    axis = _axis(times)
    axis.artifact_proximity_seed[10] = True

    exposure = compute_interval_exposure(
        axis,
        active=np.ones(21, dtype=bool),
        pit_status=np.zeros(21, dtype=int),
        segment_edge_buffer_seconds=0.0,
        artifact_buffer_seconds=0.25,
    )

    assert np.isclose(exposure.prebuffer_seconds, 2.0)
    assert np.isclose(exposure.buffered_seconds, 1.4)
