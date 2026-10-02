"""Transparent time reconstruction and causal fixed-rate resampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from brace_f1.io import DataValidationError


@dataclass(frozen=True)
class TimingDiagnostics:
    """Auditable facts and detected artifacts for a reconstructed clock."""

    timestamp_method: str
    native_frame_timestamps_available: bool
    sample_count: int
    median_source_dt_seconds: float
    estimated_source_hz: float
    lap_boundary_count: int
    clock_reset_count: int
    data_gap_count: int
    position_jump_count: int
    distance_reset_count: int
    status_transition_count: int
    hard_break_count: int


@dataclass(frozen=True)
class TimeAxis:
    """A monotone reconstructed time axis with explicit discontinuity flags."""

    time_seconds: NDArray[np.float64]
    hard_break: NDArray[np.bool_]
    lap_boundary: NDArray[np.bool_]
    clock_reset: NDArray[np.bool_]
    data_gap: NDArray[np.bool_]
    artifact_proximity_seed: NDArray[np.bool_]
    diagnostics: TimingDiagnostics

    @property
    def segment_id(self) -> NDArray[np.int64]:
        return np.cumsum(self.hard_break, dtype=np.int64) - 1


@dataclass(frozen=True)
class CausalResample:
    """Zero-order-hold samples selected at or before each fixed-grid time."""

    target_time_seconds: NDArray[np.float64]
    source_indices: NDArray[np.int64]
    source_age_seconds: NDArray[np.float64]
    segment_id: NDArray[np.int64]
    arrays: Mapping[str, NDArray[np.generic]]
    frequency_hz: float


def _one_dimensional(
    values: ArrayLike, name: str, sample_count: int | None = None
) -> NDArray[np.generic]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise DataValidationError(f"{name} must be one-dimensional")
    if sample_count is not None and array.shape[0] != sample_count:
        raise DataValidationError(f"{name} length differs from motion samples")
    return array


def reconstruct_time_axis(
    lap: Mapping[str, ArrayLike],
    positions: ArrayLike,
    *,
    clock_reset_tolerance_seconds: float = 0.05,
    position_jump_threshold_m: float = 5.0,
    distance_reset_threshold_m: float = -5.0,
) -> TimeAxis:
    """Reconstruct elapsed time from lap clocks while exposing every discontinuity.

    The compact per-car files contain only scalar header timestamps, not a native
    timestamp for every motion sample. This function therefore does not represent
    its output as native or exact-rate telemetry.
    """

    required = {
        "current_lap_times",
        "lap_numbers",
        "last_lap_times",
        "total_distances",
        "result_status",
    }
    missing = required.difference(lap)
    if missing:
        raise DataValidationError(f"time reconstruction missing fields: {sorted(missing)}")
    current = np.asarray(
        _one_dimensional(lap["current_lap_times"], "current_lap_times"), dtype=float
    )
    sample_count = current.shape[0]
    if sample_count < 2 or not np.isfinite(current).all():
        raise DataValidationError("current_lap_times must contain at least two finite samples")
    lap_numbers = np.asarray(
        _one_dimensional(lap["lap_numbers"], "lap_numbers", sample_count), dtype=np.int64
    )
    last_lap = np.asarray(
        _one_dimensional(lap["last_lap_times"], "last_lap_times", sample_count), dtype=float
    )
    total_distance = np.asarray(
        _one_dimensional(lap["total_distances"], "total_distances", sample_count), dtype=float
    )
    result_status = np.asarray(
        _one_dimensional(lap["result_status"], "result_status", sample_count), dtype=np.int64
    )
    position_array = np.asarray(positions, dtype=float)
    if position_array.shape != (sample_count, 3) or not np.isfinite(position_array).all():
        raise DataValidationError("positions must be a finite (N, 3) array")
    if not np.isfinite(total_distance).all():
        raise DataValidationError("total_distances contains non-finite values")

    current_delta = np.diff(current)
    same_lap = np.diff(lap_numbers) == 0
    cadence_candidates = current_delta[
        same_lap & (current_delta > 0.0) & (current_delta < 0.05)
    ]
    if cadence_candidates.size == 0:
        raise DataValidationError("cannot estimate cadence from the lap clock")
    median_dt = float(np.median(cadence_candidates))
    gap_threshold = max(0.05, 5.0 * median_dt)

    lap_delta = np.diff(lap_numbers, prepend=lap_numbers[0])
    lap_boundary = lap_delta == 1
    clock_reset = np.zeros(sample_count, dtype=bool)
    clock_reset[1:] = (
        ((lap_delta[1:] == 0) & (current_delta < -clock_reset_tolerance_seconds))
        | (lap_delta[1:] < 0)
        | (lap_delta[1:] > 1)
    )

    source_dt = np.full(sample_count, median_dt, dtype=float)
    source_dt[0] = 0.0
    same_indices = np.flatnonzero(lap_delta[1:] == 0) + 1
    source_dt[same_indices] = current_delta[same_indices - 1]
    boundary_indices = np.flatnonzero(lap_boundary)
    if boundary_indices.size:
        boundary_dt = (
            last_lap[boundary_indices]
            - current[boundary_indices - 1]
            + current[boundary_indices]
        )
        source_dt[boundary_indices] = boundary_dt

    data_gap = np.zeros(sample_count, dtype=bool)
    data_gap[1:] = (~np.isfinite(source_dt[1:])) | (source_dt[1:] > gap_threshold)
    position_jump = np.zeros(sample_count, dtype=bool)
    position_jump[1:] = (
        np.linalg.norm(np.diff(position_array, axis=0), axis=1)
        > position_jump_threshold_m
    )
    distance_reset = np.zeros(sample_count, dtype=bool)
    distance_reset[1:] = np.diff(total_distance) < distance_reset_threshold_m
    active = result_status == 2
    status_transition = np.zeros(sample_count, dtype=bool)
    status_transition[1:] = active[1:] != active[:-1]

    hard_break = clock_reset | data_gap | position_jump | distance_reset | status_transition
    hard_break[0] = True
    elapsed = np.zeros(sample_count, dtype=float)
    for index in range(1, sample_count):
        increment = median_dt if hard_break[index] else max(float(source_dt[index]), 0.0)
        elapsed[index] = elapsed[index - 1] + increment
    if np.any(np.diff(elapsed) < 0):
        raise DataValidationError("reconstructed time is not monotone")

    artifact_seed = hard_break | lap_boundary
    diagnostics = TimingDiagnostics(
        timestamp_method="reconstructed_from_lap_clock",
        native_frame_timestamps_available=False,
        sample_count=sample_count,
        median_source_dt_seconds=median_dt,
        estimated_source_hz=1.0 / median_dt,
        lap_boundary_count=int(np.count_nonzero(lap_boundary)),
        clock_reset_count=int(np.count_nonzero(clock_reset)),
        data_gap_count=int(np.count_nonzero(data_gap)),
        position_jump_count=int(np.count_nonzero(position_jump)),
        distance_reset_count=int(np.count_nonzero(distance_reset)),
        status_transition_count=int(np.count_nonzero(status_transition)),
        hard_break_count=int(np.count_nonzero(hard_break) - 1),
    )
    return TimeAxis(
        time_seconds=elapsed,
        hard_break=hard_break,
        lap_boundary=lap_boundary,
        clock_reset=clock_reset,
        data_gap=data_gap,
        artifact_proximity_seed=artifact_seed,
        diagnostics=diagnostics,
    )


def causal_resample(
    axis: TimeAxis,
    arrays: Mapping[str, ArrayLike],
    *,
    frequency_hz: float = 20.0,
) -> CausalResample:
    """Resample independently within each continuous segment using past-only values."""

    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise DataValidationError("resampling frequency must be positive and finite")
    source_count = axis.time_seconds.shape[0]
    converted: dict[str, NDArray[np.generic]] = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != source_count:
            raise DataValidationError(
                f"resampled array {name} must have leading length {source_count}"
            )
        converted[name] = array

    step = 1.0 / frequency_hz
    source_indices_parts: list[NDArray[np.int64]] = []
    target_parts: list[NDArray[np.float64]] = []
    segment_parts: list[NDArray[np.int64]] = []
    segment_ids = axis.segment_id
    for segment_id in np.unique(segment_ids):
        indices = np.flatnonzero(segment_ids == segment_id)
        start = float(axis.time_seconds[indices[0]])
        stop = float(axis.time_seconds[indices[-1]])
        step_count = int(np.floor((stop - start) / step + 1e-9))
        targets = start + np.arange(step_count + 1, dtype=float) * step
        local_times = axis.time_seconds[indices]
        local_positions = np.searchsorted(local_times, targets, side="right") - 1
        local_positions = np.maximum(local_positions, 0)
        source_indices_parts.append(indices[local_positions].astype(np.int64, copy=False))
        target_parts.append(targets)
        segment_parts.append(np.full(targets.shape, segment_id, dtype=np.int64))

    source_indices = np.concatenate(source_indices_parts)
    targets = np.concatenate(target_parts)
    target_segments = np.concatenate(segment_parts)
    source_age = targets - axis.time_seconds[source_indices]
    if np.any(source_age < -1e-10):
        raise DataValidationError("causal resampling selected a future source frame")
    sampled_arrays = {name: values[source_indices] for name, values in converted.items()}
    return CausalResample(
        target_time_seconds=targets,
        source_indices=source_indices,
        source_age_seconds=source_age,
        segment_id=target_segments,
        arrays=sampled_arrays,
        frequency_hz=frequency_hz,
    )
