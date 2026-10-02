"""Interval-based exposure denominators for false-alert-rate reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from brace_f1.io import DataValidationError
from brace_f1.timebase import TimeAxis


@dataclass(frozen=True)
class ExposureSummary:
    """Pre-buffer and artifact-buffered active, non-pit exposure."""

    prebuffer_seconds: float
    buffered_seconds: float
    prebuffer_interval_count: int
    buffered_interval_count: int
    prebuffer_interval_mask: NDArray[np.bool_]
    buffered_interval_mask: NDArray[np.bool_]


def _boolean_vector(values: ArrayLike, name: str, size: int) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.shape != (size,):
        raise DataValidationError(f"{name} must have shape ({size},)")
    return array.astype(bool, copy=False)


def compute_interval_exposure(
    axis: TimeAxis,
    *,
    active: ArrayLike,
    pit_status: ArrayLike,
    segment_edge_buffer_seconds: float = 0.50,
    artifact_buffer_seconds: float = 0.25,
) -> ExposureSummary:
    """Sum valid source-clock intervals before and after registered safety buffers."""

    for name, value in (
        ("segment_edge_buffer_seconds", segment_edge_buffer_seconds),
        ("artifact_buffer_seconds", artifact_buffer_seconds),
    ):
        if not np.isfinite(value) or value < 0:
            raise DataValidationError(f"{name} must be finite and non-negative")
    times = np.asarray(axis.time_seconds, dtype=float)
    size = times.shape[0]
    if size < 2 or np.any(np.diff(times) < 0):
        raise DataValidationError("exposure requires a monotone time axis with two samples")
    active_vector = _boolean_vector(active, "active", size)
    pit_vector = np.asarray(pit_status)
    if pit_vector.shape != (size,):
        raise DataValidationError(f"pit_status must have shape ({size},)")
    in_pit = pit_vector != 0
    durations = np.diff(times)
    active_pairs = active_vector[:-1] & active_vector[1:]
    valid_intervals = (durations > 0) & active_pairs & ~axis.hard_break[1:]
    prebuffer = valid_intervals & ~in_pit[:-1] & ~in_pit[1:]
    buffered = prebuffer.copy()

    active_indices = np.flatnonzero(active_pairs)
    if active_indices.size and segment_edge_buffer_seconds > 0:
        run_breaks = np.flatnonzero(np.diff(active_indices) > 1)
        run_starts = np.concatenate(([active_indices[0]], active_indices[run_breaks + 1]))
        run_ends = np.concatenate((active_indices[run_breaks], [active_indices[-1]]))
        for start, end in zip(run_starts, run_ends, strict=True):
            segment_start = times[start]
            segment_end = times[end + 1]
            interval_indices = np.arange(start, end + 1)
            edge_safe = (
                (times[interval_indices] >= segment_start + segment_edge_buffer_seconds - 1e-12)
                & (
                    times[interval_indices + 1]
                    <= segment_end - segment_edge_buffer_seconds + 1e-12
                )
            )
            buffered[interval_indices] &= edge_safe

    artifact_seeds = axis.artifact_proximity_seed
    if artifact_buffer_seconds > 0:
        interval_start = times[:-1]
        interval_end = times[1:]
        for artifact_time in times[artifact_seeds]:
            overlaps = (interval_start < artifact_time + artifact_buffer_seconds) & (
                interval_end > artifact_time - artifact_buffer_seconds
            )
            buffered[overlaps] = False

    return ExposureSummary(
        prebuffer_seconds=float(np.sum(durations[prebuffer])),
        buffered_seconds=float(np.sum(durations[buffered])),
        prebuffer_interval_count=int(np.count_nonzero(prebuffer)),
        buffered_interval_count=int(np.count_nonzero(buffered)),
        prebuffer_interval_mask=prebuffer,
        buffered_interval_mask=buffered,
    )
