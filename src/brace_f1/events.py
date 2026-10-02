"""Transparent mapped track-boundary excursion definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from brace_f1.io import DataValidationError


@dataclass(frozen=True)
class ExcursionConfig:
    """Predeclared, configurable rules for mapped boundary excursions."""

    gap_merge_seconds: float = 0.25
    min_max_depth_m: float = 0.50
    min_duration_seconds: float = 0.25
    min_lead_in_seconds: float = 0.0
    artifact_exclusion_seconds: float = 0.25
    segment_edge_exclusion_seconds: float = 0.50
    exclude_truncated: bool = True

    def __post_init__(self) -> None:
        for name in (
            "gap_merge_seconds",
            "min_max_depth_m",
            "min_duration_seconds",
            "min_lead_in_seconds",
            "artifact_exclusion_seconds",
            "segment_edge_exclusion_seconds",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise DataValidationError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ExcursionTables:
    """Frame-level exposure plus an auditable table of all event candidates."""

    frames: pd.DataFrame
    events: pd.DataFrame


_REQUIRED_FRAME_COLUMNS = {
    "circuit",
    "car_id",
    "frame_index",
    "time_seconds",
    "in_corridor",
    "outside_depth_m",
    "hard_break",
    "artifact_seed",
    "pit_status",
    "active",
    "local_side",
    "nearest_boundary_id",
    "centerline_segment_index_pcd",
    "centerline_arclength_wrapped_m",
}

_EVENT_COLUMNS = [
    "candidate_event_id",
    "circuit",
    "car_id",
    "continuous_segment_id",
    "start_frame_index",
    "end_frame_index",
    "start_time_seconds",
    "end_time_seconds",
    "duration_seconds",
    "outside_frame_count",
    "merged_gap_frame_count",
    "max_outside_depth_m",
    "side_at_onset",
    "boundary_id_at_onset",
    "centerline_segment_index_pcd_at_onset",
    "centerline_arclength_wrapped_m_at_onset",
    "segment_bin_25m_at_onset",
    "side_at_max_depth",
    "boundary_id_at_max_depth",
    "centerline_segment_index_pcd_at_max_depth",
    "centerline_arclength_wrapped_m_at_max_depth",
    "segment_bin_25m_at_max_depth",
    "start_truncated",
    "end_truncated",
    "pit_overlap",
    "inactive_overlap",
    "artifact_proximity",
    "available_lead_in_seconds",
    "active_file_lead_in_seconds",
    "active_file_followup_seconds",
    "available_followup_seconds",
    "segment_edge_proximity",
    "qualified",
    "rejection_reason",
]


def _artifact_proximity_mask(times: np.ndarray, seeds: np.ndarray, radius: float) -> np.ndarray:
    mask = np.zeros(times.shape[0], dtype=bool)
    for seed_time in times[seeds]:
        left = np.searchsorted(times, seed_time - radius - 1e-12, side="left")
        right = np.searchsorted(times, seed_time + radius + 1e-12, side="right")
        mask[left:right] = True
    return mask


def _outside_runs(indices: np.ndarray, outside: np.ndarray) -> list[tuple[int, int]]:
    outside_indices = indices[outside[indices]]
    if outside_indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(outside_indices) > 1)
    starts = np.concatenate(([outside_indices[0]], outside_indices[breaks + 1]))
    ends = np.concatenate((outside_indices[breaks], [outside_indices[-1]]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _merge_runs(
    runs: list[tuple[int, int]], times: np.ndarray, sample_interval: float, max_gap: float
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        gap_duration = max(0.0, times[start] - times[previous_end] - sample_interval)
        if gap_duration <= max_gap + 1e-12:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _active_run_bounds(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return active/file run bounds without splitting at hard breaks or pit samples."""

    run_start = np.full(active.shape[0], -1, dtype=np.int64)
    run_end = np.full(active.shape[0], -1, dtype=np.int64)
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        return run_start, run_end
    breaks = np.flatnonzero(np.diff(active_indices) > 1)
    starts = np.concatenate(([active_indices[0]], active_indices[breaks + 1]))
    ends = np.concatenate((active_indices[breaks], [active_indices[-1]]))
    for start, end in zip(starts, ends, strict=True):
        run_start[start : end + 1] = start
        run_end[start : end + 1] = end
    return run_start, run_end


def _contiguous_clean_lead_in(
    times: np.ndarray,
    start: int,
    *,
    active: np.ndarray,
    pit: np.ndarray,
    in_corridor: np.ndarray,
    hard_break: np.ndarray,
) -> float:
    """Return clean observed time immediately preceding an excursion onset."""

    if not active[start] or pit[start]:
        return 0.0
    lead_in = 0.0
    for right in range(start, 0, -1):
        left = right - 1
        if hard_break[right]:
            break
        if not (active[left] and not pit[left] and in_corridor[left]):
            break
        if right != start and not in_corridor[right]:
            break
        lead_in += max(float(times[right] - times[left]), 0.0)
    return lead_in


def _rejection_reason(
    *,
    max_depth: float,
    duration: float,
    start_truncated: bool,
    end_truncated: bool,
    pit_overlap: bool,
    inactive_overlap: bool,
    artifact_proximity: bool,
    available_lead_in: float,
    active_file_lead_in: float,
    active_file_followup: float,
    config: ExcursionConfig,
) -> str:
    if max_depth < config.min_max_depth_m:
        return "max_depth_below_threshold"
    if duration < config.min_duration_seconds:
        return "duration_below_threshold"
    if config.exclude_truncated and start_truncated:
        return "start_truncated"
    if config.exclude_truncated and end_truncated:
        return "end_truncated"
    if (
        active_file_lead_in + 1e-12 < config.segment_edge_exclusion_seconds
        or active_file_followup + 1e-12 < config.segment_edge_exclusion_seconds
    ):
        return "segment_edge_proximity"
    if pit_overlap:
        return "pit_overlap"
    if inactive_overlap:
        return "inactive_overlap"
    if artifact_proximity:
        return "artifact_proximity"
    if available_lead_in + 1e-12 < config.min_lead_in_seconds:
        return "insufficient_lead_in"
    return ""


def label_excursions(frames: pd.DataFrame, config: ExcursionConfig) -> ExcursionTables:
    """Merge and qualify mapped car-center boundary excursions for one car."""

    missing = _REQUIRED_FRAME_COLUMNS.difference(frames.columns)
    if missing:
        raise DataValidationError(f"frame table missing columns: {sorted(missing)}")
    if frames.empty:
        raise DataValidationError("frame table is empty")
    if frames["circuit"].nunique(dropna=False) != 1 or frames["car_id"].nunique(dropna=False) != 1:
        raise DataValidationError("label_excursions expects exactly one circuit/car group")

    output = frames.reset_index(drop=True).copy()
    times = output["time_seconds"].to_numpy(dtype=float)
    time_differences = np.diff(times)
    if not np.isfinite(times).all() or np.any(time_differences < 0):
        raise DataValidationError("frame times must be finite and monotone")
    positive_differences = time_differences[time_differences > 0]
    if positive_differences.size == 0:
        raise DataValidationError("frame times contain no positive sampling interval")
    sample_interval = float(np.median(positive_differences))
    outside_depth = output["outside_depth_m"].to_numpy(dtype=float)
    if not np.isfinite(outside_depth).all() or np.any(outside_depth < 0):
        raise DataValidationError("outside depth must be finite and non-negative")
    outside = ~output["in_corridor"].to_numpy(dtype=bool)
    hard_break = output["hard_break"].to_numpy(dtype=bool, copy=True)
    hard_break[0] = True
    segment_id = np.cumsum(hard_break, dtype=np.int64) - 1
    artifact_seed = output["artifact_seed"].to_numpy(dtype=bool)
    artifact_proximity_frame = _artifact_proximity_mask(
        times, artifact_seed, config.artifact_exclusion_seconds
    )
    active = output["active"].to_numpy(dtype=bool)
    pit = output["pit_status"].to_numpy(dtype=int) != 0
    in_corridor = output["in_corridor"].to_numpy(dtype=bool)
    active_transition_without_break = (active[1:] != active[:-1]) & ~hard_break[1:]
    if np.any(active_transition_without_break):
        raise DataValidationError("active transition must coincide with a hard break")
    active_run_start, active_run_end = _active_run_bounds(active)

    output["raw_outside"] = outside
    output["segment_bin_25m"] = np.floor(
        output["centerline_arclength_wrapped_m"].to_numpy(dtype=float) / 25.0
    ).astype(np.int64)
    output["continuous_segment_id"] = segment_id
    output["artifact_excluded"] = artifact_proximity_frame
    output["eligible_exposure"] = active & ~pit & ~artifact_proximity_frame
    output["candidate_event_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["qualifying_event_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["qualifying_event_onset"] = False

    event_rows: list[dict[str, object]] = []
    circuit = str(output["circuit"].iloc[0])
    car_id = str(output["car_id"].iloc[0])
    candidate_number = 0
    for segment in np.unique(segment_id):
        segment_indices = np.flatnonzero(segment_id == segment)
        segment_active = active[segment_indices]
        if not np.any(segment_active):
            continue
        if not np.all(segment_active):
            raise DataValidationError("hard-break segment mixes active and inactive samples")
        runs = _outside_runs(segment_indices, outside)
        runs = _merge_runs(runs, times, sample_interval, config.gap_merge_seconds)
        for start, end in runs:
            candidate_number += 1
            candidate_id = f"{circuit}:{car_id}:excursion_candidate_{candidate_number:04d}"
            span = np.arange(start, end + 1)
            outside_span = span[outside[span]]
            max_index = int(outside_span[np.argmax(outside_depth[outside_span])])
            max_depth = float(outside_depth[max_index])
            duration = float(times[end] - times[start] + sample_interval)
            segment_start = int(segment_indices[0])
            active_start = int(active_run_start[start])
            active_end = int(active_run_end[end])
            segment_end = int(segment_indices[-1])
            start_truncated = start == segment_start and bool(outside[start])
            end_truncated = end == segment_end and bool(outside[end])
            pit_overlap = bool(np.any(pit[span]))
            inactive_overlap = bool(np.any(~active[span]))
            proximity_left = np.searchsorted(
                times, times[start] - config.artifact_exclusion_seconds - 1e-12, side="left"
            )
            proximity_right = np.searchsorted(
                times, times[end] + config.artifact_exclusion_seconds + 1e-12, side="right"
            )
            artifact_proximity = bool(np.any(artifact_seed[proximity_left:proximity_right]))
            available_lead_in = _contiguous_clean_lead_in(
                times,
                start,
                active=active,
                pit=pit,
                in_corridor=in_corridor,
                hard_break=hard_break,
            )
            active_file_lead_in = float(times[start] - times[active_start])
            active_file_followup = float(times[active_end] - times[end])
            reason = _rejection_reason(
                max_depth=max_depth,
                duration=duration,
                start_truncated=start_truncated,
                end_truncated=end_truncated,
                pit_overlap=pit_overlap,
                inactive_overlap=inactive_overlap,
                artifact_proximity=artifact_proximity,
                available_lead_in=available_lead_in,
                active_file_lead_in=active_file_lead_in,
                active_file_followup=active_file_followup,
                config=config,
            )
            qualified = not reason
            output.loc[span, "candidate_event_id"] = candidate_id
            if qualified:
                output.loc[span, "qualifying_event_id"] = candidate_id
                output.loc[start, "qualifying_event_onset"] = True
            event_rows.append(
                {
                    "candidate_event_id": candidate_id,
                    "circuit": circuit,
                    "car_id": car_id,
                    "continuous_segment_id": int(segment),
                    "start_frame_index": int(output.loc[start, "frame_index"]),
                    "end_frame_index": int(output.loc[end, "frame_index"]),
                    "start_time_seconds": float(times[start]),
                    "end_time_seconds": float(times[end]),
                    "duration_seconds": duration,
                    "outside_frame_count": int(outside_span.size),
                    "merged_gap_frame_count": int(span.size - outside_span.size),
                    "max_outside_depth_m": max_depth,
                    "side_at_onset": str(output.loc[start, "local_side"]),
                    "boundary_id_at_onset": str(
                        output.loc[start, "nearest_boundary_id"]
                    ),
                    "centerline_segment_index_pcd_at_onset": int(
                        output.loc[start, "centerline_segment_index_pcd"]
                    ),
                    "centerline_arclength_wrapped_m_at_onset": float(
                        output.loc[start, "centerline_arclength_wrapped_m"]
                    ),
                    "segment_bin_25m_at_onset": int(output.loc[start, "segment_bin_25m"]),
                    "side_at_max_depth": str(output.loc[max_index, "local_side"]),
                    "boundary_id_at_max_depth": str(
                        output.loc[max_index, "nearest_boundary_id"]
                    ),
                    "centerline_segment_index_pcd_at_max_depth": int(
                        output.loc[max_index, "centerline_segment_index_pcd"]
                    ),
                    "centerline_arclength_wrapped_m_at_max_depth": float(
                        output.loc[max_index, "centerline_arclength_wrapped_m"]
                    ),
                    "segment_bin_25m_at_max_depth": int(
                        output.loc[max_index, "segment_bin_25m"]
                    ),
                    "start_truncated": start_truncated,
                    "end_truncated": end_truncated,
                    "pit_overlap": pit_overlap,
                    "inactive_overlap": inactive_overlap,
                    "artifact_proximity": artifact_proximity,
                    "available_lead_in_seconds": available_lead_in,
                    "active_file_lead_in_seconds": active_file_lead_in,
                    "active_file_followup_seconds": active_file_followup,
                    "available_followup_seconds": active_file_followup,
                    "segment_edge_proximity": bool(
                        active_file_lead_in + 1e-12
                        < config.segment_edge_exclusion_seconds
                        or active_file_followup + 1e-12
                        < config.segment_edge_exclusion_seconds
                    ),
                    "qualified": qualified,
                    "rejection_reason": reason,
                }
            )

    events = pd.DataFrame(event_rows, columns=_EVENT_COLUMNS)
    return ExcursionTables(frames=output, events=events)
