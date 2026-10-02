"""Projection of raw-frame excursion outcomes onto the causal 20 Hz model grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from brace_f1.io import DataValidationError


@dataclass(frozen=True)
class ProjectedEventTables:
    """Model-grid targets plus raw-qualified events with projection diagnostics."""

    frames: pd.DataFrame
    events: pd.DataFrame


_FRAME_REQUIRED = {
    "circuit",
    "car_id",
    "frame_index",
    "time_seconds",
    "continuous_segment_id",
}
_EVENT_REQUIRED = {
    "candidate_event_id",
    "circuit",
    "car_id",
    "continuous_segment_id",
    "start_time_seconds",
    "qualified",
    "side_at_onset",
    "boundary_id_at_onset",
    "centerline_arclength_wrapped_m_at_onset",
    "segment_bin_25m_at_onset",
}


def project_raw_events_to_model_grid(
    frames: pd.DataFrame, events: pd.DataFrame
) -> ProjectedEventTables:
    """Attach next-onset targets without allowing outcomes to cross hard-break segments."""

    missing_frames = _FRAME_REQUIRED.difference(frames.columns)
    missing_events = _EVENT_REQUIRED.difference(events.columns)
    if missing_frames:
        raise DataValidationError(f"model frame table missing columns: {sorted(missing_frames)}")
    if missing_events:
        raise DataValidationError(f"raw event table missing columns: {sorted(missing_events)}")
    output_frames = frames.copy()
    output_events = events.copy()
    output_frames["qualifying_event_onset_projected"] = False
    output_frames["projected_onset_event_id"] = pd.Series(
        pd.NA, index=output_frames.index, dtype="string"
    )
    output_frames["next_qualifying_event_id"] = pd.Series(
        pd.NA, index=output_frames.index, dtype="string"
    )
    output_frames["time_to_next_excursion_seconds"] = np.nan
    output_frames["next_excursion_side"] = pd.Series(
        pd.NA, index=output_frames.index, dtype="string"
    )
    output_frames["next_excursion_boundary_id"] = pd.Series(
        pd.NA, index=output_frames.index, dtype="string"
    )
    output_frames["next_excursion_arclength_wrapped_m"] = np.nan
    output_frames["next_excursion_segment_bin_25m"] = pd.Series(
        pd.NA, index=output_frames.index, dtype="Int64"
    )
    output_events["projected_onset_model_frame_index"] = pd.Series(
        pd.NA, index=output_events.index, dtype="Int64"
    )
    output_events["projected_onset_time_seconds"] = np.nan
    output_events["onset_projection_delay_seconds"] = np.nan

    qualified = output_events.loc[output_events["qualified"].astype(bool)]
    group_columns = ["circuit", "car_id", "continuous_segment_id"]
    for group_key, frame_group in output_frames.groupby(group_columns, sort=False):
        circuit, car_id, segment_id = group_key
        event_group = qualified.loc[
            (qualified["circuit"] == circuit)
            & (qualified["car_id"] == car_id)
            & (qualified["continuous_segment_id"] == segment_id)
        ].sort_values("start_time_seconds", kind="stable")
        if event_group.empty:
            continue
        frame_indices = frame_group.index.to_numpy()
        frame_times = frame_group["time_seconds"].to_numpy(dtype=float)
        onset_times = event_group["start_time_seconds"].to_numpy(dtype=float)

        for event_index, event in event_group.iterrows():
            onset = float(event["start_time_seconds"])
            local_projection = int(np.searchsorted(frame_times, onset, side="left"))
            if local_projection >= frame_times.size:
                continue
            frame_row = frame_indices[local_projection]
            projected_time = float(frame_times[local_projection])
            output_frames.loc[frame_row, "qualifying_event_onset_projected"] = True
            output_frames.loc[frame_row, "projected_onset_event_id"] = event[
                "candidate_event_id"
            ]
            output_events.loc[event_index, "projected_onset_model_frame_index"] = int(
                output_frames.loc[frame_row, "frame_index"]
            )
            output_events.loc[event_index, "projected_onset_time_seconds"] = projected_time
            output_events.loc[event_index, "onset_projection_delay_seconds"] = (
                projected_time - onset
            )

        next_positions = np.searchsorted(onset_times, frame_times, side="left")
        valid = next_positions < onset_times.size
        if not np.any(valid):
            continue
        valid_rows = frame_indices[valid]
        selected = event_group.iloc[next_positions[valid]]
        output_frames.loc[valid_rows, "next_qualifying_event_id"] = selected[
            "candidate_event_id"
        ].to_numpy()
        output_frames.loc[valid_rows, "time_to_next_excursion_seconds"] = (
            selected["start_time_seconds"].to_numpy(dtype=float) - frame_times[valid]
        )
        output_frames.loc[valid_rows, "next_excursion_side"] = selected[
            "side_at_onset"
        ].to_numpy()
        output_frames.loc[valid_rows, "next_excursion_boundary_id"] = selected[
            "boundary_id_at_onset"
        ].to_numpy()
        output_frames.loc[valid_rows, "next_excursion_arclength_wrapped_m"] = selected[
            "centerline_arclength_wrapped_m_at_onset"
        ].to_numpy(dtype=float)
        output_frames.loc[valid_rows, "next_excursion_segment_bin_25m"] = selected[
            "segment_bin_25m_at_onset"
        ].to_numpy(dtype=np.int64)

    return ProjectedEventTables(frames=output_frames, events=output_events)
