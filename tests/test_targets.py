from __future__ import annotations

import numpy as np
import pandas as pd

from brace_f1.targets import project_raw_events_to_model_grid


def test_raw_event_onset_projects_forward_to_first_model_grid_frame() -> None:
    frames = pd.DataFrame(
        {
            "circuit": ["A"] * 4,
            "car_id": ["car_1"] * 4,
            "frame_index": [0, 1, 2, 3],
            "time_seconds": [0.00, 0.05, 0.10, 0.15],
            "continuous_segment_id": [0, 0, 0, 0],
        }
    )
    events = pd.DataFrame(
        {
            "candidate_event_id": ["event_1"],
            "circuit": ["A"],
            "car_id": ["car_1"],
            "continuous_segment_id": [0],
            "start_time_seconds": [0.075],
            "qualified": [True],
            "side_at_onset": ["left"],
            "boundary_id_at_onset": ["edge_a"],
            "centerline_arclength_wrapped_m_at_onset": [51.0],
            "segment_bin_25m_at_onset": [2],
        }
    )

    projected = project_raw_events_to_model_grid(frames, events)

    event = projected.events.iloc[0]
    assert event["projected_onset_model_frame_index"] == 2
    assert np.isclose(event["projected_onset_time_seconds"], 0.10)
    assert np.isclose(event["onset_projection_delay_seconds"], 0.025)
    assert projected.frames["qualifying_event_onset_projected"].tolist() == [
        False,
        False,
        True,
        False,
    ]
    np.testing.assert_allclose(
        projected.frames.loc[:1, "time_to_next_excursion_seconds"], [0.075, 0.025]
    )
    assert projected.frames.loc[0, "next_excursion_segment_bin_25m"] == 2
    assert projected.frames.loc[0, "next_excursion_side"] == "left"
    assert projected.frames.loc[2:, "next_qualifying_event_id"].isna().all()


def test_event_targets_never_cross_continuous_segments() -> None:
    frames = pd.DataFrame(
        {
            "circuit": ["A"] * 4,
            "car_id": ["car_1"] * 4,
            "frame_index": [0, 1, 2, 3],
            "time_seconds": [0.00, 0.05, 0.10, 0.15],
            "continuous_segment_id": [0, 0, 1, 1],
        }
    )
    events = pd.DataFrame(
        {
            "candidate_event_id": ["event_1"],
            "circuit": ["A"],
            "car_id": ["car_1"],
            "continuous_segment_id": [1],
            "start_time_seconds": [0.125],
            "qualified": [True],
            "side_at_onset": ["right"],
            "boundary_id_at_onset": ["edge_b"],
            "centerline_arclength_wrapped_m_at_onset": [24.0],
            "segment_bin_25m_at_onset": [0],
        }
    )

    projected = project_raw_events_to_model_grid(frames, events)

    assert projected.frames.loc[:1, "next_qualifying_event_id"].isna().all()
    assert projected.frames.loc[2, "next_qualifying_event_id"] == "event_1"
