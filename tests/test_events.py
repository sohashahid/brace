from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brace_f1.events import ExcursionConfig, label_excursions
from brace_f1.io import DataValidationError


def _frames(outside_depth: list[float]) -> pd.DataFrame:
    n = len(outside_depth)
    depth = np.asarray(outside_depth, dtype=float)
    return pd.DataFrame(
        {
            "circuit": ["Fixture"] * n,
            "car_id": ["car_7"] * n,
            "frame_index": np.arange(n),
            "time_seconds": np.arange(n) * 0.05,
            "in_corridor": depth == 0.0,
            "outside_depth_m": depth,
            "hard_break": [True] + [False] * (n - 1),
            "artifact_seed": [False] * n,
            "pit_status": np.zeros(n, dtype=int),
            "active": np.ones(n, dtype=bool),
            "local_side": ["left"] * n,
            "nearest_boundary_id": ["boundary_a"] * n,
            "centerline_segment_index_pcd": np.arange(n) % 4,
            "centerline_arclength_wrapped_m": np.arange(n) * 25.0 + 1.0,
        }
    )


def _config(**changes: float | bool) -> ExcursionConfig:
    values: dict[str, float | bool] = {
        "gap_merge_seconds": 0.10,
        "min_max_depth_m": 0.50,
        "min_duration_seconds": 0.20,
        "min_lead_in_seconds": 0.20,
        "artifact_exclusion_seconds": 0.10,
        "segment_edge_exclusion_seconds": 0.0,
        "exclude_truncated": True,
    }
    values.update(changes)
    return ExcursionConfig(**values)


def test_primary_default_reports_lead_in_without_excluding_events() -> None:
    assert ExcursionConfig().min_lead_in_seconds == 0.0
    assert ExcursionConfig().segment_edge_exclusion_seconds == 0.5


def test_chattering_runs_merge_into_one_qualifying_excursion() -> None:
    depth = [0.0] * 5 + [0.6, 0.7] + [0.0, 0.0] + [0.8] * 5 + [0.0] * 5

    tables = label_excursions(_frames(depth), _config())

    assert len(tables.events) == 1
    event = tables.events.iloc[0]
    assert bool(event["qualified"])
    assert event["start_frame_index"] == 5
    assert event["end_frame_index"] == 13
    assert event["outside_frame_count"] == 7
    assert event["merged_gap_frame_count"] == 2
    assert event["max_outside_depth_m"] == 0.8
    assert event["side_at_onset"] == "left"
    assert event["boundary_id_at_onset"] == "boundary_a"
    assert event["centerline_arclength_wrapped_m_at_onset"] == 126.0
    assert event["segment_bin_25m_at_onset"] == 5
    assert event["side_at_max_depth"] == "left"
    assert tables.frames.loc[5:13, "candidate_event_id"].nunique() == 1


def test_shallow_noisy_excursion_is_rejected_but_audited() -> None:
    depth = [0.0] * 6 + [0.1, 0.2, 0.1, 0.2, 0.1] + [0.0] * 5

    tables = label_excursions(_frames(depth), _config())

    assert len(tables.events) == 1
    event = tables.events.iloc[0]
    assert not bool(event["qualified"])
    assert event["rejection_reason"] == "max_depth_below_threshold"
    assert tables.frames["qualifying_event_id"].isna().all()


def test_start_and_end_truncated_excursions_are_rejected() -> None:
    depth = [0.8] * 4 + [0.0] * 10 + [0.9] * 5

    tables = label_excursions(
        _frames(depth),
        _config(min_lead_in_seconds=0.0, gap_merge_seconds=0.05),
    )

    assert len(tables.events) == 2
    assert tables.events["rejection_reason"].tolist() == [
        "start_truncated",
        "end_truncated",
    ]


def test_all_negative_exposure_is_retained_without_events() -> None:
    tables = label_excursions(_frames([0.0] * 20), _config())

    assert tables.events.empty
    assert len(tables.frames) == 20
    assert tables.frames["eligible_exposure"].all()
    assert not tables.frames["raw_outside"].any()
    assert tables.frames["candidate_event_id"].isna().all()


def test_small_clock_jitter_clamped_to_equal_time_is_accepted() -> None:
    frames = _frames([0.0] * 10)
    frames.loc[4, "time_seconds"] = frames.loc[3, "time_seconds"]

    tables = label_excursions(frames, _config())

    assert tables.events.empty
    assert len(tables.frames) == 10


def test_pit_and_reset_adjacent_candidates_are_rejected() -> None:
    depth = [0.0] * 5 + [0.8] * 5 + [0.0] * 6 + [0.9] * 5 + [0.0] * 4
    frames = _frames(depth)
    frames.loc[7, "pit_status"] = 1
    frames.loc[14, "artifact_seed"] = True

    tables = label_excursions(frames, _config())

    assert tables.events["rejection_reason"].tolist() == [
        "pit_overlap",
        "artifact_proximity",
    ]


def test_candidate_without_required_clean_lead_in_is_rejected() -> None:
    depth = [0.0, 0.0] + [0.8] * 5 + [0.0] * 5

    tables = label_excursions(_frames(depth), _config(min_lead_in_seconds=0.25))

    assert len(tables.events) == 1
    assert tables.events.iloc[0]["rejection_reason"] == "insufficient_lead_in"


def test_candidate_inside_registered_segment_edge_buffer_is_rejected() -> None:
    depth = [0.0] * 8 + [0.8] * 5 + [0.0] * 28

    tables = label_excursions(
        _frames(depth),
        _config(segment_edge_exclusion_seconds=0.5, min_lead_in_seconds=0.0),
    )

    assert len(tables.events) == 1
    assert tables.events.iloc[0]["rejection_reason"] == "segment_edge_proximity"


def test_inactive_hard_break_segment_is_not_enumerated_as_a_candidate() -> None:
    depth = [0.0, 0.0] + [0.8] * 5 + [0.0] * 8 + [0.9] * 5 + [0.0] * 10
    frames = _frames(depth)
    frames.loc[:14, "active"] = False
    frames.loc[15, "hard_break"] = True

    tables = label_excursions(
        frames,
        _config(min_lead_in_seconds=0.0, segment_edge_exclusion_seconds=0.0),
    )

    assert len(tables.events) == 1
    assert tables.events.iloc[0]["start_frame_index"] == 15
    assert not tables.events["inactive_overlap"].any()


def test_hard_break_does_not_create_a_half_second_event_edge_buffer() -> None:
    depth = [0.0] * 14 + [0.8] * 7 + [0.0] * 14
    frames = _frames(depth)
    frames.loc[10, "hard_break"] = True

    tables = label_excursions(
        frames,
        _config(
            min_lead_in_seconds=0.0,
            segment_edge_exclusion_seconds=0.5,
            artifact_exclusion_seconds=0.0,
        ),
    )

    assert len(tables.events) == 1
    assert bool(tables.events.iloc[0]["qualified"])


def test_excursion_starting_at_hard_break_remains_truncated() -> None:
    depth = [0.0] * 10 + [0.8] * 7 + [0.0] * 18
    frames = _frames(depth)
    frames.loc[10, "hard_break"] = True

    tables = label_excursions(
        frames,
        _config(
            min_lead_in_seconds=0.0,
            segment_edge_exclusion_seconds=0.0,
            artifact_exclusion_seconds=0.0,
        ),
    )

    assert len(tables.events) == 1
    assert bool(tables.events.iloc[0]["start_truncated"])
    assert tables.events.iloc[0]["rejection_reason"] == "start_truncated"


def test_active_transition_without_hard_break_is_rejected() -> None:
    frames = _frames([0.0] * 20)
    frames.loc[10:, "active"] = False

    with pytest.raises(DataValidationError, match="active transition"):
        label_excursions(frames, _config())


def test_separated_excursion_has_only_contiguous_clean_lead_in() -> None:
    depth = [0.0] * 5 + [0.8] * 3 + [0.0] * 3 + [0.9] * 5 + [0.0] * 10

    tables = label_excursions(
        _frames(depth),
        _config(
            gap_merge_seconds=0.05,
            min_lead_in_seconds=0.20,
            segment_edge_exclusion_seconds=0.0,
            artifact_exclusion_seconds=0.0,
        ),
    )

    assert len(tables.events) == 2
    second = tables.events.iloc[1]
    assert np.isclose(second["available_lead_in_seconds"], 0.15)
    assert np.isclose(second["active_file_lead_in_seconds"], 0.55)
    assert second["rejection_reason"] == "insufficient_lead_in"
