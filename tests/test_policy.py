from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brace_f1.io import DataValidationError
from brace_f1.policy import (
    ProposalConfig,
    label_proposals,
    run_proposal_state_machine,
    select_calibration_threshold,
    select_neighborhood_candidates,
    summarize_detection,
)


def _score_table(
    times: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    in_corridor: np.ndarray | None = None,
    hard_break: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> pd.DataFrame:
    n = times.size
    return pd.DataFrame(
        {
            "circuit": ["Test"] * n,
            "car_id": ["car_1"] * n,
            "frame_index": np.arange(n),
            "time_seconds": times,
            "input_valid_causal": np.ones(n, dtype=bool) if valid is None else valid,
            "in_corridor": np.ones(n, dtype=bool) if in_corridor is None else in_corridor,
            "hard_break": np.zeros(n, dtype=bool) if hard_break is None else hard_break,
            "proposal_score": np.full(n, 0.9) if scores is None else scores,
            "predicted_side": ["left"] * n,
            "predicted_segment_bin": np.full(n, 3, dtype=int),
            "n_segment_bins": np.full(n, 10, dtype=int),
        }
    )


def test_neighborhood_selection_wraps_at_start_finish_and_ignores_unknown_side() -> None:
    outcome = np.zeros((2, 3, 5), dtype=float)
    outcome[0, 0, [4, 0, 1]] = [0.20, 0.30, 0.10]
    outcome[0, 1, 2] = 0.50
    outcome[0, 2, [1, 2, 3]] = 0.90
    outcome[1, 1, [1, 2, 3]] = [0.20, 0.35, 0.25]

    candidates = select_neighborhood_candidates(outcome)

    assert candidates.side.tolist() == ["left", "right"]
    assert candidates.segment_bin.tolist() == [0, 2]
    np.testing.assert_allclose(candidates.score, [0.60, 0.80])


def test_proposal_opens_on_third_completed_score_without_backdating() -> None:
    times = np.arange(0.0, 0.55, 0.05)

    proposals = run_proposal_state_machine(
        _score_table(times), threshold=0.8, config=ProposalConfig()
    )

    assert len(proposals) == 1
    proposal = proposals.iloc[0]
    assert proposal["proposal_time_seconds"] == pytest.approx(0.10)
    assert proposal["proposal_frame_index"] == 2
    assert proposal["resolution_type"] == "stream_end"
    assert bool(proposal["unresolved"])


def test_invalid_score_resets_persistence_and_hard_break_censors_open_proposal() -> None:
    times = np.arange(0.0, 0.40, 0.05)
    valid = np.ones(times.size, dtype=bool)
    valid[1] = False
    hard_break = np.zeros(times.size, dtype=bool)
    hard_break[6] = True

    proposals = run_proposal_state_machine(
        _score_table(times, valid=valid, hard_break=hard_break),
        threshold=0.8,
        config=ProposalConfig(),
    )

    assert len(proposals) == 1
    proposal = proposals.iloc[0]
    assert proposal["proposal_time_seconds"] == pytest.approx(0.20)
    assert proposal["resolution_time_seconds"] == pytest.approx(0.30)
    assert proposal["resolution_type"] == "hard_break"
    assert bool(proposal["unresolved"])


def test_invalid_current_score_does_not_censor_an_observable_open_horizon() -> None:
    times = np.arange(0.0, 1.80, 0.05)
    valid = np.ones(times.size, dtype=bool)
    valid[(times >= 0.20) & (times <= 0.40)] = False

    proposals = run_proposal_state_machine(
        _score_table(times, valid=valid),
        threshold=0.8,
        config=ProposalConfig(horizon_s=1.5),
    )

    first = proposals.iloc[0]
    assert first["proposal_time_seconds"] == pytest.approx(0.10)
    assert first["resolution_type"] == "horizon_elapsed"
    assert first["resolution_time_seconds"] == pytest.approx(1.60)
    assert not bool(first["unresolved"])


def test_horizon_resolution_requires_full_rearm_before_new_persistence_streak() -> None:
    times = np.arange(0.0, 1.65, 0.05)
    proposals = run_proposal_state_machine(
        _score_table(times),
        threshold=0.8,
        config=ProposalConfig(horizon_s=0.15, rearm_seconds=1.0),
    )

    assert proposals["proposal_time_seconds"].tolist() == pytest.approx([0.10, 1.35])
    assert proposals.iloc[0]["resolution_type"] == "horizon_elapsed"
    assert not bool(proposals.iloc[0]["unresolved"])


def test_observed_outside_resolves_proposal_and_rearm_starts_after_return_inside() -> None:
    times = np.arange(0.0, 1.60, 0.05)
    in_corridor = np.ones(times.size, dtype=bool)
    in_corridor[(times >= 0.20) & (times < 0.30)] = False
    proposals = run_proposal_state_machine(
        _score_table(times, in_corridor=in_corridor),
        threshold=0.8,
        config=ProposalConfig(horizon_s=1.5, rearm_seconds=1.0),
    )

    assert proposals["proposal_time_seconds"].tolist() == pytest.approx([0.10, 1.40])
    assert proposals.iloc[0]["resolution_type"] == "outside_observed"
    assert proposals.iloc[0]["resolution_time_seconds"] == pytest.approx(0.20)
    assert not bool(proposals.iloc[0]["unresolved"])


def test_unproposed_excursion_still_requires_full_in_corridor_rearm() -> None:
    times = np.arange(0.0, 1.60, 0.05)
    scores = np.zeros(times.size)
    scores[times >= 0.30] = 0.90
    in_corridor = np.ones(times.size, dtype=bool)
    in_corridor[(times >= 0.20) & (times < 0.30)] = False

    proposals = run_proposal_state_machine(
        _score_table(times, scores=scores, in_corridor=in_corridor),
        threshold=0.8,
        config=ProposalConfig(horizon_s=1.5, rearm_seconds=1.0),
    )

    assert proposals["proposal_time_seconds"].tolist() == pytest.approx([1.40])


def test_observed_outside_invalid_for_scoring_still_requires_rearm() -> None:
    times = np.arange(0.0, 1.60, 0.05)
    scores = np.zeros(times.size)
    scores[times >= 0.30] = 0.90
    in_corridor = np.ones(times.size, dtype=bool)
    outside = (times >= 0.20) & (times < 0.30)
    in_corridor[outside] = False
    valid = np.ones(times.size, dtype=bool)
    valid[outside] = False

    proposals = run_proposal_state_machine(
        _score_table(times, scores=scores, in_corridor=in_corridor, valid=valid),
        threshold=0.8,
        config=ProposalConfig(horizon_s=1.5, rearm_seconds=1.0),
    )

    assert proposals["proposal_time_seconds"].tolist() == pytest.approx([1.40])


def test_proposal_labels_apply_circular_localization_and_censor_unresolved() -> None:
    proposals = pd.DataFrame(
        {
            "proposal_id": ["p1", "p2", "p3", "p4"],
            "circuit": ["Test"] * 4,
            "car_id": ["car_1"] * 4,
            "proposal_time_seconds": [1.0, 3.0, 5.0, 7.0],
            "horizon_s": [1.5] * 4,
            "predicted_side": ["left", "right", "left", "left"],
            "predicted_segment_bin": [0, 5, 2, 2],
            "n_segment_bins": [10] * 4,
            "resolution_time_seconds": [2.0, 4.0, 5.5, 8.5],
            "resolution_type": [
                "outside_observed",
                "outside_observed",
                "stream_end",
                "horizon_elapsed",
            ],
            "unresolved": [False, False, True, False],
        }
    )
    events = pd.DataFrame(
        {
            "candidate_event_id": ["e1", "e2"],
            "circuit": ["Test", "Test"],
            "car_id": ["car_1", "car_1"],
            "start_time_seconds": [2.0, 4.0],
            "side_at_onset": ["left", "left"],
            "segment_bin_25m_at_onset": [9, 5],
            "qualified": [True, True],
        }
    )

    labels = label_proposals(proposals, events, segment_tolerance_bins=1)

    first = labels.set_index("proposal_id").loc["p1"]
    assert first["matched_event_id"] == "e1"
    assert first["lead_seconds"] == pytest.approx(1.0)
    assert bool(first["correct_side"])
    assert not bool(first["exact_segment"])
    assert bool(first["localized_hit"])
    assert not bool(first["false_proposal"])
    second = labels.set_index("proposal_id").loc["p2"]
    assert not bool(second["localized_hit"])
    assert bool(second["false_proposal"])
    third = labels.set_index("proposal_id").loc["p3"]
    assert bool(third["unresolved_censored"])
    assert not bool(third["false_proposal"])
    fourth = labels.set_index("proposal_id").loc["p4"]
    assert bool(fourth["false_proposal"])

    metrics = summarize_detection(
        labels,
        events,
        exposure_hours=2.0,
        required_lead_s=0.5,
    )
    assert metrics["qualified_events"] == 2
    assert metrics["localized_event_hits"] == 1
    assert metrics["localized_event_recall"] == pytest.approx(0.5)
    assert metrics["false_proposals"] == 2
    assert metrics["false_proposals_per_hour"] == pytest.approx(1.0)
    assert metrics["unresolved_proposals"] == 1

    least_favorable = summarize_detection(
        labels,
        events,
        exposure_hours=2.0,
        required_lead_s=1.25,
        count_unresolved_as_false=True,
    )
    assert least_favorable["localized_event_hits"] == 0
    assert least_favorable["false_proposals"] == 3


def test_policy_rejects_nonmonotone_stream_and_invalid_threshold() -> None:
    table = _score_table(np.asarray([0.0, 0.10, 0.05]))
    with pytest.raises(DataValidationError, match="strictly increasing"):
        run_proposal_state_machine(table, threshold=0.5)
    with pytest.raises(DataValidationError, match="threshold"):
        run_proposal_state_machine(_score_table(np.asarray([0.0, 0.05])), threshold=1.1)


def test_threshold_selection_maximizes_recall_then_uses_higher_threshold() -> None:
    times = np.arange(0.0, 4.05, 0.05)
    scores = np.zeros(times.size)
    scores[(times >= 0.50) & (times <= 0.90)] = 0.90
    scores[(times >= 2.20) & (times <= 2.40)] = 0.70
    in_corridor = np.ones(times.size, dtype=bool)
    in_corridor[(times >= 1.00) & (times < 1.10)] = False
    score_table = _score_table(times, scores=scores, in_corridor=in_corridor)
    events = pd.DataFrame(
        {
            "candidate_event_id": ["e1"],
            "circuit": ["Test"],
            "car_id": ["car_1"],
            "start_time_seconds": [1.0],
            "side_at_onset": ["left"],
            "segment_bin_25m_at_onset": [3],
            "qualified": [True],
        }
    )

    selected = select_calibration_threshold(
        score_table,
        events,
        exposure_hours=2.0,
        required_lead_s=0.25,
        false_budget_per_hour=0.5,
        candidate_thresholds=np.asarray([0.50, 0.80, 0.95]),
    )

    assert selected.threshold == pytest.approx(0.80)
    assert selected.localized_event_recall == pytest.approx(1.0)
    assert selected.false_proposals_per_hour == pytest.approx(0.0)


def test_threshold_selection_reports_not_estimable_when_no_candidate_is_feasible() -> None:
    times = np.arange(0.0, 2.05, 0.05)
    events = pd.DataFrame(
        columns=[
            "candidate_event_id",
            "circuit",
            "car_id",
            "start_time_seconds",
            "side_at_onset",
            "segment_bin_25m_at_onset",
            "qualified",
        ]
    )

    with pytest.raises(DataValidationError, match="no candidate threshold"):
        select_calibration_threshold(
            _score_table(times),
            events,
            exposure_hours=1.0,
            required_lead_s=0.25,
            false_budget_per_hour=0.0,
            candidate_thresholds=np.asarray([0.50, 0.80]),
        )
