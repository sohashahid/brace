"""Causal proposal policy and event-level localization evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from brace_f1.io import DataValidationError
from brace_f1.metrics import false_proposal_rate_upper


@dataclass(frozen=True)
class NeighborhoodCandidates:
    """Highest-probability actionable side and circular segment neighborhood."""

    score: NDArray[np.float64]
    side: NDArray[np.object_]
    segment_bin: NDArray[np.int64]


def select_neighborhood_candidates(
    outcome_probability: ArrayLike,
    *,
    side_names: tuple[str, ...] = ("left", "right", "unknown"),
    actionable_sides: tuple[str, ...] = ("left", "right"),
    radius_bins: int = 1,
) -> NeighborhoodCandidates:
    """Select a side/bin center using summed probability in a circular neighborhood."""

    outcome = np.asarray(outcome_probability, dtype=np.float64)
    if outcome.ndim != 3:
        raise DataValidationError("outcome probability must have shape (N,S,K)")
    if outcome.shape[1] != len(side_names) or outcome.shape[2] < 3:
        raise DataValidationError("outcome side names or segment-bin dimension are invalid")
    if not np.isfinite(outcome).all() or np.any(outcome < 0.0):
        raise DataValidationError("outcome probabilities must be finite and non-negative")
    if not isinstance(radius_bins, int) or radius_bins < 0:
        raise DataValidationError("radius_bins must be a non-negative integer")
    try:
        side_indices = np.asarray([side_names.index(side) for side in actionable_sides])
    except ValueError as error:
        raise DataValidationError("an actionable side is absent from side_names") from error
    if side_indices.size == 0:
        raise DataValidationError("at least one actionable side is required")

    selected = outcome[:, side_indices, :]
    neighborhood = np.zeros_like(selected)
    for shift in range(-radius_bins, radius_bins + 1):
        neighborhood += np.roll(selected, shift=shift, axis=2)
    flat_index = neighborhood.reshape(outcome.shape[0], -1).argmax(axis=1)
    side_position, segment_bin = np.divmod(flat_index, outcome.shape[2])
    sides = np.asarray(actionable_sides, dtype=object)[side_position]
    scores = neighborhood[np.arange(outcome.shape[0]), side_position.astype(np.int64), segment_bin]
    return NeighborhoodCandidates(
        score=scores,
        side=sides,
        segment_bin=segment_bin.astype(np.int64),
    )


@dataclass(frozen=True)
class ProposalConfig:
    """Frozen causal state-machine settings."""

    horizon_s: float = 1.5
    persistence_scores: int = 3
    rearm_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.horizon_s) or self.horizon_s <= 0.0:
            raise DataValidationError("horizon_s must be finite and positive")
        if not isinstance(self.persistence_scores, int) or self.persistence_scores < 1:
            raise DataValidationError("persistence_scores must be a positive integer")
        if not np.isfinite(self.rearm_seconds) or self.rearm_seconds < 0.0:
            raise DataValidationError("rearm_seconds must be finite and non-negative")


_SCORE_COLUMNS = {
    "circuit",
    "car_id",
    "frame_index",
    "time_seconds",
    "input_valid_causal",
    "in_corridor",
    "hard_break",
    "proposal_score",
    "predicted_side",
    "predicted_segment_bin",
    "n_segment_bins",
}

_PROPOSAL_TABLE_COLUMNS = [
    "proposal_id",
    "circuit",
    "car_id",
    "proposal_frame_index",
    "proposal_time_seconds",
    "proposal_score",
    "predicted_side",
    "predicted_segment_bin",
    "n_segment_bins",
    "horizon_s",
    "expiry_time_seconds",
    "resolution_time_seconds",
    "resolution_type",
    "unresolved",
]


def _proposal_row(
    row: object,
    *,
    proposal_id: str,
    config: ProposalConfig,
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "circuit": str(row.circuit),
        "car_id": str(row.car_id),
        "proposal_frame_index": int(row.frame_index),
        "proposal_time_seconds": float(row.time_seconds),
        "proposal_score": float(row.proposal_score),
        "predicted_side": str(row.predicted_side),
        "predicted_segment_bin": int(row.predicted_segment_bin),
        "n_segment_bins": int(row.n_segment_bins),
        "horizon_s": config.horizon_s,
        "expiry_time_seconds": float(row.time_seconds) + config.horizon_s,
        "resolution_time_seconds": np.nan,
        "resolution_type": "",
        "unresolved": False,
    }


def _resolve_proposal(
    proposal: dict[str, object], *, time_seconds: float, resolution_type: str, unresolved: bool
) -> None:
    proposal["resolution_time_seconds"] = float(time_seconds)
    proposal["resolution_type"] = resolution_type
    proposal["unresolved"] = unresolved


def run_proposal_state_machine(
    score_table: pd.DataFrame,
    *,
    threshold: float,
    config: ProposalConfig | None = None,
) -> pd.DataFrame:
    """Open non-overwritable proposals after three identical completed score decisions."""

    if config is None:
        config = ProposalConfig()
    missing = _SCORE_COLUMNS.difference(score_table.columns)
    if missing:
        raise DataValidationError(f"score table missing columns: {sorted(missing)}")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise DataValidationError("threshold must be finite and in [0, 1]")
    if score_table.empty:
        return pd.DataFrame(columns=_PROPOSAL_TABLE_COLUMNS)

    proposals: list[dict[str, object]] = []
    for (circuit, car_id), unordered in score_table.groupby(
        ["circuit", "car_id"], sort=False, dropna=False
    ):
        group = unordered
        times = group["time_seconds"].to_numpy(dtype=np.float64)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise DataValidationError(
                "score times must be finite and strictly increasing per stream"
            )
        scores = group["proposal_score"].to_numpy(dtype=np.float64)
        bins = group["predicted_segment_bin"].to_numpy(dtype=np.int64)
        n_bins = group["n_segment_bins"].to_numpy(dtype=np.int64)
        sides = group["predicted_side"].astype(str).to_numpy()
        if (
            not np.isfinite(scores).all()
            or np.any((scores < 0.0) | (scores > 1.0))
            or np.any(n_bins < 3)
            or np.any(bins < 0)
            or np.any(bins >= n_bins)
            or not np.isin(sides, ["left", "right"]).all()
        ):
            raise DataValidationError(
                "score decisions contain invalid probabilities, sides, or bins"
            )

        mode = "armed"
        streak_key: tuple[str, int] | None = None
        streak_count = 0
        current: dict[str, object] | None = None
        rearm_start: float | None = None
        proposal_number = 0

        for row in group.itertuples(index=False):
            now = float(row.time_seconds)
            valid = bool(row.input_valid_causal)
            inside = bool(row.in_corridor)
            hard_break = bool(row.hard_break)

            if hard_break:
                had_open = mode == "open" and current is not None
                requires_rearm = had_open or mode == "rearm" or not inside
                if had_open:
                    _resolve_proposal(
                        current,
                        time_seconds=now,
                        resolution_type="hard_break",
                        unresolved=True,
                    )
                mode = "rearm" if requires_rearm else "armed"
                current = None
                rearm_start = now if requires_rearm and valid and inside else None
                streak_key = None
                streak_count = 0
                continue

            if mode == "open":
                if current is None:
                    raise RuntimeError("open proposal state is missing its proposal record")
                if not inside:
                    _resolve_proposal(
                        current,
                        time_seconds=now,
                        resolution_type="outside_observed",
                        unresolved=False,
                    )
                    mode = "rearm"
                    current = None
                    rearm_start = None
                    continue
                expiry = float(current["expiry_time_seconds"])
                if now + 1e-12 >= expiry:
                    _resolve_proposal(
                        current,
                        time_seconds=expiry,
                        resolution_type="horizon_elapsed",
                        unresolved=False,
                    )
                    mode = "rearm"
                    current = None
                    rearm_start = expiry
                    continue
                continue

            if mode == "rearm":
                if not valid or not inside:
                    rearm_start = None
                    streak_key = None
                    streak_count = 0
                    continue
                if rearm_start is None:
                    rearm_start = now
                    continue
                if now - rearm_start + 1e-12 < config.rearm_seconds:
                    continue
                mode = "armed"
                streak_key = None
                streak_count = 0

            if not valid or not inside:
                if not inside:
                    mode = "rearm"
                    rearm_start = None
                streak_key = None
                streak_count = 0
                continue
            if float(row.proposal_score) + 1e-15 < threshold:
                streak_key = None
                streak_count = 0
                continue
            key = (str(row.predicted_side), int(row.predicted_segment_bin))
            if key == streak_key:
                streak_count += 1
            else:
                streak_key = key
                streak_count = 1
            if streak_count < config.persistence_scores:
                continue

            proposal_number += 1
            proposal_id = f"{circuit}:{car_id}:proposal_{proposal_number:04d}"
            current = _proposal_row(row, proposal_id=proposal_id, config=config)
            proposals.append(current)
            mode = "open"
            streak_key = None
            streak_count = 0

        if mode == "open" and current is not None:
            _resolve_proposal(
                current,
                time_seconds=float(times[-1]),
                resolution_type="stream_end",
                unresolved=True,
            )

    return pd.DataFrame(proposals, columns=_PROPOSAL_TABLE_COLUMNS)


_PROPOSAL_LABEL_COLUMNS = {
    "proposal_id",
    "circuit",
    "car_id",
    "proposal_time_seconds",
    "horizon_s",
    "predicted_side",
    "predicted_segment_bin",
    "n_segment_bins",
    "resolution_time_seconds",
    "resolution_type",
    "unresolved",
}

_EVENT_LABEL_COLUMNS = {
    "candidate_event_id",
    "circuit",
    "car_id",
    "start_time_seconds",
    "side_at_onset",
    "segment_bin_25m_at_onset",
    "qualified",
}


def _circular_bin_distance(first: int, second: int, n_bins: int) -> int:
    ordinary = abs(first - second)
    return min(ordinary, n_bins - ordinary)


def label_proposals(
    proposals: pd.DataFrame,
    events: pd.DataFrame,
    *,
    segment_tolerance_bins: int = 1,
) -> pd.DataFrame:
    """Label each first proposal against the first qualifying event in its open window."""

    missing_proposals = _PROPOSAL_LABEL_COLUMNS.difference(proposals.columns)
    missing_events = _EVENT_LABEL_COLUMNS.difference(events.columns)
    if missing_proposals:
        raise DataValidationError(f"proposal table missing columns: {sorted(missing_proposals)}")
    if missing_events:
        raise DataValidationError(f"event table missing columns: {sorted(missing_events)}")
    if not isinstance(segment_tolerance_bins, int) or segment_tolerance_bins < 0:
        raise DataValidationError("segment_tolerance_bins must be a non-negative integer")

    label_columns = [
        "matched_event_id",
        "matched_event_time_seconds",
        "lead_seconds",
        "correct_side",
        "exact_segment",
        "within_segment_tolerance",
        "localized_hit",
        "unresolved_censored",
        "false_proposal",
    ]
    if proposals.empty:
        return pd.DataFrame(columns=[*proposals.columns, *label_columns])

    qualified = events.loc[events["qualified"].astype(bool)].copy()
    qualified = qualified.sort_values(["circuit", "car_id", "start_time_seconds"], kind="stable")
    grouped_events = {
        key: group for key, group in qualified.groupby(["circuit", "car_id"], sort=False)
    }
    rows: list[dict[str, object]] = []
    for proposal in proposals.itertuples(index=False):
        start = float(proposal.proposal_time_seconds)
        expiry = start + float(proposal.horizon_s)
        resolution = float(proposal.resolution_time_seconds)
        observable_end = min(expiry, resolution) if np.isfinite(resolution) else expiry
        candidates = grouped_events.get((proposal.circuit, proposal.car_id))
        matched = None
        if candidates is not None:
            mask = (candidates["start_time_seconds"] >= start - 1e-12) & (
                candidates["start_time_seconds"] <= observable_end + 1e-12
            )
            if mask.any():
                matched = candidates.loc[mask].iloc[0]

        output = proposal._asdict()
        output.update(
            {
                "matched_event_id": pd.NA,
                "matched_event_time_seconds": np.nan,
                "lead_seconds": np.nan,
                "correct_side": False,
                "exact_segment": False,
                "within_segment_tolerance": False,
                "localized_hit": False,
                "unresolved_censored": False,
                "false_proposal": False,
            }
        )
        if matched is not None:
            observed_bin = int(matched["segment_bin_25m_at_onset"])
            n_bins = int(proposal.n_segment_bins)
            predicted_bin = int(proposal.predicted_segment_bin)
            if not 0 <= observed_bin < n_bins or not 0 <= predicted_bin < n_bins:
                raise DataValidationError("proposal or event segment bin falls outside the circuit")
            correct_side = str(proposal.predicted_side) == str(matched["side_at_onset"])
            distance = _circular_bin_distance(predicted_bin, observed_bin, n_bins)
            within_tolerance = distance <= segment_tolerance_bins
            localized = correct_side and within_tolerance
            output.update(
                {
                    "matched_event_id": str(matched["candidate_event_id"]),
                    "matched_event_time_seconds": float(matched["start_time_seconds"]),
                    "lead_seconds": float(matched["start_time_seconds"]) - start,
                    "correct_side": correct_side,
                    "exact_segment": correct_side and distance == 0,
                    "within_segment_tolerance": within_tolerance,
                    "localized_hit": localized,
                    "false_proposal": not localized,
                }
            )
        elif bool(proposal.unresolved):
            output["unresolved_censored"] = True
        else:
            output["false_proposal"] = True
        rows.append(output)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ThresholdSelection:
    """Calibration-only operating threshold and its observed calibration metrics."""

    threshold: float
    localized_event_recall: float
    false_proposals_per_hour: float
    false_proposals: int
    qualified_events: int
    proposal_count: int


def select_calibration_threshold(
    score_table: pd.DataFrame,
    events: pd.DataFrame,
    *,
    exposure_hours: float,
    required_lead_s: float,
    false_budget_per_hour: float,
    candidate_thresholds: ArrayLike,
    config: ProposalConfig | None = None,
    segment_tolerance_bins: int = 1,
) -> ThresholdSelection:
    """Maximize localized recall under a calibration false-proposal budget."""

    thresholds = np.asarray(candidate_thresholds, dtype=np.float64)
    if (
        thresholds.ndim != 1
        or thresholds.size == 0
        or not np.isfinite(thresholds).all()
        or np.any((thresholds < 0.0) | (thresholds > 1.0))
    ):
        raise DataValidationError("candidate thresholds must be a non-empty vector in [0, 1]")
    if not np.isfinite(false_budget_per_hour) or false_budget_per_hour < 0.0:
        raise DataValidationError("false_budget_per_hour must be finite and non-negative")

    feasible: list[ThresholdSelection] = []
    for threshold in np.unique(thresholds):
        proposals = run_proposal_state_machine(
            score_table, threshold=float(threshold), config=config
        )
        labels = label_proposals(proposals, events, segment_tolerance_bins=segment_tolerance_bins)
        summary = summarize_detection(
            labels,
            events,
            exposure_hours=exposure_hours,
            required_lead_s=required_lead_s,
        )
        rate = float(summary["false_proposals_per_hour"])
        if rate <= false_budget_per_hour + 1e-12:
            feasible.append(
                ThresholdSelection(
                    threshold=float(threshold),
                    localized_event_recall=float(summary["localized_event_recall"]),
                    false_proposals_per_hour=rate,
                    false_proposals=int(summary["false_proposals"]),
                    qualified_events=int(summary["qualified_events"]),
                    proposal_count=len(proposals),
                )
            )
    if not feasible:
        raise DataValidationError("no candidate threshold satisfies the false-proposal budget")

    def selection_key(value: ThresholdSelection) -> tuple[float, float]:
        recall = value.localized_event_recall
        return (-np.inf if np.isnan(recall) else recall, value.threshold)

    return max(feasible, key=selection_key)


def summarize_detection(
    proposal_labels: pd.DataFrame,
    events: pd.DataFrame,
    *,
    exposure_hours: float,
    required_lead_s: float,
    count_unresolved_as_false: bool = False,
) -> dict[str, float | int]:
    """Summarize localized event recall and false proposals at an actual-lead requirement."""

    if not np.isfinite(exposure_hours) or exposure_hours <= 0.0:
        raise DataValidationError("exposure_hours must be finite and positive")
    if not np.isfinite(required_lead_s) or required_lead_s < 0.0:
        raise DataValidationError("required_lead_s must be finite and non-negative")
    qualified = events.loc[events["qualified"].astype(bool)]
    event_count = int(qualified["candidate_event_id"].nunique())
    timely = proposal_labels.loc[
        proposal_labels["localized_hit"].astype(bool)
        & (proposal_labels["lead_seconds"] >= required_lead_s - 1e-12)
    ]
    localized_hits = int(timely["matched_event_id"].dropna().nunique())
    false_count = int(proposal_labels["false_proposal"].astype(bool).sum())
    unresolved_count = int(proposal_labels["unresolved_censored"].astype(bool).sum())
    if count_unresolved_as_false:
        false_count += unresolved_count
    recall = float("nan") if event_count == 0 else localized_hits / event_count
    return {
        "qualified_events": event_count,
        "localized_event_hits": localized_hits,
        "localized_event_recall": recall,
        "false_proposals": false_count,
        "false_proposals_per_hour": false_count / exposure_hours,
        "false_proposals_per_hour_upper_95": false_proposal_rate_upper(
            false_count, exposure_hours=exposure_hours, confidence=0.95
        ),
        "unresolved_proposals": unresolved_count,
        "required_lead_s": required_lead_s,
    }
