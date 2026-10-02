from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from brace_f1.bootstrap_reliability import reliability_bin_cluster_bootstrap

ROOT = Path(__file__).parents[1]
ACTUAL_POOLED = ROOT / "output" / "experiment" / "pooled-heldout"
ACTUAL_BOOTSTRAP = ROOT / "output" / "bootstrap-primary"
ACTUAL_TRANSPORT_BOOTSTRAP = ROOT / "output" / "bootstrap-two-stage"
SPEC = importlib.util.spec_from_file_location(
    "build_submission_results",
    ROOT / "scripts" / "build_submission_results.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ArtifactValidationError = MODULE.ArtifactValidationError
compile_submission = MODULE.compile_submission
METHODS = (
    "brace_bayesian",
    "posterior_mean_twin",
    "constant_velocity",
    "constant_turn_rate",
    "side_hazard",
    "calibration_mse_tuned_dynamics",
)
LEADS = (0.25, 0.5, 1.0, 1.5)
BUDGETS = (0.5, 1.0, 2.0, 5.0, 10.0)
CIRCUITS = ("Bahrain", "Britain", "Jeddah", "Monza")
_RELIABILITY_BOOTSTRAP_CACHE: pd.DataFrame | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _producer_source_tree_hash() -> str:
    module_dir = ROOT / "src" / "brace_f1"
    digest = hashlib.sha256()
    for path in sorted(
        module_dir.rglob("*.py"),
        key=lambda value: value.relative_to(module_dir).as_posix(),
    ):
        digest.update(path.relative_to(module_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _register(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in paths}


def _nested_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            for text in [str(key), *_nested_strings(item)]
        ]
    if isinstance(value, list):
        return [text for item in value for text in _nested_strings(item)]
    return []


def _canonical_json_hash(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recall(method: str, lead: float) -> float:
    values = {
        "brace_bayesian": (0.80, 0.70, 0.60, 0.40),
        "posterior_mean_twin": (0.70, 0.55, 0.45, 0.30),
        "constant_velocity": (0.45, 0.30, 0.20, 0.10),
        "constant_turn_rate": (0.50, 0.40, 0.25, 0.10),
        "side_hazard": (0.48, 0.35, 0.20, 0.10),
        "calibration_mse_tuned_dynamics": (0.60, 0.50, 0.35, 0.20),
    }
    return values[method][LEADS.index(lead)]


def _make_inputs(
    tmp_path: Path,
    *,
    invalid_per_circuit_counts: bool = False,
    source_manifest_bytes_delta: int = 0,
    examined_proposals_per_fold: int = 0,
) -> tuple[Path, Path]:
    pooled = tmp_path / "pooled-heldout"
    bootstrap = tmp_path / "bootstrap-primary"
    pooled.mkdir()
    bootstrap.mkdir()

    study_config = json.loads((ROOT / "configs" / "study.json").read_text(encoding="utf-8"))
    study_config_path = tmp_path / "configs" / "study.json"
    study_config_path.parent.mkdir(parents=True)
    study_config_path.write_text(
        json.dumps(study_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_content_hash = _canonical_json_hash(study_config)
    source_manifest_path = tmp_path / "data" / "manifests" / "deepracing-files.csv"
    source_manifest_path.parent.mkdir(parents=True)
    source_manifest_path.write_text(
        "relative_path,bytes,sha256\nfixture.bin,7," + "a" * 64 + "\n",
        encoding="utf-8",
    )
    source_manifest_hash = _sha256(source_manifest_path)
    data_content_hash = "d" * 64
    code_content_hash = _producer_source_tree_hash()

    build_output = tmp_path / "data" / "processed" / "fixture-output.csv"
    build_output.parent.mkdir(parents=True)
    build_output.write_text("fixture\n", encoding="utf-8")
    build_manifest_path = tmp_path / "data" / "manifests" / "deepracing-build.json"
    build_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    per_circuit_counts = [
        {
            "circuit": circuit,
            "cars": cars,
            "qualified_excursions": events,
            "candidate_excursions": candidates,
            "resampled_frames": frames,
            "prebuffer_exposure_seconds": prebuffer,
            "buffered_exposure_seconds": buffered,
        }
        for circuit, cars, events, candidates, frames, prebuffer, buffered in zip(
            CIRCUITS,
            (20, 11, 20, 7),
            (36, 113, 106, 39),
            (226, 325, 961, 104),
            (185_174, 68_283, 181_524, 25_288),
            (9224.805993299931, 3411.587967460975, 9032.04595878534, 1263.6139990882948),
            (9154.38796120882, 3381.621959298849, 8960.021955937147, 1245.3660022616386),
            strict=True,
        )
    ]
    build_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "counts": {
                    "input_files": 244,
                    "input_bytes": 86_360_332,
                    "circuits": 4,
                    "cars": 58,
                    "raw_frames": 2_473_009,
                    "resampled_frames": 460_269,
                    "candidate_excursions": 1_616,
                    "qualified_excursions": 294,
                    "prebuffer_exposure_hours": 6.370014977398483,
                    "buffered_exposure_hours": 6.3170549663073485,
                },
                "per_circuit_counts": (
                    [] if invalid_per_circuit_counts else per_circuit_counts
                ),
                "input_manifest": {
                    "path": str(source_manifest_path.resolve()),
                    "sha256": source_manifest_hash,
                    "bytes": source_manifest_path.stat().st_size
                    + source_manifest_bytes_delta,
                },
                "outputs": [
                    {
                        "path": str(build_output.resolve()),
                        "sha256": _sha256(build_output),
                        "bytes": build_output.stat().st_size,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    freeze_seal_path = tmp_path / "threshold-freeze-seal.json"

    circuit_brace_lstar = dict(zip(CIRCUITS, (1.0, 1.0, 0.5, 1.0), strict=True))
    circuit_twin_lstar = dict(zip(CIRCUITS, (0.5, 0.5, 0.5, 0.0), strict=True))
    primary_brace_lstar = 1.0
    primary_twin_lstar = 0.5
    units: list[dict[str, object]] = []
    for circuit_row in per_circuit_counts:
        circuit = str(circuit_row["circuit"])
        unit_count = int(circuit_row["cars"])
        event_count = int(circuit_row["qualified_excursions"])
        quotient, remainder = divmod(event_count, unit_count)
        exposure = float(circuit_row["buffered_exposure_seconds"]) / 3600.0
        for index in range(unit_count):
            units.append(
                {
                    "fold_test_circuit": circuit,
                    "circuit": circuit,
                    "source_session_id": f"session-{circuit}",
                    "car_id": f"car-{circuit}-{index:02d}",
                    "qualified_events": quotient + int(index < remainder),
                    "exposure_hours": exposure / unit_count,
                }
            )

    def allocate_hits(total: int, capacities: list[int]) -> list[int]:
        remaining = total
        result: list[int] = []
        for capacity in capacities:
            value = min(remaining, capacity)
            result.append(value)
            remaining -= value
        assert remaining == 0
        return result

    contribution_rows: list[dict[str, object]] = []
    for method in METHODS:
        for lead in LEADS:
            for budget in BUDGETS:
                if budget < 2.0:
                    continue
                for circuit in CIRCUITS:
                    circuit_units = [unit for unit in units if unit["circuit"] == circuit]
                    capacities = [int(unit["qualified_events"]) for unit in circuit_units]
                    events = sum(capacities)
                    exposure = sum(float(unit["exposure_hours"]) for unit in circuit_units)
                    if budget == 2.0 and method in {
                        "brace_bayesian",
                        "posterior_mean_twin",
                    }:
                        lstar = (
                            circuit_brace_lstar[circuit]
                            if method == "brace_bayesian"
                            else circuit_twin_lstar[circuit]
                        )
                        target_recall = 0.60 if lead <= lstar else 0.40
                    else:
                        target_recall = _recall(method, lead)
                    localized_total = int(round(events * target_recall))
                    event_total = min(events, localized_total + int(round(events * 0.10)))
                    correct_total = localized_total
                    within_total = localized_total
                    exact_total = max(0, localized_total - int(round(events * 0.10)))
                    localized = allocate_hits(localized_total, capacities)
                    event_hits = allocate_hits(event_total, capacities)
                    correct = allocate_hits(correct_total, capacities)
                    within = allocate_hits(within_total, capacities)
                    exact = allocate_hits(exact_total, capacities)
                    false_total = int(math.floor(budget * exposure * 0.8))
                    for index, unit in enumerate(circuit_units):
                        false = false_total if index == 0 else 0
                        unresolved_total = (
                            2
                            if method == "brace_bayesian" and lead >= 1.0 and budget == 2.0
                            else 1
                        )
                        unresolved = unresolved_total if index == 0 else 0
                        contribution_rows.append(
                            {
                                **unit,
                                "method": method,
                                "required_lead_s": lead,
                                "false_budget_per_hour": budget,
                                "localized_event_hits": localized[index],
                                "event_hits": event_hits[index],
                                "correct_side_event_hits": correct[index],
                                "within_segment_tolerance_event_hits": within[index],
                                "exact_bin_event_hits": exact[index],
                                "false_proposals": false,
                                "unresolved_proposals": unresolved,
                                "least_favorable_false_proposals": false + unresolved,
                                "fixed_model_and_threshold": True,
                                "operating_point_status": "estimable",
                            }
                        )

    contribution_table = pd.DataFrame(contribution_rows)
    operating_rows: list[dict[str, object]] = []
    for (method, lead, budget), group in contribution_table.groupby(
        ["method", "required_lead_s", "false_budget_per_hour"], sort=True
    ):
        events = int(group["qualified_events"].sum())
        exposure = float(group["exposure_hours"].sum())
        localized = int(group["localized_event_hits"].sum())
        event_hits = int(group["event_hits"].sum())
        correct_side_hits = int(group["correct_side_event_hits"].sum())
        within_tolerance_hits = int(group["within_segment_tolerance_event_hits"].sum())
        exact_bin_hits = int(group["exact_bin_event_hits"].sum())
        false = int(group["false_proposals"].sum())
        unresolved = int(group["unresolved_proposals"].sum())
        operating_rows.append(
            {
                "method": method,
                "required_lead_s": lead,
                "false_budget_per_hour": budget,
                "scope": "pooled_four_circuit_heldout",
                "operating_point_status": "estimable",
                "estimability_reason": "",
                "circuit_count": 4,
                "car_session_count": 58,
                "localized_event_hits": localized,
                "qualified_events": events,
                "localized_event_recall": localized / events,
                "event_hits": event_hits,
                "event_recall": event_hits / events,
                "correct_side_event_hits": correct_side_hits,
                "correct_side_event_recall": correct_side_hits / events,
                "within_segment_tolerance_event_hits": within_tolerance_hits,
                "within_segment_tolerance_event_recall": within_tolerance_hits / events,
                "exact_bin_event_hits": exact_bin_hits,
                "exact_bin_event_recall": exact_bin_hits / events,
                "false_proposals": false,
                "unresolved_proposals": unresolved,
                "least_favorable_false_proposals": false + unresolved,
                "exposure_hours": exposure,
                "false_proposals_per_hour": false / exposure,
                "false_proposals_per_hour_upper_95": (
                    0.5 * chi2.ppf(0.95, 2.0 * (false + 1)) / exposure
                ),
            }
        )
    for method in METHODS:
        for lead in LEADS:
            for budget in (0.5, 1.0):
                not_estimable_reason = json.dumps(
                    {
                        circuit: (
                            {"status": "estimable", "reason": "nan"}
                            if budget == 1.0 and circuit in {"Britain", "Monza"}
                            else {
                                "status": "not_estimable_insufficient_exposure",
                                "reason": (
                                    "false_budget_per_hour_times_"
                                    "calibration_exposure_hours_below_1"
                                ),
                            }
                        )
                        for circuit in CIRCUITS
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                operating_rows.append(
                    {
                        "method": method,
                        "required_lead_s": lead,
                        "false_budget_per_hour": budget,
                        "scope": "pooled_four_circuit_heldout",
                        "circuit_count": 4,
                        "operating_point_status": (
                            "not_estimable_insufficient_exposure"
                        ),
                        "estimability_reason": not_estimable_reason,
                    }
                )
    operating_path = pooled / "pooled-heldout-operating-metrics.csv"
    _write_csv(operating_path, operating_rows)

    probability_rows: list[dict[str, object]] = []
    brier = {
        "brace_bayesian": 0.08,
        "posterior_mean_twin": 0.10,
        "constant_velocity": 0.16,
        "constant_turn_rate": 0.14,
        "side_hazard": 0.15,
        "calibration_mse_tuned_dynamics": 0.12,
    }
    for method in METHODS:
        for horizon in LEADS:
            probability_rows.append(
                {
                    "scope": "pooled",
                    "circuit": "all",
                    "method": method,
                    "horizon_s": horizon,
                    "n": 1000,
                    "event_count": 520,
                    "brier_score": brier[method],
                    "reference_brier_score": 0.20,
                    "brier_skill_score": 1.0 - brier[method] / 0.20,
                    "log_score": 0.25 + brier[method],
                    "average_precision_stepwise": 0.72 - brier[method],
                    "calibration_intercept": -0.05,
                    "calibration_slope": 0.95,
                    "ece_10_equal_width": 0.02,
                }
            )
    probability_path = pooled / "pooled-heldout-probability-metrics.csv"
    _write_csv(probability_path, probability_rows)

    reliability_contribution_rows: list[dict[str, object]] = []
    count_allocation = [2] * 42 + [1] * 16
    for method in METHODS:
        for horizon in LEADS:
            for bin_index in range(10):
                mean_probability = bin_index / 10.0 + 0.05
                observed = min(1.0, mean_probability + 0.02)
                event_allocation = allocate_hits(
                    int(round(observed * 100)),
                    count_allocation,
                )
                for unit, count, event_count in zip(
                    units,
                    count_allocation,
                    event_allocation,
                    strict=True,
                ):
                    reliability_contribution_rows.append(
                        {
                            "fold_test_circuit": unit["circuit"],
                            "circuit": unit["circuit"],
                            "source_session_id": unit["source_session_id"],
                            "car_id": unit["car_id"],
                            "method": method,
                            "horizon_s": horizon,
                            "bin_index": bin_index,
                            "count": count,
                            "probability_sum": count * mean_probability,
                            "event_count": event_count,
                        }
                    )
    reliability_contributions = pd.DataFrame(reliability_contribution_rows)
    global _RELIABILITY_BOOTSTRAP_CACHE
    if _RELIABILITY_BOOTSTRAP_CACHE is None:
        _RELIABILITY_BOOTSTRAP_CACHE = reliability_bin_cluster_bootstrap(
            reliability_contributions,
            n_resamples=10_000,
            seed=20270927,
            horizons_s=LEADS,
            n_bins=10,
            expected_methods=METHODS,
        ).summary
    reliability_intervals = _RELIABILITY_BOOTSTRAP_CACHE.copy(deep=True)
    reliability_rows = reliability_intervals.loc[
        :,
        [
            "method",
            "horizon_s",
            "bin_index",
            "bin_left",
            "bin_right",
            "right_edge_inclusive",
            "count",
            "mean_probability",
            "observed_frequency",
        ],
    ].copy()
    reliability_rows.insert(0, "circuit", "all")
    reliability_rows.insert(0, "scope", "pooled")
    reliability_interval_rows = reliability_intervals.to_dict(orient="records")
    reliability_path = pooled / "pooled-heldout-reliability-bins.csv"
    reliability_rows.to_csv(reliability_path, index=False, lineterminator="\n")

    monotonicity_path = pooled / "pooled-heldout-horizon-monotonicity.csv"
    _write_csv(
        monotonicity_path,
        [
            {
                "scope": "pooled",
                "circuit": "all",
                "method": method,
                "score_rows": 1000,
                "violating_row_count": 5 if method == "brace_bayesian" else 8,
                "violating_row_rate": 0.005 if method == "brace_bayesian" else 0.008,
            }
            for method in METHODS
        ],
    )

    lstar_path = pooled / "pooled-heldout-l-star-sensitivity.csv"
    _write_csv(
        lstar_path,
        [
            {
                "analysis": "primary_unresolved_censored",
                "false_count_column": "false_proposals",
                "brace_method": "brace_bayesian",
                "comparator_method": "posterior_mean_twin",
                "l_star_brace_s": primary_brace_lstar,
                "l_star_comparator_s": primary_twin_lstar,
                "delta_l_star_s": primary_brace_lstar - primary_twin_lstar,
                "false_budget_per_hour": 2.0,
                "minimum_recall": 0.5,
                "models_refit": False,
                "thresholds_refit": False,
            },
            {
                "analysis": "least_favorable_unresolved_counted_as_false",
                "false_count_column": "least_favorable_false_proposals",
                "brace_method": "brace_bayesian",
                "comparator_method": "posterior_mean_twin",
                "l_star_brace_s": 0.5,
                "l_star_comparator_s": 0.5,
                "delta_l_star_s": 0.0,
                "false_budget_per_hour": 2.0,
                "minimum_recall": 0.5,
                "models_refit": False,
                "thresholds_refit": False,
            },
        ],
    )

    delay_lstar_path = pooled / "pooled-heldout-synthetic-delay-l-star.csv"
    _write_csv(
        delay_lstar_path,
        [
            {
                "synthetic_delay_ms": delay,
                "l_star_brace_s": brace,
                "l_star_comparator_s": twin,
                "delta_l_star_s": brace - twin,
                "least_favorable_l_star_brace_s": max(0.0, brace - 0.5),
                "least_favorable_l_star_comparator_s": twin,
                "least_favorable_delta_l_star_s": max(0.0, brace - 0.5) - twin,
                "brace_method": "brace_bayesian",
                "comparator_method": "posterior_mean_twin",
                "false_budget_per_hour": 2.0,
                "minimum_recall": 0.5,
                "models_refit": False,
                "thresholds_refit": False,
            }
            for delay, brace, twin in (
                (0, 1.0, 0.5),
                (40, 1.0, 0.5),
                (80, 0.5, 0.5),
                (160, 0.5, 0.25),
            )
        ],
    )

    contributions_path = pooled / "pooled-heldout-car-contributions.parquet"
    contribution_table.to_parquet(contributions_path, index=False)

    delay_lstars = {
        0: {"brace_bayesian": 1.0, "posterior_mean_twin": 0.5},
        40: {"brace_bayesian": 1.0, "posterior_mean_twin": 0.5},
        80: {"brace_bayesian": 0.5, "posterior_mean_twin": 0.5},
        160: {"brace_bayesian": 0.5, "posterior_mean_twin": 0.25},
    }
    delay_contribution_rows: list[dict[str, object]] = []
    for delay, method_lstars in delay_lstars.items():
        for method, maximum_lead in method_lstars.items():
            for lead in LEADS:
                for circuit in CIRCUITS:
                    circuit_units = [unit for unit in units if unit["circuit"] == circuit]
                    capacities = [int(unit["qualified_events"]) for unit in circuit_units]
                    events = sum(capacities)
                    exposure = sum(float(unit["exposure_hours"]) for unit in circuit_units)
                    target_recall = 0.60 if lead <= maximum_lead else 0.40
                    localized = allocate_hits(int(round(events * target_recall)), capacities)
                    false_total = int(math.floor(2.0 * exposure * 0.8))
                    for index, unit in enumerate(circuit_units):
                        delay_contribution_rows.append(
                            {
                                **unit,
                                "synthetic_delay_ms": delay,
                                "method": method,
                                "required_lead_s": lead,
                                "false_budget_per_hour": 2.0,
                                "localized_event_hits": localized[index],
                                "false_proposals": false_total if index == 0 else 0,
                                "fixed_model_and_threshold": True,
                                "operating_point_status": "estimable",
                            }
                        )
    delay_contributions_path = (
        pooled / "pooled-heldout-synthetic-delay-car-contributions.parquet"
    )
    pd.DataFrame(delay_contribution_rows).to_parquet(delay_contributions_path, index=False)

    reliability_contributions_path = (
        pooled / "pooled-heldout-reliability-cluster-contributions.parquet"
    )
    reliability_contributions.to_parquet(reliability_contributions_path, index=False)
    delay_metrics_path = pooled / "pooled-heldout-synthetic-delay-metrics.csv"
    _write_csv(delay_metrics_path, [{"synthetic_delay_ms": 0, "value": 1}])
    pooled_runtime_path = pooled / "runtime.jsonl"
    pooled_runtime_path.write_text(
        json.dumps({"step": "pooled_heldout_aggregation", "elapsed_seconds": 1.0}) + "\n",
        encoding="utf-8",
    )

    fold_records: list[dict[str, object]] = []
    seal_fold_entries: list[dict[str, object]] = []
    for circuit in CIRCUITS:
        fold_dir = tmp_path / f"fold={circuit}"
        fold_dir.mkdir()
        fold_metric_path = fold_dir / "heldout-operating-metrics.csv"
        fold_rows: list[dict[str, object]] = []
        for method, lstar in (
            ("brace_bayesian", circuit_brace_lstar[circuit]),
            ("posterior_mean_twin", circuit_twin_lstar[circuit]),
        ):
            for lead in LEADS:
                recall = 0.60 if lead <= lstar else 0.40
                fold_rows.append(
                    {
                        "fold_test_circuit": circuit,
                        "circuit": circuit,
                        "scope": "circuit",
                        "method": method,
                        "required_lead_s": lead,
                        "false_budget_per_hour": 2.0,
                        "operating_point_status": "estimable",
                        "localized_event_recall": recall,
                        "false_proposals_per_hour": 1.0,
                    }
                )
        _write_csv(fold_metric_path, fold_rows)
        proposal_labels_path = fold_dir / "heldout-proposal-labels.parquet"
        pd.DataFrame(
            {
                "method": ["brace_bayesian"] * examined_proposals_per_fold,
                "false_budget_per_hour": [2.0] * examined_proposals_per_fold,
            }
        ).to_parquet(proposal_labels_path, index=False)
        threshold_runtime = fold_dir / "runtime-threshold.jsonl"
        threshold_runtime.write_text(
            "\n".join(
                [
                    json.dumps({"step": "fit_models", "elapsed_seconds": 1.0, "fold": circuit}),
                    json.dumps(
                        {
                            "step": "score_method",
                            "elapsed_seconds": 2.0,
                            "fold": circuit,
                            "method": "brace_bayesian",
                            "rows": 100,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        heldout_runtime = fold_dir / "runtime-heldout.jsonl"
        heldout_runtime.write_text(
            json.dumps(
                {
                    "step": "heldout_evaluation_from_sealed_artifacts",
                    "elapsed_seconds": 1.0,
                    "fold": circuit,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        frozen_names = [
            "model-metadata.json",
            "compact-calibrated-scores.parquet",
            "calibrators.json",
            "calibration-thresholds.csv",
            *(f"raw-scores-{method}.parquet" for method in METHODS),
        ]
        frozen_paths: list[Path] = []
        for name in frozen_names:
            path = fold_dir / name
            if name == "calibration-thresholds.csv":
                threshold_rows = [
                    {
                        "fold_test_circuit": circuit,
                        "method": method,
                        "required_lead_s": lead,
                        "false_budget_per_hour": budget,
                        "threshold": 1.0 if examined_proposals_per_fold == 0 else 0.5,
                        "calibration_localized_event_recall": 0.0,
                        "calibration_false_proposals_per_hour": 0.0,
                        "calibration_false_proposals": 0,
                        "calibration_qualified_events": 100,
                        "calibration_proposal_count": (
                            0 if examined_proposals_per_fold == 0 else 1
                        ),
                        "calibration_exposure_hours": 1.0,
                        "calibration_false_count_capacity": 2,
                        "minimum_false_count_capacity": 1,
                        "operating_point_status": "estimable",
                        "estimability_reason": "",
                        "threshold_candidate_count": 5,
                        "threshold_source_partition": "calibration",
                    }
                    for method in METHODS
                    for lead in LEADS
                    for budget in BUDGETS
                ]
                _write_csv(path, threshold_rows)
            else:
                path.write_text(f"frozen fixture: {circuit}: {name}\n", encoding="utf-8")
            frozen_paths.append(path)
        model_identity_hash = hashlib.sha256(f"model:{circuit}".encode()).hexdigest()
        method_identities = {
            method: hashlib.sha256(f"{circuit}:{method}".encode()).hexdigest() for method in METHODS
        }
        run_identity = {
            "build_manifest_hash": _sha256(build_manifest_path),
            "code_content_hash": code_content_hash,
            "config_content_hash": config_content_hash,
            "data_content_hash": data_content_hash,
            "fold_test_circuit": circuit,
            "random_seed_base": 20270927,
            "registered_methods": list(METHODS),
            "runtime_environment": {
                "platform": "macOS-test-arm64",
                "python": "3.12.12 (test build)",
                "package_versions": {"numpy": "2.5.3"},
            },
            "source_manifest_hash": source_manifest_hash,
        }
        fold_manifest_path = fold_dir / "manifest.json"
        fold_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "heldout_complete",
                    "fold_test_circuit": circuit,
                    "code_content_hash": code_content_hash,
                    "data_content_hash": data_content_hash,
                    "config_content_hash": config_content_hash,
                    "source_manifest_hash": source_manifest_hash,
                    "build_manifest_hash": _sha256(build_manifest_path),
                    "input_content_hash": _canonical_json_hash(run_identity),
                    "run_identity": run_identity,
                    "model_identity_hash": model_identity_hash,
                    "method_identities": method_identities,
                    "completed_methods": list(METHODS),
                    "heldout_consumed_sealed_artifacts": True,
                    "models_refit_for_heldout": False,
                    "calibrators_refit_for_heldout": False,
                    "thresholds_refit_for_heldout": False,
                    "artifacts": _register(
                        [
                            fold_metric_path,
                            proposal_labels_path,
                            threshold_runtime,
                            heldout_runtime,
                            *frozen_paths,
                        ]
                    ),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        fold_records.append(
            {
                "fold_test_circuit": circuit,
                "path": str(fold_manifest_path.resolve()),
                "sha256": _sha256(fold_manifest_path),
                "model_identity_hash": model_identity_hash,
            }
        )
        seal_fold_entries.append(
            {
                "fold_test_circuit": circuit,
                "run_identity": run_identity,
                "model_identity_hash": model_identity_hash,
                "method_identities": method_identities,
                "frozen_artifacts": {path.name: _sha256(path) for path in frozen_paths},
            }
        )

    freeze_seal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "thresholds_frozen_before_heldout",
                "data_content_hash": data_content_hash,
                "source_manifest_hash": source_manifest_hash,
                "build_manifest_hash": _sha256(build_manifest_path),
                "config_content_hash": config_content_hash,
                "code_content_hash": code_content_hash,
                "registered_circuits": list(CIRCUITS),
                "registered_methods": list(METHODS),
                "fold_threshold_identities": seal_fold_entries,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pooled_artifact_paths = [
        probability_path,
        reliability_path,
        reliability_contributions_path,
        monotonicity_path,
        operating_path,
        contributions_path,
        delay_contributions_path,
        delay_metrics_path,
        delay_lstar_path,
        lstar_path,
        pooled_runtime_path,
    ]
    pooled_manifest_path = pooled / "manifest.json"
    pooled_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pooled_heldout_complete",
                "code_content_hash": code_content_hash,
                "data_content_hash": data_content_hash,
                "config_content_hash": config_content_hash,
                "source_manifest_hash": source_manifest_hash,
                "build_manifest_hash": _sha256(build_manifest_path),
                "threshold_freeze_seal": {
                    "path": str(freeze_seal_path.resolve()),
                    "sha256": _sha256(freeze_seal_path),
                    "status": "thresholds_frozen_before_heldout",
                },
                "models_refit": False,
                "thresholds_refit": False,
                "fold_manifests": fold_records,
                "artifacts": _register(pooled_artifact_paths),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    bootstrap_values = {
        "l_star_brace": 1.0,
        "l_star_brace_percentile_95": [1.0, 1.0],
        "l_star_comparator": 0.5,
        "l_star_comparator_percentile_95": [0.25, 0.5],
        "delta_l_star": 0.5,
        "delta_l_star_percentile_95": [0.5, 0.75],
    }
    least_favorable_values = {
        "l_star_brace": 0.5,
        "l_star_brace_percentile_95": [0.5, 0.5],
        "l_star_comparator": 0.5,
        "l_star_comparator_percentile_95": [0.5, 0.5],
        "delta_l_star": 0.0,
        "delta_l_star_percentile_95": [0.0, 0.0],
    }
    summary_path = bootstrap / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                **bootstrap_values,
                "n_resamples": 10_000,
                "seed": 20270927,
                "cluster_level": "car_session_within_circuit",
                "models_and_thresholds_refit": False,
                "least_favorable_unresolved_counted_as_false": {
                    **least_favorable_values,
                    "false_count_column": "least_favorable_false_proposals",
                    "models_and_thresholds_refit": False,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    reliability_interval_path = bootstrap / "reliability-bin-intervals.csv"
    _write_csv(reliability_interval_path, reliability_interval_rows)
    bootstrap_runtime_path = bootstrap / "runtime.jsonl"
    bootstrap_runtime_path.write_text(
        json.dumps(
            {
                "step": "paired_bootstrap_complete",
                "elapsed_seconds": 1.0,
                "n_resamples": 10_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    brace_draws = np.ones(10_000, dtype=float)
    comparator_draws = np.r_[np.full(5_000, 0.25), np.full(5_000, 0.50)]
    delta_draws = brace_draws - comparator_draws
    lf_brace_draws = np.full(10_000, 0.50)
    lf_comparator_draws = np.full(10_000, 0.50)
    draw_values = {
        "brace-l-star-draws.npy": brace_draws,
        "comparator-l-star-draws.npy": comparator_draws,
        "delta-l-star-draws.npy": delta_draws,
        "least-favorable-brace-l-star-draws.npy": lf_brace_draws,
        "least-favorable-comparator-l-star-draws.npy": lf_comparator_draws,
        "least-favorable-delta-l-star-draws.npy": lf_brace_draws - lf_comparator_draws,
    }
    draw_paths: list[Path] = []
    for name, values in draw_values.items():
        path = bootstrap / name
        np.save(path, values, allow_pickle=False)
        draw_paths.append(path)
    bootstrap_manifest_path = bootstrap / "manifest.json"
    bootstrap_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code_content_hash": code_content_hash,
                "input": {
                    "path": str(contributions_path.resolve()),
                    "sha256": _sha256(contributions_path),
                },
                "analysis": {
                    "n_resamples": 10_000,
                    "seed": 20270927,
                    "cluster_level": "car_session_within_circuit",
                    "models_and_thresholds_refit": False,
                },
                "interval": {
                    "construction": "percentile_95",
                    "paired": True,
                    "resampling_unit": "car_session_within_circuit",
                },
                "estimand": {
                    "brace_method": "brace_bayesian",
                    "comparator_method": "posterior_mean_twin",
                    "false_budget_per_hour": 2.0,
                    "minimum_recall": 0.5,
                    "lead_grid_s": list(LEADS),
                    "least_favorable_false_count_column": "least_favorable_false_proposals",
                },
                "experiment_manifest": {
                    "path": str(pooled_manifest_path.resolve()),
                    "sha256": _sha256(pooled_manifest_path),
                },
                "sample": {
                    "circuit_count": 4,
                    "car_session_count": 58,
                    "circuits": list(CIRCUITS),
                },
                "study_config": {
                    "path": str(study_config_path.resolve()),
                    "sha256": _sha256(study_config_path),
                    "canonical_content_hash": config_content_hash,
                },
                "reliability_bootstrap": {
                    "input_path": str(reliability_contributions_path.resolve()),
                    "input_sha256": _sha256(reliability_contributions_path),
                    "n_resamples": 10_000,
                    "seed": 20270927,
                    "cluster_level": "car_session_within_circuit",
                    "fixed_equal_width_bins": 10,
                },
                "noncanonical_synthetic": False,
                "artifacts": _register(
                    [
                        summary_path,
                        reliability_interval_path,
                        bootstrap_runtime_path,
                        *draw_paths,
                    ]
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    transport = tmp_path / "bootstrap-transport"
    transport.mkdir()
    transport_summary_path = transport / "summary.json"
    transport_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    transport_summary["cluster_level"] = "circuit_then_car_session"
    transport_summary_path.write_text(
        json.dumps(transport_summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transport_manifest = json.loads(bootstrap_manifest_path.read_text(encoding="utf-8"))
    transport_manifest["analysis"]["cluster_level"] = "circuit_then_car_session"
    transport_manifest["interval"]["resampling_unit"] = "circuit_then_car_session"
    transport_artifacts = {str(transport_summary_path.resolve()): _sha256(transport_summary_path)}
    for raw_path in transport_manifest["artifacts"]:
        source = Path(raw_path)
        if source == summary_path.resolve():
            continue
        destination = transport / source.name
        shutil.copy2(source, destination)
        transport_artifacts[str(destination.resolve())] = _sha256(destination)
    transport_manifest["artifacts"] = transport_artifacts
    (transport / "manifest.json").write_text(
        json.dumps(transport_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fixture_paths = [
        pooled,
        bootstrap,
        transport,
        study_config_path,
        source_manifest_path,
        build_output,
        build_manifest_path,
        freeze_seal_path,
        pooled_manifest_path,
        bootstrap_manifest_path,
        transport / "manifest.json",
    ]
    for record in fold_records:
        fold_manifest_path = Path(str(record["path"]))
        fixture_paths.append(fold_manifest_path)
        fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        fixture_paths.extend(Path(path) for path in fold_manifest["artifacts"])
    pooled_manifest = json.loads(pooled_manifest_path.read_text(encoding="utf-8"))
    fixture_paths.extend(Path(path) for path in pooled_manifest["artifacts"])
    fixture_paths.append(Path(pooled_manifest["threshold_freeze_seal"]["path"]))
    for manifest_path in (bootstrap_manifest_path, transport / "manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixture_paths.extend(Path(path) for path in manifest["artifacts"])
        fixture_paths.append(Path(manifest["input"]["path"]))
        fixture_paths.append(Path(manifest["experiment_manifest"]["path"]))
    _assert_within_tmp(tmp_path, *fixture_paths)
    return pooled, bootstrap


def _compile(tmp_path: Path, pooled: Path, bootstrap: Path) -> Path:
    output_root = tmp_path / "submission"
    compile_submission(
        pooled_dir=pooled,
        bootstrap_dir=bootstrap,
        transport_bootstrap_dir=tmp_path / "bootstrap-transport",
        manuscript_shell=ROOT / "paper" / "manuscript-results-shell.md",
        abstract_shell=ROOT / "paper" / "ssac27-abstract-results-shell.md",
        output_root=output_root,
    )
    return output_root


def _compile_actual(tmp_path: Path, *, output_name: str = "submission") -> Path:
    output_root = tmp_path / output_name
    compile_submission(
        pooled_dir=ACTUAL_POOLED,
        bootstrap_dir=ACTUAL_BOOTSTRAP,
        transport_bootstrap_dir=ACTUAL_TRANSPORT_BOOTSTRAP,
        manuscript_shell=ROOT / "paper" / "manuscript-results-shell.md",
        abstract_shell=ROOT / "paper" / "ssac27-abstract-results-shell.md",
        output_root=output_root,
    )
    return output_root


def _assert_within_tmp(tmp_path: Path, *paths: Path) -> None:
    tmp_root = tmp_path.resolve()
    for path in paths:
        assert path.resolve().is_relative_to(tmp_root), (
            f"synthetic-fixture write escaped tmp_path: {path.resolve()}"
        )


def _actual_evidence_with_tmp_fold_manifests(
    tmp_path: Path,
) -> tuple[object, list[dict[str, object]]]:
    evidence = MODULE._load_evidence(
        ACTUAL_POOLED,
        ACTUAL_BOOTSTRAP,
        ACTUAL_TRANSPORT_BOOTSTRAP,
    )
    pooled_payload = copy.deepcopy(dict(evidence.pooled.payload))
    raw_records = pooled_payload["fold_manifests"]
    assert isinstance(raw_records, list)
    cloned_records: list[dict[str, object]] = []
    for raw_record in raw_records:
        assert isinstance(raw_record, dict)
        record = raw_record
        source_manifest_path = Path(str(record["path"]))
        circuit = str(record["fold_test_circuit"])
        destination = tmp_path / "fold-manifests" / circuit / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        registry = manifest["artifacts"]
        assert isinstance(registry, dict)
        manifest["artifacts"] = {
            str((tmp_path / "fold-artifact-records" / circuit / Path(path).name).resolve()): digest
            for path, digest in registry.items()
        }
        destination.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record["path"] = str(destination.resolve())
        _assert_within_tmp(
            tmp_path,
            destination,
            *(Path(path) for path in manifest["artifacts"]),
        )
        cloned_records.append(record)
    cloned_evidence = replace(
        evidence,
        pooled=replace(evidence.pooled, payload=pooled_payload),
    )
    return cloned_evidence, cloned_records


def _refresh_pooled_artifact_provenance(
    tmp_path: Path,
    pooled: Path,
    bootstrap: Path,
    artifact_path: Path,
) -> None:
    pooled_manifest_path = pooled / "manifest.json"
    _assert_within_tmp(
        tmp_path,
        pooled_manifest_path,
        bootstrap / "manifest.json",
        tmp_path / "bootstrap-transport" / "manifest.json",
        artifact_path,
    )
    pooled_manifest = json.loads(pooled_manifest_path.read_text(encoding="utf-8"))
    pooled_manifest["artifacts"][str(artifact_path.resolve())] = _sha256(artifact_path)
    pooled_manifest_path.write_text(
        json.dumps(pooled_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    for manifest_path in (
        bootstrap / "manifest.json",
        tmp_path / "bootstrap-transport" / "manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["experiment_manifest"]["sha256"] = _sha256(pooled_manifest_path)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def test_compiler_authenticates_artifacts_fills_all_tokens_and_exports_bundle(
    tmp_path: Path,
) -> None:
    pooled = ACTUAL_POOLED
    bootstrap = ACTUAL_BOOTSTRAP

    output_root = _compile_actual(tmp_path)

    manuscript = (output_root / "paper" / "manuscript-final.md").read_text(encoding="utf-8")
    abstract = (output_root / "paper" / "ssac27-abstract-final.md").read_text(encoding="utf-8")
    assert "{{" not in manuscript
    assert "{{" not in abstract
    assert "Result-token contract" not in manuscript
    assert "Results-ready manuscript" not in manuscript
    assert 'date: "October 2026"' in manuscript
    assert "Results-shell word count" not in abstract
    assert "prespecified endpoint values were $L^*=0.00$ s for BRACE" in manuscript
    probability_table = (
        output_root / "analysis-output" / "tables" / "table-probability-metrics.md"
    ).read_text(encoding="utf-8")
    probability_lines = [
        line for line in probability_table.splitlines() if line.startswith("|")
    ]
    assert probability_lines[0] == (
        "| Method | Brier | Brier skill | Log score | Stepwise AP | Cal. intercept | "
        "Cal. slope | ECE |"
    )
    assert len(probability_lines) == 8
    assert all(line.count("|") == 9 for line in probability_lines)
    assert "Total rows" not in probability_lines[0]
    assert "Positive rows" not in probability_lines[0]
    assert "positive frame-horizon rows" in probability_table
    assert "not unique excursion events" in probability_table
    assert "0.020975" in probability_table
    assert "no demonstrated gain under the prespecified gates" in manuscript
    assert "forecast rows/s" in manuscript
    assert "macOS-" in manuscript
    assert MODULE._conservative_word_count(abstract) <= 450
    assert "figure-01-warning-frontier" not in abstract
    assert "Calibration-only exploratory mechanism diagnostic" in manuscript
    assert "figure-03-calibration-operating-cliff.pdf" in manuscript
    assert "figure-01-warning-frontier.pdf" not in manuscript
    assert "ten fixed equal-width bins" in manuscript
    assert "does not isolate uncertainty" in manuscript
    assert (output_root / "paper" / "references.bib").read_bytes() == (
        ROOT / "references.bib"
    ).read_bytes()

    expected_tables = {
        "table-primary-frontier.md",
        "table-probability-metrics.md",
        "table-circuit-results.md",
        "table-least-favorable.md",
        "table-delay-results.md",
    }
    assert {
        path.name for path in (output_root / "analysis-output" / "tables").glob("*.md")
    } == expected_tables
    for name in (
        "analysis-report.md",
        "stats-appendix.md",
        "figure-catalog.md",
    ):
        assert (output_root / "analysis-output" / name).is_file()
    for stem in (
        "figure-01-warning-frontier",
        "figure-02-reliability",
        "figure-03-calibration-operating-cliff",
    ):
        assert (output_root / "analysis-output" / "figures" / f"{stem}.pdf").stat().st_size > 0
        assert (output_root / "analysis-output" / "figures" / f"{stem}.png").stat().st_size > 0
    result_manifest = json.loads(
        (output_root / "analysis-output" / "submission-results-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for markdown_path in output_root.rglob("*.md"):
        assert re.search(
            r"\bregistered\b",
            markdown_path.read_text(encoding="utf-8"),
            flags=re.IGNORECASE,
        ) is None, markdown_path
    assert result_manifest["compiler"]["path"] == "scripts/build_submission_results.py"
    assert result_manifest["paper_sources"]["manuscript_shell"]["path"] == (
        "paper/manuscript-results-shell.md"
    )
    assert result_manifest["paper_sources"]["abstract_shell"]["path"] == (
        "paper/ssac27-abstract-results-shell.md"
    )
    assert result_manifest["paper_sources"]["bibliography"]["path"] == "references.bib"
    assert result_manifest["publication_environment"]["requirements"]["path"] == (
        "publication-requirements.in"
    )
    assert result_manifest["publication_environment"]["lock"]["path"] == (
        "publication-requirements.lock"
    )
    assert result_manifest["publication_environment"]["reliability_bootstrap_source"][
        "path"
    ] == "src/brace_f1/bootstrap_reliability.py"
    assert result_manifest["input_manifests"]["pooled_heldout"]["path"] == (
        "input-manifests/pooled-heldout/manifest.json"
    )
    assert result_manifest["input_manifests"]["primary_bootstrap"]["path"] == (
        "input-manifests/primary-bootstrap/manifest.json"
    )
    assert result_manifest["input_manifests"]["transport_bootstrap"]["path"] == (
        "input-manifests/transport-bootstrap/manifest.json"
    )
    cliff_sources = result_manifest["figure_sources"]["calibration_operating_cliff"]
    expected_cliff_sources = {
        "scripts/build_calibration_operating_cliff_figure.py",
        "paper/figure-source/figure-03-calibration-operating-cliff.csv",
        "paper/figures/figure-03-calibration-operating-cliff-provenance.md",
        "paper/figures/figure-03-calibration-operating-cliff.pdf",
        "paper/figures/figure-03-calibration-operating-cliff.png",
    }
    assert set(MODULE.CALIBRATION_CLIFF_SOURCE_SHA256) == expected_cliff_sources
    assert set(cliff_sources) == expected_cliff_sources
    for source_path, expected_hash in MODULE.CALIBRATION_CLIFF_SOURCE_SHA256.items():
        record = cliff_sources[source_path]
        assert record["sha256"] == expected_hash
        staged_path = output_root / record["staged_path"]
        assert staged_path.is_file()
        assert _sha256(staged_path) == expected_hash
    renderer_record = cliff_sources[
        "scripts/build_calibration_operating_cliff_figure.py"
    ]
    assert renderer_record["role"] == "archival_renderer_source"
    assert renderer_record["runnable_in_bundle"] is False
    source_manifests = {
        "pooled_heldout": pooled / "manifest.json",
        "primary_bootstrap": bootstrap / "manifest.json",
        "transport_bootstrap": ACTUAL_TRANSPORT_BOOTSTRAP / "manifest.json",
    }
    for role, source_manifest_path in source_manifests.items():
        record = result_manifest["input_manifests"][role]
        portable_manifest_path = output_root / record["path"]
        source_manifest_hash = _sha256(source_manifest_path)
        assert portable_manifest_path.is_file()
        assert record["sha256"] == _sha256(portable_manifest_path)
        assert record["authenticated_source_manifest_sha256"] == source_manifest_hash
        assert record["path"] in result_manifest["outputs"]

        portable_manifest = json.loads(portable_manifest_path.read_text(encoding="utf-8"))
        assert portable_manifest["role"] == role
        assert (
            portable_manifest["authenticated_source_manifest_sha256"]
            == source_manifest_hash
        )
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        assert {artifact["name"] for artifact in portable_manifest["artifacts"]} == {
            Path(path).name for path in source_manifest["artifacts"]
        }
        portable_strings = _nested_strings(portable_manifest)
        assert all(not Path(value).is_absolute() for value in portable_strings)
        assert "/Users/" not in json.dumps(portable_manifest, sort_keys=True)
    assert result_manifest["pandoc_render_contract"]["working_directory"] == "."
    assert "paper/manuscript-final.md" in result_manifest["outputs"]
    manifest_relative_path = "analysis-output/submission-results-manifest.json"
    published_relative_paths = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert published_relative_paths == set(result_manifest["outputs"]) | {
        manifest_relative_path
    }
    for relative_path, expected_hash in result_manifest["outputs"].items():
        assert _sha256(output_root / relative_path) == expected_hash
    assert all(not Path(path).is_absolute() for path in result_manifest["outputs"])
    manifest_strings = _nested_strings(result_manifest)
    assert all(not Path(value).is_absolute() for value in manifest_strings)
    serialized_manifest = json.dumps(result_manifest, sort_keys=True)
    assert "/Users/" not in serialized_manifest
    assert str(tmp_path.resolve()) not in serialized_manifest
    assert Path.home().name not in serialized_manifest
    assert result_manifest["conservative_abstract_word_count"] <= 450
    assert result_manifest["publication_environment"]["installed_versions"]["pubfig"] == "0.3.0"
    assert result_manifest["publication_environment"]["installed_versions"]["pyarrow"] == "25.0.1"
    assert result_manifest["publication_environment"]["installed_versions"]["pypdf"] == "6.19.0"
    assert result_manifest["publication_environment"]["parquet_read_smoke"] is True
    assert result_manifest["publication_environment"]["source_date_epoch"] == "0"
    assert len(
        result_manifest["publication_environment"]["reliability_bootstrap_source"]["sha256"]
    ) == 64
    assert (
        result_manifest["publication_environment"]["reliability_bootstrap_source"][
            "producer_source_tree_sha256"
        ]
        == _producer_source_tree_hash()
    )
    assert len(result_manifest["publication_environment"]["lock"]["sha256"]) == 64
    assert len(result_manifest["compiler"]["sha256"]) == 64
    assert len(result_manifest["paper_sources"]["manuscript_shell"]["sha256"]) == 64
    assert len(result_manifest["paper_sources"]["bibliography"]["sha256"]) == 64
    pandoc_command = result_manifest["pandoc_render_contract"]["command"]
    assert pandoc_command[1] == "--fail-if-warnings"
    assert "--citeproc" in pandoc_command
    portability_arguments = [
        argument
        for argument in pandoc_command
        if argument.startswith("--variable=header-includes:")
    ]
    assert len(portability_arguments) == 1
    assert r"\pdfsuppressptexinfo=7" in portability_arguments[0]
    assert r"\pdftrailerid" not in portability_arguments[0]
    assert result_manifest["pandoc_gate"]["bibliography_present"] is True
    assert result_manifest["pandoc_gate"]["citeproc_enabled"] is True
    assert result_manifest["pandoc_gate"]["unresolved_citation_count"] == 0
    assert result_manifest["pandoc_gate"]["pdf_page_count"] > 0
    assert result_manifest["pandoc_gate"]["pdf_image_count"] >= 2
    assert result_manifest["pandoc_gate"]["pdf_text_checked"] is True
    assert result_manifest["pandoc_gate"]["binary_portability_checked"] is True
    assert result_manifest["pandoc_gate"]["binary_portability_artifact_count"] == 7
    assert result_manifest["pandoc_gate"]["source_date_epoch"] == "0"
    manuscript_pdf = output_root / "paper" / "manuscript-final.pdf"
    assert manuscript_pdf.stat().st_size > 0
    manuscript_pdf_bytes = manuscript_pdf.read_bytes().replace(b"\\", b"/")
    assert b"/PTEX.FileName" not in manuscript_pdf_bytes
    assert b"/private/var/" not in manuscript_pdf_bytes
    assert b"/Users/" not in manuscript_pdf_bytes
    for binary_path in output_root.rglob("*"):
        if not binary_path.is_file() or binary_path.suffix.lower() not in {".pdf", ".png"}:
            continue
        MODULE._validate_portable_binary(binary_path)
    expected_pdf_id = _sha256(output_root / "paper" / "manuscript-final.md")[:32]
    assert result_manifest["pandoc_gate"]["pdf_trailer_id"] == expected_pdf_id
    first_pdf_ids = re.findall(
        rb"/ID\s*\[\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\]",
        manuscript_pdf.read_bytes(),
    )
    assert first_pdf_ids == [
        (expected_pdf_id.encode("ascii"), expected_pdf_id.encode("ascii"))
    ]

    repeat_output = tmp_path / "submission-repeat"
    compile_submission(
        pooled_dir=pooled,
        bootstrap_dir=bootstrap,
        transport_bootstrap_dir=ACTUAL_TRANSPORT_BOOTSTRAP,
        manuscript_shell=ROOT / "paper" / "manuscript-results-shell.md",
        abstract_shell=ROOT / "paper" / "ssac27-abstract-results-shell.md",
        output_root=repeat_output,
    )
    repeat_pdf_ids = re.findall(
        rb"/ID\s*\[\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\]",
        (repeat_output / "paper" / "manuscript-final.pdf").read_bytes(),
    )
    assert repeat_pdf_ids == first_pdf_ids

    full_operating_path = (
        output_root
        / "analysis-output"
        / "tables"
        / "pooled-heldout-operating-metrics.csv"
    )
    full_operating = pd.read_csv(full_operating_path)
    assert len(full_operating) == len(METHODS) * len(LEADS) * len(BUDGETS)
    assert set(full_operating["method"]) == set(METHODS)
    assert set(full_operating["required_lead_s"]) == set(LEADS)
    assert set(full_operating["false_budget_per_hour"]) == set(BUDGETS)
    frontier_table = (
        output_root / "analysis-output" / "tables" / "table-primary-frontier.md"
    ).read_text(encoding="utf-8")
    frontier_lines = [line for line in frontier_table.splitlines() if line.startswith("|")]
    assert frontier_lines[0] == (
        "| Budget / h | Method | $L^*$ (s) | Localized recall | False proposals | "
        "False rate / h | Upper 95% / h |"
    )
    assert len(frontier_lines) == 8
    assert all(line.count("|") == 8 for line in frontier_lines)
    assert "Upper 95% / h" in frontier_table
    assert "6.317055 eligible simulated car-hours" in frontier_table


def test_nonzero_nonsilent_evidence_does_not_render_silence_claims(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path, examined_proposals_per_fold=1)
    evidence = MODULE._load_evidence(
        pooled,
        bootstrap,
        tmp_path / "bootstrap-transport",
    )
    tokens, tables, _, _ = MODULE._derive_tokens(evidence)
    manuscript = MODULE._replace_tokens(
        (ROOT / "paper" / "manuscript-results-shell.md").read_text(encoding="utf-8"),
        tokens,
        "nonzero manuscript fixture",
    )
    abstract = MODULE._replace_tokens(
        (ROOT / "paper" / "ssac27-abstract-results-shell.md").read_text(encoding="utf-8"),
        tokens,
        "nonzero abstract fixture",
    )
    reports = MODULE._analysis_artifacts(evidence, tokens, tables)
    rendered = "\n".join((manuscript, abstract, *reports))
    for forbidden in (
        "Both policies were silent",
        "both policies were silent",
        "neither method emitted a held-out proposal",
        "All 10,000 paired contrasts were zero",
        "No Qualifying Warning",
        "no qualifying localized warning",
        "hiding operational failure",
        "all-zero frontier",
        "did not give the same comparison",
        "does not justify an active protection response or its additional",
        "favorable frame-level metrics can fail to yield",
    ):
        assert forbidden not in rendered
    assert "emitted 4 held-out proposals" in rendered
    assert "interval excludes zero" in rendered
    assert (
        "BRACE had higher stepwise average precision, lower log score, and a larger "
        "prespecified endpoint: $L^*=1.00$ s versus $L^*=0.50$ s for the twin."
        in rendered
    )


def test_fixed_calibration_figure_rejects_mismatched_evidence(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)

    with pytest.raises(
        ArtifactValidationError,
        match="calibration operating-cliff evidence binding mismatch",
    ):
        _compile(tmp_path, pooled, bootstrap)


@pytest.mark.parametrize(
    ("dimension", "expected_error"),
    (
        ("seal", "threshold-freeze seal"),
        ("fold_manifest", "Bahrain fold manifest"),
        ("calibrated_scores", "Bahrain scores"),
        ("calibration_thresholds", "Bahrain thresholds"),
    ),
)
def test_fixed_calibration_figure_binding_rejects_each_identity_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dimension: str,
    expected_error: str,
) -> None:
    evidence, records = _actual_evidence_with_tmp_fold_manifests(tmp_path)
    bahrain_record = next(
        record for record in records if record["fold_test_circuit"] == "Bahrain"
    )
    if dimension == "seal":
        seal_record = evidence.pooled.payload["threshold_freeze_seal"]
        assert isinstance(seal_record, dict)
        seal_record["sha256"] = "0" * 64
    elif dimension == "fold_manifest":
        bahrain_record["sha256"] = "0" * 64
    else:
        fold_manifest_path = Path(str(bahrain_record["path"]))
        _assert_within_tmp(tmp_path, fold_manifest_path)
        fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        filename = {
            "calibrated_scores": "compact-calibrated-scores.parquet",
            "calibration_thresholds": "calibration-thresholds.csv",
        }[dimension]
        artifact_path = next(
            path for path in fold_manifest["artifacts"] if Path(path).name == filename
        )
        _assert_within_tmp(tmp_path, Path(artifact_path))
        fold_manifest["artifacts"][artifact_path] = "0" * 64
        fold_manifest_path.write_text(
            json.dumps(fold_manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            MODULE,
            "_manifest_record_path",
            lambda record, _label: Path(str(record["path"])),
        )

    with pytest.raises(ArtifactValidationError, match=expected_error):
        MODULE._validate_calibration_operating_cliff_binding(evidence)


def test_fixed_calibration_figure_rejects_bad_reviewed_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = MODULE._load_evidence(
        ACTUAL_POOLED,
        ACTUAL_BOOTSTRAP,
        ACTUAL_TRANSPORT_BOOTSTRAP,
    )
    bad_hashes = dict(MODULE.CALIBRATION_CLIFF_SOURCE_SHA256)
    bad_hashes["paper/figure-source/figure-03-calibration-operating-cliff.csv"] = "0" * 64
    monkeypatch.setattr(MODULE, "CALIBRATION_CLIFF_SOURCE_SHA256", bad_hashes)

    with pytest.raises(ArtifactValidationError, match="independently reviewed digest"):
        MODULE._stage_calibration_operating_cliff(tmp_path / "stage", evidence)


def test_synthetic_fixture_manifest_paths_stay_within_tmp_path(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    pooled_manifest = json.loads((pooled / "manifest.json").read_text(encoding="utf-8"))
    recorded_paths = [
        Path(pooled_manifest["threshold_freeze_seal"]["path"]),
        *(Path(path) for path in pooled_manifest["artifacts"]),
    ]
    for record in pooled_manifest["fold_manifests"]:
        fold_manifest_path = Path(record["path"])
        recorded_paths.append(fold_manifest_path)
        fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        recorded_paths.extend(Path(path) for path in fold_manifest["artifacts"])
    for manifest_path in (
        bootstrap / "manifest.json",
        tmp_path / "bootstrap-transport" / "manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_paths.extend(Path(path) for path in manifest["artifacts"])
        recorded_paths.append(Path(manifest["input"]["path"]))
        recorded_paths.append(Path(manifest["experiment_manifest"]["path"]))

    _assert_within_tmp(tmp_path, *recorded_paths)


def test_compiler_uses_nonclaim_wording_when_primary_interval_contains_zero(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    summary_path = bootstrap / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["l_star_brace_percentile_95"] = [0.5, 1.0]
    summary["l_star_comparator_percentile_95"] = [0.5, 0.5]
    summary["delta_l_star_percentile_95"] = [0.0, 0.5]
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    brace_path = bootstrap / "brace-l-star-draws.npy"
    comparator_path = bootstrap / "comparator-l-star-draws.npy"
    delta_path = bootstrap / "delta-l-star-draws.npy"
    brace_draws = np.r_[np.full(5_000, 0.5), np.full(5_000, 1.0)]
    comparator_draws = np.full(10_000, 0.5)
    np.save(brace_path, brace_draws, allow_pickle=False)
    np.save(comparator_path, comparator_draws, allow_pickle=False)
    np.save(delta_path, brace_draws - comparator_draws, allow_pickle=False)
    manifest_path = bootstrap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][str(summary_path.resolve())] = _sha256(summary_path)
    for path in (brace_path, comparator_path, delta_path):
        manifest["artifacts"][str(path.resolve())] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    evidence = MODULE._load_evidence(
        pooled,
        bootstrap,
        tmp_path / "bootstrap-transport",
    )
    tokens, _, _, _ = MODULE._derive_tokens(evidence)
    manuscript = MODULE._replace_tokens(
        (ROOT / "paper" / "manuscript-results-shell.md").read_text(encoding="utf-8"),
        tokens,
        "interval-containing-zero manuscript fixture",
    )
    assert "no demonstrated gain under the prespecified gates." in manuscript
    assert "interval excludes zero" not in manuscript


def test_compiler_rejects_a_stale_registered_artifact(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    (pooled / "pooled-heldout-operating-metrics.csv").write_text("stale\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="hash mismatch"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_unregistered_result_tokens(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    shell = tmp_path / "unknown-shell.md"
    shell.write_text(
        (ROOT / "paper" / "manuscript-results-shell.md").read_text(encoding="utf-8")
        + "\n{{UNKNOWN_RESULT}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="unrecognized result tokens"):
        compile_submission(
            pooled_dir=pooled,
            bootstrap_dir=bootstrap,
            transport_bootstrap_dir=tmp_path / "bootstrap-transport",
            manuscript_shell=shell,
            abstract_shell=ROOT / "paper" / "ssac27-abstract-results-shell.md",
            output_root=tmp_path / "submission",
        )


def test_compiler_rejects_registered_language_in_custom_public_shell(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    shell = tmp_path / "custom-manuscript-shell.md"
    source = (ROOT / "paper" / "manuscript-results-shell.md").read_text(
        encoding="utf-8"
    )
    shell.write_text(
        source.replace(
            "# 1. Introduction",
            "# 1. Introduction\n\nThe registered analysis is described below.",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="standalone 'registered'"):
        compile_submission(
            pooled_dir=pooled,
            bootstrap_dir=bootstrap,
            transport_bootstrap_dir=tmp_path / "bootstrap-transport",
            manuscript_shell=shell,
            abstract_shell=ROOT / "paper" / "ssac27-abstract-results-shell.md",
            output_root=tmp_path / "submission",
        )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        r"\\server\share\private.csv",
        r"C:\Users\alice\private.csv",
        "~/private/private.csv",
        "~alice/private/private.csv",
        "file:///Users/alice/private.csv",
        "file://localhost/etc/private.csv",
    ],
)
def test_portable_manifest_guard_rejects_nonportable_path_forms(
    unsafe_path: str,
) -> None:
    payload = {"nested": [{"artifact": unsafe_path}]}

    with pytest.raises(ArtifactValidationError, match="local path"):
        MODULE._validate_portable_publication_manifest(payload)


@pytest.mark.parametrize(
    "unsafe_bytes",
    [
        b"/PTEX.FileName (/private/var/folders/build/figure.pdf)",
        b"binary payload with /Users/alice/private.csv",
        b"binary payload with C:\\Users\\alice\\private.csv",
    ],
)
def test_portable_binary_guard_rejects_local_paths(
    tmp_path: Path,
    unsafe_bytes: bytes,
) -> None:
    artifact = tmp_path / "artifact.pdf"
    artifact.write_bytes(unsafe_bytes)

    with pytest.raises(ArtifactValidationError, match="nonportable local path"):
        MODULE._validate_portable_binary(artifact)


def test_compiler_rejects_bootstrap_summary_that_disagrees_with_draws(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    brace_path = bootstrap / "brace-l-star-draws.npy"
    comparator_path = bootstrap / "comparator-l-star-draws.npy"
    delta_path = bootstrap / "delta-l-star-draws.npy"
    brace_draws = np.load(brace_path, allow_pickle=False)
    comparator_draws = np.load(comparator_path, allow_pickle=False)
    brace_draws[:500] = 0.5
    np.save(brace_path, brace_draws, allow_pickle=False)
    np.save(delta_path, brace_draws - comparator_draws, allow_pickle=False)
    manifest_path = bootstrap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (brace_path, delta_path):
        manifest["artifacts"][str(path.resolve())] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="summary interval"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_recomputes_least_favorable_percentile_from_exact_draws(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    brace_path = bootstrap / "least-favorable-brace-l-star-draws.npy"
    comparator_path = bootstrap / "least-favorable-comparator-l-star-draws.npy"
    delta_path = bootstrap / "least-favorable-delta-l-star-draws.npy"
    brace_draws = np.load(brace_path, allow_pickle=False)
    comparator_draws = np.load(comparator_path, allow_pickle=False)
    brace_draws[:500] = 0.25
    np.save(brace_path, brace_draws, allow_pickle=False)
    np.save(delta_path, brace_draws - comparator_draws, allow_pickle=False)
    manifest_path = bootstrap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (brace_path, delta_path):
        manifest["artifacts"][str(path.resolve())] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="least-favorable bootstrap draws disagree"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_recomputes_least_favorable_point_from_contribution_ledger(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    sensitivity_path = pooled / "pooled-heldout-l-star-sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    row = sensitivity["analysis"] == "least_favorable_unresolved_counted_as_false"
    sensitivity.loc[row, "l_star_brace_s"] = 1.0
    sensitivity.loc[row, "l_star_comparator_s"] = 0.5
    sensitivity.loc[row, "delta_l_star_s"] = 0.5
    sensitivity.to_csv(sensitivity_path, index=False, lineterminator="\n")
    _refresh_pooled_artifact_provenance(
        tmp_path,
        pooled,
        bootstrap,
        sensitivity_path,
    )
    summary_path = bootstrap / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    least_favorable = summary["least_favorable_unresolved_counted_as_false"]
    least_favorable["l_star_brace"] = 1.0
    least_favorable["l_star_comparator"] = 0.5
    least_favorable["delta_l_star"] = 0.5
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = bootstrap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][str(summary_path.resolve())] = _sha256(summary_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="least-favorable L-star disagrees"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_reliability_points_that_disagree_with_bootstrap(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    reliability_path = bootstrap / "reliability-bin-intervals.csv"
    reliability = pd.read_csv(reliability_path)
    reliability.loc[0, "mean_probability"] = 0.06
    reliability.to_csv(reliability_path, index=False, lineterminator="\n")
    manifest_path = bootstrap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][str(reliability_path.resolve())] = _sha256(reliability_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="reliability point estimates"):
        _compile(tmp_path, pooled, bootstrap)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("count", -1, "reliability point values or counts"),
        ("mean_probability", 0.95, "outside its fixed bin"),
    ],
)
def test_compiler_rejects_impossible_reliability_points_even_when_sources_agree(
    tmp_path: Path,
    column: str,
    value: float,
    message: str,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    pooled_path = pooled / "pooled-heldout-reliability-bins.csv"
    interval_path = bootstrap / "reliability-bin-intervals.csv"
    pooled_table = pd.read_csv(pooled_path)
    interval_table = pd.read_csv(interval_path)
    pooled_table.loc[0, column] = value
    interval_table.loc[0, column] = value
    pooled_table.to_csv(pooled_path, index=False, lineterminator="\n")
    interval_table.to_csv(interval_path, index=False, lineterminator="\n")
    _refresh_pooled_artifact_provenance(tmp_path, pooled, bootstrap, pooled_path)
    bootstrap_manifest_path = bootstrap / "manifest.json"
    bootstrap_manifest = json.loads(
        bootstrap_manifest_path.read_text(encoding="utf-8")
    )
    bootstrap_manifest["artifacts"][str(interval_path.resolve())] = _sha256(interval_path)
    bootstrap_manifest_path.write_text(
        json.dumps(bootstrap_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ArtifactValidationError, match=message):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_build_manifest_that_disagrees_with_pooled_hash(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    build_manifest_path = tmp_path / "data" / "manifests" / "deepracing-build.json"
    build_manifest_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="build manifest"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_missing_acquisition_source_manifest(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    source_manifest_path = tmp_path / "data" / "manifests" / "deepracing-files.csv"
    source_manifest_path.unlink()

    with pytest.raises(ArtifactValidationError, match="missing acquisition source manifest"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_tampered_acquisition_source_manifest(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    source_manifest_path = tmp_path / "data" / "manifests" / "deepracing-files.csv"
    source = source_manifest_path.read_bytes()
    source_manifest_path.write_bytes(b"X" + source[1:])

    with pytest.raises(ArtifactValidationError, match="hash mismatch.*acquisition source manifest"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_wrong_acquisition_source_manifest_byte_count(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path, source_manifest_bytes_delta=1)

    with pytest.raises(ArtifactValidationError, match="byte count changed"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_missing_per_circuit_build_counts(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path, invalid_per_circuit_counts=True)

    with pytest.raises(ArtifactValidationError, match="per-circuit counts"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_hard_coded_cohort_table_that_disagrees_with_build(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    shell = tmp_path / "manuscript-with-stale-cohort.md"
    source = (ROOT / "paper" / "manuscript-results-shell.md").read_text(encoding="utf-8")
    assert "| Bahrain | 20 | 36 | 2.542886 | 14.16 |" in source
    shell.write_text(
        source.replace(
            "| Bahrain | 20 | 36 | 2.542886 | 14.16 |",
            "| Bahrain | 20 | 36 | 2.500000 | 14.40 |",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="cohort table"):
        compile_submission(
            pooled_dir=pooled,
            bootstrap_dir=bootstrap,
            transport_bootstrap_dir=tmp_path / "bootstrap-transport",
            manuscript_shell=shell,
            abstract_shell=ROOT / "paper" / "ssac27-abstract-results-shell.md",
            output_root=tmp_path / "submission",
        )


def test_compiler_rejects_operating_counts_not_supported_by_contributions(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    operating_path = pooled / "pooled-heldout-operating-metrics.csv"
    operating = pd.read_csv(operating_path)
    row = (
        (operating["method"] == "brace_bayesian")
        & np.isclose(operating["required_lead_s"], 1.0)
        & np.isclose(operating["false_budget_per_hour"], 2.0)
    )
    assert row.sum() == 1
    operating.loc[row, "localized_event_hits"] += 1
    operating.loc[row, "localized_event_recall"] = (
        operating.loc[row, "localized_event_hits"] / operating.loc[row, "qualified_events"]
    )
    operating.to_csv(operating_path, index=False, lineterminator="\n")
    pooled_manifest_path = pooled / "manifest.json"
    pooled_manifest = json.loads(pooled_manifest_path.read_text(encoding="utf-8"))
    pooled_manifest["artifacts"][str(operating_path.resolve())] = _sha256(operating_path)
    pooled_manifest_path.write_text(
        json.dumps(pooled_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    for manifest_path in (
        bootstrap / "manifest.json",
        tmp_path / "bootstrap-transport" / "manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["experiment_manifest"]["sha256"] = _sha256(pooled_manifest_path)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="contribution ledger"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_self_consistent_impossible_event_hit_hierarchy(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    contributions_path = pooled / "pooled-heldout-car-contributions.parquet"
    contributions = pd.read_parquet(contributions_path)
    contribution_row = (
        (contributions["method"] == "brace_bayesian")
        & np.isclose(contributions["required_lead_s"], 1.0)
        & np.isclose(contributions["false_budget_per_hour"], 2.0)
    )
    contributions.loc[contribution_row, "correct_side_event_hits"] = (
        contributions.loc[contribution_row, "qualified_events"] + 1
    )
    contributions.to_parquet(contributions_path, index=False)

    operating_path = pooled / "pooled-heldout-operating-metrics.csv"
    operating = pd.read_csv(operating_path)
    operating_row = (
        (operating["method"] == "brace_bayesian")
        & np.isclose(operating["required_lead_s"], 1.0)
        & np.isclose(operating["false_budget_per_hour"], 2.0)
    )
    impossible_hits = int(
        contributions.loc[contribution_row, "correct_side_event_hits"].sum()
    )
    operating.loc[operating_row, "correct_side_event_hits"] = impossible_hits
    operating.loc[operating_row, "correct_side_event_recall"] = (
        impossible_hits / operating.loc[operating_row, "qualified_events"]
    )
    operating.to_csv(operating_path, index=False, lineterminator="\n")
    for artifact in (contributions_path, operating_path):
        _refresh_pooled_artifact_provenance(
            tmp_path,
            pooled,
            bootstrap,
            artifact,
        )
    for manifest_path in (
        bootstrap / "manifest.json",
        tmp_path / "bootstrap-transport" / "manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["input"]["sha256"] = _sha256(contributions_path)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ArtifactValidationError, match="event-hit hierarchy"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_false_rate_upper_limit_not_supported_by_counts(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    operating_path = pooled / "pooled-heldout-operating-metrics.csv"
    operating = pd.read_csv(operating_path)
    row = (
        (operating["method"] == "brace_bayesian")
        & np.isclose(operating["required_lead_s"], 1.0)
        & np.isclose(operating["false_budget_per_hour"], 2.0)
    )
    operating.loc[row, "false_proposals_per_hour_upper_95"] += 0.25
    operating.to_csv(operating_path, index=False, lineterminator="\n")
    pooled_manifest_path = pooled / "manifest.json"
    pooled_manifest = json.loads(pooled_manifest_path.read_text(encoding="utf-8"))
    pooled_manifest["artifacts"][str(operating_path.resolve())] = _sha256(operating_path)
    pooled_manifest_path.write_text(
        json.dumps(pooled_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    for manifest_path in (
        bootstrap / "manifest.json",
        tmp_path / "bootstrap-transport" / "manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["experiment_manifest"]["sha256"] = _sha256(pooled_manifest_path)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="Poisson upper limit"):
        _compile(tmp_path, pooled, bootstrap)


@pytest.mark.parametrize(
    ("method", "column", "value", "message"),
    [
        ("brace_bayesian", "circuit", "Bahrain", "probability metric scope"),
        ("brace_bayesian", "n", 1000.5, "probability metric counts"),
        ("posterior_mean_twin", "event_count", 1001, "probability metric counts"),
        ("brace_bayesian", "brier_score", np.nan, "probability metric values"),
        (
            "posterior_mean_twin",
            "average_precision_stepwise",
            1.25,
            "probability metric values",
        ),
        ("brace_bayesian", "brier_skill_score", 0.0, "Brier skill identity"),
    ],
)
def test_compiler_rejects_impossible_headline_probability_metrics(
    tmp_path: Path,
    method: str,
    column: str,
    value: object,
    message: str,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    probability_path = pooled / "pooled-heldout-probability-metrics.csv"
    probability = pd.read_csv(probability_path)
    row = (probability["method"] == method) & np.isclose(
        probability["horizon_s"], 1.5
    )
    if column in {"n", "event_count"}:
        probability[column] = probability[column].astype(float)
    probability.loc[row, column] = value
    probability.to_csv(probability_path, index=False, lineterminator="\n")
    _refresh_pooled_artifact_provenance(
        tmp_path,
        pooled,
        bootstrap,
        probability_path,
    )

    with pytest.raises(ArtifactValidationError, match=message):
        _compile(tmp_path, pooled, bootstrap)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("event_count", 519, "probability/reliability support"),
        ("ece_10_equal_width", 0.03, "probability ECE disagrees"),
    ],
)
def test_compiler_rejects_probability_metrics_inconsistent_with_reliability_bins(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    probability_path = pooled / "pooled-heldout-probability-metrics.csv"
    probability = pd.read_csv(probability_path)
    row = (probability["method"] == "brace_bayesian") & np.isclose(
        probability["horizon_s"], 1.5
    )
    probability.loc[row, column] = value
    probability.to_csv(probability_path, index=False, lineterminator="\n")
    _refresh_pooled_artifact_provenance(
        tmp_path,
        pooled,
        bootstrap,
        probability_path,
    )

    with pytest.raises(ArtifactValidationError, match=message):
        _compile(tmp_path, pooled, bootstrap)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("brace_method", "wrong_method", "estimand or no-refit"),
        ("models_refit", True, "estimand or no-refit"),
        ("l_star_brace_s", 0.75, "leaves the prespecified grid"),
        ("delta_l_star_s", 0.0, "delta identity"),
    ],
)
def test_compiler_rejects_invalid_synthetic_delay_summary_contract(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    delay_path = pooled / "pooled-heldout-synthetic-delay-l-star.csv"
    delay_table = pd.read_csv(delay_path)
    delay_table.loc[delay_table["synthetic_delay_ms"] == 160, column] = value
    delay_table.to_csv(delay_path, index=False, lineterminator="\n")
    _refresh_pooled_artifact_provenance(tmp_path, pooled, bootstrap, delay_path)

    with pytest.raises(ArtifactValidationError, match=message):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_synthetic_delay_summary_not_supported_by_ledger(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    delay_contributions_path = (
        pooled / "pooled-heldout-synthetic-delay-car-contributions.parquet"
    )
    contributions = pd.read_parquet(delay_contributions_path)
    mask = (
        (contributions["synthetic_delay_ms"] == 160)
        & (contributions["method"] == "brace_bayesian")
    )
    contributions.loc[mask, "localized_event_hits"] = 0
    contributions.to_parquet(delay_contributions_path, index=False)
    _refresh_pooled_artifact_provenance(
        tmp_path,
        pooled,
        bootstrap,
        delay_contributions_path,
    )

    with pytest.raises(ArtifactValidationError, match="disagrees with contribution ledger"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_fractional_synthetic_delay_counts(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    delay_contributions_path = (
        pooled / "pooled-heldout-synthetic-delay-car-contributions.parquet"
    )
    contributions = pd.read_parquet(delay_contributions_path)
    contributions["localized_event_hits"] = contributions[
        "localized_event_hits"
    ].astype(float)
    contributions.loc[0, "localized_event_hits"] = 0.5
    contributions.to_parquet(delay_contributions_path, index=False)
    _refresh_pooled_artifact_provenance(
        tmp_path,
        pooled,
        bootstrap,
        delay_contributions_path,
    )

    with pytest.raises(ArtifactValidationError, match="synthetic-delay contribution counts"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_self_consistent_zero_delay_result_that_differs_from_primary(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    delay_contributions_path = (
        pooled / "pooled-heldout-synthetic-delay-car-contributions.parquet"
    )
    contributions = pd.read_parquet(delay_contributions_path)
    mask = (
        (contributions["synthetic_delay_ms"] == 0)
        & (contributions["method"] == "brace_bayesian")
        & (contributions["required_lead_s"] >= 1.0)
    )
    contributions.loc[mask, "localized_event_hits"] = 0
    contributions.to_parquet(delay_contributions_path, index=False)
    delay_path = pooled / "pooled-heldout-synthetic-delay-l-star.csv"
    delay = pd.read_csv(delay_path)
    row = delay["synthetic_delay_ms"] == 0
    delay.loc[row, "l_star_brace_s"] = 0.5
    delay.loc[row, "delta_l_star_s"] = 0.0
    delay.to_csv(delay_path, index=False, lineterminator="\n")
    for artifact in (delay_contributions_path, delay_path):
        _refresh_pooled_artifact_provenance(
            tmp_path,
            pooled,
            bootstrap,
            artifact,
        )

    with pytest.raises(ArtifactValidationError, match="zero-delay L-star differs"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_abstract_missing_essential_result_token(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    abstract_shell = tmp_path / "abstract-missing-primary-delta.md"
    source = (ROOT / "paper" / "ssac27-abstract-results-shell.md").read_text(
        encoding="utf-8"
    )
    abstract_shell.write_text(
        source.replace("{{PRIMARY_DELTA_LSTAR_S}}", "0.50"),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="abstract shell is missing"):
        compile_submission(
            pooled_dir=pooled,
            bootstrap_dir=bootstrap,
            transport_bootstrap_dir=tmp_path / "bootstrap-transport",
            manuscript_shell=ROOT / "paper" / "manuscript-results-shell.md",
            abstract_shell=abstract_shell,
            output_root=tmp_path / "submission",
        )


@pytest.mark.parametrize(
    "zero_methods",
    [
        ("brace_bayesian",),
        ("posterior_mean_twin",),
        ("brace_bayesian", "posterior_mean_twin"),
    ],
)
def test_zero_lstar_uses_shortest_row_only_as_nonqualifying_description(
    tmp_path: Path,
    zero_methods: tuple[str, ...],
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    contributions_path = pooled / "pooled-heldout-car-contributions.parquet"
    contributions = pd.read_parquet(contributions_path)
    contribution_mask = contributions["method"].isin(zero_methods) & np.isclose(
        contributions["false_budget_per_hour"], 2.0
    )
    contribution_count_columns = [
        "localized_event_hits",
        "event_hits",
        "correct_side_event_hits",
        "within_segment_tolerance_event_hits",
        "exact_bin_event_hits",
    ]
    contributions.loc[contribution_mask, contribution_count_columns] = 0
    contributions.to_parquet(contributions_path, index=False)
    operating_path = pooled / "pooled-heldout-operating-metrics.csv"
    operating = pd.read_csv(operating_path)
    mask = operating["method"].isin(zero_methods) & np.isclose(
        operating["false_budget_per_hour"], 2.0
    )
    operating_count_columns = [
        "localized_event_hits",
        "event_hits",
        "correct_side_event_hits",
        "within_segment_tolerance_event_hits",
        "exact_bin_event_hits",
    ]
    operating_recall_columns = [
        "localized_event_recall",
        "event_recall",
        "correct_side_event_recall",
        "within_segment_tolerance_event_recall",
        "exact_bin_event_recall",
    ]
    operating.loc[mask, operating_count_columns] = 0
    operating.loc[mask, operating_recall_columns] = 0.0
    operating.to_csv(operating_path, index=False, lineterminator="\n")
    sensitivity_path = pooled / "pooled-heldout-l-star-sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    primary = sensitivity["analysis"] == "primary_unresolved_censored"
    brace_point = 0.0 if "brace_bayesian" in zero_methods else 1.0
    twin_point = 0.0 if "posterior_mean_twin" in zero_methods else 0.5
    delta_point = brace_point - twin_point
    sensitivity.loc[primary, "l_star_brace_s"] = brace_point
    sensitivity.loc[primary, "l_star_comparator_s"] = twin_point
    sensitivity.loc[primary, "delta_l_star_s"] = delta_point
    least_favorable = sensitivity["analysis"] == (
        "least_favorable_unresolved_counted_as_false"
    )
    least_favorable_brace = 0.0 if "brace_bayesian" in zero_methods else 0.5
    least_favorable_twin = 0.0 if "posterior_mean_twin" in zero_methods else 0.5
    least_favorable_delta = least_favorable_brace - least_favorable_twin
    sensitivity.loc[least_favorable, "l_star_brace_s"] = least_favorable_brace
    sensitivity.loc[least_favorable, "l_star_comparator_s"] = least_favorable_twin
    sensitivity.loc[least_favorable, "delta_l_star_s"] = least_favorable_delta
    sensitivity.to_csv(sensitivity_path, index=False, lineterminator="\n")
    delay_contributions_path = (
        pooled / "pooled-heldout-synthetic-delay-car-contributions.parquet"
    )
    delay_contributions = pd.read_parquet(delay_contributions_path)
    delay_mask = (delay_contributions["synthetic_delay_ms"] == 0) & (
        delay_contributions["method"].isin(zero_methods)
    )
    delay_contributions.loc[delay_mask, "localized_event_hits"] = 0
    delay_contributions.to_parquet(delay_contributions_path, index=False)
    delay_path = pooled / "pooled-heldout-synthetic-delay-l-star.csv"
    delay = pd.read_csv(delay_path)
    delay_zero = delay["synthetic_delay_ms"] == 0
    delay.loc[delay_zero, "l_star_brace_s"] = brace_point
    delay.loc[delay_zero, "l_star_comparator_s"] = twin_point
    delay.loc[delay_zero, "delta_l_star_s"] = delta_point
    delay.to_csv(delay_path, index=False, lineterminator="\n")
    pooled_manifest_path = pooled / "manifest.json"
    pooled_manifest = json.loads(pooled_manifest_path.read_text(encoding="utf-8"))
    for record in pooled_manifest["fold_manifests"]:
        fold_manifest_path = Path(record["path"])
        _assert_within_tmp(tmp_path, fold_manifest_path)
        fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        metric_path = next(
            Path(path)
            for path in fold_manifest["artifacts"]
            if Path(path).name == "heldout-operating-metrics.csv"
        )
        _assert_within_tmp(tmp_path, metric_path)
        metrics = pd.read_csv(metric_path)
        fold_mask = metrics["method"].isin(zero_methods) & np.isclose(
            metrics["false_budget_per_hour"], 2.0
        )
        metrics.loc[fold_mask, "localized_event_recall"] = 0.0
        metrics.to_csv(metric_path, index=False, lineterminator="\n")
        fold_manifest["artifacts"][str(metric_path.resolve())] = _sha256(metric_path)
        fold_manifest_path.write_text(
            json.dumps(fold_manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        record["sha256"] = _sha256(fold_manifest_path)
    for path in (
        contributions_path,
        operating_path,
        sensitivity_path,
        delay_contributions_path,
        delay_path,
    ):
        pooled_manifest["artifacts"][str(path.resolve())] = _sha256(path)
    pooled_manifest_path.write_text(
        json.dumps(pooled_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_path = bootstrap / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["l_star_brace"] = brace_point
    summary["l_star_brace_percentile_95"] = [brace_point, brace_point]
    summary["l_star_comparator"] = twin_point
    summary["l_star_comparator_percentile_95"] = [twin_point, twin_point]
    summary["delta_l_star"] = delta_point
    summary["delta_l_star_percentile_95"] = [delta_point, delta_point]
    least_favorable_summary = summary[
        "least_favorable_unresolved_counted_as_false"
    ]
    least_favorable_summary["l_star_brace"] = least_favorable_brace
    least_favorable_summary["l_star_brace_percentile_95"] = [
        least_favorable_brace,
        least_favorable_brace,
    ]
    least_favorable_summary["l_star_comparator"] = least_favorable_twin
    least_favorable_summary["l_star_comparator_percentile_95"] = [
        least_favorable_twin,
        least_favorable_twin,
    ]
    least_favorable_summary["delta_l_star"] = least_favorable_delta
    least_favorable_summary["delta_l_star_percentile_95"] = [
        least_favorable_delta,
        least_favorable_delta,
    ]
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    brace_path = bootstrap / "brace-l-star-draws.npy"
    comparator_path = bootstrap / "comparator-l-star-draws.npy"
    delta_path = bootstrap / "delta-l-star-draws.npy"
    np.save(brace_path, np.full(10_000, brace_point), allow_pickle=False)
    np.save(comparator_path, np.full(10_000, twin_point), allow_pickle=False)
    np.save(delta_path, np.full(10_000, delta_point), allow_pickle=False)
    least_favorable_brace_path = bootstrap / "least-favorable-brace-l-star-draws.npy"
    least_favorable_comparator_path = (
        bootstrap / "least-favorable-comparator-l-star-draws.npy"
    )
    least_favorable_delta_path = bootstrap / "least-favorable-delta-l-star-draws.npy"
    np.save(
        least_favorable_brace_path,
        np.full(10_000, least_favorable_brace),
        allow_pickle=False,
    )
    np.save(
        least_favorable_comparator_path,
        np.full(10_000, least_favorable_twin),
        allow_pickle=False,
    )
    np.save(
        least_favorable_delta_path,
        np.full(10_000, least_favorable_delta),
        allow_pickle=False,
    )
    bootstrap_manifest_path = bootstrap / "manifest.json"
    bootstrap_manifest = json.loads(bootstrap_manifest_path.read_text(encoding="utf-8"))
    bootstrap_manifest["experiment_manifest"]["sha256"] = _sha256(pooled_manifest_path)
    bootstrap_manifest["input"]["sha256"] = _sha256(contributions_path)
    for path in (
        summary_path,
        brace_path,
        comparator_path,
        delta_path,
        least_favorable_brace_path,
        least_favorable_comparator_path,
        least_favorable_delta_path,
    ):
        bootstrap_manifest["artifacts"][str(path.resolve())] = _sha256(path)
    bootstrap_manifest_path.write_text(
        json.dumps(bootstrap_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    transport_manifest_path = tmp_path / "bootstrap-transport" / "manifest.json"
    transport_manifest = json.loads(transport_manifest_path.read_text(encoding="utf-8"))
    transport_manifest["experiment_manifest"]["sha256"] = _sha256(pooled_manifest_path)
    transport_manifest["input"]["sha256"] = _sha256(contributions_path)
    transport_manifest_path.write_text(
        json.dumps(transport_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    evidence = MODULE._load_evidence(
        pooled,
        bootstrap,
        tmp_path / "bootstrap-transport",
    )
    tokens, _, _, _ = MODULE._derive_tokens(evidence)
    manuscript = MODULE._replace_tokens(
        (ROOT / "paper" / "manuscript-results-shell.md").read_text(encoding="utf-8"),
        tokens,
        "zero-L-star manuscript fixture",
    )
    for method in zero_methods:
        label = "BRACE" if method == "brace_bayesian" else "the twin"
        assert f"{label} had no qualifying prespecified lead ($L^*=0.00$ s)" in manuscript
    assert "0.25 s row was explicitly nonqualifying" in manuscript
    assert "shortest 0.25 s nonqualifying fallback" in manuscript
    assert "qualifying BRACE and twin operating rows" not in manuscript


def test_compiler_requires_two_stage_transport_bundle(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    with pytest.raises(ArtifactValidationError, match="transport-bootstrap-dir"):
        compile_submission(
            pooled_dir=pooled,
            bootstrap_dir=bootstrap,
            manuscript_shell=ROOT / "paper" / "manuscript-results-shell.md",
            abstract_shell=ROOT / "paper" / "ssac27-abstract-results-shell.md",
            output_root=tmp_path / "submission",
        )


def test_compiler_rejects_wrong_bootstrap_producer_identity(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    manifest_path = bootstrap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code_content_hash"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="producer code"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_fold_metric_circuit_contamination(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    pooled_manifest_path = pooled / "manifest.json"
    pooled_manifest = json.loads(pooled_manifest_path.read_text(encoding="utf-8"))
    record = pooled_manifest["fold_manifests"][0]
    fold_manifest_path = Path(record["path"])
    _assert_within_tmp(tmp_path, fold_manifest_path)
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    metric_path = next(
        Path(path)
        for path in fold_manifest["artifacts"]
        if Path(path).name == "heldout-operating-metrics.csv"
    )
    _assert_within_tmp(tmp_path, metric_path)
    metrics = pd.read_csv(metric_path)
    metrics.loc[0, "circuit"] = "Monza"
    metrics.to_csv(metric_path, index=False, lineterminator="\n")
    fold_manifest["artifacts"][str(metric_path.resolve())] = _sha256(metric_path)
    fold_manifest_path.write_text(
        json.dumps(fold_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    record["sha256"] = _sha256(fold_manifest_path)
    pooled_manifest_path.write_text(
        json.dumps(pooled_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ArtifactValidationError, match="another circuit"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_rejects_duplicate_fold_lead_grid(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    pooled_manifest_path = pooled / "manifest.json"
    pooled_manifest = json.loads(pooled_manifest_path.read_text(encoding="utf-8"))
    record = pooled_manifest["fold_manifests"][0]
    fold_manifest_path = Path(record["path"])
    _assert_within_tmp(tmp_path, fold_manifest_path)
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    metric_path = next(
        Path(path)
        for path in fold_manifest["artifacts"]
        if Path(path).name == "heldout-operating-metrics.csv"
    )
    _assert_within_tmp(tmp_path, metric_path)
    metrics = pd.read_csv(metric_path)
    row = (metrics["method"] == "brace_bayesian") & np.isclose(metrics["required_lead_s"], 0.25)
    metrics.loc[row, "required_lead_s"] = 0.5
    metrics.to_csv(metric_path, index=False, lineterminator="\n")
    fold_manifest["artifacts"][str(metric_path.resolve())] = _sha256(metric_path)
    fold_manifest_path.write_text(
        json.dumps(fold_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    record["sha256"] = _sha256(fold_manifest_path)
    pooled_manifest_path.write_text(
        json.dumps(pooled_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    for manifest_path in (
        bootstrap / "manifest.json",
        tmp_path / "bootstrap-transport" / "manifest.json",
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["experiment_manifest"]["sha256"] = _sha256(pooled_manifest_path)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="fold grid is incomplete"):
        _compile(tmp_path, pooled, bootstrap)


def test_reliability_figure_accepts_off_center_percentile_interval(tmp_path: Path) -> None:
    warning = pd.DataFrame(
        [
            {"method": method, "budget": budget, "l_star_s": lead}
            for method, lead in (("brace_bayesian", 1.0), ("posterior_mean_twin", 0.5))
            for budget in BUDGETS
        ]
    )
    reliability = pd.DataFrame(
        [
            {
                "method": method,
                "horizon_s": 1.5,
                "bin_index": 0,
                "count": 10,
                "mean_probability": 0.05,
                "observed_frequency": 0.10,
                "observed_frequency_lower_95": 0.11,
                "observed_frequency_upper_95": 0.12,
            }
            for method in ("brace_bayesian", "posterior_mean_twin")
        ]
    )

    MODULE._generate_figures(tmp_path, warning, reliability)

    assert (
        tmp_path / "analysis-output" / "figures" / "figure-02-reliability.pdf"
    ).stat().st_size > 0


def test_compiler_rejects_reliability_interval_not_recomputed_from_clusters(
    tmp_path: Path,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    interval_path = bootstrap / "reliability-bin-intervals.csv"
    intervals = pd.read_csv(interval_path)
    intervals.loc[0, "observed_frequency_lower_95"] += 0.005
    intervals.to_csv(interval_path, index=False, lineterminator="\n")
    manifest_path = bootstrap / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][str(interval_path.resolve())] = _sha256(interval_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="reliability bootstrap recomputation"):
        _compile(tmp_path, pooled, bootstrap)


def test_reliability_recomputation_source_loads_in_isolated_python(
    tmp_path: Path,
) -> None:
    script_path = ROOT / "scripts" / "build_submission_results.py"
    code = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("isolated_compiler", {str(script_path)!r})
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert callable(module._load_reliability_bootstrap_function())
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_publication_lock_must_match_every_direct_requirement_pin() -> None:
    with pytest.raises(ArtifactValidationError, match="does not match direct requirement"):
        MODULE._validate_lock_matches_requirements(
            {"pubfig": "0.3.0", "pyarrow": "25.0.1"},
            "pubfig==0.3.0\npyarrow==24.0.0\n",
        )


def test_compiler_rejects_monotonicity_rate_arithmetic_mismatch(tmp_path: Path) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    monotonicity_path = pooled / "pooled-heldout-horizon-monotonicity.csv"
    monotonicity = pd.read_csv(monotonicity_path)
    row = monotonicity["method"] == "brace_bayesian"
    monotonicity.loc[row, "violating_row_rate"] = 0.123
    monotonicity.to_csv(monotonicity_path, index=False, lineterminator="\n")
    _refresh_pooled_artifact_provenance(
        tmp_path,
        pooled,
        bootstrap,
        monotonicity_path,
    )

    with pytest.raises(ArtifactValidationError, match="monotonicity count/rate arithmetic"):
        _compile(tmp_path, pooled, bootstrap)


def test_compiler_refuses_overwrite_of_existing_publication(tmp_path: Path) -> None:
    output_root = _compile_actual(tmp_path)

    with pytest.raises(ArtifactValidationError, match="fresh dedicated directory"):
        _compile_actual(tmp_path)
    assert (output_root / "paper" / "manuscript-final.md").is_file()


def test_compiler_rolls_back_failed_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pooled, bootstrap = _make_inputs(tmp_path)
    output_root = tmp_path / "submission"

    def fail_figures(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic figure failure")

    monkeypatch.setattr(MODULE, "_generate_figures", fail_figures)
    with pytest.raises(RuntimeError, match="synthetic figure failure"):
        _compile(tmp_path, pooled, bootstrap)
    assert not output_root.exists()
    assert not list(tmp_path.glob(".submission.staging-*"))


def test_conservative_word_counter_splits_hyphenated_compounds() -> None:
    assert MODULE._conservative_word_count("state-of-the-art") == 4


@pytest.mark.parametrize(
    ("delta", "interval"),
    [
        (-0.25, (0.25, 0.50)),
        (0.25, (-0.50, -0.25)),
    ],
)
def test_directional_claim_requires_point_and_interval_sign_agreement(
    delta: float,
    interval: tuple[float, float],
) -> None:
    wording = MODULE._conditional_text(delta, interval)
    combined = " ".join(wording.values()).lower()

    assert "discord" in combined
    assert "no directional claim" in combined
    assert "brace increased" not in combined
    assert "brace reduced" not in combined
    assert "supports a positive" not in combined


def test_exact_zero_wording_requires_authenticated_silence_and_all_zero_draws() -> None:
    wording = MODULE._conditional_text(
        0.0,
        (0.0, 0.0),
        all_delta_draws_zero=True,
        both_policies_silent=True,
    )
    combined = " ".join(wording.values()).lower()

    assert "both selected policies were silent" in combined
    assert "all 10,000 paired bootstrap contrasts were zero" in combined
    assert "does not establish equivalence" in combined


def test_equal_nonzero_policies_do_not_receive_silence_wording() -> None:
    wording = MODULE._conditional_text(
        0.0,
        (0.0, 0.0),
        all_delta_draws_zero=True,
        both_policies_silent=False,
    )
    combined = " ".join(wording.values()).lower()

    assert "both selected policies were silent" not in combined
    assert "included zero" in combined


def test_zero_percentile_endpoints_with_nonzero_tail_draws_are_not_called_all_zero() -> None:
    wording = MODULE._conditional_text(
        0.0,
        (0.0, 0.0),
        all_delta_draws_zero=False,
        both_policies_silent=True,
    )
    combined = " ".join(wording.values()).lower()

    assert "all 10,000 paired bootstrap contrasts were zero" not in combined
    assert "included zero" in combined


def test_primary_proposal_count_includes_late_matched_proposal() -> None:
    labels = pd.DataFrame(
        [
            {
                "method": "brace_bayesian",
                "false_budget_per_hour": 2.0,
                "localized_hit": False,
                "false_proposal": False,
                "unresolved": False,
                "lead_seconds": 0.10,
            }
        ]
    )

    assert MODULE._primary_proposal_count(labels, "fixture proposal labels") == 1
