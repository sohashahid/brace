#!/usr/bin/env python3
# ruff: noqa: E501
"""Compile authenticated BRACE experiment artifacts into submission-ready results.

The compiler consumes only artifacts authenticated by the pooled held-out and
bootstrap manifests.  It validates every consumed byte before deriving text,
tables, or figures, and it refuses to publish a document with an unresolved
double-braced result token.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

BRACE_METHOD = "brace_bayesian"
TWIN_METHOD = "posterior_mean_twin"
REGISTERED_METHODS = (
    BRACE_METHOD,
    TWIN_METHOD,
    "constant_velocity",
    "constant_turn_rate",
    "side_hazard",
    "calibration_mse_tuned_dynamics",
)
METHOD_LABELS = {
    BRACE_METHOD: "BRACE",
    TWIN_METHOD: "Posterior-mean twin",
    "constant_velocity": "Constant velocity",
    "constant_turn_rate": "Constant turn rate",
    "side_hazard": "Side hazard",
    "calibration_mse_tuned_dynamics": "Calibration-tuned dynamics",
}
LEAD_GRID = (0.25, 0.50, 1.00, 1.50)
PRIMARY_BUDGET = 2.0
MINIMUM_RECALL = 0.50
PRIMARY_HORIZON = 1.50
REGISTERED_CIRCUITS = ("Bahrain", "Britain", "Jeddah", "Monza")
BUDGET_GRID = (0.5, 1.0, 2.0, 5.0, 10.0)
REQUIRED_PUBFIG_VERSION = "0.3.0"
REQUIRED_PYARROW_VERSION = "25.0.1"
REQUIRED_PYPDF_VERSION = "6.19.0"
PUBLICATION_SOURCE_DATE_EPOCH = "0"
MAX_CONSERVATIVE_ABSTRACT_WORDS = 450
EXPECTED_CIRCUIT_COUNTS = {
    "Bahrain": {
        "cars": 20,
        "qualified_excursions": 36,
        "candidate_excursions": 226,
        "resampled_frames": 185_174,
        "prebuffer_exposure_seconds": 9224.805993299931,
        "buffered_exposure_seconds": 9154.38796120882,
    },
    "Britain": {
        "cars": 11,
        "qualified_excursions": 113,
        "candidate_excursions": 325,
        "resampled_frames": 68_283,
        "prebuffer_exposure_seconds": 3411.587967460975,
        "buffered_exposure_seconds": 3381.621959298849,
    },
    "Jeddah": {
        "cars": 20,
        "qualified_excursions": 106,
        "candidate_excursions": 961,
        "resampled_frames": 181_524,
        "prebuffer_exposure_seconds": 9032.04595878534,
        "buffered_exposure_seconds": 8960.021955937147,
    },
    "Monza": {
        "cars": 7,
        "qualified_excursions": 39,
        "candidate_excursions": 104,
        "resampled_frames": 25_288,
        "prebuffer_exposure_seconds": 1263.6139990882948,
        "buffered_exposure_seconds": 1245.3660022616386,
    },
}
CALIBRATION_CLIFF_SOURCE_SHA256 = {
    "scripts/build_calibration_operating_cliff_figure.py": (
        "7f1dd948a7652fc34ccbb83c2cfa7a997631b5ef3018b1362265305ddfe7137b"
    ),
    "paper/figure-source/figure-03-calibration-operating-cliff.csv": (
        "4690e562ec0b5b35d4554b9f8cc21a8359642dd91113036f7c27fa6b2b209bc2"
    ),
    "paper/figures/figure-03-calibration-operating-cliff-provenance.md": (
        "fe1c8cc135702f400d3287842b4336e601b215a5b40c861480b5aee6a831a12f"
    ),
    "paper/figures/figure-03-calibration-operating-cliff.pdf": (
        "531b15d4df84bfff7253d75c989e653bbbf4103ff6e62d06a8e6ac7c5c584fb7"
    ),
    "paper/figures/figure-03-calibration-operating-cliff.png": (
        "c68dc4826e4ab7804b13793c234107167b5a9e76d4319e79de068957166dc568"
    ),
}

EXPECTED_TOKENS = {
    "PAPER_TITLE",
    "PRIMARY_LSTAR_BRACE_S",
    "PRIMARY_LSTAR_TWIN_S",
    "PRIMARY_DELTA_LSTAR_S",
    "PRIMARY_DELTA_LSTAR_CI_LOW_S",
    "PRIMARY_DELTA_LSTAR_CI_HIGH_S",
    "PRIMARY_EVIDENCE_CONCLUSION",
    "PRIMARY_POLICY_RESULT",
    "PRIMARY_BOOTSTRAP_RESULT",
    "PROBABILITY_COMPARISON_RESULT",
    "ALL_METHOD_PROBABILITY_RESULT",
    "METRIC_TO_DECISION_RESULT",
    "COHORT_FLOW_RESULT",
    "PRIMARY_GRID_ESTIMABILITY_STATUS",
    "ESTIMABILITY_DETAIL",
    "FIGURE_WARNING_FRONTIER_MARKDOWN",
    "PRIMARY_RESULT_INTERPRETATION",
    "PRIMARY_OPERATING_POINT_NARRATIVE",
    "TABLE_PRIMARY_FRONTIER_MARKDOWN",
    "OPERATING_FRONTIER_SUMMARY",
    "FIGURE_RELIABILITY_MARKDOWN",
    "BRACE_BRIER_1P50",
    "TWIN_BRIER_1P50",
    "BRACE_BRIER_SKILL_1P50",
    "BRACE_LOG_SCORE_1P50",
    "BRACE_AP_1P50",
    "BRACE_CAL_INTERCEPT_1P50",
    "BRACE_CAL_SLOPE_1P50",
    "BRACE_ECE_1P50",
    "TABLE_PROBABILITY_METRICS_MARKDOWN",
    "BRACE_MONOTONICITY_SCORE_ROWS",
    "BRACE_MONOTONICITY_VIOLATION_COUNT",
    "BRACE_MONOTONICITY_VIOLATION_RATE",
    "LOCALIZATION_COMPONENT_SUMMARY",
    "TABLE_CIRCUIT_RESULTS_MARKDOWN",
    "CIRCUIT_HETEROGENEITY_SUMMARY",
    "SECONDARY_BASELINE_SUMMARY",
    "UNRESOLVED_PROPOSAL_SUMMARY",
    "LF_LSTAR_BRACE_S",
    "LF_LSTAR_TWIN_S",
    "LF_DELTA_LSTAR_S",
    "LF_DELTA_LSTAR_CI_LOW_S",
    "LF_DELTA_LSTAR_CI_HIGH_S",
    "TABLE_LEAST_FAVORABLE_MARKDOWN",
    "TABLE_DELAY_RESULTS_MARKDOWN",
    "DELAY_160_DELTA_LSTAR_S",
    "DELAY_SENSITIVITY_SUMMARY",
    "DELAY_RESULT_INTERPRETATION",
    "TOTAL_EXPERIMENT_RUNTIME_HOURS",
    "COMPUTE_ENVIRONMENT",
    "INFERENCE_THROUGHPUT_SUMMARY",
    "POOLED_MANIFEST_SHA256",
    "BOOTSTRAP_MANIFEST_SHA256",
    "PRIMARY_DISCUSSION_PARAGRAPH",
    "BOOTSTRAP_VARIATION_LIMIT",
    "CONCLUSION_RESULT_SENTENCE",
}
ABSTRACT_REQUIRED_TOKENS = {
    "PAPER_TITLE",
    "PRIMARY_LSTAR_BRACE_S",
    "PRIMARY_LSTAR_TWIN_S",
    "PRIMARY_DELTA_LSTAR_S",
    "PRIMARY_DELTA_LSTAR_CI_LOW_S",
    "PRIMARY_DELTA_LSTAR_CI_HIGH_S",
    "PROBABILITY_COMPARISON_RESULT",
    "ALL_METHOD_PROBABILITY_RESULT",
    "PRIMARY_POLICY_RESULT",
    "PRIMARY_BOOTSTRAP_RESULT",
    "METRIC_TO_DECISION_RESULT",
    "PRIMARY_EVIDENCE_CONCLUSION",
}
TOKEN_PATTERN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_RELIABILITY_RECOMPUTATION_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_RELIABILITY_BOOTSTRAP_FUNCTION: Any | None = None


class ArtifactValidationError(RuntimeError):
    """Raised when provenance or the result-token contract is invalid."""


@dataclass(frozen=True)
class ManifestBundle:
    directory: Path
    manifest_path: Path
    manifest_hash: str
    payload: Mapping[str, Any]
    artifacts: Mapping[Path, str]


@dataclass(frozen=True)
class FoldEvidence:
    circuit: str
    metrics: pd.DataFrame
    runtime_events: tuple[Mapping[str, Any], ...]
    environment: Mapping[str, Any]
    primary_proposal_count: int
    examined_budget_proposal_count: int
    calibration_thresholds: pd.DataFrame


@dataclass(frozen=True)
class Evidence:
    pooled: ManifestBundle
    bootstrap: ManifestBundle
    transport_bootstrap: ManifestBundle | None
    build_manifest: Mapping[str, Any]
    operating: pd.DataFrame
    probability: pd.DataFrame
    reliability: pd.DataFrame
    reliability_contributions: pd.DataFrame
    monotonicity: pd.DataFrame
    l_star_sensitivity: pd.DataFrame
    delay_l_star: pd.DataFrame
    contributions: pd.DataFrame
    delay_contributions: pd.DataFrame
    bootstrap_summary: Mapping[str, Any]
    transport_summary: Mapping[str, Any] | None
    reliability_intervals: pd.DataFrame
    folds: tuple[FoldEvidence, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_project_path(path: Path, *, fallback: str) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        relative = path.resolve().relative_to(project_root)
    except ValueError:
        return fallback
    result = relative.as_posix()
    return result if result and result != "." else fallback


def _manifest_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _manifest_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _manifest_strings(item)


def _validate_portable_publication_manifest(payload: Mapping[str, Any]) -> None:
    for text in _manifest_strings(payload):
        is_tilde_path = re.match(r"^~[^/\\]*(?:[/\\]|$)", text) is not None
        is_local_file_uri = text.lower().startswith("file:")
        if (
            Path(text).is_absolute()
            or PureWindowsPath(text).is_absolute()
            or is_tilde_path
            or is_local_file_uri
        ):
            raise ArtifactValidationError(
                f"publication manifest contains a nonportable local path: {text}"
            )
        if "/Users/" in text or "\\Users\\" in text:
            raise ArtifactValidationError(
                "publication manifest contains a nonportable local user-directory path"
            )


def _validate_portable_binary(path: Path) -> None:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(f"cannot inspect binary publication artifact: {path}") from exc
    normalized = payload.replace(b"\\", b"/").replace(b"\x00", b"")
    forbidden_markers = (
        b"/PTEX.FileName",
        b"/Users/",
        b"/home/",
        b"/private/var/",
        b"/var/folders/",
        b"/tmp/",
        b"file://",
    )
    for marker in forbidden_markers:
        if marker in normalized:
            raise ArtifactValidationError(
                f"binary publication artifact contains a nonportable local path marker "
                f"{marker.decode('ascii')}: {path}"
            )


def _normalize_pdf_trailer_id(path: Path, identifier_hex: str) -> None:
    if re.fullmatch(r"[0-9a-f]{32}", identifier_hex) is None:
        raise ArtifactValidationError("normalized PDF trailer ID must be 16 lowercase hex bytes")
    temporary_path = path.with_name(f".{path.name}.id-normalized")
    try:
        from pypdf import PdfWriter
        from pypdf.generic import ArrayObject, ByteStringObject

        writer = PdfWriter(clone_from=path, keep_initial_header=True)
        identifier = bytes.fromhex(identifier_hex)
        writer._ID = ArrayObject(  # noqa: SLF001 - pinned pypdf has no public trailer-ID setter
            [ByteStringObject(identifier), ByteStringObject(identifier)]
        )
        try:
            with temporary_path.open("wb") as handle:
                writer.write(handle)
        finally:
            writer.close()
        temporary_path.replace(path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise ArtifactValidationError(f"cannot normalize PDF trailer ID: {path}") from exc


def _producer_source_tree_hash() -> str:
    module_dir = Path(__file__).resolve().parents[1] / "src" / "brace_f1"
    paths = sorted(
        module_dir.rglob("*.py"),
        key=lambda value: value.relative_to(module_dir).as_posix(),
    )
    if not paths:
        raise ArtifactValidationError(f"producer source tree is missing: {module_dir}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(module_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must be a JSON object: {path}")
    return value


def _validate_digest(path: Path, expected: object, label: str) -> str:
    if not path.is_file():
        raise ArtifactValidationError(f"missing {label}: {path}")
    expected_text = str(expected)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_text):
        raise ArtifactValidationError(f"invalid authenticated SHA-256 for {label}: {path}")
    actual = _sha256(path)
    if actual != expected_text:
        raise ArtifactValidationError(
            f"hash mismatch for {label}: expected {expected_text}, observed {actual}: {path}"
        )
    return actual


def _identity(value: object, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ArtifactValidationError(f"{label} must be a non-null lowercase SHA-256 identity")
    return text


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_manifest_bundle(directory: Path, *, label: str) -> ManifestBundle:
    resolved = Path(directory).resolve()
    manifest_path = resolved / "manifest.json"
    payload = _load_json(manifest_path, f"{label} manifest")
    registry = payload.get("artifacts")
    if not isinstance(registry, dict) or not registry:
        raise ArtifactValidationError(f"{label} manifest has no artifact registry")
    artifacts: dict[Path, str] = {}
    for raw_path, expected in registry.items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            raise ArtifactValidationError(f"{label} artifact path is not absolute: {path}")
        path = path.resolve()
        _validate_digest(path, expected, f"{label} artifact")
        artifacts[path] = str(expected)
    return ManifestBundle(
        resolved,
        manifest_path,
        _sha256(manifest_path),
        payload,
        artifacts,
    )


def _artifact(bundle: ManifestBundle, name: str) -> Path:
    matches = [path for path in bundle.artifacts if path.name == name]
    if len(matches) != 1:
        raise ArtifactValidationError(
            f"{bundle.manifest_path} must register exactly one {name}; found {len(matches)}"
        )
    return matches[0]


def _read_csv(bundle: ManifestBundle, name: str) -> pd.DataFrame:
    path = _artifact(bundle, name)
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        raise ArtifactValidationError(f"cannot read authenticated table: {path}") from exc
    if table.empty:
        raise ArtifactValidationError(f"authenticated table is empty: {path}")
    return table


def _read_parquet(bundle: ManifestBundle, name: str) -> pd.DataFrame:
    path = _artifact(bundle, name)
    try:
        table = pd.read_parquet(path)
    except Exception as exc:
        raise ArtifactValidationError(f"cannot read authenticated table: {path}") from exc
    if table.empty:
        raise ArtifactValidationError(f"authenticated table is empty: {path}")
    return table


def _require_columns(table: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns).difference(table.columns)
    if missing:
        raise ArtifactValidationError(f"{label} is missing columns: {sorted(missing)}")


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("event is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactValidationError(f"cannot read runtime JSONL: {path}") from exc
    if not rows:
        raise ArtifactValidationError(f"runtime JSONL is empty: {path}")
    return tuple(rows)


def _manifest_record_path(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        raise ArtifactValidationError(f"{label} path is not absolute: {path}")
    _validate_digest(path.resolve(), record.get("sha256"), label)
    return path.resolve()


def _primary_proposal_count(labels: pd.DataFrame, label: str) -> int:
    _require_columns(
        labels,
        {"method", "false_budget_per_hour"},
        label,
    )
    budgets = pd.to_numeric(labels["false_budget_per_hour"], errors="coerce")
    primary = labels.loc[
        labels["method"].astype(str).isin((BRACE_METHOD, TWIN_METHOD))
        & np.isclose(budgets, PRIMARY_BUDGET, atol=1e-12, rtol=0.0)
    ]
    return int(len(primary))


def _examined_budget_proposal_count(labels: pd.DataFrame, label: str) -> int:
    _require_columns(
        labels,
        {"method", "false_budget_per_hour"},
        label,
    )
    budgets = pd.to_numeric(labels["false_budget_per_hour"], errors="coerce")
    examined = labels.loc[
        labels["method"].astype(str).isin((BRACE_METHOD, TWIN_METHOD))
        & budgets.isin((2.0, 5.0, 10.0))
    ]
    return int(len(examined))


def _validate_pooled(bundle: ManifestBundle) -> tuple[FoldEvidence, ...]:
    payload = bundle.payload
    if payload.get("schema_version") != 1 or payload.get("status") != "pooled_heldout_complete":
        raise ArtifactValidationError(
            "pooled manifest is not a completed schema-v1 held-out bundle"
        )
    if payload.get("models_refit") is not False or payload.get("thresholds_refit") is not False:
        raise ArtifactValidationError("pooled aggregation reports refitting after unsealing")
    identity_keys = (
        "code_content_hash",
        "data_content_hash",
        "config_content_hash",
        "source_manifest_hash",
        "build_manifest_hash",
    )
    expected_identity = {key: _identity(payload.get(key), f"pooled {key}") for key in identity_keys}
    seal_record = payload.get("threshold_freeze_seal")
    if not isinstance(seal_record, dict):
        raise ArtifactValidationError("pooled manifest does not bind the threshold-freeze seal")
    seal_path = _manifest_record_path(seal_record, "threshold-freeze seal")
    seal = _load_json(seal_path, "threshold-freeze seal")
    expected_seal_status = "thresholds_frozen_before_heldout"
    if (
        seal.get("schema_version") != 1
        or seal_record.get("status") != expected_seal_status
        or seal.get("status") != expected_seal_status
    ):
        raise ArtifactValidationError("threshold-freeze seal status is invalid")
    for key, expected in expected_identity.items():
        if _identity(seal.get(key), f"threshold-freeze seal {key}") != expected:
            raise ArtifactValidationError(f"threshold-freeze seal has a different {key}")
    if tuple(seal.get("registered_circuits", ())) != REGISTERED_CIRCUITS:
        raise ArtifactValidationError("threshold-freeze seal has the wrong frozen circuits")
    if tuple(seal.get("registered_methods", ())) != REGISTERED_METHODS:
        raise ArtifactValidationError("threshold-freeze seal has the wrong frozen methods")
    raw_seal_folds = seal.get("fold_threshold_identities")
    if not isinstance(raw_seal_folds, list) or len(raw_seal_folds) != len(REGISTERED_CIRCUITS):
        raise ArtifactValidationError("threshold-freeze seal lacks four frozen fold identities")
    seal_folds: dict[str, Mapping[str, Any]] = {}
    for entry in raw_seal_folds:
        if not isinstance(entry, dict):
            raise ArtifactValidationError("threshold-freeze fold identity is not an object")
        circuit = str(entry.get("fold_test_circuit", ""))
        if circuit in seal_folds or circuit not in REGISTERED_CIRCUITS:
            raise ArtifactValidationError("threshold-freeze seal fold circuits are invalid")
        seal_folds[circuit] = entry
    if tuple(sorted(seal_folds)) != REGISTERED_CIRCUITS:
        raise ArtifactValidationError("threshold-freeze seal fold set is incomplete")
    fold_records = payload.get("fold_manifests")
    if not isinstance(fold_records, list) or len(fold_records) != 4:
        raise ArtifactValidationError(
            "pooled manifest must authenticate exactly four fold manifests"
        )
    folds: list[FoldEvidence] = []
    circuits: set[str] = set()
    for raw_record in fold_records:
        if not isinstance(raw_record, dict):
            raise ArtifactValidationError("fold-manifest record must be an object")
        manifest_path = _manifest_record_path(raw_record, "fold manifest")
        fold = _load_json(manifest_path, "fold manifest")
        circuit = str(raw_record.get("fold_test_circuit", ""))
        if not circuit or fold.get("fold_test_circuit") != circuit:
            raise ArtifactValidationError("fold manifest circuit identity mismatch")
        if circuit in circuits:
            raise ArtifactValidationError(f"duplicate fold manifest for {circuit}")
        circuits.add(circuit)
        if fold.get("status") != "heldout_complete":
            raise ArtifactValidationError(f"fold is not heldout_complete: {circuit}")
        if (
            fold.get("heldout_consumed_sealed_artifacts") is not True
            or fold.get("models_refit_for_heldout") is not False
            or fold.get("calibrators_refit_for_heldout") is not False
            or fold.get("thresholds_refit_for_heldout") is not False
        ):
            raise ArtifactValidationError(
                f"fold {circuit} does not prove sealed, no-refit held-out evaluation"
            )
        for key, expected in expected_identity.items():
            if fold.get(key) != expected:
                raise ArtifactValidationError(f"fold {circuit} has a different {key}")
        _identity(fold.get("input_content_hash"), f"fold {circuit} input_content_hash")
        model_identity = _identity(
            fold.get("model_identity_hash"), f"fold {circuit} model_identity_hash"
        )
        if (
            _identity(
                raw_record.get("model_identity_hash"),
                f"pooled fold record {circuit} model_identity_hash",
            )
            != model_identity
        ):
            raise ArtifactValidationError(f"fold {circuit} model identity binding differs")
        method_identities = fold.get("method_identities")
        if not isinstance(method_identities, dict) or set(method_identities) != set(
            REGISTERED_METHODS
        ):
            raise ArtifactValidationError(f"fold {circuit} method identities are incomplete")
        for method, value in method_identities.items():
            _identity(value, f"fold {circuit} method identity {method}")
        seal_fold = seal_folds[circuit]
        if (
            seal_fold.get("model_identity_hash") != model_identity
            or seal_fold.get("method_identities") != method_identities
        ):
            raise ArtifactValidationError(f"fold {circuit} differs from its frozen identities")
        registry = fold.get("artifacts")
        if not isinstance(registry, dict):
            raise ArtifactValidationError(f"fold {circuit} has no artifact registry")
        frozen_artifacts = seal_fold.get("frozen_artifacts")
        if not isinstance(frozen_artifacts, dict) or not frozen_artifacts:
            raise ArtifactValidationError(f"seal lacks frozen artifacts for {circuit}")
        expected_frozen_names = {
            "model-metadata.json",
            "compact-calibrated-scores.parquet",
            "calibrators.json",
            "calibration-thresholds.csv",
            *(f"raw-scores-{method}.parquet" for method in REGISTERED_METHODS),
        }
        if set(frozen_artifacts) != expected_frozen_names:
            raise ArtifactValidationError(
                f"seal has an incomplete frozen artifact set for {circuit}"
            )
        for name, expected_hash in frozen_artifacts.items():
            frozen_path = manifest_path.parent / str(name)
            registered_hash = registry.get(str(frozen_path.resolve()))
            if registered_hash != expected_hash:
                raise ArtifactValidationError(
                    f"fold {circuit} registry differs from sealed artifact {name}"
                )
            _validate_digest(frozen_path.resolve(), expected_hash, f"sealed {circuit} {name}")
        threshold_path = manifest_path.parent / "calibration-thresholds.csv"
        try:
            calibration_thresholds = pd.read_csv(threshold_path)
        except Exception as exc:
            raise ArtifactValidationError(
                f"cannot read authenticated calibration thresholds: {threshold_path}"
            ) from exc
        _require_columns(
            calibration_thresholds,
            {
                "fold_test_circuit",
                "method",
                "required_lead_s",
                "false_budget_per_hour",
                "threshold",
                "calibration_proposal_count",
                "operating_point_status",
                "threshold_source_partition",
            },
            f"fold {circuit} calibration thresholds",
        )
        if not (
            (calibration_thresholds["fold_test_circuit"].astype(str) == circuit).all()
            and (
                calibration_thresholds["threshold_source_partition"].astype(str)
                == "calibration"
            ).all()
        ):
            raise ArtifactValidationError(
                f"fold {circuit} calibration thresholds have an invalid fold or partition"
            )
        metric_matches = [
            Path(str(path))
            for path in registry
            if Path(str(path)).name == "heldout-operating-metrics.csv"
        ]
        if len(metric_matches) != 1:
            raise ArtifactValidationError(f"fold {circuit} does not register heldout metrics")
        metric_path = metric_matches[0].resolve()
        _validate_digest(metric_path, registry[str(metric_matches[0])], f"fold {circuit} metrics")
        try:
            metrics = pd.read_csv(metric_path)
        except Exception as exc:
            raise ArtifactValidationError(f"cannot read fold metrics: {metric_path}") from exc
        _require_columns(
            metrics,
            {"fold_test_circuit", "circuit", "scope"},
            f"fold {circuit} operating metrics",
        )
        if not (
            (metrics["fold_test_circuit"].astype(str) == circuit).all()
            and (metrics["circuit"].astype(str) == circuit).all()
            and (metrics["scope"].astype(str) == "circuit").all()
        ):
            raise ArtifactValidationError(
                f"fold {circuit} operating metrics contain another circuit"
            )
        proposal_matches = [
            Path(str(path))
            for path in registry
            if Path(str(path)).name == "heldout-proposal-labels.parquet"
        ]
        if len(proposal_matches) != 1:
            raise ArtifactValidationError(
                f"fold {circuit} does not register heldout proposal labels"
            )
        proposal_path = proposal_matches[0].resolve()
        _validate_digest(
            proposal_path,
            registry[str(proposal_matches[0])],
            f"fold {circuit} proposal labels",
        )
        try:
            proposal_labels = pd.read_parquet(proposal_path)
        except Exception as exc:
            raise ArtifactValidationError(
                f"cannot read fold proposal labels: {proposal_path}"
            ) from exc
        primary_proposal_count = _primary_proposal_count(
            proposal_labels,
            f"fold {circuit} proposal labels",
        )
        examined_budget_proposal_count = _examined_budget_proposal_count(
            proposal_labels,
            f"fold {circuit} proposal labels",
        )
        runtime_paths = [
            Path(str(path)).resolve()
            for path in registry
            if Path(str(path)).name.startswith("runtime-") and Path(str(path)).suffix == ".jsonl"
        ]
        if not runtime_paths:
            raise ArtifactValidationError(f"fold {circuit} registers no runtime evidence")
        runtime_events: list[Mapping[str, Any]] = []
        for runtime_path in sorted(runtime_paths):
            expected = registry.get(str(runtime_path))
            if expected is None:
                candidates = [
                    value
                    for key, value in registry.items()
                    if Path(str(key)).resolve() == runtime_path
                ]
                if len(candidates) != 1:
                    raise ArtifactValidationError(
                        f"ambiguous runtime registry path: {runtime_path}"
                    )
                expected = candidates[0]
            _validate_digest(runtime_path, expected, f"fold {circuit} runtime")
            runtime_events.extend(_read_jsonl(runtime_path))
        run_identity = fold.get("run_identity")
        if not isinstance(run_identity, dict) or seal_fold.get("run_identity") != run_identity:
            raise ArtifactValidationError(f"fold {circuit} run identity differs from its seal")
        if fold.get("input_content_hash") != _canonical_json_hash(run_identity):
            raise ArtifactValidationError(f"fold {circuit} input hash does not bind run identity")
        for key, expected in expected_identity.items():
            if run_identity.get(key) != expected:
                raise ArtifactValidationError(f"fold {circuit} run identity has a different {key}")
        if (
            run_identity.get("fold_test_circuit") != circuit
            or tuple(run_identity.get("registered_methods", ())) != REGISTERED_METHODS
            or run_identity.get("random_seed_base") != 20270927
            or set(str(value) for value in fold.get("completed_methods", ()))
            != set(REGISTERED_METHODS)
        ):
            raise ArtifactValidationError(f"fold {circuit} run identity is incomplete")
        environment = (
            run_identity.get("runtime_environment") if isinstance(run_identity, dict) else None
        )
        if not isinstance(environment, dict):
            raise ArtifactValidationError(f"fold {circuit} lacks a runtime environment identity")
        folds.append(
            FoldEvidence(
                circuit,
                metrics,
                tuple(runtime_events),
                environment,
                primary_proposal_count,
                examined_budget_proposal_count,
                calibration_thresholds,
            )
        )
    if tuple(sorted(circuits)) != REGISTERED_CIRCUITS:
        raise ArtifactValidationError(
            "pooled manifest does not contain the four frozen cohort circuits"
        )
    return tuple(sorted(folds, key=lambda value: value.circuit))


def _load_build_manifest(pooled: ManifestBundle) -> Mapping[str, Any]:
    candidates = [
        ancestor / "data" / "manifests" / "deepracing-build.json"
        for ancestor in (pooled.directory, *pooled.directory.parents)
    ]
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ArtifactValidationError(
            "cannot locate exactly one data/manifests/deepracing-build.json for pooled evidence"
        )
    path = matches[0]
    _validate_digest(path, pooled.payload.get("build_manifest_hash"), "build manifest")
    manifest = _load_json(path, "build manifest")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ArtifactValidationError("build manifest has no counts object")
    expected_counts = {
        "input_files": 244,
        "input_bytes": 86_360_332,
        "circuits": 4,
        "cars": 58,
        "raw_frames": 2_473_009,
        "resampled_frames": 460_269,
        "candidate_excursions": 1_616,
        "qualified_excursions": 294,
    }
    if any(int(counts.get(key, -1)) != value for key, value in expected_counts.items()):
        raise ArtifactValidationError("build manifest differs from the frozen cohort counts")
    if not math.isclose(
        _number(counts.get("buffered_exposure_hours"), "buffered exposure"),
        6.3170549663073485,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ArtifactValidationError("build manifest differs from frozen eligible exposure")
    _validate_per_circuit_counts(manifest)
    input_manifest = manifest.get("input_manifest")
    if not isinstance(input_manifest, dict):
        raise ArtifactValidationError("build manifest has no acquisition source manifest record")
    source_manifest_hash = str(input_manifest.get("sha256"))
    if source_manifest_hash != str(pooled.payload.get("source_manifest_hash")):
        raise ArtifactValidationError("build manifest source hash differs from pooled evidence")
    source_manifest_path = Path(str(input_manifest.get("path", "")))
    if not source_manifest_path.is_absolute():
        raise ArtifactValidationError(
            f"acquisition source manifest path is not absolute: {source_manifest_path}"
        )
    source_manifest_path = source_manifest_path.resolve()
    _validate_digest(
        source_manifest_path,
        source_manifest_hash,
        "acquisition source manifest",
    )
    source_manifest_bytes = input_manifest.get("bytes")
    if (
        not isinstance(source_manifest_bytes, int)
        or isinstance(source_manifest_bytes, bool)
        or source_manifest_bytes < 0
    ):
        raise ArtifactValidationError(
            "build manifest acquisition source byte count is missing or invalid"
        )
    if source_manifest_bytes != source_manifest_path.stat().st_size:
        raise ArtifactValidationError(
            f"acquisition source manifest byte count changed: {source_manifest_path}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ArtifactValidationError("build manifest has no authenticated outputs")
    for record in outputs:
        if not isinstance(record, dict):
            raise ArtifactValidationError("build output record is not an object")
        output_path = Path(str(record.get("path", "")))
        if not output_path.is_absolute():
            raise ArtifactValidationError(f"build output path is not absolute: {output_path}")
        _validate_digest(output_path.resolve(), record.get("sha256"), "build output")
        if int(record.get("bytes", -1)) != output_path.stat().st_size:
            raise ArtifactValidationError(f"build output byte count changed: {output_path}")
    return manifest


def _validate_per_circuit_counts(manifest: Mapping[str, Any]) -> None:
    raw_rows = manifest.get("per_circuit_counts")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(REGISTERED_CIRCUITS):
        raise ArtifactValidationError(
            "build manifest per-circuit counts must contain exactly four circuit rows"
        )
    rows: dict[str, Mapping[str, Any]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ArtifactValidationError("build manifest per-circuit counts contain a non-object")
        circuit = str(raw_row.get("circuit", ""))
        if circuit in rows or circuit not in EXPECTED_CIRCUIT_COUNTS:
            raise ArtifactValidationError(
                "build manifest per-circuit counts have a duplicate or unexpected circuit"
            )
        rows[circuit] = raw_row
    if set(rows) != set(REGISTERED_CIRCUITS):
        raise ArtifactValidationError("build manifest per-circuit counts omit a circuit")
    integer_fields = (
        "cars",
        "qualified_excursions",
        "candidate_excursions",
        "resampled_frames",
    )
    floating_fields = (
        "prebuffer_exposure_seconds",
        "buffered_exposure_seconds",
    )
    for circuit in REGISTERED_CIRCUITS:
        observed = rows[circuit]
        expected = EXPECTED_CIRCUIT_COUNTS[circuit]
        if any(int(observed.get(field, -1)) != int(expected[field]) for field in integer_fields):
            raise ArtifactValidationError(
                f"build manifest per-circuit counts differ for {circuit}"
            )
        if any(
            not _close(observed.get(field), float(expected[field]), atol=1e-9)
            for field in floating_fields
        ):
            raise ArtifactValidationError(
                f"build manifest per-circuit counts differ for {circuit}"
            )


def _validate_manuscript_cohort_table(
    manuscript_source: str,
    build_manifest: Mapping[str, Any],
) -> None:
    raw_rows = build_manifest.get("per_circuit_counts")
    assert isinstance(raw_rows, list)
    by_circuit = {str(row["circuit"]): row for row in raw_rows}
    expected_lines: list[str] = []
    for circuit in REGISTERED_CIRCUITS:
        row = by_circuit[circuit]
        exposure = float(row["buffered_exposure_seconds"]) / 3600.0
        events = int(row["qualified_excursions"])
        expected_lines.append(
            f"| {circuit} | {int(row['cars'])} | {events} | {exposure:.6f} | "
            f"{events / exposure:.2f} |"
        )
    counts = build_manifest["counts"]
    total_exposure = float(counts["buffered_exposure_hours"])
    total_events = int(counts["qualified_excursions"])
    expected_lines.append(
        f"| **Total** | **{int(counts['cars'])}** | **{total_events}** | "
        f"**{total_exposure:.6f}** | **{total_events / total_exposure:.2f}** |"
    )
    section_match = re.search(
        r"(?ms)^\| Held-out circuit \|.*?^Table: Frozen primary cohort\..*?$",
        manuscript_source,
    )
    if section_match is None:
        raise ArtifactValidationError("manuscript cohort table is missing")
    section = section_match.group(0)
    observed_lines = [
        line.strip()
        for line in section.splitlines()
        if any(line.startswith(f"| {circuit} ") for circuit in REGISTERED_CIRCUITS)
        or line.startswith("| **Total** ")
    ]
    if observed_lines != expected_lines:
        raise ArtifactValidationError(
            "manuscript cohort table disagrees with authenticated per-circuit build counts"
        )


def _load_draws(
    bundle: ManifestBundle,
    name: str,
    *,
    require_lstar_grid: bool = True,
) -> np.ndarray:
    path = _artifact(bundle, name)
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ArtifactValidationError(f"cannot read authenticated bootstrap draws: {path}") from exc
    if values.shape != (10_000,) or not np.isfinite(values).all():
        raise ArtifactValidationError(f"bootstrap draws have invalid shape or values: {path}")
    if require_lstar_grid and not np.isin(values, np.asarray((0.0, *LEAD_GRID), dtype=float)).all():
        raise ArtifactValidationError(f"bootstrap L-star draws leave the prespecified grid: {path}")
    return values.astype(float, copy=False)


def _validate_draw_summary(bundle: ManifestBundle, summary: Mapping[str, Any]) -> None:
    analyses = (
        (
            "",
            summary,
            "brace-l-star-draws.npy",
            "comparator-l-star-draws.npy",
            "delta-l-star-draws.npy",
        ),
        (
            "least-favorable ",
            summary.get("least_favorable_unresolved_counted_as_false"),
            "least-favorable-brace-l-star-draws.npy",
            "least-favorable-comparator-l-star-draws.npy",
            "least-favorable-delta-l-star-draws.npy",
        ),
    )
    for prefix, raw_section, brace_name, comparator_name, delta_name in analyses:
        if not isinstance(raw_section, dict):
            raise ArtifactValidationError(f"bootstrap summary lacks {prefix}draw fields")
        brace = _load_draws(bundle, brace_name)
        comparator = _load_draws(bundle, comparator_name)
        delta = _load_draws(bundle, delta_name, require_lstar_grid=False)
        if not np.array_equal(delta, brace - comparator):
            raise ArtifactValidationError(f"{prefix}bootstrap draws violate delta = brace - twin")
        intervals = {
            "l_star_brace_percentile_95": np.quantile(brace, (0.025, 0.975)),
            "l_star_comparator_percentile_95": np.quantile(comparator, (0.025, 0.975)),
            "delta_l_star_percentile_95": np.quantile(delta, (0.025, 0.975)),
        }
        for key, expected in intervals.items():
            observed = np.asarray(_interval(raw_section, key), dtype=float)
            if not np.allclose(observed, expected, atol=1e-12, rtol=0.0):
                raise ArtifactValidationError(
                    f"{prefix}bootstrap draws disagree with summary interval {key}"
                )


def _validate_bootstrap(
    bundle: ManifestBundle,
    *,
    pooled: ManifestBundle,
    expected_cluster: str,
) -> Mapping[str, Any]:
    payload = bundle.payload
    if payload.get("schema_version") != 1:
        raise ArtifactValidationError("bootstrap manifest is not schema version 1")
    if payload.get("noncanonical_synthetic") is not False:
        raise ArtifactValidationError(
            "submission compiler rejects noncanonical synthetic bootstrap"
        )
    if _identity(payload.get("code_content_hash"), "bootstrap code_content_hash") != _identity(
        pooled.payload.get("code_content_hash"), "pooled code_content_hash"
    ):
        raise ArtifactValidationError("bootstrap producer code differs from pooled evidence")
    config_record = payload.get("study_config")
    if not isinstance(config_record, dict):
        raise ArtifactValidationError("bootstrap manifest lacks frozen study_config provenance")
    config_path = _manifest_record_path(config_record, "bootstrap frozen study config")
    config = _load_json(config_path, "bootstrap frozen study config")
    canonical_config_hash = _canonical_json_hash(config)
    if config_record.get(
        "canonical_content_hash"
    ) != canonical_config_hash or canonical_config_hash != pooled.payload.get(
        "config_content_hash"
    ):
        raise ArtifactValidationError("bootstrap study config differs from pooled evidence")
    cohort = config.get("cohort")
    if not isinstance(cohort, dict) or tuple(cohort.get("circuits", ())) != REGISTERED_CIRCUITS:
        raise ArtifactValidationError("frozen study config has the wrong circuit cohort")
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise ArtifactValidationError("bootstrap manifest lacks analysis identity")
    expected_analysis = {
        "n_resamples": 10_000,
        "seed": 20270927,
        "cluster_level": expected_cluster,
        "models_and_thresholds_refit": False,
    }
    if any(analysis.get(key) != value for key, value in expected_analysis.items()):
        raise ArtifactValidationError(
            f"bootstrap analysis does not match the prespecified {expected_cluster} analysis"
        )
    expected_interval = {
        "construction": "percentile_95",
        "paired": True,
        "resampling_unit": expected_cluster,
    }
    if payload.get("interval") != expected_interval:
        raise ArtifactValidationError("bootstrap interval construction differs from frozen design")
    estimand = payload.get("estimand")
    if not isinstance(estimand, dict):
        raise ArtifactValidationError("bootstrap manifest lacks an estimand")
    expected_estimand = {
        "brace_method": BRACE_METHOD,
        "comparator_method": TWIN_METHOD,
        "false_budget_per_hour": PRIMARY_BUDGET,
        "minimum_recall": MINIMUM_RECALL,
        "lead_grid_s": list(LEAD_GRID),
        "least_favorable_false_count_column": "least_favorable_false_proposals",
    }
    if any(estimand.get(key) != value for key, value in expected_estimand.items()):
        raise ArtifactValidationError("bootstrap estimand differs from the frozen primary estimand")
    experiment_record = payload.get("experiment_manifest")
    if not isinstance(experiment_record, dict):
        raise ArtifactValidationError("bootstrap manifest does not bind a pooled experiment")
    experiment_path = _manifest_record_path(experiment_record, "bootstrap pooled manifest")
    if (
        experiment_path != pooled.manifest_path
        or str(experiment_record.get("sha256")) != pooled.manifest_hash
    ):
        raise ArtifactValidationError("bootstrap was not generated from this pooled manifest")
    input_record = payload.get("input")
    if not isinstance(input_record, dict):
        raise ArtifactValidationError("bootstrap manifest lacks its contribution input")
    contribution_path = _artifact(pooled, "pooled-heldout-car-contributions.parquet")
    if Path(str(input_record.get("path", ""))).resolve() != contribution_path:
        raise ArtifactValidationError("bootstrap contribution path differs from pooled evidence")
    if str(input_record.get("sha256")) != pooled.artifacts[contribution_path]:
        raise ArtifactValidationError("bootstrap contribution hash differs from pooled evidence")
    summary = _load_json(_artifact(bundle, "summary.json"), "bootstrap summary")
    if summary.get("n_resamples") != 10_000 or summary.get("seed") != 20270927:
        raise ArtifactValidationError("bootstrap summary has the wrong resample count or seed")
    if summary.get("cluster_level") != expected_cluster:
        raise ArtifactValidationError("bootstrap summary cluster level differs from its manifest")
    sample = payload.get("sample")
    if not isinstance(sample, dict):
        raise ArtifactValidationError("bootstrap manifest lacks sample identity")
    if (
        sample.get("circuit_count") != 4
        or sample.get("car_session_count") != 58
        or tuple(sample.get("circuits", ())) != REGISTERED_CIRCUITS
    ):
        raise ArtifactValidationError("bootstrap sample differs from the frozen cohort")
    reliability_record = payload.get("reliability_bootstrap")
    if not isinstance(reliability_record, dict):
        raise ArtifactValidationError("bootstrap manifest lacks reliability provenance")
    reliability_input = _artifact(
        pooled, "pooled-heldout-reliability-cluster-contributions.parquet"
    )
    expected_reliability = {
        "input_path": str(reliability_input),
        "input_sha256": pooled.artifacts[reliability_input],
        "n_resamples": 10_000,
        "seed": 20270927,
        "cluster_level": "car_session_within_circuit",
        "fixed_equal_width_bins": 10,
    }
    if reliability_record != expected_reliability:
        raise ArtifactValidationError(
            "bootstrap reliability provenance differs from the frozen pooled input"
        )
    _validate_draw_summary(bundle, summary)
    return summary


def _load_evidence(
    pooled_dir: Path,
    bootstrap_dir: Path,
    transport_bootstrap_dir: Path | None,
) -> Evidence:
    pooled = _load_manifest_bundle(pooled_dir, label="pooled held-out")
    folds = _validate_pooled(pooled)
    if _producer_source_tree_hash() != _identity(
        pooled.payload.get("code_content_hash"),
        "pooled code_content_hash",
    ):
        raise ArtifactValidationError(
            "current producer source-tree hash differs from authenticated experiment code"
        )
    build_manifest = _load_build_manifest(pooled)
    bootstrap = _load_manifest_bundle(bootstrap_dir, label="primary bootstrap")
    bootstrap_summary = _validate_bootstrap(
        bootstrap,
        pooled=pooled,
        expected_cluster="car_session_within_circuit",
    )
    transport: ManifestBundle | None = None
    transport_summary: Mapping[str, Any] | None = None
    if transport_bootstrap_dir is not None:
        transport = _load_manifest_bundle(
            transport_bootstrap_dir,
            label="two-stage transport bootstrap",
        )
        transport_summary = _validate_bootstrap(
            transport,
            pooled=pooled,
            expected_cluster="circuit_then_car_session",
        )
    operating = _read_csv(pooled, "pooled-heldout-operating-metrics.csv")
    probability = _read_csv(pooled, "pooled-heldout-probability-metrics.csv")
    reliability = _read_csv(pooled, "pooled-heldout-reliability-bins.csv")
    reliability_contributions_path = _artifact(
        pooled,
        "pooled-heldout-reliability-cluster-contributions.parquet",
    )
    reliability_contributions = _read_parquet(
        pooled,
        "pooled-heldout-reliability-cluster-contributions.parquet",
    )
    monotonicity = _read_csv(pooled, "pooled-heldout-horizon-monotonicity.csv")
    l_star_sensitivity = _read_csv(pooled, "pooled-heldout-l-star-sensitivity.csv")
    delay_l_star = _read_csv(pooled, "pooled-heldout-synthetic-delay-l-star.csv")
    contributions = _read_parquet(pooled, "pooled-heldout-car-contributions.parquet")
    delay_contributions = _read_parquet(
        pooled,
        "pooled-heldout-synthetic-delay-car-contributions.parquet",
    )
    reliability_intervals = _read_csv(bootstrap, "reliability-bin-intervals.csv")
    _validate_operating_against_contributions(operating, contributions, folds)
    _validate_delay_results(
        delay_l_star,
        delay_contributions,
        operating,
        contributions,
    )
    _validate_reliability_points(
        reliability,
        reliability_intervals,
        reliability_contributions,
        pooled.artifacts[reliability_contributions_path],
    )
    _validate_probability_reliability_support(
        probability,
        reliability,
        reliability_contributions,
    )
    return Evidence(
        pooled,
        bootstrap,
        transport,
        build_manifest,
        operating,
        probability,
        reliability,
        reliability_contributions,
        monotonicity,
        l_star_sensitivity,
        delay_l_star,
        contributions,
        delay_contributions,
        bootstrap_summary,
        transport_summary,
        reliability_intervals,
        folds,
    )


def _reliability_bootstrap_source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "brace_f1" / "bootstrap_reliability.py"


def _load_reliability_bootstrap_function() -> Any:
    global _RELIABILITY_BOOTSTRAP_FUNCTION
    if _RELIABILITY_BOOTSTRAP_FUNCTION is not None:
        return _RELIABILITY_BOOTSTRAP_FUNCTION
    source_path = _reliability_bootstrap_source_path()
    if not source_path.is_file():
        raise ArtifactValidationError(
            f"producer reliability-bootstrap source is missing: {source_path}"
        )
    package = types.ModuleType("brace_f1")
    package.__path__ = [str(source_path.parent)]
    experiment = types.ModuleType("brace_f1.experiment")
    experiment.HORIZONS_S = LEAD_GRID
    io_module = types.ModuleType("brace_f1.io")

    class ProducerDataValidationError(ValueError):
        pass

    io_module.DataValidationError = ProducerDataValidationError
    module_name = "_brace_publication_bootstrap_reliability"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ArtifactValidationError(
            f"cannot load producer reliability-bootstrap source: {source_path}"
        )
    module = importlib.util.module_from_spec(spec)
    replaced = {
        name: sys.modules.get(name)
        for name in ("brace_f1", "brace_f1.experiment", "brace_f1.io", module_name)
    }
    try:
        sys.modules["brace_f1"] = package
        sys.modules["brace_f1.experiment"] = experiment
        sys.modules["brace_f1.io"] = io_module
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ArtifactValidationError(
            f"cannot execute producer reliability-bootstrap source: {source_path}"
        ) from exc
    finally:
        for name, previous in replaced.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    function = getattr(module, "reliability_bin_cluster_bootstrap", None)
    if not callable(function):
        raise ArtifactValidationError(
            "producer source lacks reliability_bin_cluster_bootstrap"
        )
    _RELIABILITY_BOOTSTRAP_FUNCTION = function
    return function


def _recompute_reliability_intervals(
    contributions: pd.DataFrame,
    contribution_sha256: str,
) -> pd.DataFrame:
    source_path = _reliability_bootstrap_source_path()
    source_sha256 = _sha256(source_path)
    cache_key = (source_sha256, contribution_sha256)
    cached = _RELIABILITY_RECOMPUTATION_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy(deep=True)
    function = _load_reliability_bootstrap_function()
    try:
        result = function(
            contributions,
            n_resamples=10_000,
            seed=20270927,
            horizons_s=LEAD_GRID,
            n_bins=10,
            expected_methods=REGISTERED_METHODS,
        )
    except Exception as exc:
        raise ArtifactValidationError(
            "producer reliability-bootstrap recomputation failed"
        ) from exc
    summary = getattr(result, "summary", None)
    if not isinstance(summary, pd.DataFrame):
        raise ArtifactValidationError(
            "producer reliability-bootstrap recomputation returned no summary"
        )
    _RELIABILITY_RECOMPUTATION_CACHE[cache_key] = summary.copy(deep=True)
    return summary


def _validate_reliability_points(
    pooled: pd.DataFrame,
    intervals: pd.DataFrame,
    contributions: pd.DataFrame,
    contribution_sha256: str,
) -> None:
    keys = ["method", "horizon_s", "bin_index"]
    values = ["count", "mean_probability", "observed_frequency"]
    _require_columns(
        pooled,
        {"scope", "circuit", *keys, *values},
        "pooled reliability table",
    )
    _require_columns(
        intervals,
        {
            *keys,
            *values,
            "bin_left",
            "bin_right",
            "right_edge_inclusive",
            "mean_probability_lower_95",
            "mean_probability_upper_95",
            "observed_frequency_lower_95",
            "observed_frequency_upper_95",
            "valid_probability_resamples",
            "valid_frequency_resamples",
            "circuit_count",
            "car_session_count",
            "cluster_level",
            "transport_uncertainty_included",
        },
        "bootstrap reliability table",
    )
    bin_index = pd.to_numeric(intervals["bin_index"], errors="coerce").to_numpy(dtype=float)
    expected_left = bin_index / 10.0
    expected_right = (bin_index + 1.0) / 10.0
    inclusive = intervals["right_edge_inclusive"].astype(str).str.lower().isin(("true", "1"))
    if (
        not np.isin(bin_index, np.arange(10, dtype=float)).all()
        or not np.allclose(intervals["bin_left"], expected_left, atol=1e-12, rtol=0.0)
        or not np.allclose(intervals["bin_right"], expected_right, atol=1e-12, rtol=0.0)
        or not np.array_equal(inclusive.to_numpy(), bin_index == 9.0)
        or not (pd.to_numeric(intervals["circuit_count"], errors="coerce") == 4).all()
        or not (pd.to_numeric(intervals["car_session_count"], errors="coerce") == 58).all()
        or not (intervals["cluster_level"].astype(str) == "car_session_within_circuit").all()
        or intervals["transport_uncertainty_included"]
        .astype(str)
        .str.lower()
        .isin(("true", "1"))
        .any()
    ):
        raise ArtifactValidationError(
            "reliability interval bin or resampling provenance is invalid"
        )
    actual_grid = {
        (str(row.method), float(row.horizon_s), int(row.bin_index))
        for row in intervals.itertuples(index=False)
    }
    expected_grid = {
        (method, horizon, bin_number)
        for method in REGISTERED_METHODS
        for horizon in LEAD_GRID
        for bin_number in range(10)
    }
    if len(intervals) != len(expected_grid) or actual_grid != expected_grid:
        raise ArtifactValidationError("reliability interval grid is incomplete or duplicated")
    counts = pd.to_numeric(intervals["count"], errors="coerce").to_numpy(dtype=float)
    mean_probability = pd.to_numeric(
        intervals["mean_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    observed_frequency = pd.to_numeric(
        intervals["observed_frequency"], errors="coerce"
    ).to_numpy(dtype=float)
    nonempty = counts > 0.0
    empty = ~nonempty
    if (
        not np.isfinite(counts).all()
        or not np.equal(counts, np.floor(counts)).all()
        or np.any(counts < 0.0)
        or not np.isfinite(mean_probability[nonempty]).all()
        or not np.isfinite(observed_frequency[nonempty]).all()
        or np.any(mean_probability[nonempty] < 0.0)
        or np.any(mean_probability[nonempty] > 1.0)
        or np.any(observed_frequency[nonempty] < 0.0)
        or np.any(observed_frequency[nonempty] > 1.0)
        or not np.isnan(mean_probability[empty]).all()
        or not np.isnan(observed_frequency[empty]).all()
    ):
        raise ArtifactValidationError("reliability point values or counts are invalid")
    bin_left = pd.to_numeric(intervals["bin_left"], errors="coerce").to_numpy(dtype=float)
    bin_right = pd.to_numeric(intervals["bin_right"], errors="coerce").to_numpy(dtype=float)
    within_bin = (mean_probability >= bin_left - 1e-12) & (
        (mean_probability < bin_right - 1e-12)
        | ((bin_index == 9.0) & (mean_probability <= bin_right + 1e-12))
    )
    if not within_bin[nonempty].all():
        raise ArtifactValidationError("reliability mean probability lies outside its fixed bin")
    for prefix, valid_column in (
        ("mean_probability", "valid_probability_resamples"),
        ("observed_frequency", "valid_frequency_resamples"),
    ):
        valid = pd.to_numeric(intervals[valid_column], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(intervals[f"{prefix}_lower_95"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(intervals[f"{prefix}_upper_95"], errors="coerce").to_numpy(dtype=float)
        present = valid > 0
        if (
            not np.isfinite(valid).all()
            or not np.equal(valid, np.floor(valid)).all()
            or np.any(valid < 0)
            or np.any(valid > 10_000)
            or not np.isfinite(low[present]).all()
            or not np.isfinite(high[present]).all()
            or np.any(low[present] < 0.0)
            or np.any(high[present] > 1.0)
            or np.any(low[present] > high[present])
            or np.any(valid[nonempty] <= 0.0)
            or np.any(valid[empty] != 0.0)
            or not np.isnan(low[~present]).all()
            or not np.isnan(high[~present]).all()
        ):
            raise ArtifactValidationError("reliability percentile bounds are invalid")
    points = pooled.loc[
        (pooled["scope"].astype(str) == "pooled") & (pooled["circuit"].astype(str) == "all"),
        [*keys, *values],
    ].copy()
    try:
        merged = points.merge(
            intervals.loc[:, [*keys, *values]],
            on=keys,
            suffixes=("_pooled", "_bootstrap"),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    except pd.errors.MergeError as exc:
        raise ArtifactValidationError("reliability point grid is duplicated") from exc
    if not (merged["_merge"] == "both").all():
        raise ArtifactValidationError("reliability point grids are not identical")
    if not np.array_equal(
        merged["count_pooled"].to_numpy(dtype=int),
        merged["count_bootstrap"].to_numpy(dtype=int),
    ):
        raise ArtifactValidationError("reliability point estimates disagree on bin counts")
    for column in ("mean_probability", "observed_frequency"):
        if not np.allclose(
            merged[f"{column}_pooled"].to_numpy(dtype=float),
            merged[f"{column}_bootstrap"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        ):
            raise ArtifactValidationError(f"reliability point estimates disagree on {column}")
    recomputed = _recompute_reliability_intervals(contributions, contribution_sha256)
    ordered_intervals = intervals.sort_values(keys, kind="stable").reset_index(drop=True)
    ordered_recomputed = recomputed.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(ordered_intervals) != len(ordered_recomputed):
        raise ArtifactValidationError(
            "reliability bootstrap recomputation has a different row count"
        )
    exact_columns = (
        "method",
        "bin_index",
        "count",
        "right_edge_inclusive",
        "valid_probability_resamples",
        "valid_frequency_resamples",
        "circuit_count",
        "car_session_count",
        "cluster_level",
        "transport_uncertainty_included",
    )
    for column in exact_columns:
        if not np.array_equal(
            ordered_intervals[column].astype(str).to_numpy(),
            ordered_recomputed[column].astype(str).to_numpy(),
        ):
            raise ArtifactValidationError(
                f"reliability bootstrap recomputation disagrees on {column}"
            )
    numeric_columns = (
        "horizon_s",
        "bin_left",
        "bin_right",
        "mean_probability",
        "mean_probability_lower_95",
        "mean_probability_upper_95",
        "observed_frequency",
        "observed_frequency_lower_95",
        "observed_frequency_upper_95",
    )
    for column in numeric_columns:
        if not np.allclose(
            pd.to_numeric(ordered_intervals[column], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(ordered_recomputed[column], errors="coerce").to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        ):
            raise ArtifactValidationError(
                f"reliability bootstrap recomputation disagrees on {column}"
            )


def _validate_probability_reliability_support(
    probability: pd.DataFrame,
    reliability: pd.DataFrame,
    contributions: pd.DataFrame,
) -> None:
    keys = ["method", "horizon_s"]
    _require_columns(
        probability,
        {"scope", "circuit", *keys, "n", "event_count", "ece_10_equal_width"},
        "probability metric table",
    )
    _require_columns(
        reliability,
        {
            "scope",
            "circuit",
            *keys,
            "bin_index",
            "count",
            "mean_probability",
            "observed_frequency",
        },
        "pooled reliability table",
    )
    _require_columns(
        contributions,
        {*keys, "bin_index", "count", "event_count"},
        "reliability contribution table",
    )
    pooled_scope = probability["scope"].astype(str) == "pooled"
    if not (probability.loc[pooled_scope, "circuit"].astype(str) == "all").all():
        raise ArtifactValidationError(
            "pooled probability metric scope has a non-pooled circuit"
        )
    pooled_probability = probability.loc[pooled_scope].copy()
    pooled_probability["horizon_s"] = [
        _registered_grid_value(value, LEAD_GRID, "probability horizon")
        for value in pooled_probability["horizon_s"]
    ]
    expected_grid = {
        (method, horizon) for method in REGISTERED_METHODS for horizon in LEAD_GRID
    }
    probability_grid = {
        (str(row.method), float(row.horizon_s))
        for row in pooled_probability.itertuples(index=False)
    }
    if (
        len(pooled_probability) != len(expected_grid)
        or probability_grid != expected_grid
        or pooled_probability.duplicated(keys).any()
    ):
        raise ArtifactValidationError(
            "pooled probability metric method/horizon grid is incomplete or duplicated"
        )
    probability_counts = pooled_probability[["n", "event_count"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        not np.isfinite(probability_counts.to_numpy(dtype=float)).all()
        or not np.equal(
            probability_counts.to_numpy(dtype=float),
            np.floor(probability_counts.to_numpy(dtype=float)),
        ).all()
        or (probability_counts["n"] <= 0).any()
        or (probability_counts["event_count"] < 0).any()
        or (probability_counts["event_count"] > probability_counts["n"]).any()
    ):
        raise ArtifactValidationError("pooled probability metric counts are invalid")

    contribution_counts = contributions[["count", "event_count"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        not np.isfinite(contribution_counts.to_numpy(dtype=float)).all()
        or not np.equal(
            contribution_counts.to_numpy(dtype=float),
            np.floor(contribution_counts.to_numpy(dtype=float)),
        ).all()
        or (contribution_counts < 0).any(axis=None)
        or (
            contribution_counts["event_count"] > contribution_counts["count"]
        ).any()
    ):
        raise ArtifactValidationError(
            "reliability contribution counts are invalid for probability support"
        )
    support = (
        contributions.assign(
            method=contributions["method"].astype(str),
            horizon_s=[
                _registered_grid_value(value, LEAD_GRID, "reliability contribution horizon")
                for value in contributions["horizon_s"]
            ],
            count=contribution_counts["count"].to_numpy(dtype=int),
            event_count=contribution_counts["event_count"].to_numpy(dtype=int),
        )
        .groupby(keys, sort=True, as_index=False)[["count", "event_count"]]
        .sum()
    )
    support_grid = {
        (str(row.method), float(row.horizon_s)) for row in support.itertuples(index=False)
    }
    if len(support) != len(expected_grid) or support_grid != expected_grid:
        raise ArtifactValidationError(
            "reliability contribution support grid is incomplete or duplicated"
        )
    shared_support = support.groupby("horizon_s", sort=True)[
        ["count", "event_count"]
    ].nunique()
    if (shared_support != 1).any(axis=None):
        raise ArtifactValidationError(
            "reliability contribution support is not shared across methods"
        )
    merged_support = pooled_probability.loc[:, [*keys, "n", "event_count"]].merge(
        support,
        on=keys,
        how="outer",
        suffixes=("_probability", "_reliability"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged_support["_merge"] == "both").all() or not np.array_equal(
        merged_support[["n", "event_count_probability"]].to_numpy(dtype=int),
        merged_support[["count", "event_count_reliability"]].to_numpy(dtype=int),
    ):
        raise ArtifactValidationError(
            "probability/reliability support counts disagree"
        )

    pooled_reliability = reliability.loc[
        (reliability["scope"].astype(str) == "pooled")
        & (reliability["circuit"].astype(str) == "all")
    ].copy()
    pooled_reliability["horizon_s"] = [
        _registered_grid_value(value, LEAD_GRID, "pooled reliability horizon")
        for value in pooled_reliability["horizon_s"]
    ]
    ece_rows: list[dict[str, object]] = []
    for (method, horizon), group in pooled_reliability.groupby(keys, sort=True):
        counts = pd.to_numeric(group["count"], errors="coerce").to_numpy(dtype=float)
        nonempty = counts > 0.0
        mean_probability = pd.to_numeric(
            group.loc[nonempty, "mean_probability"],
            errors="coerce",
        ).to_numpy(dtype=float)
        observed_frequency = pd.to_numeric(
            group.loc[nonempty, "observed_frequency"],
            errors="coerce",
        ).to_numpy(dtype=float)
        total = float(counts.sum())
        if total <= 0.0:
            raise ArtifactValidationError("pooled reliability support is empty")
        ece_rows.append(
            {
                "method": str(method),
                "horizon_s": float(horizon),
                "recomputed_ece": float(
                    np.sum(
                        counts[nonempty]
                        / total
                        * np.abs(mean_probability - observed_frequency)
                    )
                ),
            }
        )
    ece = pd.DataFrame(ece_rows)
    merged_ece = pooled_probability.loc[:, [*keys, "ece_10_equal_width"]].merge(
        ece,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    reported_ece = pd.to_numeric(
        merged_ece["ece_10_equal_width"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if (
        not (merged_ece["_merge"] == "both").all()
        or not np.isfinite(reported_ece).all()
        or not np.allclose(
            reported_ece,
            merged_ece["recomputed_ece"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise ArtifactValidationError(
            "pooled probability ECE disagrees with exact reliability bins"
        )


def _registered_grid_value(value: object, grid: Sequence[float], label: str) -> float:
    numeric = _number(value, label)
    matches = [
        candidate
        for candidate in grid
        if math.isclose(numeric, candidate, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ArtifactValidationError(f"{label} leaves the prespecified grid: {value!r}")
    return float(matches[0])


def _validate_operating_against_contributions(
    operating: pd.DataFrame,
    contributions: pd.DataFrame,
    folds: Sequence[FoldEvidence],
) -> None:
    grid_columns = ["method", "required_lead_s", "false_budget_per_hour"]
    unit_columns = ["circuit", "source_session_id", "car_id"]
    count_columns = [
        "localized_event_hits",
        "qualified_events",
        "event_hits",
        "correct_side_event_hits",
        "within_segment_tolerance_event_hits",
        "exact_bin_event_hits",
        "false_proposals",
        "unresolved_proposals",
        "least_favorable_false_proposals",
    ]
    _require_columns(
        contributions,
        {
            *grid_columns,
            *unit_columns,
            "fold_test_circuit",
            "exposure_hours",
            "fixed_model_and_threshold",
            "operating_point_status",
            *count_columns,
        },
        "car contribution table",
    )
    _require_columns(
        operating,
        {
            *grid_columns,
            "scope",
            "operating_point_status",
            "estimability_reason",
            "circuit_count",
            "exposure_hours",
            "localized_event_recall",
            "event_recall",
            "correct_side_event_recall",
            "within_segment_tolerance_event_recall",
            "exact_bin_event_recall",
            "false_proposals_per_hour",
            "false_proposals_per_hour_upper_95",
            *count_columns,
        },
        "pooled operating table",
    )
    ledger = contributions.copy()
    ledger["required_lead_s"] = [
        _registered_grid_value(value, LEAD_GRID, "contribution lead")
        for value in ledger["required_lead_s"]
    ]
    ledger["false_budget_per_hour"] = [
        _registered_grid_value(value, BUDGET_GRID, "contribution false budget")
        for value in ledger["false_budget_per_hour"]
    ]
    if set(ledger["method"].astype(str)) != set(REGISTERED_METHODS):
        raise ArtifactValidationError("car contribution ledger has an unexpected method grid")
    if set(ledger["circuit"].astype(str)) != set(REGISTERED_CIRCUITS):
        raise ArtifactValidationError("car contribution ledger has an unexpected circuit grid")
    if not (
        ledger["fold_test_circuit"].astype(str).to_numpy()
        == ledger["circuit"].astype(str).to_numpy()
    ).all():
        raise ArtifactValidationError("car contribution ledger mixes held-out circuits")
    fixed = ledger["fixed_model_and_threshold"].astype(str).str.lower().isin(("true", "1"))
    if not fixed.all() or not (
        ledger["operating_point_status"].astype(str) == "estimable"
    ).all():
        raise ArtifactValidationError(
            "car contribution ledger is not a fixed, estimable held-out grid"
        )
    numeric_counts = ledger[count_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    exposure = pd.to_numeric(ledger["exposure_hours"], errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(numeric_counts).all()
        or np.any(numeric_counts < 0.0)
        or not np.equal(numeric_counts, np.floor(numeric_counts)).all()
        or not np.isfinite(exposure).all()
        or np.any(exposure <= 0.0)
    ):
        raise ArtifactValidationError(
            "car contribution ledger has invalid counts or exposure"
        )
    event_hits = ledger["event_hits"].to_numpy(dtype=float)
    qualified_events = ledger["qualified_events"].to_numpy(dtype=float)
    correct_side_hits = ledger["correct_side_event_hits"].to_numpy(dtype=float)
    within_tolerance_hits = ledger[
        "within_segment_tolerance_event_hits"
    ].to_numpy(dtype=float)
    localized_hits = ledger["localized_event_hits"].to_numpy(dtype=float)
    exact_bin_hits = ledger["exact_bin_event_hits"].to_numpy(dtype=float)
    if (
        np.any(event_hits > qualified_events)
        or np.any(correct_side_hits > event_hits)
        or np.any(within_tolerance_hits > event_hits)
        or np.any(localized_hits > correct_side_hits)
        or np.any(localized_hits > within_tolerance_hits)
        or np.any(exact_bin_hits > within_tolerance_hits)
        or np.any(exact_bin_hits > event_hits)
    ):
        raise ArtifactValidationError(
            "car contribution ledger violates the prespecified event-hit hierarchy"
        )
    if not np.array_equal(
        ledger["least_favorable_false_proposals"].to_numpy(dtype=int),
        (
            ledger["false_proposals"].to_numpy(dtype=int)
            + ledger["unresolved_proposals"].to_numpy(dtype=int)
        ),
    ):
        raise ArtifactValidationError(
            "car contribution ledger violates the least-favorable false-count identity"
        )
    full_keys = [*unit_columns, *grid_columns]
    if ledger.loc[:, full_keys].isna().any(axis=None) or ledger.duplicated(full_keys).any():
        raise ArtifactValidationError("car contribution ledger has duplicate or null grid keys")
    grid_sizes = ledger.groupby(grid_columns, sort=True).size()
    expected_grid = {
        (method, lead, budget)
        for method in REGISTERED_METHODS
        for lead in LEAD_GRID
        for budget in BUDGET_GRID
    }
    observed_grid = {
        (str(method), float(lead), float(budget))
        for method, lead, budget in grid_sizes.index
    }
    pooled = operating.copy()
    pooled["required_lead_s"] = [
        _registered_grid_value(value, LEAD_GRID, "pooled operating lead")
        for value in pooled["required_lead_s"]
    ]
    pooled["false_budget_per_hour"] = [
        _registered_grid_value(value, BUDGET_GRID, "pooled operating false budget")
        for value in pooled["false_budget_per_hour"]
    ]
    pooled_grid = {
        (str(row.method), float(row.required_lead_s), float(row.false_budget_per_hour))
        for row in pooled.itertuples(index=False)
    }
    if len(pooled) != len(expected_grid) or pooled_grid != expected_grid:
        raise ArtifactValidationError("pooled operating table grid is incomplete or duplicated")
    statuses = pooled["operating_point_status"].astype(str)
    allowed_statuses = {"estimable", "not_estimable_insufficient_exposure"}
    if not (pooled["scope"].astype(str) == "pooled_four_circuit_heldout").all() or not set(
        statuses
    ).issubset(allowed_statuses):
        raise ArtifactValidationError("pooled operating table has invalid scope or status")
    estimable_rows = pooled.loc[statuses == "estimable"]
    estimable_grid = {
        (str(row.method), float(row.required_lead_s), float(row.false_budget_per_hour))
        for row in estimable_rows.itertuples(index=False)
    }
    primary_grid = {
        (method, lead, PRIMARY_BUDGET)
        for method in REGISTERED_METHODS
        for lead in LEAD_GRID
    }
    if not primary_grid.issubset(estimable_grid):
        raise ArtifactValidationError("pooled primary operating grid is not fully estimable")
    expected_estimable_grid = {
        (method, lead, budget)
        for method in REGISTERED_METHODS
        for lead in LEAD_GRID
        for budget in (2.0, 5.0, 10.0)
    }
    if estimable_grid != expected_estimable_grid:
        raise ArtifactValidationError(
            "pooled operating estimability grid differs from frozen calibration exposure"
        )
    if observed_grid != estimable_grid or len(grid_sizes) != len(estimable_grid):
        raise ArtifactValidationError(
            "car contribution ledger grid differs from estimable pooled operating rows"
        )
    if set(grid_sizes.astype(int)) != {58}:
        raise ArtifactValidationError(
            "car contribution ledger does not contain 58 car sessions per estimable row"
        )
    if "circuit_count" not in pooled or not (
        pd.to_numeric(pooled["circuit_count"], errors="coerce") == 4
    ).all():
        raise ArtifactValidationError("pooled operating table has invalid circuit counts")
    not_estimable = pooled.loc[statuses == "not_estimable_insufficient_exposure"]
    ne_numeric_columns = [
        *count_columns,
        "exposure_hours",
        "localized_event_recall",
        "event_recall",
        "correct_side_event_recall",
        "within_segment_tolerance_event_recall",
        "exact_bin_event_recall",
        "false_proposals_per_hour",
        "false_proposals_per_hour_upper_95",
    ]
    if not not_estimable.loc[:, ne_numeric_columns].isna().all(axis=None):
        raise ArtifactValidationError("pooled N/E operating rows contain outcome values")
    expected_ne_reason = "false_budget_per_hour_times_calibration_exposure_hours_below_1"
    for raw_reason in not_estimable["estimability_reason"]:
        try:
            reasons = json.loads(str(raw_reason))
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError("pooled N/E estimability reason is not JSON") from exc
        if not isinstance(reasons, dict) or set(reasons) != set(REGISTERED_CIRCUITS):
            raise ArtifactValidationError(
                "pooled N/E estimability reason lacks exact circuit provenance"
            )
        observed_nonestimable = False
        for circuit, reason in reasons.items():
            if not isinstance(reason, dict):
                raise ArtifactValidationError(
                    f"pooled N/E estimability reason for {circuit} is not an object"
                )
            status = str(reason.get("status", ""))
            detail = str(reason.get("reason", ""))
            if status == "not_estimable_insufficient_exposure":
                observed_nonestimable = True
                if detail != expected_ne_reason:
                    raise ArtifactValidationError(
                        f"pooled N/E estimability reason differs for {circuit}"
                    )
            elif status == "estimable":
                if detail.strip().lower() not in {"", "nan"}:
                    raise ArtifactValidationError(
                        f"pooled estimable fold has an N/E reason for {circuit}"
                    )
            else:
                raise ArtifactValidationError(
                    f"pooled N/E estimability status differs for {circuit}"
                )
        if not observed_nonestimable:
            raise ArtifactValidationError("pooled N/E row has no non-estimable circuit")

    aggregate = (
        ledger.groupby(grid_columns, sort=True, as_index=False)[[*count_columns, "exposure_hours"]]
        .sum()
        .set_index(grid_columns)
    )
    pooled_indexed = pooled.set_index(grid_columns)
    for key in sorted(estimable_grid):
        source = aggregate.loc[key]
        reported = pooled_indexed.loc[key]
        for column in count_columns:
            if int(source[column]) != int(reported[column]):
                raise ArtifactValidationError(
                    f"pooled operating {column} disagrees with contribution ledger at {key}"
                )
        source_exposure = float(source["exposure_hours"])
        if not _close(reported["exposure_hours"], source_exposure):
            raise ArtifactValidationError(
                f"pooled operating exposure disagrees with contribution ledger at {key}"
            )
        events = int(source["qualified_events"])
        recall_checks = {
            "localized_event_recall": int(source["localized_event_hits"]) / events,
            "event_recall": int(source["event_hits"]) / events,
            "correct_side_event_recall": int(source["correct_side_event_hits"]) / events,
            "within_segment_tolerance_event_recall": int(
                source["within_segment_tolerance_event_hits"]
            )
            / events,
            "exact_bin_event_recall": int(source["exact_bin_event_hits"]) / events,
            "false_proposals_per_hour": int(source["false_proposals"]) / source_exposure,
        }
        if any(not _close(reported[column], value) for column, value in recall_checks.items()):
            raise ArtifactValidationError(
                f"pooled operating rates disagree with contribution ledger at {key}"
            )
        false_count = int(source["false_proposals"])
        expected_upper = (
            0.5 * float(chi2.ppf(0.95, 2.0 * (false_count + 1))) / source_exposure
        )
        if not _close(reported["false_proposals_per_hour_upper_95"], expected_upper):
            raise ArtifactValidationError(
                f"pooled operating Poisson upper limit disagrees with counts at {key}"
            )

    for fold in folds:
        circuit_rows = ledger.loc[
            (ledger["circuit"].astype(str) == fold.circuit)
            & np.isclose(ledger["false_budget_per_hour"], PRIMARY_BUDGET)
        ]
        for method in (BRACE_METHOD, TWIN_METHOD):
            method_rows = circuit_rows.loc[circuit_rows["method"].astype(str) == method]
            summaries: list[dict[str, object]] = []
            for lead, group in method_rows.groupby("required_lead_s", sort=True):
                events = int(group["qualified_events"].sum())
                group_exposure = float(group["exposure_hours"].sum())
                summaries.append(
                    {
                        "method": method,
                        "required_lead_s": float(lead),
                        "false_budget_per_hour": PRIMARY_BUDGET,
                        "operating_point_status": "estimable",
                        "localized_event_recall": int(group["localized_event_hits"].sum())
                        / events,
                        "false_proposals_per_hour": int(group["false_proposals"].sum())
                        / group_exposure,
                    }
                )
            contribution_lstar = _fold_lstar(pd.DataFrame(summaries), method)
            fold_lstar = _fold_lstar(fold.metrics, method)
            if not _close(contribution_lstar, fold_lstar):
                raise ArtifactValidationError(
                    f"fold {fold.circuit} {method} L-star disagrees with contribution ledger"
                )


def _validate_delay_results(
    delay_table: pd.DataFrame,
    contributions: pd.DataFrame,
    operating: pd.DataFrame,
    primary_contributions: pd.DataFrame,
) -> None:
    _require_columns(
        delay_table,
        {
            "synthetic_delay_ms",
            "brace_method",
            "comparator_method",
            "l_star_brace_s",
            "l_star_comparator_s",
            "delta_l_star_s",
            "least_favorable_l_star_brace_s",
            "least_favorable_l_star_comparator_s",
            "least_favorable_delta_l_star_s",
            "false_budget_per_hour",
            "minimum_recall",
            "models_refit",
            "thresholds_refit",
        },
        "synthetic-delay L-star table",
    )
    delays = pd.to_numeric(delay_table["synthetic_delay_ms"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        len(delay_table) != 4
        or not np.isfinite(delays).all()
        or not np.equal(delays, np.floor(delays)).all()
        or set(delays.astype(int)) != {0, 40, 80, 160}
    ):
        raise ArtifactValidationError("synthetic-delay L-star grid is incomplete or duplicated")
    if (
        not (delay_table["brace_method"].astype(str) == BRACE_METHOD).all()
        or not (delay_table["comparator_method"].astype(str) == TWIN_METHOD).all()
        or not np.isclose(
            pd.to_numeric(delay_table["false_budget_per_hour"], errors="coerce"),
            PRIMARY_BUDGET,
            atol=1e-12,
            rtol=0.0,
        ).all()
        or not np.isclose(
            pd.to_numeric(delay_table["minimum_recall"], errors="coerce"),
            MINIMUM_RECALL,
            atol=1e-12,
            rtol=0.0,
        ).all()
        or not (delay_table["models_refit"].astype(str).str.lower() == "false").all()
        or not (delay_table["thresholds_refit"].astype(str).str.lower() == "false").all()
    ):
        raise ArtifactValidationError("synthetic-delay estimand or no-refit identity is invalid")
    lstar_columns = (
        "l_star_brace_s",
        "l_star_comparator_s",
        "least_favorable_l_star_brace_s",
        "least_favorable_l_star_comparator_s",
    )
    allowed_lstars = (0.0, *LEAD_GRID)
    normalized_lstars: dict[str, np.ndarray] = {}
    for column in lstar_columns:
        normalized_lstars[column] = np.asarray(
            [
                _registered_grid_value(value, allowed_lstars, f"synthetic-delay {column}")
                for value in delay_table[column]
            ],
            dtype=float,
        )
    delta = pd.to_numeric(delay_table["delta_l_star_s"], errors="coerce").to_numpy(
        dtype=float
    )
    least_favorable_delta = pd.to_numeric(
        delay_table["least_favorable_delta_l_star_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(delta).all()
        or not np.isfinite(least_favorable_delta).all()
        or not np.allclose(
            delta,
            normalized_lstars["l_star_brace_s"]
            - normalized_lstars["l_star_comparator_s"],
            atol=1e-12,
            rtol=0.0,
        )
        or not np.allclose(
            least_favorable_delta,
            normalized_lstars["least_favorable_l_star_brace_s"]
            - normalized_lstars["least_favorable_l_star_comparator_s"],
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise ArtifactValidationError("synthetic-delay delta identity is invalid")

    unit_columns = ["circuit", "source_session_id", "car_id"]
    _require_columns(
        contributions,
        {
            *unit_columns,
            "fold_test_circuit",
            "synthetic_delay_ms",
            "method",
            "required_lead_s",
            "false_budget_per_hour",
            "localized_event_hits",
            "qualified_events",
            "false_proposals",
            "exposure_hours",
            "fixed_model_and_threshold",
            "operating_point_status",
        },
        "synthetic-delay contribution table",
    )
    ledger = contributions.loc[
        contributions["method"].astype(str).isin((BRACE_METHOD, TWIN_METHOD))
        & np.isclose(
            pd.to_numeric(contributions["false_budget_per_hour"], errors="coerce"),
            PRIMARY_BUDGET,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()
    delay_count_columns = [
        "localized_event_hits",
        "qualified_events",
        "false_proposals",
    ]
    numeric_counts = ledger[delay_count_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    exposure = pd.to_numeric(ledger["exposure_hours"], errors="coerce")
    if (
        not np.isfinite(numeric_counts.to_numpy(dtype=float)).all()
        or not np.equal(
            numeric_counts.to_numpy(dtype=float),
            np.floor(numeric_counts.to_numpy(dtype=float)),
        ).all()
        or (numeric_counts < 0.0).any(axis=None)
        or not np.isfinite(exposure.to_numpy(dtype=float)).all()
        or (exposure <= 0.0).any()
        or (
            numeric_counts["localized_event_hits"]
            > numeric_counts["qualified_events"]
        ).any()
    ):
        raise ArtifactValidationError(
            "synthetic-delay contribution counts or exposure are invalid"
        )
    ledger.loc[:, delay_count_columns] = numeric_counts.to_numpy(dtype=np.int64)
    ledger.loc[:, "exposure_hours"] = exposure.to_numpy(dtype=float)
    ledger["required_lead_s"] = [
        _registered_grid_value(value, LEAD_GRID, "synthetic-delay contribution lead")
        for value in ledger["required_lead_s"]
    ]
    ledger_delays = pd.to_numeric(ledger["synthetic_delay_ms"], errors="coerce")
    if set(ledger_delays.astype(int)) != {0, 40, 80, 160} or not np.equal(
        ledger_delays, np.floor(ledger_delays)
    ).all():
        raise ArtifactValidationError("synthetic-delay contribution delay grid is invalid")
    ledger["synthetic_delay_ms"] = ledger_delays.astype(int)
    full_keys = [*unit_columns, "synthetic_delay_ms", "method", "required_lead_s"]
    if ledger.loc[:, full_keys].isna().any(axis=None) or ledger.duplicated(full_keys).any():
        raise ArtifactValidationError("synthetic-delay contribution keys are invalid")
    expected_cells = 4 * 2 * len(LEAD_GRID)
    cell_sizes = ledger.groupby(
        ["synthetic_delay_ms", "method", "required_lead_s"], sort=True
    ).size()
    if len(cell_sizes) != expected_cells or set(cell_sizes.astype(int)) != {58}:
        raise ArtifactValidationError("synthetic-delay contribution grid is incomplete")
    reference_denominators = (
        primary_contributions.groupby(unit_columns, sort=True)[
            ["qualified_events", "exposure_hours"]
        ]
        .first()
        .sort_index()
    )
    reference_units = set(reference_denominators.index)
    for _, group in ledger.groupby(
        ["synthetic_delay_ms", "method", "required_lead_s"],
        sort=True,
    ):
        units = set(group.set_index(unit_columns).index)
        if units != reference_units:
            raise ArtifactValidationError(
                "synthetic-delay contribution cell differs from authoritative car units"
            )
    delay_denominator_variation = ledger.groupby(unit_columns, sort=True)[
        ["qualified_events", "exposure_hours"]
    ].nunique(dropna=False)
    if (delay_denominator_variation > 1).any(axis=None):
        raise ArtifactValidationError(
            "synthetic-delay contribution denominators vary across cells"
        )
    delay_denominators = (
        ledger.groupby(unit_columns, sort=True)[["qualified_events", "exposure_hours"]]
        .first()
        .sort_index()
    )
    if not np.array_equal(
        delay_denominators["qualified_events"].to_numpy(dtype=int),
        reference_denominators["qualified_events"].to_numpy(dtype=int),
    ) or not np.allclose(
        delay_denominators["exposure_hours"].to_numpy(dtype=float),
        reference_denominators["exposure_hours"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ArtifactValidationError(
            "synthetic-delay contribution denominators differ from primary ledger"
        )
    fixed = ledger["fixed_model_and_threshold"].astype(str).str.lower().isin(("true", "1"))
    if (
        not fixed.all()
        or not (ledger["operating_point_status"].astype(str) == "estimable").all()
        or not (
            ledger["fold_test_circuit"].astype(str).to_numpy()
            == ledger["circuit"].astype(str).to_numpy()
        ).all()
    ):
        raise ArtifactValidationError("synthetic-delay contribution identity is invalid")
    table_by_delay = delay_table.set_index("synthetic_delay_ms")
    for delay in (0, 40, 80, 160):
        delay_rows = ledger.loc[ledger["synthetic_delay_ms"] == delay]
        reported = table_by_delay.loc[delay]
        for method, column in (
            (BRACE_METHOD, "l_star_brace_s"),
            (TWIN_METHOD, "l_star_comparator_s"),
        ):
            summaries: list[dict[str, object]] = []
            method_rows = delay_rows.loc[delay_rows["method"].astype(str) == method]
            for lead, group in method_rows.groupby("required_lead_s", sort=True):
                events = int(group["qualified_events"].sum())
                exposure = float(group["exposure_hours"].sum())
                summaries.append(
                    {
                        "method": method,
                        "required_lead_s": float(lead),
                        "false_budget_per_hour": PRIMARY_BUDGET,
                        "operating_point_status": "estimable",
                        "localized_event_recall": int(group["localized_event_hits"].sum())
                        / events,
                        "false_proposals_per_hour": int(group["false_proposals"].sum())
                        / exposure,
                    }
                )
            recomputed = _fold_lstar(pd.DataFrame(summaries), method)
            if not _close(reported[column], recomputed):
                raise ArtifactValidationError(
                    f"synthetic-delay {delay} ms {method} L-star disagrees with contribution ledger"
                )
    zero_delay = table_by_delay.loc[0]
    primary_brace = _l_star(_method_budget_rows(operating, BRACE_METHOD, PRIMARY_BUDGET))
    primary_twin = _l_star(_method_budget_rows(operating, TWIN_METHOD, PRIMARY_BUDGET))
    if not _close(zero_delay["l_star_brace_s"], primary_brace) or not _close(
        zero_delay["l_star_comparator_s"], primary_twin
    ):
        raise ArtifactValidationError(
            "zero-delay L-star differs from the authenticated primary operating result"
        )


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ArtifactValidationError(f"{label} is not finite: {value!r}")
    return result


def _interval(summary: Mapping[str, Any], key: str) -> tuple[float, float]:
    raw = summary.get(key)
    if not isinstance(raw, list) or len(raw) != 2:
        raise ArtifactValidationError(f"bootstrap summary has no two-sided {key}")
    low, high = (_number(raw[0], key), _number(raw[1], key))
    if low > high:
        raise ArtifactValidationError(f"bootstrap interval is reversed: {key}")
    return low, high


def _close(left: object, right: float, *, atol: float = 1e-9) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _registered_primary_rows(operating: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        operating,
        {
            "method",
            "required_lead_s",
            "false_budget_per_hour",
            "operating_point_status",
            "localized_event_recall",
            "false_proposals_per_hour",
        },
        "pooled operating table",
    )
    rows = operating.loc[
        np.isclose(
            pd.to_numeric(operating["false_budget_per_hour"], errors="coerce"),
            PRIMARY_BUDGET,
            atol=1e-12,
            rtol=0.0,
        )
        & operating["method"].astype(str).isin(REGISTERED_METHODS)
    ].copy()
    keys = {(str(row.method), float(row.required_lead_s)) for row in rows.itertuples(index=False)}
    expected = {(method, lead) for method in REGISTERED_METHODS for lead in LEAD_GRID}
    if len(rows) != len(expected) or keys != expected:
        raise ArtifactValidationError("pooled primary operating grid is incomplete or duplicated")
    if not (rows["operating_point_status"].astype(str) == "estimable").all():
        raise ArtifactValidationError("pooled primary operating grid contains a non-estimable row")
    return rows


def _method_budget_rows(operating: pd.DataFrame, method: str, budget: float) -> pd.DataFrame:
    rows = operating.loc[
        (operating["method"].astype(str) == method)
        & np.isclose(
            pd.to_numeric(operating["false_budget_per_hour"], errors="coerce"),
            budget,
            atol=1e-12,
            rtol=0.0,
        )
        & (operating["operating_point_status"].astype(str) == "estimable")
    ].copy()
    return rows


def _l_star(rows: pd.DataFrame) -> float:
    passing = rows.loc[
        (pd.to_numeric(rows["localized_event_recall"], errors="coerce") >= MINIMUM_RECALL - 1e-12)
        & (
            pd.to_numeric(rows["false_proposals_per_hour"], errors="coerce")
            <= pd.to_numeric(rows["false_budget_per_hour"], errors="coerce") + 1e-12
        )
    ]
    return 0.0 if passing.empty else float(passing["required_lead_s"].max())


def _row_at_lead(rows: pd.DataFrame, lead: float) -> pd.Series | None:
    selected = rows.loc[
        np.isclose(
            pd.to_numeric(rows["required_lead_s"], errors="coerce"),
            lead,
            atol=1e-12,
            rtol=0.0,
        )
    ]
    if len(selected) != 1:
        return None
    return selected.iloc[0]


def _validate_primary_points(evidence: Evidence) -> tuple[float, float, float]:
    primary_rows = _registered_primary_rows(evidence.operating)
    brace = _l_star(_method_budget_rows(primary_rows, BRACE_METHOD, PRIMARY_BUDGET))
    twin = _l_star(_method_budget_rows(primary_rows, TWIN_METHOD, PRIMARY_BUDGET))
    delta = brace - twin
    pooled = evidence.l_star_sensitivity.loc[
        evidence.l_star_sensitivity["analysis"].astype(str) == "primary_unresolved_censored"
    ]
    if len(pooled) != 1:
        raise ArtifactValidationError("pooled L-star sensitivity table lacks one primary row")
    row = pooled.iloc[0]
    checks = {
        "l_star_brace_s": brace,
        "l_star_comparator_s": twin,
        "delta_l_star_s": delta,
    }
    if any(not _close(row.get(key), value) for key, value in checks.items()):
        raise ArtifactValidationError("pooled L-star table disagrees with pooled operating metrics")
    summary_checks = {
        "l_star_brace": brace,
        "l_star_comparator": twin,
        "delta_l_star": delta,
    }
    if any(
        not _close(evidence.bootstrap_summary.get(key), value)
        for key, value in summary_checks.items()
    ):
        raise ArtifactValidationError(
            "bootstrap point estimates disagree with pooled held-out data"
        )
    return brace, twin, delta


def _fmt(value: object, digits: int = 3) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return "N/E"
    return f"{numeric:.{digits}f}"


def _fmt_lead(value: object) -> str:
    return _fmt(value, 2)


def _fmt_count(value: object) -> str:
    return f"{int(round(float(value))):,}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(value) for value in headers) + " |"
    divider = "|" + "|".join(":--" if index == 0 else "--:" for index in range(len(headers))) + "|"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _with_caption(table: str, caption: str, identifier: str) -> str:
    return f"{table}\n\nTable: {caption} {{#{identifier}}}"


def _frontier(evidence: Evidence) -> pd.DataFrame:
    budgets = sorted(
        float(value) for value in evidence.operating["false_budget_per_hour"].dropna().unique()
    )
    rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in REGISTERED_METHODS:
            candidates = _method_budget_rows(evidence.operating, method, budget)
            if candidates.empty:
                rows.append({"budget": budget, "method": method, "l_star_s": np.nan})
                continue
            rows.append({"budget": budget, "method": method, "l_star_s": _l_star(candidates)})
    return pd.DataFrame(rows)


def _primary_frontier_table(evidence: Evidence, frontier: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    omitted_budgets: set[float] = set()
    for budget in sorted(frontier["budget"].unique()):
        for method in (BRACE_METHOD, TWIN_METHOD):
            lstar_row = frontier.loc[
                np.isclose(frontier["budget"], budget) & (frontier["method"] == method)
            ]
            lstar = float(lstar_row.iloc[0]["l_star_s"])
            operating = _method_budget_rows(evidence.operating, method, float(budget))
            source = _row_at_lead(operating, max(lstar, LEAD_GRID[0]))
            if source is None:
                omitted_budgets.add(float(budget))
                continue
            rows.append(
                [
                    _fmt(budget, 1),
                    METHOD_LABELS[method],
                    _fmt_lead(lstar),
                    _fmt(source["localized_event_recall"]),
                    _fmt_count(source.get("false_proposals", np.nan)),
                    _fmt(source["false_proposals_per_hour"], 2),
                    _fmt(source["false_proposals_per_hour_upper_95"], 2),
                ]
            )
    table = _markdown_table(
        (
            "Budget / h",
            "Method",
            "$L^*$ (s)",
            "Localized recall",
            "False proposals",
            "False rate / h",
            "Upper 95% / h",
        ),
        rows,
    )
    omitted_text = ", ".join(_fmt(value, 1) for value in sorted(omitted_budgets))
    estimability_note = (
        f" Budgets {omitted_text}/h were not estimable under the frozen exposure rule and are "
        "omitted here."
        if omitted_text
        else ""
    )
    return _with_caption(
        table,
        "Held-out BRACE-versus-twin warning-frontier summary over 6.317055 eligible simulated car-hours. Rows use the qualifying $L^*$ operating point when positive and the shortest 0.25 s nonqualifying fallback when $L^*=0$. Raw false counts, rates, and one-sided Poisson-model 95% upper limits are shown."
        + estimability_note
        + " The authenticated full 120-row table is included in the reproducibility package as `pooled-heldout-operating-metrics.csv`.",
        "tbl:primary-frontier",
    )


def _probability_rows(evidence: Evidence) -> pd.DataFrame:
    _require_columns(
        evidence.probability,
        {
            "scope",
            "circuit",
            "method",
            "horizon_s",
            "n",
            "event_count",
            "brier_score",
            "reference_brier_score",
            "brier_skill_score",
            "log_score",
            "average_precision_stepwise",
            "calibration_intercept",
            "calibration_slope",
            "ece_10_equal_width",
        },
        "probability metric table",
    )
    rows = evidence.probability.loc[
        (evidence.probability["scope"].astype(str) == "pooled")
        & np.isclose(
            pd.to_numeric(evidence.probability["horizon_s"], errors="coerce"),
            PRIMARY_HORIZON,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()
    if len(rows) != len(REGISTERED_METHODS) or set(rows["method"].astype(str)) != set(
        REGISTERED_METHODS
    ):
        raise ArtifactValidationError("pooled 1.50-second probability metric grid is incomplete")
    if not (rows["circuit"].astype(str) == "all").all():
        raise ArtifactValidationError("pooled probability metric scope has a non-pooled circuit")
    counts = rows[["n", "event_count"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    if (
        not np.isfinite(counts).all()
        or not np.equal(counts, np.floor(counts)).all()
        or np.any(counts[:, 0] <= 0.0)
        or np.any(counts[:, 1] < 0.0)
        or np.any(counts[:, 1] > counts[:, 0])
    ):
        raise ArtifactValidationError("pooled probability metric counts are invalid")
    metric_columns = [
        "brier_score",
        "reference_brier_score",
        "brier_skill_score",
        "log_score",
        "average_precision_stepwise",
        "calibration_intercept",
        "calibration_slope",
        "ece_10_equal_width",
    ]
    metrics = rows.loc[:, metric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(metrics.to_numpy()).all():
        raise ArtifactValidationError("pooled probability metric values are not finite")
    unit_interval_columns = [
        "brier_score",
        "reference_brier_score",
        "average_precision_stepwise",
        "ece_10_equal_width",
    ]
    unit_values = metrics.loc[:, unit_interval_columns].to_numpy(dtype=float)
    if (
        np.any(unit_values < 0.0)
        or np.any(unit_values > 1.0)
        or np.any(metrics["reference_brier_score"].to_numpy(dtype=float) <= 0.0)
        or np.any(metrics["log_score"].to_numpy(dtype=float) < 0.0)
        or np.any(metrics["brier_skill_score"].to_numpy(dtype=float) > 1.0)
    ):
        raise ArtifactValidationError("pooled probability metric values leave valid ranges")
    expected_skill = 1.0 - (
        metrics["brier_score"].to_numpy(dtype=float)
        / metrics["reference_brier_score"].to_numpy(dtype=float)
    )
    if not np.allclose(
        metrics["brier_skill_score"].to_numpy(dtype=float),
        expected_skill,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ArtifactValidationError("pooled probability Brier skill identity is invalid")
    return rows.set_index("method").loc[list(REGISTERED_METHODS)].reset_index()


def _probability_table(rows: pd.DataFrame) -> str:
    total_counts = {int(value) for value in rows["n"]}
    positive_counts = {int(value) for value in rows["event_count"]}
    if len(total_counts) != 1 or len(positive_counts) != 1:
        raise ArtifactValidationError(
            "pooled probability methods do not share frame-horizon row counts"
        )
    total_count = next(iter(total_counts))
    positive_count = next(iter(positive_counts))
    values = [
        [
            METHOD_LABELS[str(row.method)],
            _fmt(row.brier_score, 6),
            _fmt(row.brier_skill_score, 6),
            _fmt(row.log_score, 6),
            _fmt(row.average_precision_stepwise, 6),
            _fmt(row.calibration_intercept, 6),
            _fmt(row.calibration_slope, 6),
            _fmt(row.ece_10_equal_width, 6),
        ]
        for row in rows.itertuples(index=False)
    ]
    table = _markdown_table(
        (
            "Method",
            "Brier",
            "Brier skill",
            "Log score",
            "Stepwise AP",
            "Cal. intercept",
            "Cal. slope",
            "ECE",
        ),
        values,
    )
    return _with_caption(
        table,
        f"Pooled held-out frame-horizon probability quality at 1.50 s over {total_count:,} score rows. The {positive_count:,} positive frame-horizon rows are score times followed by an excursion within 1.50 s, not unique excursion events. Lower Brier, log score, and ECE are better; higher Brier skill and stepwise average precision are better.",
        "tbl:probability-metrics",
    )


def _fold_lstar(metrics: pd.DataFrame, method: str) -> float:
    _require_columns(
        metrics,
        {
            "method",
            "required_lead_s",
            "false_budget_per_hour",
            "operating_point_status",
            "localized_event_recall",
            "false_proposals_per_hour",
        },
        "fold operating table",
    )
    rows = _method_budget_rows(metrics, method, PRIMARY_BUDGET)
    leads = pd.to_numeric(rows["required_lead_s"], errors="coerce").to_numpy(dtype=float)
    if (
        len(rows) != len(LEAD_GRID)
        or len(np.unique(leads)) != len(LEAD_GRID)
        or not np.allclose(np.sort(leads), np.asarray(LEAD_GRID), atol=1e-12, rtol=0.0)
    ):
        raise ArtifactValidationError(f"fold grid is incomplete for {method}")
    return _l_star(rows)


def _circuit_table(evidence: Evidence) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for fold in evidence.folds:
        brace = _fold_lstar(fold.metrics, BRACE_METHOD)
        twin = _fold_lstar(fold.metrics, TWIN_METHOD)
        rows.append({"circuit": fold.circuit, "brace": brace, "twin": twin, "delta": brace - twin})
    table_data = pd.DataFrame(rows).sort_values("circuit", kind="stable")
    table = _markdown_table(
        ("Held-out circuit", "BRACE $L^*$ (s)", "Twin $L^*$ (s)", r"$\Delta L^*$ (s)"),
        [
            [row.circuit, _fmt_lead(row.brace), _fmt_lead(row.twin), _fmt_lead(row.delta)]
            for row in table_data.itertuples(index=False)
        ],
    )
    return (
        _with_caption(
            table,
            "Circuit-specific primary warning-time results at two false proposals per eligible simulated car-hour and at least 0.50 localized event recall.",
            "tbl:circuit-results",
        ),
        table_data,
    )


def _least_favorable(evidence: Evidence) -> tuple[float, float, float, tuple[float, float], str]:
    required_metadata = {
        "analysis",
        "false_count_column",
        "brace_method",
        "comparator_method",
        "l_star_brace_s",
        "l_star_comparator_s",
        "delta_l_star_s",
        "false_budget_per_hour",
        "minimum_recall",
        "models_refit",
        "thresholds_refit",
    }
    _require_columns(
        evidence.l_star_sensitivity,
        required_metadata,
        "pooled L-star sensitivity table",
    )
    row = evidence.l_star_sensitivity.loc[
        evidence.l_star_sensitivity["analysis"].astype(str)
        == "least_favorable_unresolved_counted_as_false"
    ]
    if len(row) != 1:
        raise ArtifactValidationError(
            "pooled L-star sensitivity table lacks one least-favorable row"
        )
    source = row.iloc[0]
    if (
        str(source["false_count_column"]) != "least_favorable_false_proposals"
        or str(source["brace_method"]) != BRACE_METHOD
        or str(source["comparator_method"]) != TWIN_METHOD
        or not _close(source["false_budget_per_hour"], PRIMARY_BUDGET)
        or not _close(source["minimum_recall"], MINIMUM_RECALL)
        or str(source["models_refit"]).lower() != "false"
        or str(source["thresholds_refit"]).lower() != "false"
    ):
        raise ArtifactValidationError("least-favorable sensitivity estimand is invalid")
    brace = _number(source["l_star_brace_s"], "least-favorable BRACE L-star")
    twin = _number(source["l_star_comparator_s"], "least-favorable twin L-star")
    delta = _number(source["delta_l_star_s"], "least-favorable delta L-star")
    recomputed: dict[str, float] = {}
    for method, label in ((BRACE_METHOD, "brace"), (TWIN_METHOD, "twin")):
        selected = evidence.contributions.loc[
            (evidence.contributions["method"].astype(str) == method)
            & np.isclose(
                pd.to_numeric(
                    evidence.contributions["false_budget_per_hour"],
                    errors="coerce",
                ),
                PRIMARY_BUDGET,
                atol=1e-12,
                rtol=0.0,
            )
        ]
        summaries: list[dict[str, object]] = []
        for lead, group in selected.groupby("required_lead_s", sort=True):
            events = int(group["qualified_events"].sum())
            exposure = float(group["exposure_hours"].sum())
            summaries.append(
                {
                    "method": method,
                    "required_lead_s": float(lead),
                    "false_budget_per_hour": PRIMARY_BUDGET,
                    "operating_point_status": "estimable",
                    "localized_event_recall": int(group["localized_event_hits"].sum())
                    / events,
                    "false_proposals_per_hour": int(
                        group["least_favorable_false_proposals"].sum()
                    )
                    / exposure,
                }
            )
        recomputed[label] = _fold_lstar(pd.DataFrame(summaries), method)
    if (
        not _close(brace, recomputed["brace"])
        or not _close(twin, recomputed["twin"])
        or not _close(delta, recomputed["brace"] - recomputed["twin"])
    ):
        raise ArtifactValidationError(
            "least-favorable L-star disagrees with authenticated contribution ledger"
        )
    sensitivity = evidence.bootstrap_summary.get("least_favorable_unresolved_counted_as_false")
    if not isinstance(sensitivity, dict):
        raise ArtifactValidationError("bootstrap summary lacks least-favorable sensitivity")
    if (
        sensitivity.get("false_count_column") != "least_favorable_false_proposals"
        or sensitivity.get("models_and_thresholds_refit") is not False
    ):
        raise ArtifactValidationError("least-favorable bootstrap estimand is invalid")
    if any(
        not _close(sensitivity.get(key), expected)
        for key, expected in {
            "l_star_brace": brace,
            "l_star_comparator": twin,
            "delta_l_star": delta,
        }.items()
    ):
        raise ArtifactValidationError("least-favorable bootstrap disagrees with pooled evidence")
    interval = _interval(sensitivity, "delta_l_star_percentile_95")
    table = _markdown_table(
        (
            "Rule",
            "BRACE $L^*$ (s)",
            "Twin $L^*$ (s)",
            r"$\Delta L^*$ (s)",
            "Paired 95% interval (s)",
        ),
        [
            [
                "Primary: unresolved censored",
                _fmt_lead(evidence.bootstrap_summary["l_star_brace"]),
                _fmt_lead(evidence.bootstrap_summary["l_star_comparator"]),
                _fmt_lead(evidence.bootstrap_summary["delta_l_star"]),
                f"{_fmt_lead(_interval(evidence.bootstrap_summary, 'delta_l_star_percentile_95')[0])} to {_fmt_lead(_interval(evidence.bootstrap_summary, 'delta_l_star_percentile_95')[1])}",
            ],
            [
                "Least favorable: unresolved false",
                _fmt_lead(brace),
                _fmt_lead(twin),
                _fmt_lead(delta),
                f"{_fmt_lead(interval[0])} to {_fmt_lead(interval[1])}",
            ],
        ],
    )
    return (
        brace,
        twin,
        delta,
        interval,
        _with_caption(
            table,
            "Primary and least-favorable unresolved-proposal analyses with fixed fitted models, calibrators, and thresholds.",
            "tbl:least-favorable",
        ),
    )


def _delay_table(evidence: Evidence) -> tuple[str, pd.Series]:
    required = {
        "synthetic_delay_ms",
        "l_star_brace_s",
        "l_star_comparator_s",
        "delta_l_star_s",
    }
    _require_columns(evidence.delay_l_star, required, "synthetic-delay L-star table")
    delays = evidence.delay_l_star.sort_values("synthetic_delay_ms", kind="stable")
    if set(delays["synthetic_delay_ms"].astype(int)) != {0, 40, 80, 160} or len(delays) != 4:
        raise ArtifactValidationError("synthetic-delay L-star grid is incomplete or duplicated")
    row160 = delays.loc[delays["synthetic_delay_ms"].astype(int) == 160].iloc[0]
    table = _markdown_table(
        (
            "Added delay (ms)",
            "BRACE $L^*$ (s)",
            "Twin $L^*$ (s)",
            r"$\Delta L^*$ (s)",
        ),
        [
            [
                int(row.synthetic_delay_ms),
                _fmt_lead(row.l_star_brace_s),
                _fmt_lead(row.l_star_comparator_s),
                _fmt_lead(row.delta_l_star_s),
            ]
            for row in delays.itertuples(index=False)
        ],
    )
    return (
        _with_caption(
            table,
            "Synthetic issue-delay sensitivity. Models, calibrators, and thresholds were not refitted.",
            "tbl:delay-results",
        ),
        row160,
    )


def _cohort_summary(
    build_manifest: Mapping[str, Any],
    contributions: pd.DataFrame,
) -> tuple[int, int, float, int, int, int, float]:
    unit_columns = ["circuit", "source_session_id", "car_id"]
    _require_columns(
        contributions,
        {*unit_columns, "qualified_events", "exposure_hours"},
        "car contribution table",
    )
    grouped = contributions.groupby(unit_columns, sort=True, dropna=False)
    denominator_counts = grouped[["qualified_events", "exposure_hours"]].nunique(dropna=False)
    if (denominator_counts > 1).any(axis=None):
        raise ArtifactValidationError(
            "car contribution denominators vary across method or lead rows"
        )
    units = grouped[["qualified_events", "exposure_hours"]].first().reset_index()
    counts = build_manifest["counts"]
    expected_car_count = int(counts["cars"])
    if len(units) != expected_car_count:
        raise ArtifactValidationError(
            f"car contribution ledger has {len(units)} units; expected {expected_car_count}"
        )
    expected_events = int(counts["qualified_excursions"])
    expected_exposure = float(counts["buffered_exposure_hours"])
    if int(round(float(units["qualified_events"].sum()))) != expected_events:
        raise ArtifactValidationError("car contribution event denominator differs from build")
    if not math.isclose(
        float(units["exposure_hours"].sum()),
        expected_exposure,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ArtifactValidationError("car contribution exposure differs from build")
    return (
        expected_car_count,
        expected_events,
        expected_exposure,
        int(counts["circuits"]),
        int(counts["raw_frames"]),
        int(counts["resampled_frames"]),
        float(counts["prebuffer_exposure_hours"]),
    )


def _runtime_summary(folds: Sequence[FoldEvidence]) -> tuple[float, float, int, float, str]:
    elapsed = 0.0
    score_seconds = 0.0
    score_rows = 0
    environments: list[str] = []
    for fold in folds:
        normalized = json.dumps(fold.environment, sort_keys=True, separators=(",", ":"))
        environments.append(normalized)
        for event in fold.runtime_events:
            seconds = _number(event.get("elapsed_seconds"), "runtime elapsed_seconds")
            if seconds < 0.0:
                raise ArtifactValidationError("runtime elapsed_seconds cannot be negative")
            elapsed += seconds
            if event.get("step") == "score_method":
                rows = int(_number(event.get("rows"), "score_method rows"))
                if rows < 0:
                    raise ArtifactValidationError("score_method rows cannot be negative")
                score_seconds += seconds
                score_rows += rows
    if len(set(environments)) != 1:
        raise ArtifactValidationError("fold runtime environments are not identical")
    if score_seconds <= 0.0 or score_rows <= 0:
        raise ArtifactValidationError("fold runtime logs contain no measured scoring throughput")
    environment = folds[0].environment
    platform_name = str(environment.get("platform", "unknown platform"))
    python_version = str(environment.get("python", "unknown Python")).split(" (")[0]
    packages = environment.get("package_versions")
    numpy_version = packages.get("numpy") if isinstance(packages, dict) else None
    environment_text = f"{platform_name} with Python {python_version}"
    if numpy_version:
        environment_text += f" and NumPy {numpy_version}"
    return elapsed / 3600.0, score_seconds, score_rows, score_rows / score_seconds, environment_text


def _monotonicity(evidence: Evidence) -> pd.Series:
    _require_columns(
        evidence.monotonicity,
        {"scope", "method", "score_rows", "violating_row_count", "violating_row_rate"},
        "horizon monotonicity table",
    )
    row = evidence.monotonicity.loc[
        (evidence.monotonicity["scope"].astype(str) == "pooled")
        & (evidence.monotonicity["method"].astype(str) == BRACE_METHOD)
    ]
    if len(row) != 1:
        raise ArtifactValidationError("monotonicity table lacks one pooled BRACE row")
    selected = row.iloc[0]
    score_rows = _number(selected["score_rows"], "monotonicity score_rows")
    violating_rows = _number(
        selected["violating_row_count"],
        "monotonicity violating_row_count",
    )
    rate = _number(selected["violating_row_rate"], "monotonicity violating_row_rate")
    if (
        not score_rows.is_integer()
        or not violating_rows.is_integer()
        or score_rows <= 0.0
        or violating_rows < 0.0
        or violating_rows > score_rows
        or rate < 0.0
        or rate > 1.0
        or not math.isclose(rate, violating_rows / score_rows, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ArtifactValidationError("monotonicity count/rate arithmetic is invalid")
    return selected


def _primary_row(evidence: Evidence, method: str, lead: float) -> tuple[pd.Series, bool]:
    rows = _method_budget_rows(evidence.operating, method, PRIMARY_BUDGET)
    qualifying = lead > 0.0
    row = _row_at_lead(rows, lead if qualifying else LEAD_GRID[0])
    if row is None:
        raise ArtifactValidationError(
            f"primary operating row missing for {method}/{lead if qualifying else LEAD_GRID[0]}"
        )
    return row, qualifying


def _conditional_text(
    delta: float,
    interval: tuple[float, float],
    *,
    all_delta_draws_zero: bool = False,
    both_policies_silent: bool = False,
) -> dict[str, str]:
    low, high = interval
    if (
        delta == 0.0
        and low == 0.0
        and high == 0.0
        and all_delta_draws_zero
        and both_policies_silent
    ):
        return {
            "evidence": (
                "Both selected policies were silent: no demonstrated gain under the prespecified "
                "gates. The [0,0] interval reflects endpoint degeneracy, not equivalence."
            ),
            "interpretation": (
                "Both selected policies were silent, so every paired bootstrap contrast was zero: "
                "no demonstrated gain under the prespecified gates. The [0,0] interval is not "
                "evidence of precise equivalence."
            ),
            "discussion": (
                "Both fitted policies were silent under the prespecified gates. Consequently, all "
                "10,000 paired bootstrap contrasts were zero and the percentile interval collapsed "
                "to [0,0]. This is a structural consequence of the selected no-proposal policies and "
                "the discrete endpoint, not evidence that the population difference is known "
                "precisely to be zero. The comparison covers complete fitted pipelines and does not "
                "isolate uncertainty propagation because calibrators and thresholds were separately "
                "fitted."
            ),
            "conclusion": (
                "Both policies were silent: no demonstrated gain under the prespecified gates. The "
                "degenerate [0,0] bootstrap interval does not establish equivalence."
            ),
        }
    sign_discordant = (low > 0.0 and delta <= 0.0) or (high < 0.0 and delta >= 0.0)
    if sign_discordant:
        interval_direction = "above" if low > 0.0 else "below"
        return {
            "evidence": (
                "The original point estimate and paired percentile interval had discordant "
                "directions, so no directional claim is made for the complete BRACE-versus-twin "
                "comparison. This does not isolate uncertainty because calibration and thresholds "
                "differed."
            ),
            "interpretation": (
                "The point estimate and paired percentile interval were directionally discordant; "
                "the prespecified analysis therefore supports no directional claim."
            ),
            "discussion": (
                f"The original point contrast was {_fmt_lead(delta)} s, while the conditional "
                f"paired percentile interval ({_fmt_lead(low)} to {_fmt_lead(high)} s) lay wholly "
                f"{interval_direction} zero. Because a percentile interval need not contain the "
                "original estimate, these directions are discordant and no directional claim is "
                "made. The comparison covers complete fitted pipelines and does not isolate "
                "uncertainty propagation because calibrators and thresholds were separately fitted."
            ),
            "conclusion": (
                "The point estimate and percentile interval were directionally discordant, so the "
                "experiment supports no directional claim between the complete pipelines."
            ),
        }
    if low > 0.0 and delta > 0.0:
        return {
            "evidence": (
                "In this fixed four-circuit cohort, the complete BRACE pipeline had longer "
                "localized warning than its matched twin; the paired interval excludes zero. "
                "This does not isolate uncertainty because calibration and thresholds differed."
            ),
            "interpretation": (
                "The paired interval excluded zero, so the prespecified analysis supports a "
                "positive warning-time difference for BRACE in these four simulated circuits."
            ),
            "discussion": (
                f"Under the prespecified gates, BRACE increased the grid-valued warning frontier by "
                f"{_fmt_lead(delta)} s relative to its matched deterministic twin, and the "
                f"conditional paired interval ({_fmt_lead(low)} to {_fmt_lead(high)} s) excluded "
                "zero. This supports a difference between the complete fitted pipelines on the "
                "observed simulated circuit set. It does not isolate uncertainty propagation: "
                "BRACE and the twin used separately fitted calibrators and thresholds. It also "
                "does not establish transport, real crash prediction, or intervention efficacy."
            ),
            "conclusion": (
                "The conditional interval excluded zero, supporting a warning-time difference "
                "between the complete BRACE and twin pipelines, not an isolated uncertainty effect."
            ),
        }
    if high < 0.0 and delta < 0.0:
        return {
            "evidence": (
                "In this fixed four-circuit cohort, the complete BRACE pipeline had shorter "
                "localized warning than its matched twin; the paired interval lies below zero. "
                "This does not isolate uncertainty because calibration and thresholds differed."
            ),
            "interpretation": (
                "The paired interval lay below zero, so BRACE performed worse than its matched "
                "deterministic twin on the prespecified warning frontier."
            ),
            "discussion": (
                f"Under the prespecified gates, BRACE reduced the grid-valued warning frontier by "
                f"{_fmt_lead(abs(delta))} s relative to its matched deterministic twin; the "
                f"conditional paired interval ({_fmt_lead(low)} to {_fmt_lead(high)} s) lay below "
                "zero. This compares complete pipelines and cannot isolate uncertainty because "
                "their calibrators and thresholds were separately fitted."
            ),
            "conclusion": (
                "The conditional interval lay below zero for the complete BRACE-versus-twin "
                "comparison; it does not identify an isolated uncertainty effect."
            ),
        }
    return {
        "evidence": (
            "In this fixed four-circuit cohort, the paired interval included zero: no demonstrated "
            "gain under the prespecified gates. "
            "This does not isolate uncertainty because calibration and thresholds differed."
        ),
        "interpretation": (
            "The paired interval included zero: no demonstrated gain under the prespecified gates."
        ),
        "discussion": (
            f"The point contrast was {_fmt_lead(delta)} s, but the conditional paired interval "
            f"({_fmt_lead(low)} to {_fmt_lead(high)} s) included zero. The prespecified analysis "
            "therefore did not demonstrate a warning-time difference between the complete fitted "
            "pipelines in these four simulated circuits. Because calibrators and thresholds were "
            "separately fitted, the contrast does not isolate uncertainty propagation."
        ),
        "conclusion": (
            "The conditional interval included zero: no demonstrated gain under the prespecified "
            "gates."
        ),
    }


def _figure_markdown(stem: str, caption: str) -> str:
    return f"![{caption}](../analysis-output/figures/{stem}.pdf)"


def _comparison_word(left: float, right: float, *, lower_is_better: bool) -> str:
    if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        return "equal"
    left_is_better = left < right if lower_is_better else left > right
    if lower_is_better:
        return "lower" if left_is_better else "higher"
    return "higher" if left_is_better else "lower"


def _probability_result_text(
    rows: pd.DataFrame,
    brace: pd.Series,
    twin: pd.Series,
) -> tuple[str, str]:
    brace_ap = float(brace["average_precision_stepwise"])
    twin_ap = float(twin["average_precision_stepwise"])
    brace_log = float(brace["log_score"])
    twin_log = float(twin["log_score"])
    brace_brier = float(brace["brier_score"])
    twin_brier = float(twin["brier_score"])
    comparison = (
        "At 1.50 s, BRACE had "
        f"{_comparison_word(brace_ap, twin_ap, lower_is_better=False)} stepwise average "
        f"precision ({brace_ap:.4f} versus {twin_ap:.4f}) and "
        f"{_comparison_word(brace_log, twin_log, lower_is_better=True)} logarithmic score "
        f"({brace_log:.4f} versus {twin_log:.4f}); its Brier score was "
        f"{_comparison_word(brace_brier, twin_brier, lower_is_better=True)} "
        f"({brace_brier:.4f} versus {twin_brier:.4f})."
    )

    best_ap_method = str(rows.loc[rows["average_precision_stepwise"].idxmax(), "method"])
    best_log_method = str(rows.loc[rows["log_score"].idxmin(), "method"])
    worst_brier_method = str(rows.loc[rows["brier_score"].idxmax(), "method"])
    skill = pd.to_numeric(rows["brier_skill_score"], errors="coerce").to_numpy(dtype=float)
    if np.all(skill < 0.0):
        skill_clause = "All six methods had negative pooled Brier skill."
    else:
        nonnegative = int(np.sum(skill >= 0.0))
        skill_clause = (
            "All six methods had nonnegative pooled Brier skill."
            if nonnegative == len(REGISTERED_METHODS)
            else f"{nonnegative} of the six methods had nonnegative pooled Brier skill."
        )
    all_methods = (
        f"{METHOD_LABELS[best_ap_method]} had the highest pooled stepwise average precision; "
        f"{METHOD_LABELS[best_log_method]} had the lowest logarithmic score; and "
        f"{METHOD_LABELS[worst_brier_method]} had the highest Brier score. {skill_clause}"
    )
    return comparison, all_methods


def _policy_result_text(evidence: Evidence) -> tuple[str, bool]:
    selected_rows: list[pd.DataFrame] = []
    for fold in evidence.folds:
        table = fold.calibration_thresholds
        budgets = pd.to_numeric(table["false_budget_per_hour"], errors="coerce")
        leads = pd.to_numeric(table["required_lead_s"], errors="coerce")
        rows = table.loc[
            table["method"].astype(str).isin((BRACE_METHOD, TWIN_METHOD))
            & budgets.isin((2.0, 5.0, 10.0))
            & leads.isin(LEAD_GRID)
            & (table["operating_point_status"].astype(str) == "estimable")
        ].copy()
        expected_rows = 2 * 3 * len(LEAD_GRID)
        if len(rows) != expected_rows:
            raise ArtifactValidationError(
                f"fold {fold.circuit} lacks the complete estimable 2/5/10-h threshold grid"
            )
        selected_rows.append(rows)
    selected = pd.concat(selected_rows, ignore_index=True)
    thresholds = pd.to_numeric(selected["threshold"], errors="coerce").to_numpy(dtype=float)
    calibration_proposals = pd.to_numeric(
        selected["calibration_proposal_count"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(thresholds).all() or not np.isfinite(calibration_proposals).all():
        raise ArtifactValidationError("examined calibration thresholds contain non-finite values")
    heldout_proposals = sum(fold.examined_budget_proposal_count for fold in evidence.folds)
    all_silent = bool(
        np.allclose(thresholds, 1.0, atol=1e-12, rtol=0.0)
        and np.allclose(calibration_proposals, 0.0, atol=1e-12, rtol=0.0)
        and heldout_proposals == 0
    )
    if all_silent:
        return (
            "Calibration selected threshold 1.0 for both methods in every fold at each "
            "estimable budget of 2, 5, and 10 false proposals per car-hour, so neither "
            "method emitted a held-out proposal at those budgets.",
            True,
        )
    return (
        "Across the complete estimable 2, 5, and 10 false-proposals-per-hour grid, "
        f"calibration-selected thresholds ranged from {thresholds.min():.6f} to "
        f"{thresholds.max():.6f}; the two methods emitted {heldout_proposals:,} held-out "
        "proposals at those budgets.",
        False,
    )


def _bootstrap_result_text(
    delta_draws: np.ndarray,
    *,
    both_policies_silent: bool,
) -> tuple[str, str]:
    if bool(np.all(delta_draws == 0.0)) and both_policies_silent:
        return (
            "All 10,000 paired contrasts were zero because both policies were silent, not "
            "because equivalence was established.",
            "Here every paired draw was zero, so the bootstrap revealed no sampling variation. ",
        )
    return (
        f"Across 10,000 paired resamples, the contrast ranged from "
        f"{_fmt_lead(delta_draws.min())} to {_fmt_lead(delta_draws.max())} s.",
        "The paired bootstrap quantified car-session variation conditional on the observed "
        "circuits and fixed fitted pipeline. ",
    )


def _metric_to_decision_text(
    brace_probability: pd.Series,
    twin_probability: pd.Series,
    *,
    brace_lstar: float,
    twin_lstar: float,
) -> str:
    better_ranking = float(brace_probability["average_precision_stepwise"]) > float(
        twin_probability["average_precision_stepwise"]
    )
    better_log = float(brace_probability["log_score"]) < float(twin_probability["log_score"])
    if better_ranking and better_log and brace_lstar == 0.0 and twin_lstar == 0.0:
        return (
            "Better ranking and log loss did not produce a qualifying localized warning. "
            "This metric-to-decision gap is the main empirical result."
        )
    if better_ranking and better_log and brace_lstar > twin_lstar:
        return (
            "BRACE had higher stepwise average precision, lower log score, and a larger "
            f"prespecified endpoint: $L^*={_fmt_lead(brace_lstar)}$ s versus "
            f"$L^*={_fmt_lead(twin_lstar)}$ s for the twin."
        )
    if better_ranking and better_log and brace_lstar < twin_lstar:
        return (
            "BRACE had higher stepwise average precision and lower log score but a smaller "
            f"prespecified endpoint: $L^*={_fmt_lead(brace_lstar)}$ s versus "
            f"$L^*={_fmt_lead(twin_lstar)}$ s for the twin."
        )
    if better_ranking and better_log:
        return (
            "BRACE had higher stepwise average precision and lower log score, while the "
            f"prespecified endpoint was tied at $L^*={_fmt_lead(brace_lstar)}$ s."
        )
    return (
        "The probability metrics and prespecified endpoint must be interpreted separately: "
        f"BRACE achieved $L^*={_fmt_lead(brace_lstar)}$ s and the twin achieved "
        f"$L^*={_fmt_lead(twin_lstar)}$ s."
    )


def _operating_clause(
    label: str,
    row: pd.Series,
    lead: float,
    qualifies: bool,
) -> str:
    counts = (
        f"localized {_fmt_count(row['localized_event_hits'])}/"
        f"{_fmt_count(row['qualified_events'])} events with "
        f"{_fmt_count(row['false_proposals'])} false proposals"
    )
    if qualifies:
        return f"At qualifying {_fmt_lead(lead)} s, {label} {counts}"
    return (
        f"{label} had no qualifying prespecified lead ($L^*=0.00$ s); its shortest "
        f"0.25 s row was explicitly nonqualifying and {counts}"
    )


def _derive_tokens(
    evidence: Evidence,
) -> tuple[dict[str, str], dict[str, str], pd.DataFrame, pd.DataFrame]:
    brace, twin, delta = _validate_primary_points(evidence)
    interval = _interval(evidence.bootstrap_summary, "delta_l_star_percentile_95")
    frontier = _frontier(evidence)
    primary_frontier_table = _primary_frontier_table(evidence, frontier)
    probability_rows = _probability_rows(evidence)
    probability_table = _probability_table(probability_rows)
    circuit_table, circuit_rows = _circuit_table(evidence)
    lf_brace, lf_twin, lf_delta, lf_interval, lf_table = _least_favorable(evidence)
    delay_table, delay160 = _delay_table(evidence)
    (
        car_count,
        event_count,
        exposure_hours,
        circuit_count,
        raw_frames,
        resampled_frames,
        prebuffer_hours,
    ) = _cohort_summary(evidence.build_manifest, evidence.contributions)
    runtime_hours, score_seconds, score_rows, throughput, environment = _runtime_summary(
        evidence.folds
    )
    monotonicity = _monotonicity(evidence)
    brace_probability = probability_rows.loc[probability_rows["method"] == BRACE_METHOD].iloc[0]
    twin_probability = probability_rows.loc[probability_rows["method"] == TWIN_METHOD].iloc[0]
    brace_primary, brace_qualifies = _primary_row(evidence, BRACE_METHOD, brace)
    twin_primary, twin_qualifies = _primary_row(evidence, TWIN_METHOD, twin)
    delta_draws = _load_draws(
        evidence.bootstrap,
        "delta-l-star-draws.npy",
        require_lstar_grid=False,
    )
    all_delta_draws_zero = bool(np.all(delta_draws == 0.0))
    primary_policies_silent = all(
        fold.primary_proposal_count == 0 for fold in evidence.folds
    )
    policy_result, examined_policies_silent = _policy_result_text(evidence)
    probability_comparison, all_method_probability = _probability_result_text(
        probability_rows,
        brace_probability,
        twin_probability,
    )
    bootstrap_result, bootstrap_variation_limit = _bootstrap_result_text(
        delta_draws,
        both_policies_silent=primary_policies_silent,
    )
    metric_to_decision = _metric_to_decision_text(
        brace_probability,
        twin_probability,
        brace_lstar=brace,
        twin_lstar=twin,
    )
    conditional = _conditional_text(
        delta,
        interval,
        all_delta_draws_zero=all_delta_draws_zero,
        both_policies_silent=primary_policies_silent,
    )
    primary_rows = _registered_primary_rows(evidence.operating)
    secondary = frontier.loc[
        np.isclose(frontier["budget"], PRIMARY_BUDGET)
        & ~frontier["method"].isin((BRACE_METHOD, TWIN_METHOD))
    ].sort_values(["l_star_s", "method"], ascending=[False, True], kind="stable")
    secondary_text = ", ".join(
        f"{METHOD_LABELS[str(row.method)]} {_fmt_lead(row.l_star_s)} s"
        for row in secondary.itertuples(index=False)
    )
    circuit_deltas = circuit_rows["delta"].to_numpy(dtype=float)
    positive_circuits = int(np.sum(circuit_deltas > 0.0))
    tied_circuits = int(np.sum(np.isclose(circuit_deltas, 0.0)))
    negative_circuits = int(np.sum(circuit_deltas < 0.0))
    budget_rows = frontier.loc[frontier["method"].isin((BRACE_METHOD, TWIN_METHOD))]
    budget_summary = "; ".join(
        f"{_fmt(float(budget), 1)}/h: BRACE {_fmt_lead(group.loc[group['method'] == BRACE_METHOD, 'l_star_s'].iloc[0])} s versus twin {_fmt_lead(group.loc[group['method'] == TWIN_METHOD, 'l_star_s'].iloc[0])} s"
        for budget, group in budget_rows.groupby("budget", sort=True)
    )
    unresolved_brace = int(round(float(brace_primary.get("unresolved_proposals", 0))))
    unresolved_twin = int(round(float(twin_primary.get("unresolved_proposals", 0))))
    if brace_qualifies and twin_qualifies:
        unresolved_basis = "the qualifying BRACE and twin operating rows"
    else:
        brace_basis_text = (
            "the qualifying BRACE row"
            if brace_qualifies
            else "BRACE's 0.25 s nonqualifying fallback row"
        )
        twin_basis_text = (
            "the qualifying twin row"
            if twin_qualifies
            else "the twin's 0.25 s nonqualifying fallback row"
        )
        unresolved_basis = f"{brace_basis_text} and {twin_basis_text}"
    delay_delta = _number(delay160["delta_l_star_s"], "160 ms delta L-star")
    delay_deltas = pd.to_numeric(
        evidence.delay_l_star["delta_l_star_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if examined_policies_silent and np.allclose(
        delay_deltas,
        delay_deltas[0],
        atol=1e-12,
        rtol=0.0,
    ):
        delay_interpretation = (
            "Delay and unresolved-proposal sensitivities were unchanged because there were no "
            "proposals to shift or reclassify; they provide no evidence of latency robustness."
        )
    else:
        delay_interpretation = (
            "The synthetic-delay grid changed or retained the endpoint as reported above; it "
            "does not provide evidence of measured latency robustness."
        )
    if evidence.transport_summary is None:
        raise ArtifactValidationError("publication requires a two-stage transport bootstrap")
    _interval(evidence.transport_summary, "delta_l_star_percentile_95")
    brace_clause = _operating_clause("BRACE", brace_primary, brace, brace_qualifies)
    twin_clause = _operating_clause("the twin", twin_primary, twin, twin_qualifies)
    operating_sentence = f"{brace_clause}; {twin_clause}."
    brace_basis = (
        "At the BRACE qualifying point"
        if brace_qualifies
        else "At BRACE's shortest 0.25 s nonqualifying fallback row"
    )
    localization_sentence = (
        f"{brace_basis}, event recall was "
        f"{_fmt(brace_primary.get('event_recall', np.nan))}, correct-side recall was "
        f"{_fmt(brace_primary.get('correct_side_event_recall', np.nan))}, within-one-bin "
        f"localized recall was {_fmt(brace_primary['localized_event_recall'])}, and exact-bin "
        f"recall was {_fmt(brace_primary.get('exact_bin_event_recall', np.nan))}."
    )
    paper_title = (
        "Better Ranking, No Qualifying Warning: An Operational Audit of BRACE in Simulated "
        "Open-Wheel Racing"
        if (
            float(brace_probability["average_precision_stepwise"])
            > float(twin_probability["average_precision_stepwise"])
            and float(brace_probability["log_score"]) < float(twin_probability["log_score"])
            and brace == 0.0
            and twin == 0.0
        )
        else "BRACE: An Operational Audit in Simulated Open-Wheel Racing"
    )
    tokens = {
        "PAPER_TITLE": paper_title,
        "PRIMARY_LSTAR_BRACE_S": _fmt_lead(brace),
        "PRIMARY_LSTAR_TWIN_S": _fmt_lead(twin),
        "PRIMARY_DELTA_LSTAR_S": _fmt_lead(delta),
        "PRIMARY_DELTA_LSTAR_CI_LOW_S": _fmt_lead(interval[0]),
        "PRIMARY_DELTA_LSTAR_CI_HIGH_S": _fmt_lead(interval[1]),
        "PRIMARY_EVIDENCE_CONCLUSION": conditional["evidence"],
        "PRIMARY_POLICY_RESULT": policy_result,
        "PRIMARY_BOOTSTRAP_RESULT": bootstrap_result,
        "PROBABILITY_COMPARISON_RESULT": probability_comparison,
        "ALL_METHOD_PROBABILITY_RESULT": all_method_probability,
        "METRIC_TO_DECISION_RESULT": metric_to_decision,
        "COHORT_FLOW_RESULT": (
            f"The authenticated build converted {raw_frames:,} raw rows into "
            f"{resampled_frames:,} causal 20 Hz frames, reduced {prebuffer_hours:.6f} "
            f"pre-buffer car-hours to {exposure_hours:.6f} eligible car-hours, and retained "
            f"{event_count} qualifying excursions from {car_count} car sessions across "
            f"{circuit_count} circuits."
        ),
        "PRIMARY_GRID_ESTIMABILITY_STATUS": "estimable",
        "ESTIMABILITY_DETAIL": (
            f"All {len(primary_rows)} prespecified method-by-lead primary rows were estimable; no "
            "primary threshold was imputed or borrowed across circuits."
        ),
        "FIGURE_WARNING_FRONTIER_MARKDOWN": _figure_markdown(
            "figure-03-calibration-operating-cliff",
            "Calibration-only exploratory mechanism diagnostic; the primary held-out result is unchanged. Each point is the highest active frozen-grid threshold below the silent threshold for one outer fold. All points lie beyond the examined gates at or below 10 false proposals per eligible simulated car-hour. These values are not held-out estimates and were not used to revise the primary analysis.",
        ),
        "PRIMARY_RESULT_INTERPRETATION": conditional["interpretation"],
        "PRIMARY_OPERATING_POINT_NARRATIVE": operating_sentence,
        "TABLE_PRIMARY_FRONTIER_MARKDOWN": primary_frontier_table,
        "OPERATING_FRONTIER_SUMMARY": budget_summary,
        "FIGURE_RELIABILITY_MARKDOWN": _figure_markdown(
            "figure-02-reliability",
            "Held-out 1.50-second reliability in ten fixed equal-width bins. Intervals condition on the four observed circuits and resample car sessions; empty bins are omitted.",
        ),
        "BRACE_BRIER_1P50": _fmt(brace_probability["brier_score"]),
        "TWIN_BRIER_1P50": _fmt(twin_probability["brier_score"]),
        "BRACE_BRIER_SKILL_1P50": _fmt(brace_probability["brier_skill_score"]),
        "BRACE_LOG_SCORE_1P50": _fmt(brace_probability["log_score"]),
        "BRACE_AP_1P50": _fmt(brace_probability["average_precision_stepwise"]),
        "BRACE_CAL_INTERCEPT_1P50": _fmt(brace_probability["calibration_intercept"]),
        "BRACE_CAL_SLOPE_1P50": _fmt(brace_probability["calibration_slope"]),
        "BRACE_ECE_1P50": _fmt(brace_probability["ece_10_equal_width"]),
        "TABLE_PROBABILITY_METRICS_MARKDOWN": probability_table,
        "BRACE_MONOTONICITY_SCORE_ROWS": _fmt_count(monotonicity["score_rows"]),
        "BRACE_MONOTONICITY_VIOLATION_COUNT": _fmt_count(monotonicity["violating_row_count"]),
        "BRACE_MONOTONICITY_VIOLATION_RATE": _fmt(monotonicity["violating_row_rate"], 4),
        "LOCALIZATION_COMPONENT_SUMMARY": localization_sentence,
        "TABLE_CIRCUIT_RESULTS_MARKDOWN": circuit_table,
        "CIRCUIT_HETEROGENEITY_SUMMARY": (
            f"Circuit-specific contrasts were positive in {positive_circuits} circuits, tied in "
            f"{tied_circuits}, and negative in {negative_circuits}; the range was "
            f"{_fmt_lead(circuit_deltas.min())} to {_fmt_lead(circuit_deltas.max())} s."
        ),
        "SECONDARY_BASELINE_SUMMARY": secondary_text,
        "UNRESOLVED_PROPOSAL_SUMMARY": (
            f"{unresolved_basis} contained {unresolved_brace} and "
            f"{unresolved_twin} unresolved proposals, respectively"
        ),
        "LF_LSTAR_BRACE_S": _fmt_lead(lf_brace),
        "LF_LSTAR_TWIN_S": _fmt_lead(lf_twin),
        "LF_DELTA_LSTAR_S": _fmt_lead(lf_delta),
        "LF_DELTA_LSTAR_CI_LOW_S": _fmt_lead(lf_interval[0]),
        "LF_DELTA_LSTAR_CI_HIGH_S": _fmt_lead(lf_interval[1]),
        "TABLE_LEAST_FAVORABLE_MARKDOWN": lf_table,
        "TABLE_DELAY_RESULTS_MARKDOWN": delay_table,
        "DELAY_160_DELTA_LSTAR_S": _fmt_lead(delay_delta),
        "DELAY_SENSITIVITY_SUMMARY": (
            f"Across the prespecified 0--160 ms grid, the contrast ranged from "
            f"{_fmt_lead(evidence.delay_l_star['delta_l_star_s'].min())} to "
            f"{_fmt_lead(evidence.delay_l_star['delta_l_star_s'].max())} s."
        ),
        "DELAY_RESULT_INTERPRETATION": delay_interpretation,
        "TOTAL_EXPERIMENT_RUNTIME_HOURS": _fmt(runtime_hours, 3),
        "COMPUTE_ENVIRONMENT": environment,
        "INFERENCE_THROUGHPUT_SUMMARY": (
            f"The six-method scoring stages processed {score_rows:,} forecast rows in "
            f"{score_seconds:.2f} s ({throughput:.2f} forecast rows/s)."
        ),
        "POOLED_MANIFEST_SHA256": evidence.pooled.manifest_hash,
        "BOOTSTRAP_MANIFEST_SHA256": evidence.bootstrap.manifest_hash,
        "PRIMARY_DISCUSSION_PARAGRAPH": conditional["discussion"],
        "BOOTSTRAP_VARIATION_LIMIT": bootstrap_variation_limit,
        "CONCLUSION_RESULT_SENTENCE": conditional["conclusion"],
    }
    tables = {
        "table-primary-frontier.md": primary_frontier_table,
        "table-probability-metrics.md": probability_table,
        "table-circuit-results.md": circuit_table,
        "table-least-favorable.md": lf_table,
        "table-delay-results.md": delay_table,
    }
    return tokens, tables, frontier, probability_rows


def _replace_tokens(source: str, tokens: Mapping[str, str], label: str) -> str:
    present = set(TOKEN_PATTERN.findall(source))
    unknown = present.difference(EXPECTED_TOKENS)
    if unknown:
        raise ArtifactValidationError(f"unrecognized result tokens in {label}: {sorted(unknown)}")
    missing_values = present.difference(tokens)
    if missing_values:
        raise ArtifactValidationError(
            f"no derived values for tokens in {label}: {sorted(missing_values)}"
        )
    rendered = TOKEN_PATTERN.sub(lambda match: tokens[match.group(1)], source)
    rendered = rendered.replace("{{...}}", "double-braced result field")
    leftovers = re.findall(r"\{\{[^{}]+\}\}", rendered)
    if leftovers:
        raise ArtifactValidationError(
            f"unresolved result tokens in {label}: {sorted(set(leftovers))}"
        )
    return rendered


def _validate_publication_language(sources: Mapping[str, str]) -> None:
    internal = re.compile(r"\b(?:result[- ]shell|result[- ]token|placeholder)s?\b", re.I)
    registered = re.compile(r"\bregistered\b", re.I)
    for label, source in sources.items():
        match = internal.search(source)
        if match:
            raise ArtifactValidationError(
                f"publication {label} retains internal authoring language: {match.group(0)}"
            )
        match = registered.search(source)
        if match:
            raise ArtifactValidationError(
                f"publication {label} contains prohibited standalone 'registered'"
            )


def _publication_cleanup(manuscript: str, abstract: str) -> tuple[str, str]:
    manuscript = re.sub(
        r"(?ms)^# Appendix B\. Result-token contract\s*.*?(?=^# References\s*$)",
        "",
        manuscript,
    )
    manuscript = manuscript.replace(
        'date: "Results-ready manuscript"',
        'date: "October 2026"',
    )
    manuscript = manuscript.replace(
        "aggregate hours over their registered timed stages",
        "aggregate hours over their recorded timed stages",
    )
    abstract = abstract.replace(
        "A circuit-side warning is useful only if it identifies where a vehicle will leave the "
        "mapped racing envelope, early enough to act, without flooding operators with false "
        "proposals.",
        "Useful warnings must be early, localized, and sparse.",
    )
    abstract = abstract.replace(
        "In four leave-one-circuit-out folds, BRACE propagated 256 particles from a "
        "constant-turn-rate-and-acceleration transition corrected by a cluster-weighted "
        "Bayesian-bootstrap residual model. A matched twin used the posterior-mean residual "
        "coefficients and no process noise.",
        "Across four held-out circuits, BRACE propagated 256 particles through a motion model "
        "with cluster-weighted Bayesian-bootstrap residuals; its twin used posterior-mean "
        "residuals without process noise.",
    )
    abstract = abstract.replace(
        "The evidence is limited to simulated mapped boundary excursions; it does not establish "
        "Formula 1 crash prediction, physical barrier-contact prediction, or the effectiveness "
        "of active padding. BRACE should therefore be evaluated as a probabilistic "
        "decision-support prototype whose next validation step is prospective replay with "
        "measured end-to-end latency and adjudicated contact outcomes.",
        "Evidence is limited to simulated boundary excursions, not crashes, impacts, or "
        "active-padding effectiveness; prospective replay remains necessary.",
    )
    abstract = re.sub(
        r"A circuit-side forecast is useful only if it identifies where a vehicle will leave "
        r"the mapped racing envelope before the crossing while limiting false proposals\.",
        "Useful warnings must be early, localized, and sparse.",
        abstract,
    )
    abstract = re.sub(
        r"That budget equals 40 car-level proposals per 20-car field-hour before deduplication "
        r"and is an analytical gate, not a deployment requirement\.",
        "The budget is analytical, not a deployment requirement.",
        abstract,
    )
    abstract = re.sub(
        r"The evidence is limited to a high-event-rate simulator benchmark and does not "
        r"establish Formula 1 crash prediction, physical barrier-contact prediction, "
        r"active-padding effectiveness, or actionability within 1\.50 seconds\. The next "
        r"validation step is prospective shadow-mode replay with native clocks, field-level "
        r"alert accounting, surveyed geometry, and adjudicated contact outcomes\.",
        "Evidence is limited to a high-event-rate simulator, not crashes, impacts, active-padding "
        "efficacy, or deployment readiness; prospective shadow-mode replay remains necessary.",
        abstract,
    )
    abstract = re.sub(
        r"(?ms)\n?<!--\s*Results-shell word count:.*?-->\s*$",
        "\n",
        abstract,
    )
    _validate_publication_language(
        {
            "manuscript": manuscript,
            "abstract": abstract,
        }
    )
    return manuscript, abstract


def _conservative_word_count(source: str) -> int:
    without_comments = re.sub(r"<!--.*?-->", "", source, flags=re.DOTALL)
    without_links = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", without_comments)
    without_markup = re.sub(r"[`#*_{}$\\]", " ", without_links)
    return len(re.findall(r"\b[\w']+\b", without_markup))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_portable_input_manifest(
    stage: Path,
    *,
    role: str,
    directory_name: str,
    bundle: ManifestBundle,
) -> Mapping[str, str]:
    artifact_names = [path.name for path in bundle.artifacts]
    if len(artifact_names) != len(set(artifact_names)):
        raise ArtifactValidationError(
            f"cannot create a portable {role} manifest with duplicate artifact names"
        )
    source_identity = {
        key: bundle.payload[key]
        for key in (
            "schema_version",
            "status",
            "code_content_hash",
            "data_content_hash",
            "config_content_hash",
            "source_manifest_hash",
            "build_manifest_hash",
        )
        if key in bundle.payload
    }
    study_config = bundle.payload.get("study_config")
    if isinstance(study_config, Mapping):
        for source_key, target_key in (
            ("sha256", "study_config_sha256"),
            ("canonical_content_hash", "config_content_hash"),
        ):
            if source_key in study_config:
                source_identity[target_key] = study_config[source_key]
    analysis = bundle.payload.get("analysis")
    if isinstance(analysis, Mapping):
        source_identity["analysis"] = {
            key: analysis[key]
            for key in (
                "n_resamples",
                "seed",
                "cluster_level",
                "models_and_thresholds_refit",
            )
            if key in analysis
        }
    experiment_manifest = bundle.payload.get("experiment_manifest")
    if isinstance(experiment_manifest, Mapping) and "sha256" in experiment_manifest:
        source_identity["upstream_experiment_manifest_sha256"] = experiment_manifest["sha256"]

    payload = {
        "schema_version": 1,
        "derivative_kind": "authenticated_input_manifest_receipt",
        "role": role,
        "authenticated_source_manifest_sha256": bundle.manifest_hash,
        "source_identity": source_identity,
        "artifacts": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
            for path, digest in sorted(
                bundle.artifacts.items(),
                key=lambda item: item[0].name,
            )
        ],
    }
    _validate_portable_publication_manifest(payload)
    relative_path = Path("input-manifests") / directory_name / "manifest.json"
    destination = stage / relative_path
    _write_text(destination, json.dumps(payload, indent=2, sort_keys=True))
    return {
        "path": relative_path.as_posix(),
        "sha256": _sha256(destination),
        "authenticated_source_manifest_sha256": bundle.manifest_hash,
    }


def _figure_sources(
    evidence: Evidence, frontier: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    warning = frontier.loc[frontier["method"].isin((BRACE_METHOD, TWIN_METHOD))].copy()
    reliability = evidence.reliability_intervals.loc[
        evidence.reliability_intervals["method"].astype(str).isin((BRACE_METHOD, TWIN_METHOD))
        & np.isclose(
            pd.to_numeric(evidence.reliability_intervals["horizon_s"], errors="coerce"),
            PRIMARY_HORIZON,
            atol=1e-12,
            rtol=0.0,
        )
        & (pd.to_numeric(evidence.reliability_intervals["count"], errors="coerce") > 0)
    ].copy()
    _require_columns(
        reliability,
        {
            "method",
            "horizon_s",
            "bin_index",
            "count",
            "mean_probability",
            "observed_frequency",
            "observed_frequency_lower_95",
            "observed_frequency_upper_95",
        },
        "reliability interval table",
    )
    if set(reliability["method"].astype(str)) != {BRACE_METHOD, TWIN_METHOD}:
        raise ArtifactValidationError("reliability intervals lack BRACE or twin 1.50-second bins")
    return warning, reliability


def _registered_digest_by_name(
    registry: object,
    *,
    filename: str,
    label: str,
) -> str:
    if not isinstance(registry, Mapping):
        raise ArtifactValidationError(f"{label} lacks an artifact registry")
    matches = [
        str(value)
        for key, value in registry.items()
        if Path(str(key)).name == filename
    ]
    if len(matches) != 1:
        raise ArtifactValidationError(
            f"{label} must register {filename!r} exactly once"
        )
    return matches[0]


def _validate_calibration_operating_cliff_binding(evidence: Evidence) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_relative = "paper/figure-source/figure-03-calibration-operating-cliff.csv"
    source = project_root / source_relative
    expected_source_hash = CALIBRATION_CLIFF_SOURCE_SHA256[source_relative]
    if not source.is_file() or _sha256(source) != expected_source_hash:
        raise ArtifactValidationError(
            "calibration operating-cliff source differs from its independently reviewed digest"
        )
    try:
        fixed = pd.read_csv(source)
    except Exception as exc:
        raise ArtifactValidationError(
            "cannot read the fixed calibration operating-cliff source table"
        ) from exc
    _require_columns(
        fixed,
        {
            "fold_test_circuit",
            "fold_manifest_sha256",
            "source_calibrated_scores_sha256",
            "source_calibration_thresholds_sha256",
            "threshold_freeze_seal_sha256",
        },
        "fixed calibration operating-cliff source table",
    )
    seal_hashes = set(fixed["threshold_freeze_seal_sha256"].astype(str))
    if len(seal_hashes) != 1:
        raise ArtifactValidationError(
            "fixed calibration operating-cliff rows do not share one threshold seal"
        )
    expected_seal_hash = next(iter(seal_hashes))
    seal_record = evidence.pooled.payload.get("threshold_freeze_seal")
    if not isinstance(seal_record, Mapping) or str(seal_record.get("sha256")) != (
        expected_seal_hash
    ):
        raise ArtifactValidationError(
            "calibration operating-cliff evidence binding mismatch: threshold-freeze seal"
        )

    raw_records = evidence.pooled.payload.get("fold_manifests")
    if not isinstance(raw_records, list):
        raise ArtifactValidationError(
            "calibration operating-cliff evidence binding lacks pooled fold manifests"
        )
    pooled_records = {
        str(record.get("fold_test_circuit")): record
        for record in raw_records
        if isinstance(record, Mapping)
    }
    if set(pooled_records) != set(REGISTERED_CIRCUITS):
        raise ArtifactValidationError(
            "calibration operating-cliff evidence binding has an incomplete fold set"
        )
    for circuit in REGISTERED_CIRCUITS:
        rows = fixed.loc[fixed["fold_test_circuit"].astype(str) == circuit]
        if len(rows) != 2:
            raise ArtifactValidationError(
                f"fixed calibration operating-cliff source has the wrong rows for {circuit}"
            )
        expected_values: dict[str, str] = {}
        for column in (
            "fold_manifest_sha256",
            "source_calibrated_scores_sha256",
            "source_calibration_thresholds_sha256",
        ):
            values = set(rows[column].astype(str))
            if len(values) != 1:
                raise ArtifactValidationError(
                    f"fixed calibration operating-cliff rows disagree on {column}/{circuit}"
                )
            expected_values[column] = next(iter(values))
        record = pooled_records[circuit]
        if str(record.get("sha256")) != expected_values["fold_manifest_sha256"]:
            raise ArtifactValidationError(
                f"calibration operating-cliff evidence binding mismatch: {circuit} fold manifest"
            )
        manifest_path = _manifest_record_path(record, f"{circuit} fold manifest")
        manifest = _load_json(manifest_path, f"{circuit} fold manifest")
        registry = manifest.get("artifacts")
        actual_score_hash = _registered_digest_by_name(
            registry,
            filename="compact-calibrated-scores.parquet",
            label=f"{circuit} fold manifest",
        )
        actual_threshold_hash = _registered_digest_by_name(
            registry,
            filename="calibration-thresholds.csv",
            label=f"{circuit} fold manifest",
        )
        if actual_score_hash != expected_values["source_calibrated_scores_sha256"]:
            raise ArtifactValidationError(
                f"calibration operating-cliff evidence binding mismatch: {circuit} scores"
            )
        if actual_threshold_hash != expected_values["source_calibration_thresholds_sha256"]:
            raise ArtifactValidationError(
                f"calibration operating-cliff evidence binding mismatch: {circuit} thresholds"
            )


def _stage_calibration_operating_cliff(
    stage: Path,
    evidence: Evidence,
) -> Mapping[str, Mapping[str, Any]]:
    _validate_calibration_operating_cliff_binding(evidence)
    project_root = Path(__file__).resolve().parents[1]
    destinations = {
        "scripts/build_calibration_operating_cliff_figure.py": (
            "analysis-output/figure-source/build-calibration-operating-cliff.py"
        ),
        "paper/figure-source/figure-03-calibration-operating-cliff.csv": (
            "analysis-output/figure-source/figure-03-calibration-operating-cliff.csv"
        ),
        "paper/figures/figure-03-calibration-operating-cliff-provenance.md": (
            "analysis-output/figure-source/figure-03-calibration-operating-cliff-provenance.md"
        ),
        "paper/figures/figure-03-calibration-operating-cliff.pdf": (
            "analysis-output/figures/figure-03-calibration-operating-cliff.pdf"
        ),
        "paper/figures/figure-03-calibration-operating-cliff.png": (
            "analysis-output/figures/figure-03-calibration-operating-cliff.png"
        ),
    }
    records: dict[str, Mapping[str, Any]] = {}
    for source_relative, expected_hash in CALIBRATION_CLIFF_SOURCE_SHA256.items():
        source = project_root / source_relative
        if not source.is_file():
            raise ArtifactValidationError(
                f"calibration operating-cliff source is missing: {source}"
            )
        actual_hash = _sha256(source)
        if actual_hash != expected_hash:
            raise ArtifactValidationError(
                "calibration operating-cliff source differs from its independently reviewed "
                f"digest: {source_relative}"
            )
        destination_relative = destinations[source_relative]
        destination = stage / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if _sha256(destination) != expected_hash:
            raise ArtifactValidationError(
                f"staged calibration operating-cliff byte check failed: {destination_relative}"
            )
        record: dict[str, Any] = {
            "path": source_relative,
            "sha256": expected_hash,
            "staged_path": destination_relative,
            "role": (
                "archival_renderer_source"
                if source_relative == "scripts/build_calibration_operating_cliff_figure.py"
                else "reviewed_figure_bundle_artifact"
            ),
        }
        if source_relative == "scripts/build_calibration_operating_cliff_figure.py":
            record["runnable_in_bundle"] = False
        records[source_relative] = record
    return records


def _generate_figures(stage: Path, warning: pd.DataFrame, reliability: pd.DataFrame) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["SOURCE_DATE_EPOCH"] = PUBLICATION_SOURCE_DATE_EPOCH
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pubfig as pf

    figure_dir = stage / "analysis-output" / "figures"
    source_dir = stage / "analysis-output" / "figure-source"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    warning.sort_values(["method", "budget"], kind="stable").to_csv(
        source_dir / "figure-01-warning-frontier.csv",
        index=False,
        lineterminator="\n",
    )
    reliability.sort_values(["method", "bin_index"], kind="stable").to_csv(
        source_dir / "figure-02-reliability.csv",
        index=False,
        lineterminator="\n",
    )

    budgets = np.sort(warning["budget"].unique().astype(float))
    values = np.vstack(
        [
            warning.loc[warning["method"] == method]
            .set_index("budget")
            .loc[budgets, "l_star_s"]
            .to_numpy(dtype=float)
            for method in (BRACE_METHOD, TWIN_METHOD)
        ]
    )
    figure = pf.line(
        values.T,
        x=budgets,
        series_names=[METHOD_LABELS[BRACE_METHOD], METHOD_LABELS[TWIN_METHOD]],
        x_label="False-proposal budget (per eligible simulated car-hour)",
        y_label="Greatest qualifying actual lead, $L^*$ (s)",
        marker="auto",
        color_palette=["#0072B2", "#D55E00"],
        show_y_grid=True,
    )
    axis = figure.axes[0]
    axis.set_xscale("log")
    axis.set_xticks(budgets)
    axis.set_xticklabels([f"{value:g}" for value in budgets])
    axis.set_ylim(-0.03, max(1.55, float(np.nanmax(values)) + 0.05))
    pf.batch_export(
        figure,
        figure_dir / "figure-01-warning-frontier",
        formats=("pdf", "png"),
        spec="nature",
        width="double",
        dpi=300,
        trim=True,
    )
    plt.close(figure)

    ordered = reliability.sort_values(["method", "bin_index"], kind="stable")
    x = ordered["mean_probability"].to_numpy(dtype=float)
    y = ordered["observed_frequency"].to_numpy(dtype=float)
    labels = ordered["method"].map(METHOD_LABELS).to_numpy(dtype=str)
    figure = pf.scatter(
        x,
        y,
        labels=labels,
        x_label="Mean forecast probability",
        y_label="Observed excursion frequency",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        show_y_equal_x=False,
        color_palette=["#0072B2", "#D55E00"],
        scatter_size=5.0,
    )
    axis = figure.axes[0]
    axis.plot([0.0, 1.0], [0.0, 1.0], color="0.65", linewidth=0.8, linestyle="--")
    for method, color in ((BRACE_METHOD, "#0072B2"), (TWIN_METHOD, "#D55E00")):
        group = ordered.loc[ordered["method"].astype(str) == method].sort_values(
            "mean_probability", kind="stable"
        )
        gx = group["mean_probability"].to_numpy(dtype=float)
        gy = group["observed_frequency"].to_numpy(dtype=float)
        lower = group["observed_frequency_lower_95"].to_numpy(dtype=float)
        upper = group["observed_frequency_upper_95"].to_numpy(dtype=float)
        axis.plot(gx, gy, color=color, linewidth=1.0, alpha=0.85)
        axis.vlines(gx, lower, upper, color=color, linewidth=0.7, alpha=0.75)
        cap = 0.008
        axis.hlines(lower, gx - cap, gx + cap, color=color, linewidth=0.7, alpha=0.75)
        axis.hlines(upper, gx - cap, gx + cap, color=color, linewidth=0.7, alpha=0.75)
    pf.batch_export(
        figure,
        figure_dir / "figure-02-reliability",
        formats=("pdf", "png"),
        spec="nature",
        width="double",
        dpi=300,
        trim=True,
    )
    plt.close(figure)


def _analysis_artifacts(
    evidence: Evidence,
    tokens: Mapping[str, str],
    tables: Mapping[str, str],
) -> tuple[str, str, str]:
    low = tokens["PRIMARY_DELTA_LSTAR_CI_LOW_S"]
    high = tokens["PRIMARY_DELTA_LSTAR_CI_HIGH_S"]
    analysis_report = f"""# BRACE strict analysis report

## Analysis question

Does the complete uncertainty-propagating BRACE pipeline create more correctly localized warning lead than its matched posterior-mean deterministic twin at no more than two false proposals per eligible simulated car-hour and at least 0.50 localized event recall?

## Provenance gate

- Pooled held-out manifest role: `input-manifests/pooled-heldout/manifest.json`
- Pooled manifest SHA-256: `{evidence.pooled.manifest_hash}`
- Primary bootstrap manifest role: `input-manifests/primary-bootstrap/manifest.json`
- Primary bootstrap manifest SHA-256: `{evidence.bootstrap.manifest_hash}`
- Pooled and bootstrap artifacts sealed before held-out outcome access: hash-verified before analysis
- Held-out fold manifests: {len(evidence.folds)} hash-verified circuit folds

## Key findings

- BRACE $L^*$: {tokens["PRIMARY_LSTAR_BRACE_S"]} s.
- Matched twin $L^*$: {tokens["PRIMARY_LSTAR_TWIN_S"]} s.
- Paired difference: {tokens["PRIMARY_DELTA_LSTAR_S"]} s; conditional 95% percentile interval {low} to {high} s.
- Least-favorable difference: {tokens["LF_DELTA_LSTAR_S"]} s; interval {tokens["LF_DELTA_LSTAR_CI_LOW_S"]} to {tokens["LF_DELTA_LSTAR_CI_HIGH_S"]} s.
- 160 ms synthetic-delay difference: {tokens["DELAY_160_DELTA_LSTAR_S"]} s.

## Claim candidates

- Claim:
  - Source evidence: authenticated pooled operating metrics and primary paired bootstrap.
  - Allowed wording: {tokens["PRIMARY_EVIDENCE_CONCLUSION"]}
  - Forbidden stronger wording: BRACE predicts Formula 1 crashes, prevents injury, or validates active padding.
  - Uncertainty: conditional on four observed circuit sessions and fixed fitted models, calibrators, and thresholds.
  - Next check: prospective replay with native packet clocks and adjudicated contact outcomes.
  - Decision: keep with the stated scope.

## Limits

The unit resampled by the primary interval is the complete car session within each fixed circuit. The interval does not include model-, calibration-, or threshold-selection uncertainty and does not estimate transport to a broad population of circuits. The endpoint is a simulated mapped car-center boundary excursion, not a crash or barrier contact.
"""
    stats_appendix = f"""# BRACE statistical appendix

## Frozen estimand

- Methods: BRACE versus posterior-mean deterministic twin.
- Required-lead grid: 0.25, 0.50, 1.00, 1.50 s.
- False-proposal budget: 2 per eligible simulated car-hour.
- Minimum correct-side, within-one-bin event recall: 0.50.
- Primary uncertainty: 10,000 paired car-session-within-circuit bootstrap resamples, seed 20270927.

## Primary result

| Quantity | Estimate |
|:--|--:|
| BRACE $L^*$ | {tokens["PRIMARY_LSTAR_BRACE_S"]} s |
| Twin $L^*$ | {tokens["PRIMARY_LSTAR_TWIN_S"]} s |
| $\\Delta L^*$ | {tokens["PRIMARY_DELTA_LSTAR_S"]} s |
| Conditional 95% percentile interval | {low} to {high} s |

The interval is discrete because $L^*$ is evaluated on the frozen four-point lead grid. No null-hypothesis p-value is reported; the decision wording follows whether the paired interval is entirely above zero, entirely below zero, or includes zero.

## Probability quality

{tables["table-probability-metrics.md"]}

## Sensitivities

{tables["table-least-favorable.md"]}

{tables["table-delay-results.md"]}

## Dependence and transport

Repeated frames and events are not treated as independent replicates. Primary resampling retains the four circuit sessions and resamples complete car sessions within circuit. Any optional circuit-then-car interval is descriptive because only four top-level circuit clusters are available.
"""
    figure_catalog = f"""# BRACE figure catalog

## Figure 1 — Calibration-only operating cliff

- Filename: `figures/figure-03-calibration-operating-cliff.pdf` and `.png`.
- Purpose: show the highest active frozen-grid threshold below the silent threshold in each outer fold for BRACE and its matched twin; this exploratory mechanism diagnostic does not replace the primary held-out result.
- Data source: independently recomputed calibration rows authenticated through the threshold seal, fold manifests, build manifest, and processed events, timing, and split inputs; exact plotted values are in `figure-source/figure-03-calibration-operating-cliff.csv`.
- Renderer status: `figure-source/build-calibration-operating-cliff.py` is an archival copy of the independently reviewed renderer. It is not runnable from this compact publication bundle because the authenticated frozen experiment and processed input tree are intentionally not duplicated here; rebuild it from the repository root described in the provenance note.
- Key observation: every active point lies beyond the examined gate region at or below 10 false proposals per eligible simulated car-hour.
- Interpretation check: these are calibration diagnostics, not held-out performance estimates, and they were not used to revise the frozen analysis.
- Caveat: unexamined off-grid thresholds are not ruled out.

## Figure 2 — Reliability

- Filename: `figures/figure-02-reliability.pdf` and `.png`.
- Purpose: compare 1.50-second marginal excursion-probability reliability for BRACE and the matched twin.
- Data source: authenticated primary-bootstrap `reliability-bin-intervals.csv`; exact plotted rows are in `figure-source/figure-02-reliability.csv`.
- Error bars: 95% percentile intervals from paired complete-car-session resampling within the four fixed circuits.
- Caption requirement: state fixed equal-width bins, horizon, cluster unit, and that empty bins are omitted.
- Key observation: BRACE Brier score was {tokens["BRACE_BRIER_1P50"]} and ECE was {tokens["BRACE_ECE_1P50"]}.
- Interpretation check: calibration is diagnostic and does not by itself establish operational warning lead.
- Caveat: the interval is conditional on the observed circuit set.

## Supplemental asset — Held-out warning frontier

- Filename: `figures/figure-01-warning-frontier.pdf` and `.png`.
- Purpose: preserve the complete BRACE-versus-twin $L^*$ frontier across prespecified false-proposal budgets.
- Data source: authenticated `pooled-heldout-operating-metrics.csv`; exact plotted rows are in `figure-source/figure-01-warning-frontier.csv`.
- Key observation: at the primary two-per-hour budget, BRACE was {tokens["PRIMARY_LSTAR_BRACE_S"]} s and the twin was {tokens["PRIMARY_LSTAR_TWIN_S"]} s.
- Caveat: the complete frontier is reported numerically in the manuscript and is retained here for reproducibility rather than used as a main display.
"""
    return analysis_report, stats_appendix, figure_catalog


def _validate_lock_matches_requirements(
    requirement_pins: Mapping[str, str],
    lock_text: str,
) -> None:
    locked: dict[str, str] = {}
    for line in lock_text.splitlines():
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([^\s\\]+)(?:\s+\\)?\s*",
            line,
        )
        if match is None:
            continue
        normalized = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        version = match.group(2)
        if normalized in locked and locked[normalized] != version:
            raise ArtifactValidationError(
                f"publication lock contains conflicting versions for {normalized}"
            )
        locked[normalized] = version
    for distribution, expected in requirement_pins.items():
        normalized = re.sub(r"[-_.]+", "-", distribution).lower()
        if locked.get(normalized) != expected:
            raise ArtifactValidationError(
                "publication lock does not match direct requirement pin "
                f"{distribution}=={expected}"
            )


def _publication_environment() -> Mapping[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    requirements_path = project_root / "publication-requirements.in"
    lock_path = project_root / "publication-requirements.lock"
    if not requirements_path.is_file() or not lock_path.is_file():
        raise ArtifactValidationError(
            "publication-requirements.in and its hashed lock are required"
        )
    lock_text = lock_path.read_text(encoding="utf-8")
    if "--generate-hashes" not in lock_text or "--hash=sha256:" not in lock_text:
        raise ArtifactValidationError("publication dependency lock is not hash-pinned")
    pins: dict[str, str] = {}
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if match:
            pins[match.group(1)] = match.group(2)
    _validate_lock_matches_requirements(pins, lock_text)
    if pins.get("pubfig") != REQUIRED_PUBFIG_VERSION:
        raise ArtifactValidationError(
            f"publication requirements must pin pubfig=={REQUIRED_PUBFIG_VERSION}"
        )
    if pins.get("pyarrow") != REQUIRED_PYARROW_VERSION:
        raise ArtifactValidationError(
            f"publication requirements must pin pyarrow=={REQUIRED_PYARROW_VERSION}"
        )
    if pins.get("pypdf") != REQUIRED_PYPDF_VERSION:
        raise ArtifactValidationError(
            f"publication requirements must pin pypdf=={REQUIRED_PYPDF_VERSION}"
        )
    installed: dict[str, str] = {}
    for distribution, expected in pins.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ArtifactValidationError(
                f"publication dependency is missing: {distribution}=={expected}; "
                "install publication-requirements.lock in a separate environment"
            ) from exc
        if observed != expected:
            raise ArtifactValidationError(
                f"publication dependency mismatch for {distribution}: "
                f"expected {expected}, observed {observed}"
            )
        installed[distribution] = observed
    parquet_buffer = io.BytesIO()
    try:
        pd.DataFrame({"publication_smoke": [1]}).to_parquet(
            parquet_buffer,
            index=False,
            engine="pyarrow",
        )
        parquet_buffer.seek(0)
        parquet_smoke = pd.read_parquet(parquet_buffer, engine="pyarrow")
    except Exception as exc:
        raise ArtifactValidationError(
            "pinned pyarrow publication environment cannot round-trip Parquet"
        ) from exc
    if parquet_smoke.to_dict(orient="list") != {"publication_smoke": [1]}:
        raise ArtifactValidationError("pinned pyarrow Parquet smoke result is invalid")
    try:
        pandoc = subprocess.run(
            ["pandoc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        raise ArtifactValidationError("Pandoc is required for the publication render gate") from exc
    return {
        "requirements": {
            "path": _logical_project_path(
                requirements_path,
                fallback="publication-requirements.in",
            ),
            "sha256": _sha256(requirements_path),
        },
        "lock": {
            "path": _logical_project_path(
                lock_path,
                fallback="publication-requirements.lock",
            ),
            "sha256": _sha256(lock_path),
        },
        "installed_versions": installed,
        "parquet_read_smoke": True,
        "source_date_epoch": PUBLICATION_SOURCE_DATE_EPOCH,
        "pandoc": pandoc,
        "reliability_bootstrap_source": {
            "path": _logical_project_path(
                _reliability_bootstrap_source_path(),
                fallback="src/brace_f1/bootstrap_reliability.py",
            ),
            "sha256": _sha256(_reliability_bootstrap_source_path()),
            "producer_source_tree_sha256": _producer_source_tree_hash(),
        },
    }


def _pandoc_resource_gate(stage: Path) -> Mapping[str, Any]:
    render_environment = os.environ.copy()
    render_environment["SOURCE_DATE_EPOCH"] = PUBLICATION_SOURCE_DATE_EPOCH
    check_path = stage / ".pandoc-manuscript-check.html"
    command = [
        "pandoc",
        "--fail-if-warnings",
        "--citeproc",
        "--resource-path=paper",
        "--from=markdown",
        "--to=html",
        "paper/manuscript-final.md",
        "--output",
        str(check_path.name),
    ]
    try:
        subprocess.run(
            command,
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
            env=render_environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise ArtifactValidationError(f"Pandoc manuscript resource gate failed: {stderr}") from exc
    rendered = check_path.read_text(encoding="utf-8")
    if re.search(r"\[@[^]]+\]", rendered) or 'id="refs"' not in rendered:
        raise ArtifactValidationError(
            "Pandoc citation gate found unresolved citations or no rendered bibliography"
        )
    check_path.unlink()
    pdf_path = stage / "paper" / "manuscript-final.pdf"
    trailer_id = _sha256(stage / "paper" / "manuscript-final.md")[:32]
    portability_header = r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=7\relax\fi"
    pdf_command = [
        "pandoc",
        "--fail-if-warnings",
        "--citeproc",
        "--resource-path=paper",
        f"--variable=header-includes:{portability_header}",
        "paper/manuscript-final.md",
        "--output=paper/manuscript-final.pdf",
    ]
    try:
        subprocess.run(
            pdf_command,
            cwd=stage,
            check=True,
            capture_output=True,
            text=True,
            env=render_environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise ArtifactValidationError(f"Pandoc manuscript PDF gate failed: {stderr}") from exc
    _normalize_pdf_trailer_id(pdf_path, trailer_id)
    binary_paths = sorted(
        path
        for path in stage.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}
    )
    for binary_path in binary_paths:
        _validate_portable_binary(binary_path)
    try:
        from pypdf import PdfReader

        document = PdfReader(str(pdf_path))
        page_texts = [page.extract_text() or "" for page in document.pages]
        graphic_count = 0
        for page in document.pages:
            resources = page.get("/Resources")
            if resources is not None and hasattr(resources, "get_object"):
                resources = resources.get_object()
            xobjects = resources.get("/XObject") if resources is not None else None
            if xobjects is not None and hasattr(xobjects, "get_object"):
                xobjects = xobjects.get_object()
            graphic_count += len(xobjects or {})
    except Exception as exc:
        raise ArtifactValidationError(f"cannot inspect rendered manuscript PDF: {pdf_path}") from exc
    if not page_texts or any(not text.strip() for text in page_texts):
        raise ArtifactValidationError("rendered manuscript PDF has no pages or a text-empty page")
    pdf_text = "\n".join(page_texts)
    if (
        re.search(r"\[@[^]]+\]", pdf_text)
        or "References" not in pdf_text
        or "Figure 1" not in pdf_text
        or "Figure 2" not in pdf_text
    ):
        raise ArtifactValidationError(
            "rendered manuscript PDF lacks resolved references or both figure captions"
        )
    if graphic_count < 2:
        raise ArtifactValidationError(
            "rendered manuscript PDF does not contain both publication graphics"
        )
    page_word_counts = [len(re.findall(r"\b[\w']+\b", text)) for text in page_texts]
    return {
        "citeproc_enabled": True,
        "bibliography_present": True,
        "unresolved_citation_count": 0,
        "pdf_page_count": len(page_texts),
        "pdf_image_count": graphic_count,
        "pdf_text_checked": True,
        "binary_portability_checked": True,
        "binary_portability_artifact_count": len(binary_paths),
        "pdf_trailer_id": trailer_id,
        "source_date_epoch": PUBLICATION_SOURCE_DATE_EPOCH,
        "command": pdf_command,
        "pdf_page_word_counts": page_word_counts,
        "pdf_sparse_text_pages_under_75_words": [
            index + 1 for index, count in enumerate(page_word_counts) if count < 75
        ],
    }


def compile_submission(
    *,
    pooled_dir: Path,
    bootstrap_dir: Path,
    manuscript_shell: Path,
    abstract_shell: Path,
    output_root: Path,
    transport_bootstrap_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Compile authenticated result artifacts and return every published path."""

    if transport_bootstrap_dir is None:
        raise ArtifactValidationError("publication compilation requires --transport-bootstrap-dir")
    output = Path(output_root).resolve()
    if output.exists():
        raise ArtifactValidationError(
            f"publication output must be a fresh dedicated directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    publication_environment = _publication_environment()
    manuscript_path = Path(manuscript_shell).resolve()
    abstract_path = Path(abstract_shell).resolve()
    references_path = Path(__file__).resolve().parents[1] / "references.bib"
    if not references_path.is_file():
        raise ArtifactValidationError(f"publication bibliography is missing: {references_path}")
    try:
        manuscript_source = manuscript_path.read_text(encoding="utf-8")
        abstract_source = abstract_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactValidationError("cannot read manuscript or abstract result shell") from exc
    manuscript_tokens = set(TOKEN_PATTERN.findall(manuscript_source))
    unknown = manuscript_tokens.difference(EXPECTED_TOKENS)
    missing = EXPECTED_TOKENS.difference(manuscript_tokens)
    if unknown:
        raise ArtifactValidationError(
            f"unrecognized result tokens in manuscript: {sorted(unknown)}"
        )
    if missing:
        raise ArtifactValidationError(
            f"manuscript shell is missing result tokens: {sorted(missing)}"
        )
    abstract_tokens_present = set(TOKEN_PATTERN.findall(abstract_source))
    abstract_unknown = abstract_tokens_present.difference(EXPECTED_TOKENS)
    abstract_missing = ABSTRACT_REQUIRED_TOKENS.difference(abstract_tokens_present)
    if abstract_unknown:
        raise ArtifactValidationError(
            f"unrecognized result tokens in abstract: {sorted(abstract_unknown)}"
        )
    if abstract_missing:
        raise ArtifactValidationError(
            f"abstract shell is missing essential result tokens: {sorted(abstract_missing)}"
        )
    evidence = _load_evidence(
        Path(pooled_dir),
        Path(bootstrap_dir),
        Path(transport_bootstrap_dir),
    )
    _validate_manuscript_cohort_table(manuscript_source, evidence.build_manifest)
    tokens, tables, frontier, _ = _derive_tokens(evidence)
    if set(tokens) != EXPECTED_TOKENS:
        raise ArtifactValidationError(
            f"compiler derived {len(tokens)} tokens; expected {len(EXPECTED_TOKENS)}"
        )
    manuscript = _replace_tokens(manuscript_source, tokens, "manuscript")
    abstract_tokens = dict(tokens)
    abstract_tokens["FIGURE_WARNING_FRONTIER_MARKDOWN"] = ""
    abstract_tokens["FIGURE_RELIABILITY_MARKDOWN"] = ""
    abstract = _replace_tokens(abstract_source, abstract_tokens, "abstract")
    manuscript, abstract = _publication_cleanup(manuscript, abstract)
    abstract_count = _conservative_word_count(abstract)
    if abstract_count > MAX_CONSERVATIVE_ABSTRACT_WORDS:
        raise ArtifactValidationError(
            f"rendered abstract has {abstract_count} conservative words; "
            f"publication gate is {MAX_CONSERVATIVE_ABSTRACT_WORDS}"
        )
    warning_source, reliability_source = _figure_sources(evidence, frontier)
    report_names = ("analysis-report.md", "stats-appendix.md", "figure-catalog.md")
    reports = _analysis_artifacts(evidence, tokens, tables)
    _validate_publication_language(
        {
            **{f"table {name}": table for name, table in tables.items()},
            **{
                f"report {name}": report
                for name, report in zip(report_names, reports, strict=True)
            },
        }
    )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        _write_text(stage / "paper" / "manuscript-final.md", manuscript)
        _write_text(stage / "paper" / "ssac27-abstract-final.md", abstract)
        shutil.copy2(references_path, stage / "paper" / "references.bib")
        for name, table in tables.items():
            _write_text(stage / "analysis-output" / "tables" / name, table)
        shutil.copy2(
            _artifact(evidence.pooled, "pooled-heldout-operating-metrics.csv"),
            stage
            / "analysis-output"
            / "tables"
            / "pooled-heldout-operating-metrics.csv",
        )
        for name, report in zip(report_names, reports, strict=True):
            _write_text(stage / "analysis-output" / name, report)
        _generate_figures(stage, warning_source, reliability_source)
        calibration_cliff_sources = _stage_calibration_operating_cliff(stage, evidence)
        pandoc_gate = _pandoc_resource_gate(stage)
        assert evidence.transport_bootstrap is not None
        input_manifests = {
            "pooled_heldout": _write_portable_input_manifest(
                stage,
                role="pooled_heldout",
                directory_name="pooled-heldout",
                bundle=evidence.pooled,
            ),
            "primary_bootstrap": _write_portable_input_manifest(
                stage,
                role="primary_bootstrap",
                directory_name="primary-bootstrap",
                bundle=evidence.bootstrap,
            ),
            "transport_bootstrap": _write_portable_input_manifest(
                stage,
                role="transport_bootstrap",
                directory_name="transport-bootstrap",
                bundle=evidence.transport_bootstrap,
            ),
        }
        output_manifest_path = stage / "analysis-output" / "submission-results-manifest.json"
        staged_files = sorted(path for path in stage.rglob("*") if path.is_file())
        output_manifest = {
            "schema_version": 1,
            "compiler": {
                "path": _logical_project_path(
                    Path(__file__),
                    fallback="scripts/build_submission_results.py",
                ),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "paper_sources": {
                "manuscript_shell": {
                    "path": _logical_project_path(
                        manuscript_path,
                        fallback="paper-sources/manuscript-shell.md",
                    ),
                    "sha256": _sha256(manuscript_path),
                },
                "abstract_shell": {
                    "path": _logical_project_path(
                        abstract_path,
                        fallback="paper-sources/abstract-shell.md",
                    ),
                    "sha256": _sha256(abstract_path),
                },
                "bibliography": {
                    "path": _logical_project_path(
                        references_path,
                        fallback="references.bib",
                    ),
                    "sha256": _sha256(references_path),
                },
            },
            "publication_environment": publication_environment,
            "input_manifests": input_manifests,
            "figure_sources": {
                "calibration_operating_cliff": calibration_cliff_sources,
            },
            "result_token_count": len(tokens),
            "conservative_abstract_word_count": abstract_count,
            "conservative_abstract_word_limit": MAX_CONSERVATIVE_ABSTRACT_WORDS,
            "pandoc_render_contract": {
                "working_directory": ".",
                "command": pandoc_gate["command"],
            },
            "pandoc_gate": pandoc_gate,
            "outputs": {
                path.relative_to(stage).as_posix(): _sha256(path) for path in staged_files
            },
        }
        _validate_portable_publication_manifest(output_manifest)
        _write_text(
            output_manifest_path,
            json.dumps(output_manifest, indent=2, sort_keys=True),
        )
        stage.replace(output)
        published = sorted(path for path in output.rglob("*") if path.is_file())
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return tuple(published)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile authenticated BRACE held-out results into paper artifacts."
    )
    parser.add_argument("--pooled-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--transport-bootstrap-dir", type=Path, required=True)
    parser.add_argument(
        "--manuscript-shell",
        type=Path,
        default=Path("paper/manuscript-results-shell.md"),
    )
    parser.add_argument(
        "--abstract-shell",
        type=Path,
        default=Path("paper/ssac27-abstract-results-shell.md"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    compile_submission(
        pooled_dir=args.pooled_dir,
        bootstrap_dir=args.bootstrap_dir,
        transport_bootstrap_dir=args.transport_bootstrap_dir,
        manuscript_shell=args.manuscript_shell,
        abstract_shell=args.abstract_shell,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
