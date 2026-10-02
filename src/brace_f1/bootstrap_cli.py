"""Auditable command-line orchestration for fixed-artifact BRACE bootstraps.

The statistical resampling lives in :mod:`brace_f1.bootstrap`; this module
owns input authentication, frozen-cohort checks, and transactional publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brace_f1.bootstrap import (
    BootstrapResult,
    _BootstrapArrays,
    _validated_arrays,
    paired_car_session_bootstrap,
    paired_two_stage_circuit_car_bootstrap,
)
from brace_f1.bootstrap_reliability import (
    ReliabilityBootstrapResult,
    reliability_bin_cluster_bootstrap,
)
from brace_f1.experiment import (
    HORIZONS_S,
    PRIMARY_BAYESIAN_METHOD,
    PRIMARY_DETERMINISTIC_METHOD,
    REGISTERED_METHODS,
    _canonical_json_hash,
    _code_content_hash,
    _runtime_environment_identity,
    _validate_frozen_config,
)
from brace_f1.io import DataValidationError

_UNIT_KEYS = ("circuit", "source_session_id", "car_id")
_CANONICAL_BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class _LoadedInputs:
    table: pd.DataFrame
    contribution_path: Path
    contribution_hash: str
    contribution_bytes: int
    config_path: Path
    config_file_hash: str
    study_config: dict[str, Any]
    analysis_arrays: _BootstrapArrays
    expected_circuits: tuple[str, ...]
    observed_circuits: tuple[str, ...]
    expected_car_sessions: int
    current_code_hash: str


@dataclass(frozen=True)
class _AuthenticatedInputs:
    experiment_manifest_path: Path
    experiment_manifest_hash: str | None
    experiment_manifest_record: dict[str, object] | None
    authenticated_artifacts: dict[Path, str]
    reliability_path: Path | None


@dataclass(frozen=True)
class _AnalysisOutputs:
    primary: BootstrapResult
    least_favorable: BootstrapResult | None
    reliability: ReliabilityBootstrapResult | None
    reliability_hash: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_contributions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise DataValidationError("bootstrap contributions must be CSV or Parquet")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired BRACE car-session bootstraps.")
    parser.add_argument("contributions", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270927)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--false-budget", type=float, default=2.0)
    parser.add_argument("--minimum-recall", type=float, default=0.50)
    parser.add_argument("--lead-grid", type=float, nargs="+", default=list(HORIZONS_S))
    parser.add_argument("--brace-method", default=PRIMARY_BAYESIAN_METHOD)
    parser.add_argument("--comparator-method", default=PRIMARY_DETERMINISTIC_METHOD)
    parser.add_argument("--study-config", type=Path, default=Path("configs/study.json"))
    parser.add_argument("--experiment-manifest", type=Path, default=None)
    parser.add_argument(
        "--reliability-contributions",
        type=Path,
        default=None,
        help=(
            "fixed-bin car-session sufficient statistics; required automatically "
            "for the canonical pooled analysis"
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-synthetic",
        action="store_true",
        help="test-only escape hatch; production requires the frozen four-circuit cohort",
    )
    return parser


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{label} must contain a JSON object")
    return value


def _validate_analysis_contract(
    args: argparse.Namespace,
    study_config: dict[str, Any],
) -> None:
    if args.allow_noncanonical_synthetic:
        return
    primary = study_config["primary_endpoint"]
    scoring = study_config["scoring"]
    expected = {
        "resamples": _CANONICAL_BOOTSTRAP_RESAMPLES,
        "seed": int(study_config["random_seed"]),
        "false_budget": float(primary["false_alerts_per_eligible_car_hour_max"]),
        "minimum_recall": float(primary["localized_event_recall_min"]),
        "lead_grid": tuple(float(value) for value in scoring["required_actual_lead_s"]),
        "brace_method": PRIMARY_BAYESIAN_METHOD,
        "comparator_method": PRIMARY_DETERMINISTIC_METHOD,
    }
    observed = {
        "resamples": args.resamples,
        "seed": args.seed,
        "false_budget": args.false_budget,
        "minimum_recall": args.minimum_recall,
        "lead_grid": tuple(float(value) for value in args.lead_grid),
        "brace_method": args.brace_method,
        "comparator_method": args.comparator_method,
    }
    mismatches = [name for name, value in observed.items() if value != expected[name]]
    if mismatches:
        raise DataValidationError(
            "canonical bootstrap requires the frozen production analysis contract; "
            f"noncanonical overrides {mismatches} require "
            "--allow-noncanonical-synthetic"
        )


def _load_inputs(args: argparse.Namespace) -> _LoadedInputs:
    contribution_path = args.contributions.resolve()
    if not contribution_path.is_file():
        raise DataValidationError(
            f"bootstrap contribution file does not exist: {contribution_path}"
        )
    contribution_hash = _sha256(contribution_path)
    table = _read_contributions(contribution_path)
    config_path = args.study_config.resolve()
    if not config_path.is_file():
        raise DataValidationError(f"frozen study config does not exist: {config_path}")
    config_file_hash = _sha256(config_path)
    study_config = _load_json(config_path, label="frozen study config")
    _validate_frozen_config(study_config)
    _validate_analysis_contract(args, study_config)
    arrays = _validated_arrays(
        table,
        false_budget_per_hour=args.false_budget,
        methods=(args.brace_method, args.comparator_method),
        lead_grid_s=args.lead_grid,
    )
    cohort = study_config["cohort"]
    expected_circuits = tuple(sorted(str(value) for value in cohort["circuits"]))
    observed_circuits = tuple(sorted(arrays.circuits))
    expected_car_sessions = int(cohort["car_sessions"])
    if not args.allow_noncanonical_synthetic and (
        observed_circuits != expected_circuits or len(arrays.units) != expected_car_sessions
    ):
        raise DataValidationError(
            "bootstrap CLI requires the canonical frozen cohort; use the explicit "
            "synthetic escape hatch only in tests"
        )
    return _LoadedInputs(
        table=table,
        contribution_path=contribution_path,
        contribution_hash=contribution_hash,
        contribution_bytes=contribution_path.stat().st_size,
        config_path=config_path,
        config_file_hash=config_file_hash,
        study_config=study_config,
        analysis_arrays=arrays,
        expected_circuits=expected_circuits,
        observed_circuits=observed_circuits,
        expected_car_sessions=expected_car_sessions,
        current_code_hash=_code_content_hash(),
    )


def _validate_manifest_artifacts(manifest: dict[str, Any]) -> dict[Path, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataValidationError("pooled experiment manifest has no artifact registry")
    authenticated: dict[Path, str] = {}
    for raw_path, expected_hash in artifacts.items():
        artifact_path = Path(str(raw_path))
        if not artifact_path.is_absolute():
            raise DataValidationError("pooled artifact registry must use absolute paths")
        artifact_path = artifact_path.resolve()
        if (
            not artifact_path.is_file()
            or not isinstance(expected_hash, str)
            or _sha256(artifact_path) != expected_hash
        ):
            raise DataValidationError(
                f"pooled experiment artifact is missing or stale: {artifact_path}"
            )
        authenticated[artifact_path] = expected_hash
    return authenticated


def _authenticate_inputs(
    args: argparse.Namespace,
    loaded: _LoadedInputs,
) -> _AuthenticatedInputs:
    manifest_path = (
        args.experiment_manifest.resolve()
        if args.experiment_manifest is not None
        else loaded.contribution_path.parent / "manifest.json"
    )
    reliability_path = (
        args.reliability_contributions.resolve()
        if args.reliability_contributions is not None
        else loaded.contribution_path.parent
        / "pooled-heldout-reliability-cluster-contributions.parquet"
    )
    if args.allow_noncanonical_synthetic:
        return _AuthenticatedInputs(
            manifest_path,
            None,
            None,
            {},
            reliability_path if args.reliability_contributions is not None else None,
        )
    if not manifest_path.is_file():
        raise DataValidationError(f"pooled experiment manifest does not exist: {manifest_path}")
    manifest_hash = _sha256(manifest_path)
    manifest = _load_json(manifest_path, label="pooled experiment manifest")
    if manifest.get("status") != "pooled_heldout_complete":
        raise DataValidationError("bootstrap requires a completed pooled held-out manifest")
    if manifest.get("config_content_hash") != _canonical_json_hash(loaded.study_config):
        raise DataValidationError(
            "pooled experiment manifest does not match the frozen study config"
        )
    if manifest.get("code_content_hash") != loaded.current_code_hash:
        raise DataValidationError(
            "pooled experiment manifest does not match the current analysis code"
        )
    authenticated = _validate_manifest_artifacts(manifest)
    if authenticated.get(loaded.contribution_path) != loaded.contribution_hash:
        raise DataValidationError(
            "bootstrap contribution file is not authenticated by the pooled manifest"
        )
    if reliability_path not in authenticated:
        raise DataValidationError(
            "reliability contribution file is not authenticated by the pooled manifest"
        )
    return _AuthenticatedInputs(
        experiment_manifest_path=manifest_path,
        experiment_manifest_hash=manifest_hash,
        experiment_manifest_record={
            "path": str(manifest_path),
            "sha256": manifest_hash,
        },
        authenticated_artifacts=authenticated,
        reliability_path=reliability_path,
    )


def _bootstrap_function(args: argparse.Namespace) -> Callable[..., BootstrapResult]:
    if args.two_stage:
        return paired_two_stage_circuit_car_bootstrap
    return paired_car_session_bootstrap


def _run_l_star_bootstraps(
    args: argparse.Namespace,
    loaded: _LoadedInputs,
) -> tuple[BootstrapResult, BootstrapResult | None]:
    function = _bootstrap_function(args)
    kwargs = {
        "n_resamples": args.resamples,
        "seed": args.seed,
        "false_budget_per_hour": args.false_budget,
        "minimum_recall": args.minimum_recall,
        "lead_grid_s": args.lead_grid,
        "brace_method": args.brace_method,
        "comparator_method": args.comparator_method,
    }
    primary = function(loaded.table, **kwargs)
    least_favorable: BootstrapResult | None = None
    if "least_favorable_false_proposals" in loaded.table:
        least_favorable = function(
            loaded.table,
            **kwargs,
            false_count_column="least_favorable_false_proposals",
        )
    elif not args.allow_noncanonical_synthetic:
        raise DataValidationError(
            "canonical bootstrap requires least-favorable false-count contributions"
        )
    return primary, least_favorable


def _run_reliability_bootstrap(
    args: argparse.Namespace,
    loaded: _LoadedInputs,
    authenticated: _AuthenticatedInputs,
) -> tuple[ReliabilityBootstrapResult | None, str | None]:
    path = authenticated.reliability_path
    if path is None:
        return None, None
    if not path.is_file():
        raise DataValidationError(f"reliability contribution file does not exist: {path}")
    source_hash = _sha256(path)
    table = _read_contributions(path)
    result = reliability_bin_cluster_bootstrap(
        table,
        n_resamples=args.resamples,
        seed=args.seed,
        horizons_s=HORIZONS_S,
        n_bins=10,
        expected_methods=REGISTERED_METHODS,
    )
    units = table.loc[:, _UNIT_KEYS].drop_duplicates()
    circuits = tuple(sorted(units["circuit"].astype(str).unique()))
    if not args.allow_noncanonical_synthetic and (
        circuits != loaded.expected_circuits or len(units) != loaded.expected_car_sessions
    ):
        raise DataValidationError("reliability bootstrap requires the canonical frozen cohort")
    return result, source_hash


def _verify_stable_inputs(
    loaded: _LoadedInputs,
    authenticated: _AuthenticatedInputs,
    reliability_hash: str | None,
) -> None:
    if (
        _sha256(loaded.contribution_path) != loaded.contribution_hash
        or _sha256(loaded.config_path) != loaded.config_file_hash
    ):
        raise DataValidationError("bootstrap input or study config changed during analysis")
    if authenticated.experiment_manifest_hash is not None:
        if (
            _sha256(authenticated.experiment_manifest_path)
            != authenticated.experiment_manifest_hash
            or _code_content_hash() != loaded.current_code_hash
        ):
            raise DataValidationError("bootstrap provenance changed during analysis")
        for path, expected_hash in authenticated.authenticated_artifacts.items():
            if not path.is_file() or _sha256(path) != expected_hash:
                raise DataValidationError(f"pooled artifact changed during bootstrap: {path}")
    if authenticated.reliability_path is not None and (
        reliability_hash is None or _sha256(authenticated.reliability_path) != reliability_hash
    ):
        raise DataValidationError("reliability contribution file changed during analysis")


def _result_fields(result: BootstrapResult) -> dict[str, object]:
    return {
        "l_star_brace": result.l_star_brace,
        "l_star_brace_percentile_95": list(result.l_star_brace_percentile_95),
        "l_star_comparator": result.l_star_comparator,
        "l_star_comparator_percentile_95": list(result.l_star_comparator_percentile_95),
        "delta_l_star": result.delta_l_star,
        "delta_l_star_percentile_95": list(result.percentile_95),
    }


def _summary_payload(outputs: _AnalysisOutputs) -> dict[str, object]:
    payload = {
        **_result_fields(outputs.primary),
        "n_resamples": outputs.primary.n_resamples,
        "seed": outputs.primary.seed,
        "cluster_level": outputs.primary.cluster_level,
        "models_and_thresholds_refit": outputs.primary.models_and_thresholds_refit,
        "least_favorable_unresolved_counted_as_false": None,
    }
    if outputs.least_favorable is not None:
        payload["least_favorable_unresolved_counted_as_false"] = {
            **_result_fields(outputs.least_favorable),
            "false_count_column": "least_favorable_false_proposals",
            "models_and_thresholds_refit": False,
        }
    return payload


def _save_draws(
    staging_dir: Path,
    outputs: _AnalysisOutputs,
) -> list[Path]:
    pairs = (
        ("delta-l-star-draws.npy", outputs.primary.delta_l_star_draws),
        ("brace-l-star-draws.npy", outputs.primary.l_star_brace_draws),
        ("comparator-l-star-draws.npy", outputs.primary.l_star_comparator_draws),
    )
    paths: list[Path] = []
    for name, draws in pairs:
        path = staging_dir / name
        np.save(path, draws, allow_pickle=False)
        paths.append(path)
    if outputs.least_favorable is not None:
        sensitivity_pairs = (
            (
                "least-favorable-delta-l-star-draws.npy",
                outputs.least_favorable.delta_l_star_draws,
            ),
            (
                "least-favorable-brace-l-star-draws.npy",
                outputs.least_favorable.l_star_brace_draws,
            ),
            (
                "least-favorable-comparator-l-star-draws.npy",
                outputs.least_favorable.l_star_comparator_draws,
            ),
        )
        for name, draws in sensitivity_pairs:
            path = staging_dir / name
            np.save(path, draws, allow_pickle=False)
            paths.append(path)
    return paths


def _write_result_artifacts(
    staging_dir: Path,
    outputs: _AnalysisOutputs,
    *,
    started: float,
) -> list[Path]:
    summary_path = staging_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_summary_payload(outputs), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = [summary_path, *_save_draws(staging_dir, outputs)]
    runtime_path = staging_dir / "runtime.jsonl"
    runtime_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "step": "paired_bootstrap_complete",
                "elapsed_seconds": time.perf_counter() - started,
                "n_resamples": outputs.primary.n_resamples,
                "cluster_level": outputs.primary.cluster_level,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(runtime_path)
    if outputs.reliability is not None:
        path = staging_dir / "reliability-bin-intervals.csv"
        outputs.reliability.summary.to_csv(path, index=False, lineterminator="\n")
        artifacts.append(path)
    return artifacts


def _warning(args: argparse.Namespace, circuit_count: int) -> str:
    if args.two_stage:
        return (
            f"Two-stage transport sensitivity uses only {circuit_count} circuit "
            "clusters; report this limitation with every interval."
        )
    return (
        "The primary bootstrap conditions on the observed circuit set and does "
        "not estimate transport uncertainty to new circuits."
    )


def _manifest_payload(
    args: argparse.Namespace,
    loaded: _LoadedInputs,
    authenticated: _AuthenticatedInputs,
    outputs: _AnalysisOutputs,
    output_dir: Path,
    staged_artifacts: Sequence[Path],
) -> dict[str, object]:
    least_favorable = outputs.least_favorable is not None
    reliability = outputs.reliability
    return {
        "schema_version": 1,
        "input": {
            "path": str(loaded.contribution_path),
            "bytes": loaded.contribution_bytes,
            "sha256": loaded.contribution_hash,
        },
        "analysis": {
            "n_resamples": outputs.primary.n_resamples,
            "seed": outputs.primary.seed,
            "cluster_level": outputs.primary.cluster_level,
            "models_and_thresholds_refit": False,
        },
        "estimand": {
            "brace_method": args.brace_method,
            "comparator_method": args.comparator_method,
            "false_budget_per_hour": args.false_budget,
            "minimum_recall": args.minimum_recall,
            "lead_grid_s": [float(value) for value in args.lead_grid],
            "least_favorable_false_count_column": (
                "least_favorable_false_proposals" if least_favorable else None
            ),
        },
        "interval": {
            "construction": "percentile_95",
            "paired": True,
            "resampling_unit": outputs.primary.cluster_level,
        },
        "sample": {
            "circuit_count": len(loaded.analysis_arrays.circuits),
            "car_session_count": int(len(loaded.analysis_arrays.units)),
            "circuits": list(loaded.observed_circuits),
        },
        "study_config": {
            "path": str(loaded.config_path),
            "sha256": loaded.config_file_hash,
            "canonical_content_hash": _canonical_json_hash(loaded.study_config),
        },
        "code_content_hash": loaded.current_code_hash,
        "runtime_environment": _runtime_environment_identity(),
        "experiment_manifest": authenticated.experiment_manifest_record,
        "noncanonical_synthetic": bool(args.allow_noncanonical_synthetic),
        "warnings": [_warning(args, len(loaded.analysis_arrays.circuits))],
        "reliability_bootstrap": (
            None
            if reliability is None or authenticated.reliability_path is None
            else {
                "input_path": str(authenticated.reliability_path),
                "input_sha256": outputs.reliability_hash,
                "n_resamples": reliability.n_resamples,
                "seed": reliability.seed,
                "cluster_level": reliability.cluster_level,
                "fixed_equal_width_bins": 10,
            }
        ),
        "artifacts": {str(output_dir / path.name): _sha256(path) for path in staged_artifacts},
    }


def _publish_bundle(
    args: argparse.Namespace,
    loaded: _LoadedInputs,
    authenticated: _AuthenticatedInputs,
    outputs: _AnalysisOutputs,
    output_dir: Path,
    *,
    started: float,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent)
    )
    try:
        artifacts = _write_result_artifacts(staging_dir, outputs, started=started)
        manifest = _manifest_payload(
            args,
            loaded,
            authenticated,
            outputs,
            output_dir,
            artifacts,
        )
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run paired L-star and reliability bootstraps from immutable artifacts."""

    started = time.perf_counter()
    args = _parser().parse_args(argv)
    if args.output is None:
        raise DataValidationError("bootstrap CLI requires --output for auditable artifacts")
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise DataValidationError(f"bootstrap output already exists: {output_dir}")
    loaded = _load_inputs(args)
    authenticated = _authenticate_inputs(args, loaded)
    primary, least_favorable = _run_l_star_bootstraps(args, loaded)
    reliability, reliability_hash = _run_reliability_bootstrap(args, loaded, authenticated)
    outputs = _AnalysisOutputs(
        primary=primary,
        least_favorable=least_favorable,
        reliability=reliability,
        reliability_hash=reliability_hash,
    )
    _verify_stable_inputs(loaded, authenticated, reliability_hash)
    _publish_bundle(
        args,
        loaded,
        authenticated,
        outputs,
        output_dir,
        started=started,
    )
    return 0
