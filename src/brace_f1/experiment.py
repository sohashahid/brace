"""Leakage-safe leave-one-circuit-out experiment orchestration for BRACE.

The functions in this module separate causal score artifacts from future truth.
Loading and fitting may inspect only the declared fit partition; calibration
labels are joined transiently; held-out outcomes are not needed until every
fold-specific calibrator and operating threshold has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from brace_f1.baselines import (
    RegularizedSideHazard,
    TunedDeterministicResidualDynamics,
    forecast_constant_turn_rate,
    forecast_constant_velocity,
)
from brace_f1.calibration import MonotonePlattCalibrator
from brace_f1.features import FEATURE_COLUMNS, TrackReference, derive_causal_features
from brace_f1.forecast import (
    canonical_forecast_row_ids,
    forecast_ctra_particles,
    planar_states_from_features,
)
from brace_f1.geometry import CircuitCorridor
from brace_f1.io import DataValidationError, read_ascii_pcd
from brace_f1.metrics import (
    calibration_intercept_slope,
    expected_calibration_error,
    false_proposal_rate_upper,
)
from brace_f1.policy import (
    ProposalConfig,
    label_proposals,
    run_proposal_state_machine,
    select_neighborhood_candidates,
    summarize_detection,
)
from brace_f1.residual import (
    RESIDUAL_FEATURE_COLUMNS,
    BayesianResidualDynamics,
    DeterministicResidualDynamics,
    ResidualTrainingSet,
    build_residual_training_pairs,
    residual_design_from_features,
)

FRAME_TARGET_KEYS: tuple[str, ...] = (
    "circuit",
    "car_id",
    "frame_index",
    "time_seconds",
)
UNIT_KEYS: tuple[str, ...] = (
    "circuit",
    "source_session_id",
    "car_id",
    "source_revision",
    "source_unit_sha256",
)
HORIZONS_S: tuple[float, ...] = (0.25, 0.50, 1.00, 1.50)
DECLARED_SCORING_RATE_HZ = 20.0
RIDGE_ALPHA_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
PRIMARY_BAYESIAN_METHOD = "brace_bayesian"
PRIMARY_DETERMINISTIC_METHOD = "posterior_mean_twin"
CONSTANT_VELOCITY_METHOD = "constant_velocity"
CONSTANT_TURN_RATE_METHOD = "constant_turn_rate"
SIDE_HAZARD_METHOD = "side_hazard"
CALIBRATION_TUNED_DETERMINISTIC_METHOD = "calibration_mse_tuned_dynamics"
MIN_FALSE_COUNT_CAPACITY_FOR_ESTIMABILITY = 1.0
REGISTERED_METHODS: tuple[str, ...] = (
    PRIMARY_BAYESIAN_METHOD,
    PRIMARY_DETERMINISTIC_METHOD,
    CONSTANT_VELOCITY_METHOD,
    CONSTANT_TURN_RATE_METHOD,
    SIDE_HAZARD_METHOD,
    CALIBRATION_TUNED_DETERMINISTIC_METHOD,
)
FROZEN_STUDY_CONFIG_SHA256 = "ef8c3ebc58de1122bab1dadfbbad977ae10ce65b24653ef09040243cd7e6c00c"

_TARGET_PREFIXES = (
    "next_",
    "time_to_next_",
    "outcome_evaluable_",
    "projected_onset_",
    "qualifying_event_",
)
_TARGET_NAMES = {
    "candidate_event_id_at_source",
    "offline_artifact_excluded",
    "offline_artifact_seed",
    "offline_eligible_grid",
    "raw_outside_at_source",
    "continuous_segment_end_time_seconds",
    "continuous_segment_remaining_seconds",
}


@dataclass(frozen=True)
class ExperimentPaths:
    """The immutable inputs required by the public DeepRacing experiment."""

    frames: Path
    targets: Path
    events: Path
    timing: Path
    splits: Path
    source_manifest: Path
    build_manifest: Path | None = None


@dataclass(frozen=True)
class ExperimentData:
    """Validated experiment inputs; score-time truth remains a separate table."""

    frames: pd.DataFrame
    targets: pd.DataFrame
    frame_targets: pd.DataFrame
    events: pd.DataFrame
    timing: pd.DataFrame
    splits: pd.DataFrame
    corridors: Mapping[str, CircuitCorridor]
    input_content_hash: str
    source_manifest_hash: str
    build_manifest_hash: str | None = None


@dataclass(frozen=True)
class RidgeSelection:
    """Recorded inner-LOCO Gaussian-NLL ridge selection."""

    selected_alpha: float
    mean_gaussian_nll_by_alpha: Mapping[float, float]
    fold_scores: pd.DataFrame


@dataclass(frozen=True)
class _EqualClusterResidualFit:
    """Deterministic prior-mean analogue of the production cluster fit."""

    coefficients: NDArray[np.float64]
    process_covariance: NDArray[np.float64]
    feature_center: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    cluster_count: int

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        design = np.asarray(X, dtype=np.float64)
        if design.ndim != 2 or design.shape[1] != self.feature_center.size:
            raise DataValidationError("residual validation design has the wrong shape")
        if not np.isfinite(design).all():
            raise DataValidationError("residual validation design contains non-finite values")
        scaled = (design - self.feature_center) / self.feature_scale
        augmented = np.column_stack((np.ones(design.shape[0]), scaled))
        return augmented @ self.coefficients


@dataclass(frozen=True)
class HazardTrainingRows:
    """Fit-partition one-step competing-onset labels for the side hazard."""

    X: NDArray[np.float64]
    labels: NDArray[np.str_]
    feature_names: tuple[str, ...]
    row_keys: pd.DataFrame


@dataclass(frozen=True)
class ScaledSideHazard:
    """Side hazard with fit-bound median/IQR scaling."""

    model: RegularizedSideHazard
    feature_names: tuple[str, ...]
    feature_center: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    content_hash: str

    def __post_init__(self) -> None:
        center = np.array(self.feature_center, dtype=np.float64, copy=True)
        scale = np.array(self.feature_scale, dtype=np.float64, copy=True)
        if center.shape != (len(self.feature_names),) or scale.shape != center.shape:
            raise DataValidationError("hazard scaling dimensions do not match feature names")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise DataValidationError("hazard scaling must be finite with positive IQR values")
        center.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)

    @classmethod
    def fit(
        cls,
        X: ArrayLike,
        labels: ArrayLike,
        *,
        feature_names: Sequence[str],
        regularization: float = 1.0,
    ) -> ScaledSideHazard:
        design = np.asarray(X, dtype=np.float64)
        if design.ndim != 2 or design.shape[0] == 0 or not np.isfinite(design).all():
            raise DataValidationError("hazard X must be a non-empty finite matrix")
        names = tuple(str(name) for name in feature_names)
        if len(names) != design.shape[1]:
            raise DataValidationError("hazard feature names do not match X")
        center = np.median(design, axis=0)
        quartiles = np.percentile(design, [25.0, 75.0], axis=0)
        scale = np.where(quartiles[1] - quartiles[0] > 0.0, quartiles[1] - quartiles[0], 1.0)
        model = RegularizedSideHazard.fit(
            (design - center) / scale,
            labels,
            feature_names=names,
            regularization=regularization,
        )
        digest = hashlib.sha256()
        digest.update(model.content_hash.encode())
        digest.update(np.ascontiguousarray(center).tobytes())
        digest.update(np.ascontiguousarray(scale).tobytes())
        return cls(model, names, center, scale, digest.hexdigest())

    def _transform(self, X: ArrayLike) -> NDArray[np.float64]:
        design = np.asarray(X, dtype=np.float64)
        if design.ndim != 2 or design.shape[1] != len(self.feature_names):
            raise DataValidationError("hazard X feature count does not match model")
        if not np.isfinite(design).all():
            raise DataValidationError("hazard X contains non-finite values")
        return (design - self.feature_center) / self.feature_scale

    def predict_horizon_probabilities(
        self,
        X: ArrayLike,
        *,
        horizons_s: Sequence[float],
        dt_s: float,
    ) -> NDArray[np.float64]:
        return self.model.predict_horizon_probabilities(
            self._transform(X), horizons_s=horizons_s, dt_s=dt_s
        )


@dataclass(frozen=True)
class FittedFoldModels:
    """Fit-only model bundle for one outer fold."""

    ridge_selection: RidgeSelection
    bayesian: BayesianResidualDynamics
    posterior_mean_twin: DeterministicResidualDynamics
    side_hazard: ScaledSideHazard
    calibration_tuned_dynamics: TunedDeterministicResidualDynamics | None
    models: Mapping[str, object]


@dataclass(frozen=True)
class HeldoutEvaluation:
    """Post-threshold held-out summaries and bootstrap-ready car contributions."""

    metrics: pd.DataFrame
    car_contributions: pd.DataFrame
    proposal_labels: pd.DataFrame
    delay_metrics: pd.DataFrame
    delay_car_contributions: pd.DataFrame


@dataclass(frozen=True)
class PreparedOuterFold:
    """Causal features and separately held future targets for one outer fold."""

    causal_features: pd.DataFrame
    targets: pd.DataFrame

    def fit_model_table(self) -> pd.DataFrame:
        """Join fit/calibration onset fields while keeping held-out truth sealed."""

        columns = [
            *FRAME_TARGET_KEYS,
            "qualifying_event_onset_projected",
            "next_excursion_side",
        ]
        fit = self.causal_features.loc[
            self.causal_features["partition"].astype(str).isin(["fit", "calibration"])
        ].copy()
        target = self.targets.loc[
            self.targets["partition"].astype(str).isin(["fit", "calibration"]), columns
        ]
        return fit.merge(
            target,
            on=list(FRAME_TARGET_KEYS),
            how="left",
            validate="one_to_one",
        )


@dataclass(frozen=True)
class FoldRunResult:
    """Paths and state for one independently resumable outer fold."""

    fold: str
    status: str
    manifest_path: Path
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class HeldoutAggregateResult:
    """Auditable pooled outputs created only after every circuit fold is complete."""

    status: str
    manifest_path: Path
    artifact_paths: tuple[Path, ...]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise DataValidationError(f"{name} missing columns: {sorted(missing)}")


def _read_table(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"missing {name} table: {path}")
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise DataValidationError(f"cannot read {name} table {path}: {exc}") from exc
    raise DataValidationError(f"{name} table must be CSV or Parquet: {path}")


def join_frames_targets(frames: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Perform the registered one-to-one join on all four score-time keys."""

    required = set(FRAME_TARGET_KEYS)
    _require_columns(frames, required, "frame table")
    _require_columns(targets, required, "target table")
    if frames.empty or targets.empty:
        raise DataValidationError("frame and target tables must be non-empty")
    if (
        frames.duplicated(list(FRAME_TARGET_KEYS)).any()
        or targets.duplicated(list(FRAME_TARGET_KEYS)).any()
    ):
        raise DataValidationError("frame-target four-key rows must be one-to-one")
    leaked = {
        column
        for column in frames.columns
        if column in _TARGET_NAMES or column.startswith(_TARGET_PREFIXES)
    }
    if leaked:
        raise DataValidationError(
            f"future/offline target fields leaked into frames: {sorted(leaked)}"
        )
    frame_keys = pd.MultiIndex.from_frame(frames.loc[:, FRAME_TARGET_KEYS])
    target_keys = pd.MultiIndex.from_frame(targets.loc[:, FRAME_TARGET_KEYS])
    if len(frame_keys) != len(target_keys) or set(frame_keys) != set(target_keys):
        raise DataValidationError("frame and target tables do not contain the same four-key rows")
    joined = frames.merge(
        targets,
        on=list(FRAME_TARGET_KEYS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_target"),
    )
    if len(joined) != len(frames):
        raise DataValidationError("frame-target join changed row accounting")
    return joined


def attach_partition_once(
    frames: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    fold_test_circuit: str,
) -> pd.DataFrame:
    """Attach exactly one whole-car partition for one outer fold."""

    _require_columns(frames, set(UNIT_KEYS), "frame table")
    _require_columns(
        splits,
        {"fold_test_circuit", "partition", *UNIT_KEYS},
        "split table",
    )
    fold = splits.loc[splits["fold_test_circuit"].astype(str) == str(fold_test_circuit)].copy()
    if fold.empty:
        raise DataValidationError(f"split table has no outer fold {fold_test_circuit}")
    if fold.duplicated(list(UNIT_KEYS)).any():
        raise DataValidationError("a complete car-session has multiple assignments within a fold")
    partition = fold["partition"].astype(str)
    invalid = sorted(set(partition).difference({"fit", "calibration", "test"}))
    if invalid:
        raise DataValidationError(f"split table contains invalid partitions: {invalid}")
    heldout = fold["circuit"].astype(str) == str(fold_test_circuit)
    if not (partition[heldout] == "test").all() or (partition[~heldout] == "test").any():
        raise DataValidationError("held-out circuit must be the entire and only test partition")
    attached = frames.merge(
        fold.loc[:, [*UNIT_KEYS, "fold_test_circuit", "partition"]],
        on=list(UNIT_KEYS),
        how="left",
        validate="many_to_one",
    )
    if attached["partition"].isna().any() or len(attached) != len(frames):
        raise DataValidationError("some frame rows lack exactly one split assignment")
    if (
        attached.groupby(list(UNIT_KEYS), sort=False, dropna=False)["partition"].nunique() > 1
    ).any():
        raise DataValidationError("a complete car-session was split across partitions")
    return attached


def _resolve_manifest_path(raw: object, *, manifest_path: Path, base_dir: Path) -> Path:
    value = Path(str(raw))
    if value.is_absolute():
        return value
    candidates = (base_dir / value, manifest_path.parent / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _load_corridors(
    source_manifest: Path,
    *,
    circuits: set[str],
    base_dir: Path,
) -> tuple[dict[str, CircuitCorridor], list[Path]]:
    manifest = _read_table(source_manifest, "source manifest")
    _require_columns(
        manifest,
        {"circuit", "role", "filename", "local_path", "bytes", "sha256"},
        "source manifest",
    )
    boundary = manifest.loc[manifest["role"].astype(str) == "boundary"].copy()
    required_files = {"center_line.pcd", "inner_boundary.pcd", "outer_boundary.pcd"}
    corridors: dict[str, CircuitCorridor] = {}
    paths: list[Path] = []
    for circuit in sorted(circuits):
        rows = boundary.loc[boundary["circuit"].astype(str) == circuit]
        if len(rows) != 3 or set(rows["filename"].astype(str)) != required_files:
            raise DataValidationError(
                f"circuit {circuit} must have exactly one center and two boundary PCD files"
            )
        if rows["filename"].duplicated().any():
            raise DataValidationError(f"circuit {circuit} has duplicate boundary manifest rows")
        clouds = {}
        for row in rows.itertuples(index=False):
            path = _resolve_manifest_path(
                row.local_path,
                manifest_path=source_manifest,
                base_dir=base_dir,
            )
            if not path.is_file():
                raise DataValidationError(f"boundary PCD is missing: {path}")
            try:
                expected_bytes = int(row.bytes)
            except (TypeError, ValueError) as exc:
                raise DataValidationError("boundary manifest byte counts must be integers") from exc
            if path.stat().st_size != expected_bytes:
                raise DataValidationError(f"boundary PCD byte count mismatch: {path}")
            expected_hash = str(row.sha256).lower()
            if _sha256(path) != expected_hash:
                raise DataValidationError(f"boundary PCD SHA-256 mismatch: {path}")
            clouds[str(row.filename)] = read_ascii_pcd(path)
            paths.append(path)
        corridors[circuit] = CircuitCorridor.from_point_clouds(
            centerline=clouds["center_line.pcd"],
            boundary_a=clouds["inner_boundary.pcd"],
            boundary_b=clouds["outer_boundary.pcd"],
            boundary_a_id="inner_boundary.pcd",
            boundary_b_id="outer_boundary.pcd",
        )
    return corridors, paths


def _input_content_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths}, key=lambda value: str(value)):
        digest.update(str(path).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _validate_build_manifest(paths: ExperimentPaths) -> str | None:
    if paths.build_manifest is None:
        return None
    manifest_path = Path(paths.build_manifest)
    if not manifest_path.is_file():
        raise DataValidationError(f"missing authoritative build manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read authoritative build manifest: {exc}") from exc
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise DataValidationError("authoritative build manifest has no output records")
    for table_path in (paths.frames, paths.targets, paths.events, paths.timing, paths.splits):
        table = Path(table_path).resolve()
        if not table.is_file():
            raise DataValidationError(
                f"processed table bound by build manifest is missing: {table}"
            )
        matching = [
            record
            for record in outputs
            if isinstance(record, dict) and Path(str(record.get("path", ""))).name == table.name
        ]
        if len(matching) != 1:
            raise DataValidationError(
                f"authoritative build manifest does not uniquely bind {table.name}"
            )
        record = matching[0]
        if table.stat().st_size != int(record.get("bytes", -1)) or _sha256(table) != str(
            record.get("sha256", "")
        ):
            raise DataValidationError(f"processed table differs from build manifest: {table}")
    input_manifest = payload.get("input_manifest")
    if not isinstance(input_manifest, dict) or _sha256(Path(paths.source_manifest)) != str(
        input_manifest.get("sha256", "")
    ):
        raise DataValidationError("source manifest differs from authoritative build manifest")
    return _sha256(manifest_path)


def load_experiment_data(
    paths: ExperimentPaths,
    *,
    base_dir: Path | None = None,
) -> ExperimentData:
    """Load and validate cohort tables and PCD geometry without scoring held-out rows."""

    resolved_values: dict[str, Path | None] = {}
    for name in paths.__dataclass_fields__:
        value = getattr(paths, name)
        resolved_values[name] = None if value is None else Path(value).resolve()
    resolved = ExperimentPaths(**resolved_values)  # type: ignore[arg-type]
    root = Path.cwd() if base_dir is None else Path(base_dir).resolve()
    build_manifest_hash = _validate_build_manifest(resolved)
    frames = _read_table(resolved.frames, "frames")
    targets = _read_table(resolved.targets, "targets")
    events = _read_table(resolved.events, "events")
    timing = _read_table(resolved.timing, "timing")
    splits = _read_table(resolved.splits, "splits")
    frame_targets = join_frames_targets(frames, targets)
    _require_columns(
        events,
        {
            "candidate_event_id",
            "circuit",
            "car_id",
            "start_time_seconds",
            "side_at_onset",
            "segment_bin_25m_at_onset",
            "qualified",
        },
        "event table",
    )
    _require_columns(timing, {"buffered_exposure_seconds", *UNIT_KEYS}, "timing table")
    if timing.duplicated(list(UNIT_KEYS)).any():
        raise DataValidationError("timing table must contain one row per car-session")
    circuits = set(frames["circuit"].astype(str))
    if set(events["circuit"].astype(str)).difference(circuits):
        raise DataValidationError("event table contains circuits absent from frames")
    if set(timing["circuit"].astype(str)) != circuits:
        raise DataValidationError("timing and frame circuits differ")
    fold_names = set(splits["fold_test_circuit"].astype(str))
    if fold_names != circuits:
        raise DataValidationError("LOCO folds must name every and only frame circuit")
    for fold_name in sorted(fold_names):
        attach_partition_once(frames, splits, fold_test_circuit=fold_name)
    corridors, pcd_paths = _load_corridors(
        resolved.source_manifest,
        circuits=circuits,
        base_dir=root,
    )
    input_paths = [
        resolved.frames,
        resolved.targets,
        resolved.events,
        resolved.timing,
        resolved.splits,
        resolved.source_manifest,
        *pcd_paths,
    ]
    if resolved.build_manifest is not None:
        input_paths.append(resolved.build_manifest)
    return ExperimentData(
        frames=frames,
        targets=targets,
        frame_targets=frame_targets,
        events=events,
        timing=timing,
        splits=splits,
        corridors=corridors,
        input_content_hash=_input_content_hash(input_paths),
        source_manifest_hash=_sha256(resolved.source_manifest),
        build_manifest_hash=build_manifest_hash,
    )


def prepare_outer_fold(data: ExperimentData, *, outer_fold: str) -> PreparedOuterFold:
    """Derive causal features and attach whole-car partitions for one LOCO fold."""

    attached = attach_partition_once(
        data.frames,
        data.splits,
        fold_test_circuit=outer_fold,
    )
    references = {
        circuit: TrackReference(corridor.centerline) for circuit, corridor in data.corridors.items()
    }
    derived = derive_causal_features(attached, references)
    feature_values = derived.loc[:, [*FRAME_TARGET_KEYS, *FEATURE_COLUMNS]]
    causal = attached.merge(
        feature_values,
        on=list(FRAME_TARGET_KEYS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_derived"),
    )
    for column in FEATURE_COLUMNS:
        derived_column = f"{column}_derived"
        if derived_column in causal:
            causal[column] = causal.pop(derived_column)
    leaked = {
        column
        for column in causal.columns
        if column in _TARGET_NAMES or column.startswith(_TARGET_PREFIXES)
    }
    if leaked:
        raise DataValidationError(
            f"future truth leaked into prepared causal features: {sorted(leaked)}"
        )
    partition_keys = causal.loc[:, [*FRAME_TARGET_KEYS, "fold_test_circuit", "partition"]]
    targets = data.targets.merge(
        partition_keys,
        on=list(FRAME_TARGET_KEYS),
        how="left",
        validate="one_to_one",
    )
    if targets["partition"].isna().any():
        raise DataValidationError("prepared targets lack a fold partition")
    return PreparedOuterFold(causal.reset_index(drop=True), targets.reset_index(drop=True))


def _mean_gaussian_nll(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> float:
    residual = np.asarray(observed, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if residual.ndim != 2 or matrix.shape != (residual.shape[1], residual.shape[1]):
        raise DataValidationError("Gaussian NLL residual and covariance dimensions differ")
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-9)
    eigenvalues = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
    log_determinant = float(np.log(eigenvalues).sum())
    mahalanobis = np.einsum("ni,ij,nj->n", residual, inverse, residual)
    dimension = residual.shape[1]
    return float(np.mean(0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + mahalanobis)))


def _fit_equal_cluster_residual(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    circuits: ArrayLike,
    cars: ArrayLike,
    ridge_alpha: float,
) -> _EqualClusterResidualFit:
    """Fit the deterministic equal-circuit/equal-car analogue used for ridge tuning.

    Scaling remains the registered rowwise fit-partition median/IQR rule.  The
    normal equations then give equal total weight to circuits and equal total
    weight to cars within a circuit, matching the prior mean of the production
    two-stage Bayesian bootstrap.  Multiplication by the number of car clusters
    preserves the production interpretation of ``ridge_alpha``.
    """

    design = np.asarray(X, dtype=np.float64)
    target = np.asarray(Y, dtype=np.float64)
    circuit_array = np.asarray(circuits, dtype=np.str_)
    car_array = np.asarray(cars, dtype=np.str_)
    if (
        design.ndim != 2
        or target.ndim != 2
        or design.shape[0] == 0
        or target.shape[0] != design.shape[0]
        or circuit_array.shape != (design.shape[0],)
        or car_array.shape != circuit_array.shape
        or not np.isfinite(design).all()
        or not np.isfinite(target).all()
    ):
        raise DataValidationError("equal-cluster residual inputs are not finite aligned matrices")
    if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
        raise DataValidationError("ridge alpha must be finite and non-negative")

    center = np.median(design, axis=0)
    quartiles = np.percentile(design, [25.0, 75.0], axis=0)
    scale = np.where(quartiles[1] - quartiles[0] > 0.0, quartiles[1] - quartiles[0], 1.0)
    scaled = (design - center) / scale
    augmented = np.column_stack((np.ones(design.shape[0]), scaled))

    circuit_names = np.unique(circuit_array)
    normalized_row_weight = np.zeros(design.shape[0], dtype=np.float64)
    cluster_count = 0
    for circuit in circuit_names:
        circuit_rows = np.flatnonzero(circuit_array == circuit)
        circuit_cars = np.unique(car_array[circuit_rows])
        cluster_count += int(circuit_cars.size)
        for car in circuit_cars:
            rows = np.flatnonzero((circuit_array == circuit) & (car_array == car))
            normalized_row_weight[rows] = 1.0 / (
                circuit_names.size * circuit_cars.size * rows.size
            )
    if cluster_count == 0 or not np.isclose(normalized_row_weight.sum(), 1.0):
        raise DataValidationError("equal-cluster residual weights do not sum to one")

    fit_weight = normalized_row_weight * cluster_count
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    normal = augmented.T @ (augmented * fit_weight[:, None]) + penalty
    rhs = augmented.T @ (target * fit_weight[:, None])
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]

    residual = target - augmented @ coefficients
    covariance = (residual * normalized_row_weight[:, None]).T @ residual
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    covariance += np.eye(target.shape[1], dtype=np.float64) * 1e-9
    return _EqualClusterResidualFit(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        process_covariance=np.asarray(covariance, dtype=np.float64),
        feature_center=np.asarray(center, dtype=np.float64),
        feature_scale=np.asarray(scale, dtype=np.float64),
        cluster_count=cluster_count,
    )


def select_ridge_alpha_inner_loco(
    pairs: ResidualTrainingSet,
    *,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
) -> RidgeSelection:
    """Select ridge alpha by equal-circuit mean inner-LOCO Gaussian NLL."""

    design = np.asarray(pairs.X, dtype=np.float64)
    target = np.asarray(pairs.Y, dtype=np.float64)
    circuits = np.asarray(pairs.circuits, dtype=np.str_)
    cars = np.asarray(pairs.cars, dtype=np.str_)
    if design.ndim != 2 or target.ndim != 2 or target.shape[0] != design.shape[0]:
        raise DataValidationError("residual training matrices are not aligned")
    if circuits.shape != (design.shape[0],) or cars.shape != circuits.shape:
        raise DataValidationError("residual circuit/car labels are not aligned")
    circuit_names = np.unique(circuits)
    if circuit_names.size < 2:
        raise DataValidationError("inner leave-one-circuit-out selection needs two circuits")
    grid = tuple(float(alpha) for alpha in alpha_grid)
    if (
        not grid
        or len(grid) != len(set(grid))
        or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in grid)
    ):
        raise DataValidationError("ridge alpha grid must contain unique non-negative values")
    rows: list[dict[str, float | int | str]] = []
    for alpha in grid:
        for validation_circuit in circuit_names:
            validation = circuits == validation_circuit
            fit = ~validation
            if not np.any(fit) or not np.any(validation):
                raise DataValidationError("an inner LOCO fold has no fit or validation rows")
            model = _fit_equal_cluster_residual(
                design[fit],
                target[fit],
                circuits=circuits[fit],
                cars=cars[fit],
                ridge_alpha=alpha,
            )
            score = _mean_gaussian_nll(
                target[validation],
                model.predict(design[validation]),
                model.process_covariance,
            )
            rows.append(
                {
                    "ridge_alpha": alpha,
                    "validation_circuit": str(validation_circuit),
                    "fit_circuit_count": int(np.unique(circuits[fit]).size),
                    "fit_car_cluster_count": model.cluster_count,
                    "fit_row_count": int(np.count_nonzero(fit)),
                    "validation_row_count": int(np.count_nonzero(validation)),
                    "mean_gaussian_nll": score,
                }
            )
    fold_scores = pd.DataFrame(rows)
    mean_scores = {
        alpha: float(
            fold_scores.loc[
                np.isclose(fold_scores["ridge_alpha"], alpha), "mean_gaussian_nll"
            ].mean()
        )
        for alpha in grid
    }
    selected = min(grid, key=lambda alpha: (mean_scores[alpha], alpha))
    return RidgeSelection(selected, mean_scores, fold_scores)


def build_hazard_training_rows(fold_rows: pd.DataFrame) -> HazardTrainingRows:
    """Build one-step onset-side labels using fit cars and causal sources only."""

    required = {
        "partition",
        "circuit",
        "car_id",
        "frame_index",
        "time_seconds",
        "continuous_segment_id",
        "input_valid_causal",
        "in_corridor",
        "qualifying_event_onset_projected",
        "next_excursion_side",
        *RESIDUAL_FEATURE_COLUMNS[:-1],
    }
    _require_columns(fold_rows, required, "hazard training table")
    fit_rows = fold_rows.loc[fold_rows["partition"].astype(str) == "fit"].copy()
    if fit_rows.empty:
        raise DataValidationError("hazard training has no fit-partition rows")
    designs: list[NDArray[np.float64]] = []
    labels: list[str] = []
    keys: list[dict[str, object]] = []
    for (circuit, car, segment), group in fit_rows.groupby(
        ["circuit", "car_id", "continuous_segment_id"],
        sort=False,
        dropna=False,
    ):
        ordered = group.sort_values("time_seconds", kind="stable")
        if len(ordered) < 2:
            continue
        source = ordered.iloc[:-1]
        following = ordered.iloc[1:]
        dt = following["time_seconds"].to_numpy(dtype=np.float64) - source["time_seconds"].to_numpy(
            dtype=np.float64
        )
        valid = source["input_valid_causal"].astype(bool).to_numpy(copy=True)
        valid &= source["in_corridor"].astype(bool).to_numpy()
        valid &= following["input_valid_causal"].astype(bool).to_numpy()
        valid &= np.isfinite(dt) & (dt > 0.0)
        for position in np.flatnonzero(valid):
            current = source.iloc[int(position)]
            next_row = following.iloc[int(position)]
            design = np.asarray(
                [float(current[column]) for column in RESIDUAL_FEATURE_COLUMNS[:-1]]
                + [float(dt[int(position)])],
                dtype=np.float64,
            )
            if not np.isfinite(design).all():
                continue
            label = "no_exit"
            if bool(next_row["qualifying_event_onset_projected"]):
                side = str(current["next_excursion_side"])
                if side not in {"left", "right"}:
                    raise DataValidationError("qualifying onset has no registered left/right side")
                label = side
            designs.append(design)
            labels.append(label)
            keys.append(
                {
                    "circuit": str(circuit),
                    "car_id": str(car),
                    "continuous_segment_id": int(segment),
                    "source_frame_index": int(current["frame_index"]),
                    "target_frame_index": int(next_row["frame_index"]),
                }
            )
    if not designs:
        raise DataValidationError("hazard training contains no valid fit transitions")
    return HazardTrainingRows(
        X=np.stack(designs),
        labels=np.asarray(labels, dtype=np.str_),
        feature_names=RESIDUAL_FEATURE_COLUMNS,
        row_keys=pd.DataFrame(keys),
    )


def fit_fold_models(
    fold_rows: pd.DataFrame,
    *,
    seed: int,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    posterior_draws: int = 256,
    hazard_regularization: float = 1.0,
) -> FittedFoldModels:
    """Fit residual and hazard models using only whole fit-partition cars."""

    if "partition" not in fold_rows:
        raise DataValidationError("fold model table is missing partition")
    fit_rows = fold_rows.loc[fold_rows["partition"].astype(str) == "fit"].copy()
    if fit_rows.empty:
        raise DataValidationError("outer fold has no fit-partition rows")
    residual_pairs = build_residual_training_pairs(fit_rows)
    ridge = select_ridge_alpha_inner_loco(residual_pairs, alpha_grid=alpha_grid)
    bayesian = BayesianResidualDynamics.fit(
        residual_pairs.X,
        residual_pairs.Y,
        circuits=residual_pairs.circuits,
        cars=residual_pairs.cars,
        feature_names=residual_pairs.feature_names,
        target_names=residual_pairs.target_names,
        n_draws=posterior_draws,
        ridge_alpha=ridge.selected_alpha,
        seed=int(seed),
    )
    twin = bayesian.deterministic_twin()
    hazard_rows = build_hazard_training_rows(fold_rows)
    hazard = ScaledSideHazard.fit(
        hazard_rows.X,
        hazard_rows.labels,
        feature_names=hazard_rows.feature_names,
        regularization=hazard_regularization,
    )
    calibration_rows = fold_rows.loc[fold_rows["partition"].astype(str) == "calibration"].copy()
    if calibration_rows.empty:
        raise DataValidationError(
            "registered calibration-MSE comparator has no calibration-partition rows"
        )
    calibration_pairs = build_residual_training_pairs(calibration_rows)
    tuned = TunedDeterministicResidualDynamics.fit(
        residual_pairs.X,
        residual_pairs.Y,
        X_validation=calibration_pairs.X,
        Y_validation=calibration_pairs.Y,
        feature_names=residual_pairs.feature_names,
        target_names=residual_pairs.target_names,
        alpha_grid=alpha_grid,
    )
    tuned_model = tuned
    models: dict[str, object] = {
        PRIMARY_BAYESIAN_METHOD: bayesian,
        PRIMARY_DETERMINISTIC_METHOD: twin,
        SIDE_HAZARD_METHOD: hazard,
    }
    models[CALIBRATION_TUNED_DETERMINISTIC_METHOD] = tuned_model
    return FittedFoldModels(ridge, bayesian, twin, hazard, tuned_model, models)


def _horizon_suffix(horizon_s: float) -> str:
    if not np.isfinite(horizon_s) or horizon_s <= 0.0:
        raise DataValidationError("horizon must be finite and positive")
    return f"{float(horizon_s):.2f}".replace(".", "p")


def calibration_threshold_grid(
    calibrated_scores: ArrayLike,
    *,
    bulk_quantiles: int = 100,
    bulk_quantile_range: tuple[float, float] = (0.0, 0.99),
    upper_tail_quantiles: int = 401,
    upper_tail_survival_exponents: tuple[float, float] = (2.0, 6.0),
    include_endpoints: tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.float64]:
    """Return the frozen 100-bulk + 401-tail calibration-score threshold grid."""

    scores = np.asarray(calibrated_scores, dtype=np.float64)
    if (
        scores.ndim != 1
        or scores.size == 0
        or not np.isfinite(scores).all()
        or np.any((scores < 0.0) | (scores > 1.0))
    ):
        raise DataValidationError("calibrated scores must be a non-empty vector in [0, 1]")
    if not isinstance(bulk_quantiles, int) or bulk_quantiles < 2:
        raise DataValidationError("bulk_quantiles must be an integer of at least two")
    if not isinstance(upper_tail_quantiles, int) or upper_tail_quantiles < 2:
        raise DataValidationError("upper_tail_quantiles must be an integer of at least two")
    bulk_start, bulk_stop = map(float, bulk_quantile_range)
    exponent_start, exponent_stop = map(float, upper_tail_survival_exponents)
    if not 0.0 <= bulk_start <= bulk_stop <= 1.0:
        raise DataValidationError("bulk quantile range must lie in [0, 1]")
    if not 0.0 <= exponent_start <= exponent_stop:
        raise DataValidationError("tail survival exponents must be ordered and non-negative")
    endpoints = np.asarray(include_endpoints, dtype=np.float64)
    if endpoints.shape != (2,) or np.any((endpoints < 0.0) | (endpoints > 1.0)):
        raise DataValidationError("threshold endpoints must contain two values in [0, 1]")
    bulk_q = np.linspace(bulk_start, bulk_stop, bulk_quantiles)
    tail_q = 1.0 - 10.0 ** (-np.linspace(exponent_start, exponent_stop, upper_tail_quantiles))
    values = np.quantile(scores, np.concatenate((bulk_q, tail_q)))
    return np.unique(np.concatenate((values, endpoints)))


def hazard_segment_bins(
    *,
    arclength_m: ArrayLike,
    longitudinal_speed_mps: ArrayLike,
    horizons_s: Sequence[float] = HORIZONS_S,
    track_length_m: float,
    segment_length_m: float = 25.0,
) -> NDArray[np.int64]:
    """Apply the frozen current-arclength plus half-horizon hazard bin adapter."""

    arclength = np.asarray(arclength_m, dtype=np.float64)
    speed = np.asarray(longitudinal_speed_mps, dtype=np.float64)
    horizons = np.asarray(tuple(horizons_s), dtype=np.float64)
    if arclength.ndim != 1 or speed.shape != arclength.shape or arclength.size == 0:
        raise DataValidationError("arclength and speed must be non-empty aligned vectors")
    if not np.isfinite(arclength).all() or not np.isfinite(speed).all():
        raise DataValidationError("hazard adapter inputs must be finite")
    if horizons.ndim != 1 or horizons.size == 0 or np.any(horizons <= 0.0):
        raise DataValidationError("hazard horizons must be positive")
    if not np.isfinite(track_length_m) or track_length_m <= 0.0:
        raise DataValidationError("track_length_m must be finite and positive")
    if not np.isfinite(segment_length_m) or segment_length_m <= 0.0:
        raise DataValidationError("segment_length_m must be finite and positive")
    projected = arclength[:, None] + np.maximum(speed, 0.0)[:, None] * horizons[None, :] / 2.0
    wrapped = np.mod(projected, float(track_length_m))
    bins = np.floor(wrapped / float(segment_length_m)).astype(np.int64)
    n_bins = int(np.ceil(float(track_length_m) / float(segment_length_m)))
    return np.mod(bins, n_bins)


def compact_forecast_batch(
    score_keys: pd.DataFrame,
    forecast: object,
    *,
    method: str,
    primary_horizon_s: float = 1.50,
    neighborhood_radius_bins: int = 1,
) -> pd.DataFrame:
    """Release compact probabilities and one primary decision from a dense batch."""

    required = {
        "circuit",
        "source_session_id",
        "car_id",
        "frame_index",
        "time_seconds",
        "input_valid_causal",
        "in_corridor",
        "hard_break",
    }
    _require_columns(score_keys, required, "score key table")
    forbidden = {
        column
        for column in score_keys.columns
        if column in _TARGET_NAMES or column.startswith(_TARGET_PREFIXES)
    }
    if forbidden:
        raise DataValidationError(
            f"future truth cannot enter a score artifact: {sorted(forbidden)}"
        )
    horizons = np.asarray(forecast.horizons_s, dtype=np.float64)
    no_exit = np.asarray(forecast.no_exit_probability, dtype=np.float64)
    outcome = np.asarray(forecast.outcome_probability, dtype=np.float64)
    if no_exit.shape != (len(score_keys), horizons.size):
        raise DataValidationError("forecast totals do not align with score keys")
    if outcome.ndim != 4 or outcome.shape[:2] != no_exit.shape:
        raise DataValidationError("forecast outcome tensor does not align with totals")
    if not np.allclose(no_exit + outcome.sum(axis=(2, 3)), 1.0, atol=1e-8, rtol=0.0):
        raise DataValidationError("forecast probability mass does not sum to one")
    primary_matches = np.flatnonzero(np.isclose(horizons, primary_horizon_s, atol=1e-12))
    if primary_matches.size != 1:
        raise DataValidationError("primary horizon is absent or duplicated in forecast")
    primary_index = int(primary_matches[0])
    candidates = select_neighborhood_candidates(
        outcome[:, primary_index],
        side_names=tuple(forecast.side_order),
        radius_bins=neighborhood_radius_bins,
    )
    keep = [
        "circuit",
        "source_session_id",
        "car_id",
        "frame_index",
        "time_seconds",
        "input_valid_causal",
        "in_corridor",
        "hard_break",
    ]
    for optional in ("fold_test_circuit", "partition"):
        if optional in score_keys:
            keep.append(optional)
    compact = score_keys.loc[:, keep].copy().reset_index(drop=True)
    compact["method"] = str(method)
    totals = outcome.sum(axis=(2, 3))
    for index, horizon in enumerate(horizons):
        compact[f"raw_exit_probability_{_horizon_suffix(float(horizon))}s"] = totals[:, index]
    compact["raw_primary_neighborhood_probability"] = candidates.score
    compact["predicted_side"] = candidates.side
    compact["predicted_segment_bin"] = candidates.segment_bin
    compact["n_segment_bins"] = int(forecast.segment_bin_count)
    return compact


def _hazard_forecast(
    rows: pd.DataFrame,
    model: RegularizedSideHazard | ScaledSideHazard,
    corridor: CircuitCorridor,
    *,
    horizons_s: Sequence[float],
    dt_s: float,
    segment_length_m: float,
) -> object:
    """Adapt a side-hazard forecast to the common batch probability contract."""

    design = residual_design_from_features(rows, transition_dt_s=dt_s)
    probability = model.predict_horizon_probabilities(
        design,
        horizons_s=horizons_s,
        dt_s=dt_s,
    )
    bins = hazard_segment_bins(
        arclength_m=rows["centerline_arclength_wrapped_m"].to_numpy(dtype=np.float64),
        longitudinal_speed_mps=rows["body_speed_longitudinal_mps"].to_numpy(dtype=np.float64),
        horizons_s=horizons_s,
        track_length_m=corridor.track_length_m,
        segment_length_m=segment_length_m,
    )
    n_bins = int(np.ceil(corridor.track_length_m / segment_length_m))
    outcome = np.zeros((len(rows), len(horizons_s), 3, n_bins), dtype=np.float64)
    for row_index in range(len(rows)):
        for horizon_index in range(len(horizons_s)):
            segment_bin = int(bins[row_index, horizon_index])
            outcome[row_index, horizon_index, 0, segment_bin] = probability[
                row_index, horizon_index, 1
            ]
            outcome[row_index, horizon_index, 1, segment_bin] = probability[
                row_index, horizon_index, 2
            ]

    # A lightweight namespace avoids exporting first-crossing particle arrays
    # that a discrete hazard does not define.
    class _HazardForecast:
        pass

    result = _HazardForecast()
    result.horizons_s = np.asarray(horizons_s, dtype=np.float64)
    result.no_exit_probability = probability[:, :, 0]
    result.outcome_probability = outcome
    result.side_order = ("left", "right", "unknown")
    result.segment_bin_count = n_bins
    return result


def score_partition_methods(
    fold_features: pd.DataFrame,
    *,
    corridors: Mapping[str, CircuitCorridor],
    models: Mapping[str, object],
    methods: Sequence[str] = (
        PRIMARY_BAYESIAN_METHOD,
        PRIMARY_DETERMINISTIC_METHOD,
        CONSTANT_VELOCITY_METHOD,
        CONSTANT_TURN_RATE_METHOD,
        SIDE_HAZARD_METHOD,
        CALIBRATION_TUNED_DETERMINISTIC_METHOD,
    ),
    outer_fold: str,
    base_seed: int,
    batch_size: int = 128,
    horizons_s: Sequence[float] = HORIZONS_S,
    dt_s: float = 0.05,
    segment_length_m: float = 25.0,
) -> pd.DataFrame:
    """Batch-score calibration and test rows with frozen particles and row RNG IDs."""

    required = {
        "fold_test_circuit",
        "partition",
        "circuit",
        "source_session_id",
        "car_id",
        "frame_index",
        "time_seconds",
        "input_valid_causal",
        "in_corridor",
        "hard_break",
        "map_x_m",
        "map_y_m",
        "yaw_rad",
        "yaw_rate_radps",
        "body_speed_longitudinal_mps",
        "body_speed_lateral_mps",
        "body_acceleration_longitudinal_mps2",
        "body_acceleration_lateral_mps2",
        "heading_error_rad",
        "track_offset_m",
        "track_curvature_per_m",
    }
    _require_columns(fold_features, required, "score feature table")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise DataValidationError("batch_size must be a positive integer")
    selected = fold_features.loc[
        fold_features["partition"].astype(str).isin(["calibration", "test"])
    ].copy()
    if selected.empty:
        raise DataValidationError("score table has no calibration or test rows")
    if set(selected["fold_test_circuit"].astype(str)) != {str(outer_fold)}:
        raise DataValidationError("score rows do not belong to exactly the requested outer fold")
    method_names = tuple(str(method) for method in methods)
    registered = {
        PRIMARY_BAYESIAN_METHOD,
        PRIMARY_DETERMINISTIC_METHOD,
        CONSTANT_VELOCITY_METHOD,
        CONSTANT_TURN_RATE_METHOD,
        SIDE_HAZARD_METHOD,
        CALIBRATION_TUNED_DETERMINISTIC_METHOD,
    }
    invalid = sorted(set(method_names).difference(registered))
    if invalid:
        raise DataValidationError(f"unregistered experiment methods: {invalid}")
    selected = selected.sort_values(
        ["circuit", "source_session_id", "car_id", "time_seconds"], kind="stable"
    ).reset_index(drop=True)
    outputs: list[pd.DataFrame] = []
    key_columns = [
        "fold_test_circuit",
        "partition",
        "circuit",
        "source_session_id",
        "car_id",
        "frame_index",
        "time_seconds",
        "input_valid_causal",
        "in_corridor",
        "hard_break",
    ]
    for method in method_names:
        if (
            method
            in {
                PRIMARY_BAYESIAN_METHOD,
                PRIMARY_DETERMINISTIC_METHOD,
                SIDE_HAZARD_METHOD,
                CALIBRATION_TUNED_DETERMINISTIC_METHOD,
            }
            and method not in models
        ):
            raise DataValidationError(f"missing fitted model for method {method}")
        for circuit, circuit_rows in selected.groupby("circuit", sort=True):
            circuit_name = str(circuit)
            if circuit_name not in corridors:
                raise DataValidationError(f"missing forecast corridor for circuit {circuit_name}")
            group = circuit_rows.reset_index(drop=True)
            corridor = corridors[circuit_name]
            stable_ids = canonical_forecast_row_ids(
                base_seed=base_seed,
                outer_fold=str(outer_fold),
                method=method,
                circuits=group["circuit"].astype(str).to_numpy(),
                cars=group["car_id"].astype(str).to_numpy(),
                frame_indices=group["frame_index"].to_numpy(),
            )
            for start in range(0, len(group), batch_size):
                stop = min(start + batch_size, len(group))
                batch = group.iloc[start:stop].copy()
                states = planar_states_from_features(batch)
                row_ids = stable_ids[start:stop]
                common = {
                    "horizons_s": horizons_s,
                    "dt_s": dt_s,
                    "seed": int(base_seed),
                    "segment_length_m": segment_length_m,
                    "row_ids": row_ids,
                }
                if method == PRIMARY_BAYESIAN_METHOD:
                    model = models[method]
                    if not isinstance(model, BayesianResidualDynamics):
                        raise DataValidationError("BRACE method requires BayesianResidualDynamics")
                    forecast = forecast_ctra_particles(
                        states,
                        corridor,
                        residual_model=model,
                        residual_features=residual_design_from_features(
                            batch, transition_dt_s=dt_s
                        ),
                        n_particles=256,
                        **common,
                    )
                elif method in {
                    PRIMARY_DETERMINISTIC_METHOD,
                    CALIBRATION_TUNED_DETERMINISTIC_METHOD,
                }:
                    model = models[method]
                    if not isinstance(
                        model,
                        (
                            DeterministicResidualDynamics,
                            TunedDeterministicResidualDynamics,
                        ),
                    ):
                        raise DataValidationError(
                            "posterior-mean twin requires DeterministicResidualDynamics"
                        )
                    forecast = forecast_ctra_particles(
                        states,
                        corridor,
                        residual_model=model,
                        residual_features=residual_design_from_features(
                            batch, transition_dt_s=dt_s
                        ),
                        n_particles=1,
                        **common,
                    )
                elif method == CONSTANT_VELOCITY_METHOD:
                    forecast = forecast_constant_velocity(
                        states,
                        corridor,
                        n_particles=1,
                        **common,
                    )
                elif method == CONSTANT_TURN_RATE_METHOD:
                    forecast = forecast_constant_turn_rate(
                        states,
                        corridor,
                        n_particles=1,
                        **common,
                    )
                else:
                    model = models[method]
                    if not isinstance(model, (RegularizedSideHazard, ScaledSideHazard)):
                        raise DataValidationError(
                            "side hazard method requires a registered side-hazard model"
                        )
                    _require_columns(
                        batch,
                        {"centerline_arclength_wrapped_m"},
                        "hazard score feature table",
                    )
                    forecast = _hazard_forecast(
                        batch,
                        model,
                        corridor,
                        horizons_s=horizons_s,
                        dt_s=dt_s,
                        segment_length_m=segment_length_m,
                    )
                outputs.append(
                    compact_forecast_batch(
                        batch.loc[:, key_columns],
                        forecast,
                        method=method,
                        primary_horizon_s=max(horizons_s),
                    )
                )
    if not outputs:
        raise DataValidationError("no method scores were produced")
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(
        ["method", "circuit", "source_session_id", "car_id", "time_seconds"],
        kind="stable",
    ).reset_index(drop=True)


def _horizon_target_columns(horizon_s: float) -> tuple[str, str]:
    suffix = _horizon_suffix(horizon_s)
    return f"outcome_evaluable_{suffix}s", f"raw_exit_probability_{suffix}s"


def _evaluation_truth_columns(horizons_s: Sequence[float]) -> list[str]:
    return [
        "next_qualifying_event_id",
        "time_to_next_excursion_seconds",
        *(_horizon_target_columns(float(horizon))[0] for horizon in horizons_s),
    ]


def _join_score_truth(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    truth_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    _require_columns(scores, {"method", "partition", *FRAME_TARGET_KEYS}, "score table")
    _require_columns(
        targets,
        {"next_qualifying_event_id", "time_to_next_excursion_seconds", *FRAME_TARGET_KEYS},
        "target table",
    )
    if targets.duplicated(list(FRAME_TARGET_KEYS)).any():
        raise DataValidationError("target rows must be one-to-one before calibration")
    truth = targets
    if truth_columns is not None:
        selected_truth_columns = list(dict.fromkeys([*FRAME_TARGET_KEYS, *truth_columns]))
        _require_columns(targets, set(selected_truth_columns), "target table")
        truth = targets.loc[:, selected_truth_columns]
    score_key = ["method", *FRAME_TARGET_KEYS]
    if "fold_test_circuit" in scores:
        score_key.insert(0, "fold_test_circuit")
    if scores.duplicated(score_key).any():
        raise DataValidationError("score rows must be unique per method and four-key row")
    joined = scores.merge(
        truth,
        on=list(FRAME_TARGET_KEYS),
        how="left",
        validate="many_to_one",
        suffixes=("", "_truth"),
        indicator="_truth_merge",
    )
    if not (joined["_truth_merge"] == "both").all():
        raise DataValidationError("every score row must match exactly one target truth row")
    return joined.drop(columns="_truth_merge")


def fit_horizon_calibrators(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizons_s: Sequence[float] = HORIZONS_S,
) -> dict[tuple[str, float], MonotonePlattCalibrator]:
    """Fit one Monotone Platt map per method/horizon on calibration cars only."""

    calibration = scores.loc[scores["partition"].astype(str) == "calibration"].copy()
    if calibration.empty:
        raise DataValidationError("calibrator fit has no calibration-partition rows")
    joined = _join_score_truth(
        calibration,
        targets,
        truth_columns=_evaluation_truth_columns(horizons_s),
    )
    output: dict[tuple[str, float], MonotonePlattCalibrator] = {}
    for method, method_rows in joined.groupby("method", sort=True):
        for horizon_raw in horizons_s:
            horizon = float(horizon_raw)
            evaluable_column, probability_column = _horizon_target_columns(horizon)
            _require_columns(
                method_rows,
                {evaluable_column, probability_column},
                "calibration score/target table",
            )
            mask = method_rows[evaluable_column].astype(bool).to_numpy(copy=True)
            if "input_valid_causal" in method_rows:
                mask &= method_rows["input_valid_causal"].astype(bool).to_numpy()
            if "in_corridor" in method_rows:
                mask &= method_rows["in_corridor"].astype(bool).to_numpy()
            if not np.any(mask):
                raise DataValidationError(
                    f"no evaluable calibration rows for method {method} horizon {horizon}"
                )
            selected = method_rows.loc[mask]
            time_to_event = selected["time_to_next_excursion_seconds"].to_numpy(dtype=np.float64)
            has_event = selected["next_qualifying_event_id"].notna().to_numpy()
            labels = (
                has_event
                & np.isfinite(time_to_event)
                & (time_to_event >= -1e-12)
                & (time_to_event <= horizon + 1e-12)
            )
            output[(str(method), horizon)] = MonotonePlattCalibrator.fit(
                selected[probability_column].to_numpy(dtype=np.float64),
                labels.astype(np.int8),
            )
    return output


def calibration_reference_prevalence(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizons_s: Sequence[float] = HORIZONS_S,
) -> pd.DataFrame:
    """Record outer-fold calibration prevalence for every method and horizon."""

    calibration = scores.loc[scores["partition"].astype(str) == "calibration"].copy()
    if calibration.empty:
        raise DataValidationError("reference prevalence has no calibration rows")
    joined = _join_score_truth(
        calibration,
        targets,
        truth_columns=_evaluation_truth_columns(horizons_s),
    )
    rows: list[dict[str, object]] = []
    group_columns = ["fold_test_circuit", "method"]
    for (fold, method), group in joined.groupby(group_columns, sort=True):
        for horizon_raw in horizons_s:
            horizon = float(horizon_raw)
            evaluable_column, _ = _horizon_target_columns(horizon)
            _require_columns(group, {evaluable_column}, "calibration target table")
            mask = group[evaluable_column].astype(bool).to_numpy(copy=True)
            mask &= group["input_valid_causal"].astype(bool).to_numpy()
            mask &= group["in_corridor"].astype(bool).to_numpy()
            selected = group.loc[mask]
            if selected.empty:
                raise DataValidationError("reference prevalence has no evaluable rows")
            lead = selected["time_to_next_excursion_seconds"].to_numpy(dtype=np.float64)
            labels = (
                selected["next_qualifying_event_id"].notna().to_numpy()
                & np.isfinite(lead)
                & (lead >= -1e-12)
                & (lead <= horizon + 1e-12)
            )
            rows.append(
                {
                    "fold_test_circuit": str(fold),
                    "method": str(method),
                    "horizon_s": horizon,
                    "reference_prevalence": float(labels.mean()),
                    "calibration_evaluable_rows": int(labels.size),
                    "calibration_events": int(labels.sum()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_probability_performance(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    reference_prevalence: pd.DataFrame,
    *,
    horizons_s: Sequence[float] = HORIZONS_S,
    include_pooled: bool = True,
) -> pd.DataFrame:
    """Compute per-circuit and pooled held-out probability/calibration metrics."""

    test = scores.loc[scores["partition"].astype(str) == "test"].copy()
    if test.empty:
        raise DataValidationError("probability evaluation has no held-out score rows")
    joined = _join_score_truth(
        test,
        targets,
        truth_columns=_evaluation_truth_columns(horizons_s),
    )
    _require_columns(
        reference_prevalence,
        {"fold_test_circuit", "method", "horizon_s", "reference_prevalence"},
        "reference prevalence table",
    )
    reference_lookup = {
        (str(row.fold_test_circuit), str(row.method), float(row.horizon_s)): float(
            row.reference_prevalence
        )
        for row in reference_prevalence.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for method, method_rows in joined.groupby("method", sort=True):
        for horizon_raw in horizons_s:
            horizon = float(horizon_raw)
            evaluable_column, _ = _horizon_target_columns(horizon)
            probability_column = f"exit_probability_{_horizon_suffix(horizon)}s"
            _require_columns(
                method_rows,
                {evaluable_column, probability_column},
                "held-out probability table",
            )
            actionable_mask = method_rows["input_valid_causal"].astype(bool).to_numpy(copy=True)
            actionable_mask &= method_rows["in_corridor"].astype(bool).to_numpy()
            actionable_rows = method_rows.loc[actionable_mask].copy()
            unresolved_rows = actionable_rows.loc[~actionable_rows[evaluable_column].astype(bool)]
            mask = method_rows[evaluable_column].astype(bool).to_numpy(copy=True)
            mask &= method_rows["input_valid_causal"].astype(bool).to_numpy()
            mask &= method_rows["in_corridor"].astype(bool).to_numpy()
            horizon_rows = method_rows.loc[mask].copy()
            lead = horizon_rows["time_to_next_excursion_seconds"].to_numpy(dtype=np.float64)
            horizon_rows["_label"] = (
                horizon_rows["next_qualifying_event_id"].notna().to_numpy()
                & np.isfinite(lead)
                & (lead >= -1e-12)
                & (lead <= horizon + 1e-12)
            ).astype(np.int8)
            scopes: list[tuple[str, str, pd.DataFrame]] = [
                ("circuit", str(circuit), group)
                for circuit, group in horizon_rows.groupby("circuit", sort=True)
            ]
            if include_pooled:
                scopes.append(("pooled", "all", horizon_rows))
            for scope, circuit, group in scopes:
                if group.empty:
                    continue
                labels = group["_label"].to_numpy(dtype=np.float64)
                probability = group[probability_column].to_numpy(dtype=np.float64)
                if scope == "pooled":
                    unresolved_count = int(len(unresolved_rows))
                else:
                    unresolved_count = int(
                        (unresolved_rows["circuit"].astype(str).to_numpy() == str(circuit)).sum()
                    )
                reference = np.asarray(
                    [
                        reference_lookup[(str(fold), str(method), horizon)]
                        for fold in group["fold_test_circuit"]
                    ],
                    dtype=np.float64,
                )
                brier = float(np.mean((probability - labels) ** 2))
                reference_brier = float(np.mean((reference - labels) ** 2))
                clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
                log_score = float(
                    -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
                )
                reliability = expected_calibration_error(labels, probability, n_bins=10)
                intercept, slope = calibration_intercept_slope(labels, probability)
                rows.append(
                    {
                        "scope": scope,
                        "circuit": circuit,
                        "method": str(method),
                        "horizon_s": horizon,
                        "n": int(labels.size),
                        "declared_rate_hz": DECLARED_SCORING_RATE_HZ,
                        "evaluable_exposure_hours": (
                            float(labels.size) / DECLARED_SCORING_RATE_HZ / 3600.0
                        ),
                        "unresolved_outcome_rows": unresolved_count,
                        "unresolved_outcome_definition": (
                            "input_valid_causal_and_in_corridor_but_not_"
                            "outcome_evaluable_by_horizon"
                        ),
                        "event_count": int(labels.sum()),
                        "brier_score": brier,
                        "reference_brier_score": reference_brier,
                        "brier_skill_score": (
                            float("nan")
                            if reference_brier <= 0.0
                            else 1.0 - brier / reference_brier
                        ),
                        "log_score": log_score,
                        "average_precision_stepwise": _average_precision(labels, probability),
                        "calibration_intercept": intercept,
                        "calibration_slope": slope,
                        "ece_10_equal_width": reliability.ece,
                        "reliability_bin_counts": json.dumps(
                            reliability.counts.tolist(), separators=(",", ":")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def horizon_monotonicity_diagnostics(
    scores: pd.DataFrame,
    *,
    horizons_s: Sequence[float] = HORIZONS_S,
    include_pooled: bool = True,
) -> pd.DataFrame:
    """Count decreases in separately calibrated cumulative exit probabilities."""

    test = scores.loc[scores["partition"].astype(str) == "test"].copy()
    if test.empty:
        raise DataValidationError("horizon monotonicity diagnostic has no test rows")
    horizons = tuple(float(value) for value in horizons_s)
    if (
        not horizons
        or any(not np.isfinite(value) or value <= 0.0 for value in horizons)
        or any(second <= first for first, second in zip(horizons, horizons[1:], strict=False))
    ):
        raise DataValidationError("diagnostic horizons must be finite, positive, and increasing")
    probability_columns = [f"exit_probability_{_horizon_suffix(horizon)}s" for horizon in horizons]
    _require_columns(test, set(probability_columns), "calibrated score table")
    rows: list[dict[str, object]] = []
    for method, method_rows in test.groupby("method", sort=True):
        scopes: list[tuple[str, str, pd.DataFrame]] = [
            ("circuit", str(circuit), group)
            for circuit, group in method_rows.groupby("circuit", sort=True)
        ]
        if include_pooled:
            scopes.append(("pooled", "all", method_rows))
        for scope, circuit, group in scopes:
            probability = group.loc[:, probability_columns].to_numpy(dtype=np.float64)
            if not np.isfinite(probability).all():
                raise DataValidationError("calibrated horizon probabilities must be finite")
            adjacent_violations = np.diff(probability, axis=1) < -1e-12
            violating_rows = adjacent_violations.any(axis=1)
            rows.append(
                {
                    "scope": scope,
                    "circuit": circuit,
                    "method": str(method),
                    "score_rows": int(len(group)),
                    "horizon_order": json.dumps(horizons, separators=(",", ":")),
                    "violation_tolerance": 1e-12,
                    "violating_row_count": int(violating_rows.sum()),
                    "violating_row_rate": float(violating_rows.mean()),
                    "adjacent_pair_comparisons": int(adjacent_violations.size),
                    "adjacent_pair_violation_count": int(adjacent_violations.sum()),
                    "coherence_correction_applied": False,
                }
            )
    return pd.DataFrame(rows)


def _average_precision(labels: ArrayLike, probability: ArrayLike) -> float:
    """Return threshold-grouped average precision without a scikit-learn dependency."""

    outcome = np.asarray(labels, dtype=np.int8)
    forecast = np.asarray(probability, dtype=np.float64)
    if outcome.ndim != 1 or forecast.shape != outcome.shape or outcome.size == 0:
        raise DataValidationError(
            "average-precision labels and probabilities must be aligned vectors"
        )
    positives = int(outcome.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-forecast, kind="stable")
    sorted_outcome = outcome[order]
    sorted_probability = forecast[order]
    ends = np.r_[
        np.flatnonzero(np.diff(sorted_probability) != 0.0),
        sorted_probability.size - 1,
    ]
    cumulative_true = np.cumsum(sorted_outcome)
    previous_true = 0
    average_precision = 0.0
    for end in ends:
        true_positive = int(cumulative_true[int(end)])
        precision = true_positive / (int(end) + 1)
        average_precision += (true_positive - previous_true) / positives * precision
        previous_true = true_positive
    return float(average_precision)


def reliability_bin_source_table(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizons_s: Sequence[float] = HORIZONS_S,
    n_bins: int = 10,
    include_pooled: bool = True,
) -> pd.DataFrame:
    """Persist plot-ready held-out equal-width reliability-bin sufficient summaries."""

    if not isinstance(n_bins, int) or n_bins < 2:
        raise DataValidationError("n_bins must be an integer of at least two")
    test = scores.loc[scores["partition"].astype(str) == "test"].copy()
    if test.empty:
        raise DataValidationError("reliability evaluation has no held-out score rows")
    joined = _join_score_truth(
        test,
        targets,
        truth_columns=_evaluation_truth_columns(horizons_s),
    )
    rows: list[dict[str, object]] = []
    for method, method_rows in joined.groupby("method", sort=True):
        for horizon_raw in horizons_s:
            horizon = float(horizon_raw)
            evaluable_column, _ = _horizon_target_columns(horizon)
            probability_column = f"exit_probability_{_horizon_suffix(horizon)}s"
            _require_columns(
                method_rows,
                {evaluable_column, probability_column},
                "held-out reliability table",
            )
            mask = method_rows[evaluable_column].astype(bool).to_numpy(copy=True)
            mask &= method_rows["input_valid_causal"].astype(bool).to_numpy()
            mask &= method_rows["in_corridor"].astype(bool).to_numpy()
            horizon_rows = method_rows.loc[mask].copy()
            lead = horizon_rows["time_to_next_excursion_seconds"].to_numpy(dtype=np.float64)
            horizon_rows["_label"] = (
                horizon_rows["next_qualifying_event_id"].notna().to_numpy()
                & np.isfinite(lead)
                & (lead >= -1e-12)
                & (lead <= horizon + 1e-12)
            ).astype(np.int8)
            scopes: list[tuple[str, str, pd.DataFrame]] = [
                ("circuit", str(circuit), group)
                for circuit, group in horizon_rows.groupby("circuit", sort=True)
            ]
            if include_pooled:
                scopes.append(("pooled", "all", horizon_rows))
            for scope, circuit, group in scopes:
                if group.empty:
                    continue
                reliability = expected_calibration_error(
                    group["_label"].to_numpy(dtype=np.int8),
                    group[probability_column].to_numpy(dtype=np.float64),
                    n_bins=n_bins,
                )
                for bin_index in range(n_bins):
                    rows.append(
                        {
                            "scope": scope,
                            "circuit": circuit,
                            "method": str(method),
                            "horizon_s": horizon,
                            "bin_index": bin_index,
                            "bin_left": float(reliability.bin_edges[bin_index]),
                            "bin_right": float(reliability.bin_edges[bin_index + 1]),
                            "right_edge_inclusive": bin_index == n_bins - 1,
                            "count": int(reliability.counts[bin_index]),
                            "mean_probability": float(reliability.mean_probability[bin_index]),
                            "observed_frequency": float(reliability.observed_frequency[bin_index]),
                        }
                    )
    return pd.DataFrame(rows)


def reliability_cluster_source_table(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizons_s: Sequence[float] = HORIZONS_S,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return car-session bin contributions for cluster-aware reliability intervals."""

    if not isinstance(n_bins, int) or n_bins < 2:
        raise DataValidationError("n_bins must be an integer of at least two")
    test = scores.loc[scores["partition"].astype(str) == "test"].copy()
    _require_columns(
        test,
        {"fold_test_circuit", "source_session_id", "car_id"},
        "held-out reliability score table",
    )
    if test.empty:
        raise DataValidationError("reliability evaluation has no held-out score rows")
    joined = _join_score_truth(
        test,
        targets,
        truth_columns=_evaluation_truth_columns(horizons_s),
    )
    rows: list[dict[str, object]] = []
    unit_columns = ["fold_test_circuit", "circuit", "source_session_id", "car_id"]
    for method, method_rows in joined.groupby("method", sort=True):
        for horizon_raw in horizons_s:
            horizon = float(horizon_raw)
            evaluable_column, _ = _horizon_target_columns(horizon)
            probability_column = f"exit_probability_{_horizon_suffix(horizon)}s"
            _require_columns(
                method_rows,
                {evaluable_column, probability_column},
                "held-out reliability table",
            )
            mask = method_rows[evaluable_column].astype(bool).to_numpy(copy=True)
            mask &= method_rows["input_valid_causal"].astype(bool).to_numpy()
            mask &= method_rows["in_corridor"].astype(bool).to_numpy()
            selected = method_rows.loc[mask].copy()
            lead = selected["time_to_next_excursion_seconds"].to_numpy(dtype=np.float64)
            selected["_label"] = (
                selected["next_qualifying_event_id"].notna().to_numpy()
                & np.isfinite(lead)
                & (lead >= -1e-12)
                & (lead <= horizon + 1e-12)
            ).astype(np.int8)
            probability = selected[probability_column].to_numpy(dtype=np.float64)
            selected["_bin_index"] = np.minimum((probability * n_bins).astype(np.int64), n_bins - 1)
            all_units = (
                method_rows.loc[:, unit_columns]
                .drop_duplicates()
                .sort_values(unit_columns, kind="stable")
            )
            selected_groups = {
                tuple(str(value) for value in key): group
                for key, group in selected.groupby(unit_columns, sort=False)
            }
            for unit_key_raw in all_units.itertuples(index=False, name=None):
                unit_key = tuple(str(value) for value in unit_key_raw)
                unit_rows = selected_groups.get(unit_key, selected.iloc[0:0])
                for bin_index in range(n_bins):
                    bin_rows = unit_rows.loc[unit_rows["_bin_index"] == bin_index]
                    rows.append(
                        {
                            **dict(zip(unit_columns, unit_key, strict=True)),
                            "method": str(method),
                            "horizon_s": horizon,
                            "bin_index": bin_index,
                            "count": int(len(bin_rows)),
                            "probability_sum": float(bin_rows[probability_column].sum()),
                            "event_count": int(bin_rows["_label"].sum()),
                            "uncertainty_cluster": "car_session_within_circuit",
                        }
                    )
    return pd.DataFrame(rows)


def apply_horizon_calibrators(
    scores: pd.DataFrame,
    calibrators: Mapping[tuple[str, float], MonotonePlattCalibrator],
    *,
    horizons_s: Sequence[float] = HORIZONS_S,
    primary_horizon_s: float | None = None,
) -> pd.DataFrame:
    """Apply frozen calibrators while preserving only compact score-time fields."""

    forbidden = {
        column
        for column in scores.columns
        if column in _TARGET_NAMES or column.startswith(_TARGET_PREFIXES)
    }
    if forbidden:
        raise DataValidationError(
            f"future truth cannot enter a score artifact: {sorted(forbidden)}"
        )
    output = scores.copy()
    primary = float(max(horizons_s) if primary_horizon_s is None else primary_horizon_s)
    for method, index in output.groupby("method", sort=False).groups.items():
        rows = np.asarray(index, dtype=np.int64)
        for horizon_raw in horizons_s:
            horizon = float(horizon_raw)
            key = (str(method), horizon)
            if key not in calibrators:
                raise DataValidationError(f"missing calibrator for {key}")
            suffix = _horizon_suffix(horizon)
            raw_column = f"raw_exit_probability_{suffix}s"
            _require_columns(output, {raw_column}, "score table")
            raw_probability = output.loc[rows, raw_column].to_numpy(dtype=np.float64)
            calibrated_probability = calibrators[key].transform(raw_probability)
            output.loc[rows, f"exit_probability_{suffix}s"] = np.where(
                raw_probability > 0.0,
                calibrated_probability,
                0.0,
            )
    primary_suffix = _horizon_suffix(primary)
    raw_total = output[f"raw_exit_probability_{primary_suffix}s"].to_numpy(dtype=np.float64)
    calibrated_total = output[f"exit_probability_{primary_suffix}s"].to_numpy(dtype=np.float64)
    raw_neighborhood = output["raw_primary_neighborhood_probability"].to_numpy(dtype=np.float64)
    scale = np.divide(
        calibrated_total,
        raw_total,
        out=np.zeros_like(calibrated_total),
        where=raw_total > 0.0,
    )
    output["proposal_score"] = np.clip(raw_neighborhood * scale, 0.0, 1.0)
    return output


def join_exposure_to_split(
    timing: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    fold_test_circuit: str,
) -> pd.DataFrame:
    """Join interval exposure to its whole-car partition using immutable unit keys."""

    _require_columns(timing, {"buffered_exposure_seconds", *UNIT_KEYS}, "timing table")
    _require_columns(
        splits,
        {"fold_test_circuit", "partition", *UNIT_KEYS},
        "split table",
    )
    if timing.duplicated(list(UNIT_KEYS)).any():
        raise DataValidationError("timing table must contain one row per car-session")
    fold = splits.loc[splits["fold_test_circuit"].astype(str) == str(fold_test_circuit)].copy()
    if fold.duplicated(list(UNIT_KEYS)).any():
        raise DataValidationError("split table must contain one assignment per car-session")
    joined = timing.merge(
        fold.loc[:, [*UNIT_KEYS, "fold_test_circuit", "partition"]],
        on=list(UNIT_KEYS),
        how="left",
        validate="one_to_one",
    )
    if joined["partition"].isna().any() or len(joined) != len(timing):
        raise DataValidationError("timing exposure does not match every split car-session")
    seconds = joined["buffered_exposure_seconds"].to_numpy(dtype=np.float64)
    if not np.isfinite(seconds).all() or np.any(seconds < 0.0):
        raise DataValidationError("buffered exposure must be finite and non-negative")
    joined["exposure_hours"] = seconds / 3600.0
    return joined


def _events_for_units(events: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    _require_columns(events, {"circuit", "car_id"}, "event table")
    _require_columns(units, {"circuit", "car_id"}, "unit table")
    pairs = units.loc[:, ["circuit", "car_id"]].drop_duplicates()
    return events.merge(pairs, on=["circuit", "car_id"], how="inner", validate="many_to_one")


def select_calibration_operating_points(
    scores: pd.DataFrame,
    events: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    outer_fold: str,
    required_leads_s: Sequence[float] = HORIZONS_S,
    false_budgets_per_hour: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    proposal_config: ProposalConfig | None = None,
    segment_tolerance_bins: int = 1,
) -> pd.DataFrame:
    """Freeze every fold/method/lead/budget threshold from calibration cars only."""

    required_scores = {
        "fold_test_circuit",
        "partition",
        "method",
        "circuit",
        "source_session_id",
        "car_id",
        "proposal_score",
    }
    _require_columns(scores, required_scores, "score table")
    _require_columns(
        exposure,
        {
            "fold_test_circuit",
            "partition",
            "circuit",
            "source_session_id",
            "car_id",
            "exposure_hours",
        },
        "exposure table",
    )
    fold_scores = scores.loc[scores["fold_test_circuit"].astype(str) == str(outer_fold)].copy()
    calibration_scores = fold_scores.loc[
        fold_scores["partition"].astype(str) == "calibration"
    ].copy()
    if calibration_scores.empty:
        raise DataValidationError("threshold selection has no calibration score rows")
    calibration_exposure = exposure.loc[
        (exposure["fold_test_circuit"].astype(str) == str(outer_fold))
        & (exposure["partition"].astype(str) == "calibration")
    ].copy()
    exposure_hours = float(calibration_exposure["exposure_hours"].sum())
    if not np.isfinite(exposure_hours) or exposure_hours <= 0.0:
        raise DataValidationError("calibration exposure must be finite and positive")
    calibration_events = _events_for_units(events, calibration_exposure)
    config = ProposalConfig() if proposal_config is None else proposal_config
    output: list[dict[str, object]] = []
    leads = tuple(float(value) for value in required_leads_s)
    budgets = tuple(float(value) for value in false_budgets_per_hour)
    if not leads or any(not np.isfinite(value) or value < 0.0 for value in leads):
        raise DataValidationError("required leads must be finite and non-negative")
    if not budgets or any(not np.isfinite(value) or value < 0.0 for value in budgets):
        raise DataValidationError("false-alert budgets must be finite and non-negative")
    for method, method_scores in calibration_scores.groupby("method", sort=True):
        candidate_thresholds = calibration_threshold_grid(
            method_scores["proposal_score"].to_numpy(dtype=np.float64)
        )
        cached: list[dict[str, float | int]] = []
        for threshold in candidate_thresholds:
            proposals = run_proposal_state_machine(
                method_scores,
                threshold=float(threshold),
                config=config,
            )
            labels = label_proposals(
                proposals,
                calibration_events,
                segment_tolerance_bins=segment_tolerance_bins,
            )
            for lead in leads:
                summary = summarize_detection(
                    labels,
                    calibration_events,
                    exposure_hours=exposure_hours,
                    required_lead_s=lead,
                )
                cached.append(
                    {
                        "threshold": float(threshold),
                        "required_lead_s": lead,
                        "localized_event_recall": float(summary["localized_event_recall"]),
                        "false_proposals_per_hour": float(summary["false_proposals_per_hour"]),
                        "false_proposals": int(summary["false_proposals"]),
                        "qualified_events": int(summary["qualified_events"]),
                        "proposal_count": len(proposals),
                    }
                )
        cached_table = pd.DataFrame(cached)
        for lead in leads:
            lead_rows = cached_table.loc[
                np.isclose(
                    cached_table["required_lead_s"].to_numpy(dtype=np.float64),
                    lead,
                    atol=1e-12,
                    rtol=0.0,
                )
            ]
            for budget in budgets:
                capacity = budget * exposure_hours
                if capacity + 1e-12 < MIN_FALSE_COUNT_CAPACITY_FOR_ESTIMABILITY:
                    output.append(
                        {
                            "fold_test_circuit": str(outer_fold),
                            "method": str(method),
                            "required_lead_s": lead,
                            "false_budget_per_hour": budget,
                            "threshold": "NOT_ESTIMABLE",
                            "calibration_localized_event_recall": None,
                            "calibration_false_proposals_per_hour": None,
                            "calibration_false_proposals": None,
                            "calibration_qualified_events": int(
                                calibration_events["qualified"].astype(bool).sum()
                            ),
                            "calibration_proposal_count": None,
                            "calibration_exposure_hours": exposure_hours,
                            "calibration_false_count_capacity": capacity,
                            "minimum_false_count_capacity": (
                                MIN_FALSE_COUNT_CAPACITY_FOR_ESTIMABILITY
                            ),
                            "operating_point_status": ("not_estimable_insufficient_exposure"),
                            "estimability_reason": (
                                "false_budget_per_hour_times_calibration_exposure_hours_below_1"
                            ),
                            "threshold_candidate_count": int(candidate_thresholds.size),
                            "threshold_source_partition": "calibration",
                        }
                    )
                    continue
                feasible = lead_rows.loc[
                    lead_rows["false_proposals_per_hour"].to_numpy(dtype=np.float64)
                    <= budget + 1e-12
                ].copy()
                if feasible.empty:
                    raise DataValidationError(
                        "no candidate threshold satisfies the false-proposal budget"
                    )
                feasible["_selection_recall"] = feasible["localized_event_recall"].fillna(-np.inf)
                selection = feasible.sort_values(
                    ["_selection_recall", "threshold"], kind="stable"
                ).iloc[-1]
                output.append(
                    {
                        "fold_test_circuit": str(outer_fold),
                        "method": str(method),
                        "required_lead_s": lead,
                        "false_budget_per_hour": budget,
                        "threshold": float(selection["threshold"]),
                        "calibration_localized_event_recall": (
                            float(selection["localized_event_recall"])
                        ),
                        "calibration_false_proposals_per_hour": (
                            float(selection["false_proposals_per_hour"])
                        ),
                        "calibration_false_proposals": int(selection["false_proposals"]),
                        "calibration_qualified_events": int(selection["qualified_events"]),
                        "calibration_proposal_count": int(selection["proposal_count"]),
                        "calibration_exposure_hours": exposure_hours,
                        "calibration_false_count_capacity": capacity,
                        "minimum_false_count_capacity": (MIN_FALSE_COUNT_CAPACITY_FOR_ESTIMABILITY),
                        "operating_point_status": "estimable",
                        "estimability_reason": "",
                        "threshold_candidate_count": int(candidate_thresholds.size),
                        "threshold_source_partition": "calibration",
                    }
                )
    return pd.DataFrame(output)


def _add_exact_bin_diagnostic(
    labels: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Add side-independent exact-bin truth only to post-unsealing label artifacts."""

    output = labels.copy()
    output["matched_side_at_onset"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["matched_segment_bin"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["exact_bin"] = False
    if output.empty:
        return output
    matched_events = events.loc[
        events["qualified"].astype(bool),
        ["candidate_event_id", "side_at_onset", "segment_bin_25m_at_onset"],
    ].copy()
    if matched_events["candidate_event_id"].duplicated().any():
        raise DataValidationError("qualified event identifiers must be unique")
    side_lookup = matched_events.set_index("candidate_event_id")["side_at_onset"]
    bin_lookup = matched_events.set_index("candidate_event_id")["segment_bin_25m_at_onset"]
    matched = output["matched_event_id"].notna()
    event_ids = output.loc[matched, "matched_event_id"].astype(str)
    if not event_ids.isin(side_lookup.index.astype(str)).all():
        raise DataValidationError("proposal label references an unknown qualified event")
    side_by_string = {str(key): value for key, value in side_lookup.items()}
    bin_by_string = {str(key): int(value) for key, value in bin_lookup.items()}
    output.loc[matched, "matched_side_at_onset"] = event_ids.map(side_by_string).to_numpy()
    output.loc[matched, "matched_segment_bin"] = pd.array(
        event_ids.map(bin_by_string).to_numpy(), dtype="Int64"
    )
    output.loc[matched, "exact_bin"] = output.loc[matched, "predicted_segment_bin"].to_numpy(
        dtype=np.int64
    ) == output.loc[matched, "matched_segment_bin"].to_numpy(dtype=np.int64)
    return output


def apply_synthetic_proposal_delays(
    proposal_labels: pd.DataFrame,
    *,
    delays_ms: Sequence[int] = (0, 40, 80, 160),
) -> pd.DataFrame:
    """Apply fixed postprocessing delays to immutable proposals without any refit."""

    _require_columns(
        proposal_labels,
        {
            "proposal_time_seconds",
            "matched_event_id",
            "matched_event_time_seconds",
            "lead_seconds",
            "localized_hit",
            "false_proposal",
            "unresolved_censored",
        },
        "proposal label table",
    )
    delay_values = tuple(int(value) for value in delays_ms)
    if (
        not delay_values
        or len(delay_values) != len(set(delay_values))
        or any(value < 0 for value in delay_values)
    ):
        raise DataValidationError("synthetic delays must be unique non-negative milliseconds")
    outputs: list[pd.DataFrame] = []
    for delay_ms in delay_values:
        delay_s = delay_ms / 1000.0
        delayed = proposal_labels.copy()
        delayed["synthetic_delay_ms"] = delay_ms
        delayed["effective_proposal_time_seconds"] = (
            delayed["proposal_time_seconds"].to_numpy(dtype=np.float64) + delay_s
        )
        original_lead = delayed["lead_seconds"].to_numpy(dtype=np.float64)
        effective_lead = original_lead - delay_s
        delayed["effective_lead_seconds"] = effective_lead
        matched = delayed["matched_event_id"].notna().to_numpy()
        still_precedes = matched & np.isfinite(effective_lead) & (effective_lead >= -1e-12)
        delayed["delay_still_precedes_event"] = still_precedes
        delayed["delayed_localized_hit"] = (
            delayed["localized_hit"].astype(bool).to_numpy() & still_precedes
        )
        delayed["delayed_false_proposal"] = delayed["false_proposal"].astype(bool).to_numpy() | (
            matched & ~delayed["delayed_localized_hit"].to_numpy(dtype=bool)
        )
        delayed["models_refit"] = False
        delayed["thresholds_refit"] = False
        outputs.append(delayed)
    return pd.concat(outputs, ignore_index=True)


def evaluate_heldout_operating_points(
    scores: pd.DataFrame,
    events: pd.DataFrame,
    exposure: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    outer_fold: str,
    proposal_config: ProposalConfig | None = None,
    segment_tolerance_bins: int = 1,
) -> HeldoutEvaluation:
    """Apply already-frozen thresholds, then inspect held-out outcomes."""

    required_thresholds = {
        "fold_test_circuit",
        "method",
        "required_lead_s",
        "false_budget_per_hour",
        "threshold",
        "threshold_source_partition",
    }
    _require_columns(thresholds, required_thresholds, "threshold table")
    selected_thresholds = thresholds.loc[
        thresholds["fold_test_circuit"].astype(str) == str(outer_fold)
    ].copy()
    if selected_thresholds.empty:
        raise DataValidationError("held-out evaluation has no frozen thresholds")
    if not (selected_thresholds["threshold_source_partition"].astype(str) == "calibration").all():
        raise DataValidationError("held-out evaluation requires calibration-frozen thresholds")
    threshold_key = [
        "fold_test_circuit",
        "method",
        "required_lead_s",
        "false_budget_per_hour",
    ]
    if selected_thresholds.duplicated(threshold_key).any():
        raise DataValidationError("held-out threshold keys must be unique")
    _require_columns(
        scores,
        {
            "fold_test_circuit",
            "partition",
            "method",
            "circuit",
            "source_session_id",
            "car_id",
            "proposal_score",
        },
        "score table",
    )
    test_scores = scores.loc[
        (scores["fold_test_circuit"].astype(str) == str(outer_fold))
        & (scores["partition"].astype(str) == "test")
    ].copy()
    if test_scores.empty:
        raise DataValidationError("held-out evaluation has no test score rows")
    test_exposure = exposure.loc[
        (exposure["fold_test_circuit"].astype(str) == str(outer_fold))
        & (exposure["partition"].astype(str) == "test")
    ].copy()
    if test_exposure.empty:
        raise DataValidationError("held-out evaluation has no test exposure")
    if test_exposure.duplicated(["circuit", "car_id"]).any():
        raise DataValidationError(
            "proposal policy requires one source car-session per circuit/car identifier"
        )
    test_events = _events_for_units(events, test_exposure)
    config = ProposalConfig() if proposal_config is None else proposal_config
    metric_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    delay_metric_rows: list[dict[str, object]] = []
    delay_contribution_rows: list[dict[str, object]] = []
    label_tables: list[pd.DataFrame] = []
    for threshold_row in selected_thresholds.itertuples(index=False):
        method = str(threshold_row.method)
        method_scores = test_scores.loc[test_scores["method"].astype(str) == method]
        if method_scores.empty:
            raise DataValidationError(f"test score table has no rows for method {method}")
        required_lead = float(threshold_row.required_lead_s)
        budget = float(threshold_row.false_budget_per_hour)
        total_exposure = float(test_exposure["exposure_hours"].sum())
        operating_status = str(getattr(threshold_row, "operating_point_status", "estimable"))
        if operating_status != "estimable":
            metric_rows.append(
                {
                    "fold_test_circuit": str(outer_fold),
                    "scope": "circuit",
                    "circuit": str(outer_fold),
                    "method": method,
                    "required_lead_s": required_lead,
                    "false_budget_per_hour": budget,
                    "threshold": "NOT_ESTIMABLE",
                    "exposure_hours": total_exposure,
                    "calibration_false_count_capacity": float(
                        getattr(
                            threshold_row,
                            "calibration_false_count_capacity",
                            float("nan"),
                        )
                    ),
                    "heldout_false_count_capacity": budget * total_exposure,
                    "operating_point_status": operating_status,
                    "estimability_reason": str(getattr(threshold_row, "estimability_reason", "")),
                }
            )
            continue
        proposals = run_proposal_state_machine(
            method_scores,
            threshold=float(threshold_row.threshold),
            config=config,
        )
        labels = label_proposals(
            proposals,
            test_events,
            segment_tolerance_bins=segment_tolerance_bins,
        )
        labels = _add_exact_bin_diagnostic(labels, test_events)
        summary = summarize_detection(
            labels,
            test_events,
            exposure_hours=total_exposure,
            required_lead_s=required_lead,
        )
        least_favorable_summary = summarize_detection(
            labels,
            test_events,
            exposure_hours=total_exposure,
            required_lead_s=required_lead,
            count_unresolved_as_false=True,
        )
        qualified_events = int(
            test_events.loc[test_events["qualified"].astype(bool), "candidate_event_id"].nunique()
        )
        timely = labels.loc[
            labels["matched_event_id"].notna()
            & (labels["lead_seconds"].to_numpy(dtype=np.float64) >= required_lead - 1e-12)
        ]

        def event_recall(
            rows: pd.DataFrame,
            event_count: int,
            column: str | None = None,
        ) -> float:
            selected = rows if column is None else rows.loc[rows[column].astype(bool)]
            hits = int(selected["matched_event_id"].dropna().nunique())
            return float("nan") if event_count == 0 else hits / event_count

        invalid = ~method_scores["input_valid_causal"].astype(bool)
        outside = ~method_scores["in_corridor"].astype(bool)
        hard_break = method_scores["hard_break"].astype(bool)
        invalid_reason = invalid
        hard_break_reason = ~invalid_reason & hard_break
        outside_reason = ~invalid_reason & ~hard_break_reason & outside
        abstention_mask = invalid_reason | hard_break_reason | outside_reason
        abstentions = int(abstention_mask.sum())
        false_count = int(summary["false_proposals"])
        metric_rows.append(
            {
                "fold_test_circuit": str(outer_fold),
                "scope": "circuit",
                "circuit": str(outer_fold),
                "method": method,
                "required_lead_s": required_lead,
                "false_budget_per_hour": float(threshold_row.false_budget_per_hour),
                "threshold": float(threshold_row.threshold),
                "exposure_hours": total_exposure,
                "calibration_false_count_capacity": float(
                    getattr(
                        threshold_row,
                        "calibration_false_count_capacity",
                        float("nan"),
                    )
                ),
                "heldout_false_count_capacity": budget * total_exposure,
                "operating_point_status": "estimable",
                "estimability_reason": "",
                "event_recall": event_recall(timely, qualified_events),
                "correct_side_event_recall": event_recall(timely, qualified_events, "correct_side"),
                "within_segment_tolerance_event_recall": event_recall(
                    timely, qualified_events, "within_segment_tolerance"
                ),
                "exact_bin_event_recall": event_recall(timely, qualified_events, "exact_bin"),
                "false_proposals_per_hour_upper_95": false_proposal_rate_upper(
                    false_count,
                    exposure_hours=total_exposure,
                ),
                "least_favorable_false_proposals": int(least_favorable_summary["false_proposals"]),
                "least_favorable_false_proposals_per_hour": float(
                    least_favorable_summary["false_proposals_per_hour"]
                ),
                "least_favorable_false_proposals_per_hour_upper_95": float(
                    least_favorable_summary["false_proposals_per_hour_upper_95"]
                ),
                "unresolved_sensitivity_rule": "count_every_unresolved_proposal_as_false",
                "score_rows": int(len(method_scores)),
                "abstention_rows": abstentions,
                "abstention_rate": abstentions / len(method_scores),
                "abstention_reason_counts_are_disjoint": True,
                "abstention_reason_counts": json.dumps(
                    {
                        "input_invalid": int(invalid_reason.sum()),
                        "hard_break": int(hard_break_reason.sum()),
                        "outside_corridor": int(outside_reason.sum()),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **summary,
            }
        )
        annotated = labels.copy()
        annotated["fold_test_circuit"] = str(outer_fold)
        annotated["method"] = method
        annotated["required_lead_s"] = required_lead
        annotated["false_budget_per_hour"] = float(threshold_row.false_budget_per_hour)
        label_tables.append(annotated)
        delayed_labels = apply_synthetic_proposal_delays(labels)
        for delay_ms in (0, 40, 80, 160):
            delayed_group = delayed_labels.loc[delayed_labels["synthetic_delay_ms"] == delay_ms]
            delayed_timely = delayed_group.loc[
                delayed_group["delayed_localized_hit"].astype(bool)
                & (
                    delayed_group["effective_lead_seconds"].to_numpy(dtype=np.float64)
                    >= required_lead - 1e-12
                )
            ]
            delayed_hits = int(delayed_timely["matched_event_id"].dropna().nunique())
            delayed_false_count = int(delayed_group["delayed_false_proposal"].astype(bool).sum())
            delayed_unresolved_count = int(delayed_group["unresolved_censored"].astype(bool).sum())
            delayed_least_favorable_false_count = delayed_false_count + delayed_unresolved_count
            delay_metric_rows.append(
                {
                    "fold_test_circuit": str(outer_fold),
                    "scope": "circuit",
                    "circuit": str(outer_fold),
                    "method": method,
                    "required_lead_s": required_lead,
                    "false_budget_per_hour": float(threshold_row.false_budget_per_hour),
                    "synthetic_delay_ms": int(delay_ms),
                    "localized_event_hits": delayed_hits,
                    "qualified_events": qualified_events,
                    "localized_event_recall": (
                        float("nan") if qualified_events == 0 else delayed_hits / qualified_events
                    ),
                    "false_proposals": delayed_false_count,
                    "false_proposals_per_hour": delayed_false_count / total_exposure,
                    "unresolved_proposals": delayed_unresolved_count,
                    "least_favorable_false_proposals": (delayed_least_favorable_false_count),
                    "least_favorable_false_proposals_per_hour": (
                        delayed_least_favorable_false_count / total_exposure
                    ),
                    "unresolved_sensitivity_rule": ("count_every_unresolved_proposal_as_false"),
                    "exposure_hours": total_exposure,
                    "models_refit": False,
                    "thresholds_refit": False,
                }
            )
        for unit in test_exposure.itertuples(index=False):
            unit_events = test_events.loc[
                (test_events["circuit"].astype(str) == str(unit.circuit))
                & (test_events["car_id"].astype(str) == str(unit.car_id))
                & test_events["qualified"].astype(bool)
            ]
            unit_labels = labels.loc[
                (labels["circuit"].astype(str) == str(unit.circuit))
                & (labels["car_id"].astype(str) == str(unit.car_id))
            ]
            timely = unit_labels.loc[
                unit_labels["matched_event_id"].notna()
                & (unit_labels["lead_seconds"].to_numpy(dtype=np.float64) >= required_lead - 1e-12)
            ]
            localized_timely = timely.loc[timely["localized_hit"].astype(bool)]
            contribution_rows.append(
                {
                    "fold_test_circuit": str(outer_fold),
                    "circuit": str(unit.circuit),
                    "source_session_id": str(unit.source_session_id),
                    "car_id": str(unit.car_id),
                    "method": method,
                    "required_lead_s": required_lead,
                    "false_budget_per_hour": float(threshold_row.false_budget_per_hour),
                    "localized_event_hits": int(
                        localized_timely["matched_event_id"].dropna().nunique()
                    ),
                    "event_hits": int(timely["matched_event_id"].dropna().nunique()),
                    "correct_side_event_hits": int(
                        timely.loc[timely["correct_side"].astype(bool), "matched_event_id"]
                        .dropna()
                        .nunique()
                    ),
                    "within_segment_tolerance_event_hits": int(
                        timely.loc[
                            timely["within_segment_tolerance"].astype(bool),
                            "matched_event_id",
                        ]
                        .dropna()
                        .nunique()
                    ),
                    "exact_bin_event_hits": int(
                        timely.loc[timely["exact_bin"].astype(bool), "matched_event_id"]
                        .dropna()
                        .nunique()
                    ),
                    "qualified_events": int(unit_events["candidate_event_id"].nunique()),
                    "false_proposals": int(unit_labels["false_proposal"].astype(bool).sum()),
                    "unresolved_proposals": int(
                        unit_labels["unresolved_censored"].astype(bool).sum()
                    ),
                    "least_favorable_false_proposals": int(
                        unit_labels["false_proposal"].astype(bool).sum()
                        + unit_labels["unresolved_censored"].astype(bool).sum()
                    ),
                    "exposure_hours": float(unit.exposure_hours),
                    "fixed_model_and_threshold": True,
                    "operating_point_status": "estimable",
                }
            )
            for delay_ms in (0, 40, 80, 160):
                delayed_group = delayed_labels.loc[delayed_labels["synthetic_delay_ms"] == delay_ms]
                unit_delayed = delayed_group.loc[
                    (delayed_group["circuit"].astype(str) == str(unit.circuit))
                    & (delayed_group["car_id"].astype(str) == str(unit.car_id))
                ]
                timely_delayed = unit_delayed.loc[
                    unit_delayed["delayed_localized_hit"].astype(bool)
                    & (
                        unit_delayed["effective_lead_seconds"].to_numpy(dtype=np.float64)
                        >= required_lead - 1e-12
                    )
                ]
                delay_contribution_rows.append(
                    {
                        "fold_test_circuit": str(outer_fold),
                        "circuit": str(unit.circuit),
                        "source_session_id": str(unit.source_session_id),
                        "car_id": str(unit.car_id),
                        "method": method,
                        "required_lead_s": required_lead,
                        "false_budget_per_hour": float(threshold_row.false_budget_per_hour),
                        "synthetic_delay_ms": int(delay_ms),
                        "localized_event_hits": int(
                            timely_delayed["matched_event_id"].dropna().nunique()
                        ),
                        "qualified_events": int(unit_events["candidate_event_id"].nunique()),
                        "false_proposals": int(
                            unit_delayed["delayed_false_proposal"].astype(bool).sum()
                        ),
                        "unresolved_proposals": int(
                            unit_labels["unresolved_censored"].astype(bool).sum()
                        ),
                        "least_favorable_false_proposals": int(
                            unit_delayed["delayed_false_proposal"].astype(bool).sum()
                            + unit_labels["unresolved_censored"].astype(bool).sum()
                        ),
                        "exposure_hours": float(unit.exposure_hours),
                        "fixed_model_and_threshold": True,
                        "operating_point_status": "estimable",
                        "models_refit": False,
                        "thresholds_refit": False,
                    }
                )
    labels = pd.concat(label_tables, ignore_index=True) if label_tables else pd.DataFrame()
    return HeldoutEvaluation(
        metrics=pd.DataFrame(metric_rows),
        car_contributions=pd.DataFrame(contribution_rows),
        proposal_labels=labels,
        delay_metrics=pd.DataFrame(delay_metric_rows),
        delay_car_contributions=pd.DataFrame(delay_contribution_rows),
    )


def l_star_from_contributions(
    contributions: pd.DataFrame,
    *,
    method: str,
    false_budget_per_hour: float = 2.0,
    minimum_recall: float = 0.50,
    lead_grid_s: Sequence[float] = HORIZONS_S,
    false_count_column: str = "false_proposals",
) -> float:
    """Compute pooled held-out L-star from post-threshold cluster contributions."""

    unit_key = ["circuit", "source_session_id", "car_id"]
    required = {
        *unit_key,
        "method",
        "required_lead_s",
        "false_budget_per_hour",
        "localized_event_hits",
        "qualified_events",
        false_count_column,
        "exposure_hours",
        "operating_point_status",
    }
    _require_columns(contributions, required, "contribution table")
    if not np.isfinite(false_budget_per_hour) or false_budget_per_hour < 0.0:
        raise DataValidationError("false budget must be finite and non-negative")
    if not np.isfinite(minimum_recall) or not 0.0 <= minimum_recall <= 1.0:
        raise DataValidationError("minimum recall must lie in [0, 1]")
    leads = np.asarray(tuple(float(value) for value in lead_grid_s), dtype=np.float64)
    if (
        leads.ndim != 1
        or leads.size == 0
        or not np.isfinite(leads).all()
        or np.any(leads <= 0.0)
        or np.any(np.diff(leads) <= 0.0)
    ):
        raise DataValidationError("L-star lead grid must be finite, positive, and increasing")
    selected = contributions.loc[
        (contributions["method"].astype(str) == str(method))
        & np.isclose(
            contributions["false_budget_per_hour"].to_numpy(dtype=np.float64),
            false_budget_per_hour,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()
    if selected.empty:
        raise DataValidationError("L-star has no rows for the requested method and budget")
    if not (selected["operating_point_status"].astype(str) == "estimable").all():
        raise DataValidationError("L-star cannot use a not-estimable operating point")
    matches = np.isclose(
        selected["required_lead_s"].to_numpy(dtype=np.float64)[:, None],
        leads[None, :],
        atol=1e-12,
        rtol=0.0,
    )
    if not (matches.sum(axis=1) == 1).all():
        raise DataValidationError("L-star rows must match exactly one registered lead")
    selected["lead_slot"] = np.argmax(matches, axis=1)
    if selected.loc[:, unit_key].isna().any(axis=None):
        raise DataValidationError("L-star cluster keys must be non-null")
    if selected.duplicated([*unit_key, "lead_slot"]).any():
        raise DataValidationError("L-star needs one row per car-session and registered lead")
    units = selected.loc[:, unit_key].drop_duplicates()
    if len(selected) != len(units) * len(leads):
        raise DataValidationError("L-star contribution table is missing a registered lead")
    numeric = selected.loc[
        :, ["localized_event_hits", "qualified_events", false_count_column, "exposure_hours"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
        raise DataValidationError("L-star counts and exposure must be finite and non-negative")
    if not np.equal(numeric[:, :3], np.floor(numeric[:, :3])).all():
        raise DataValidationError("L-star count fields must be integer counts")
    if np.any(numeric[:, 0] > numeric[:, 1]) or np.any(numeric[:, 3] <= 0.0):
        raise DataValidationError("L-star hits cannot exceed events and exposure must be positive")
    passing: list[float] = []
    for lead_slot, lead in enumerate(leads):
        rows = selected.loc[selected["lead_slot"] == lead_slot]
        hits = float(rows["localized_event_hits"].sum())
        events = float(rows["qualified_events"].sum())
        false = float(rows[false_count_column].sum())
        exposure = float(rows["exposure_hours"].sum())
        if events <= 0.0 or exposure <= 0.0:
            continue
        if hits / events >= minimum_recall - 1e-12 and false / exposure <= (
            false_budget_per_hour + 1e-12
        ):
            passing.append(float(lead))
    return 0.0 if not passing else float(max(passing))


def paired_l_star_sensitivity_table(
    contributions: pd.DataFrame,
    *,
    brace_method: str = PRIMARY_BAYESIAN_METHOD,
    comparator_method: str = PRIMARY_DETERMINISTIC_METHOD,
    false_budget_per_hour: float = 2.0,
    minimum_recall: float = 0.50,
    lead_grid_s: Sequence[float] = HORIZONS_S,
) -> pd.DataFrame:
    """Compute primary and least-favorable fixed-policy pooled L-star contrasts."""

    rows: list[dict[str, object]] = []
    for analysis, false_column in (
        ("primary_unresolved_censored", "false_proposals"),
        (
            "least_favorable_unresolved_counted_as_false",
            "least_favorable_false_proposals",
        ),
    ):
        brace = l_star_from_contributions(
            contributions,
            method=brace_method,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
            lead_grid_s=lead_grid_s,
            false_count_column=false_column,
        )
        comparator = l_star_from_contributions(
            contributions,
            method=comparator_method,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
            lead_grid_s=lead_grid_s,
            false_count_column=false_column,
        )
        rows.append(
            {
                "analysis": analysis,
                "false_count_column": false_column,
                "brace_method": brace_method,
                "comparator_method": comparator_method,
                "l_star_brace_s": brace,
                "l_star_comparator_s": comparator,
                "delta_l_star_s": brace - comparator,
                "false_budget_per_hour": false_budget_per_hour,
                "minimum_recall": minimum_recall,
                "models_refit": False,
                "thresholds_refit": False,
            }
        )
    return pd.DataFrame(rows)


def pooled_operating_metrics_from_contributions(
    contributions: pd.DataFrame,
    *,
    synthetic_delay: bool = False,
) -> pd.DataFrame:
    """Pool immutable car-session numerators and denominators across held-out circuits."""

    required = {
        "method",
        "required_lead_s",
        "false_budget_per_hour",
        "localized_event_hits",
        "qualified_events",
        "false_proposals",
        "exposure_hours",
    }
    if synthetic_delay:
        required.add("synthetic_delay_ms")
    _require_columns(contributions, required, "held-out contribution table")
    group_columns = ["method", "required_lead_s", "false_budget_per_hour"]
    if synthetic_delay:
        group_columns.insert(0, "synthetic_delay_ms")
    component_columns = {
        "event_hits": "event_recall",
        "correct_side_event_hits": "correct_side_event_recall",
        "within_segment_tolerance_event_hits": ("within_segment_tolerance_event_recall"),
        "exact_bin_event_hits": "exact_bin_event_recall",
    }
    rows: list[dict[str, object]] = []
    for key, group in contributions.groupby(group_columns, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        output: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
        hits = int(group["localized_event_hits"].sum())
        events = int(group["qualified_events"].sum())
        false = int(group["false_proposals"].sum())
        exposure_hours = float(group["exposure_hours"].sum())
        if exposure_hours <= 0.0:
            raise DataValidationError("pooled held-out exposure must be positive")
        output.update(
            {
                "scope": "pooled_four_circuit_heldout",
                "operating_point_status": "estimable",
                "estimability_reason": "",
                "circuit_count": int(group["circuit"].nunique()) if "circuit" in group else 0,
                "car_session_count": int(
                    group[["circuit", "source_session_id", "car_id"]].drop_duplicates().shape[0]
                )
                if {"circuit", "source_session_id", "car_id"}.issubset(group.columns)
                else 0,
                "localized_event_hits": hits,
                "qualified_events": events,
                "localized_event_recall": (float("nan") if events == 0 else hits / events),
                "false_proposals": false,
                "exposure_hours": exposure_hours,
                "false_proposals_per_hour": false / exposure_hours,
                "false_proposals_per_hour_upper_95": false_proposal_rate_upper(
                    false, exposure_hours=exposure_hours
                ),
            }
        )
        if "unresolved_proposals" in group:
            unresolved = int(group["unresolved_proposals"].sum())
            output["unresolved_proposals"] = unresolved
        if "least_favorable_false_proposals" in group:
            least_favorable_false = int(group["least_favorable_false_proposals"].sum())
            output.update(
                {
                    "least_favorable_false_proposals": least_favorable_false,
                    "least_favorable_false_proposals_per_hour": (
                        least_favorable_false / exposure_hours
                    ),
                    "least_favorable_false_proposals_per_hour_upper_95": (
                        false_proposal_rate_upper(
                            least_favorable_false,
                            exposure_hours=exposure_hours,
                        )
                    ),
                    "unresolved_sensitivity_rule": ("count_every_unresolved_proposal_as_false"),
                }
            )
        for count_column, recall_column in component_columns.items():
            if count_column in group:
                component_hits = int(group[count_column].sum())
                output[count_column] = component_hits
                output[recall_column] = float("nan") if events == 0 else component_hits / events
        rows.append(output)
    return pd.DataFrame(rows)


def synthetic_delay_l_star_table(
    contributions: pd.DataFrame,
    *,
    brace_method: str = PRIMARY_BAYESIAN_METHOD,
    comparator_method: str = PRIMARY_DETERMINISTIC_METHOD,
    false_budget_per_hour: float = 2.0,
    minimum_recall: float = 0.50,
    lead_grid_s: Sequence[float] = HORIZONS_S,
) -> pd.DataFrame:
    """Summarize delay-level pooled L-star and the frozen paired method contrast."""

    _require_columns(
        contributions,
        {"synthetic_delay_ms", "method"},
        "synthetic-delay contribution table",
    )
    rows: list[dict[str, object]] = []
    for delay_ms, delay_rows in contributions.groupby("synthetic_delay_ms", sort=True):
        brace = l_star_from_contributions(
            delay_rows,
            method=brace_method,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
            lead_grid_s=lead_grid_s,
        )
        least_favorable_brace = l_star_from_contributions(
            delay_rows,
            method=brace_method,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
            lead_grid_s=lead_grid_s,
            false_count_column="least_favorable_false_proposals",
        )
        least_favorable_comparator = l_star_from_contributions(
            delay_rows,
            method=comparator_method,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
            lead_grid_s=lead_grid_s,
            false_count_column="least_favorable_false_proposals",
        )
        comparator = l_star_from_contributions(
            delay_rows,
            method=comparator_method,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
            lead_grid_s=lead_grid_s,
        )
        rows.append(
            {
                "synthetic_delay_ms": int(delay_ms),
                "brace_method": brace_method,
                "comparator_method": comparator_method,
                "l_star_brace_s": brace,
                "l_star_comparator_s": comparator,
                "delta_l_star_s": brace - comparator,
                "least_favorable_l_star_brace_s": least_favorable_brace,
                "least_favorable_l_star_comparator_s": (least_favorable_comparator),
                "least_favorable_delta_l_star_s": (
                    least_favorable_brace - least_favorable_comparator
                ),
                "false_budget_per_hour": false_budget_per_hour,
                "minimum_recall": minimum_recall,
                "models_refit": False,
                "thresholds_refit": False,
            }
        )
    return pd.DataFrame(rows)


def validate_resume_manifest(
    manifest_path: Path,
    *,
    expected_input_content_hash: str,
    expected_run_identity: Mapping[str, object] | None = None,
    expected_model_identity_hash: str | None = None,
) -> dict[str, object]:
    """Refuse resume if any input or saved artifact byte has changed."""

    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read resume manifest: {exc}") from exc
    if payload.get("input_content_hash") != expected_input_content_hash:
        raise DataValidationError("resume input content hash differs from current inputs")
    if expected_run_identity is not None and payload.get("run_identity") != dict(
        expected_run_identity
    ):
        raise DataValidationError("resume run identity differs from current run")
    if (
        expected_model_identity_hash is not None
        and payload.get("model_identity_hash") != expected_model_identity_hash
    ):
        raise DataValidationError("resume model identity differs from current fitted models")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataValidationError("resume manifest artifacts must be a path-to-hash mapping")
    for raw_path, expected_hash in artifacts.items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = Path(manifest_path).parent / path
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise DataValidationError(f"stale resume artifact: {path}")
    return payload


def _canonical_json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime_event(path: Path, *, step: str, elapsed_seconds: float, **fields: object) -> None:
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "step": step,
        "elapsed_seconds": float(elapsed_seconds),
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _code_content_hash() -> str:
    module_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(
        module_dir.rglob("*.py"), key=lambda value: value.relative_to(module_dir).as_posix()
    ):
        digest.update(path.relative_to(module_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_environment_identity() -> dict[str, object]:
    project_dir = Path(__file__).resolve().parents[2]
    project_files = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = project_dir / name
        if path.is_file():
            project_files[name] = _sha256(path)
    package_versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for name in ("numpy", "pandas", "pyarrow", "scipy", "shapely"):
            try:
                package_versions[name] = version(name)
            except PackageNotFoundError:
                package_versions[name] = "missing"
    except ImportError:  # pragma: no cover - Python 3.12 always supplies importlib.metadata
        package_versions = {"status": "unavailable"}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "project_file_hashes": project_files,
    }


def _validate_frozen_config(config: Mapping[str, object]) -> None:
    if _canonical_json_hash(config) != FROZEN_STUDY_CONFIG_SHA256:
        raise DataValidationError("config does not match the canonical frozen study config")
    try:
        model = config["model"]
        calibration = config["calibration"]
        scoring = config["scoring"]
        primary = config["primary_endpoint"]
        threshold_grid = calibration["threshold_grid"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise DataValidationError("study config is missing frozen experiment fields") from exc
    checks = {
        "posterior_coefficient_draws": model["posterior_coefficient_draws"],  # type: ignore[index]
        "forecast_particles": model["forecast_particles"],  # type: ignore[index]
        "integration_step_s": model["integration_step_s"],  # type: ignore[index]
        "ridge_alpha_grid": tuple(model["ridge_alpha_grid"]),  # type: ignore[index]
        "horizons_s": tuple(scoring["horizons_s"]),  # type: ignore[index]
        "segment_length_m": scoring["segment_length_m"],  # type: ignore[index]
        "bulk_quantiles": threshold_grid["bulk_quantiles"],  # type: ignore[index]
        "bulk_quantile_range": tuple(threshold_grid["bulk_quantile_range"]),  # type: ignore[index]
        "upper_tail_quantiles": threshold_grid["upper_tail_quantiles"],  # type: ignore[index]
        "upper_tail_survival_exponents": tuple(
            threshold_grid["upper_tail_survival_exponents"]  # type: ignore[index]
        ),
        "include_endpoints": tuple(threshold_grid["include_endpoints"]),  # type: ignore[index]
        "false_budget": primary["false_alerts_per_eligible_car_hour_max"],  # type: ignore[index]
    }
    expected = {
        "posterior_coefficient_draws": 256,
        "forecast_particles": 256,
        "integration_step_s": 0.05,
        "ridge_alpha_grid": RIDGE_ALPHA_GRID,
        "horizons_s": HORIZONS_S,
        "segment_length_m": 25.0,
        "bulk_quantiles": 100,
        "bulk_quantile_range": (0.0, 0.99),
        "upper_tail_quantiles": 401,
        "upper_tail_survival_exponents": (2.0, 6.0),
        "include_endpoints": (0.0, 1.0),
        "false_budget": 2.0,
    }
    if checks != expected:
        raise DataValidationError(f"study config differs from frozen experiment: {checks}")


def _model_metadata(fitted: FittedFoldModels) -> dict[str, object]:
    output: dict[str, object] = {
        "ridge_selected_alpha": fitted.ridge_selection.selected_alpha,
        "ridge_mean_gaussian_nll_by_alpha": {
            str(key): value
            for key, value in fitted.ridge_selection.mean_gaussian_nll_by_alpha.items()
        },
        "ridge_inner_fold_scores": fitted.ridge_selection.fold_scores.to_dict(orient="records"),
        "brace": {
            "content_hash": fitted.bayesian.content_hash,
            "posterior_draws": int(fitted.bayesian.coefficient_draws.shape[0]),
            "fit_row_count": fitted.bayesian.fit_row_count,
            "cluster_count": fitted.bayesian.cluster_count,
            "feature_center": fitted.bayesian.feature_center.tolist(),
            "feature_scale": fitted.bayesian.feature_scale.tolist(),
        },
        "posterior_mean_twin": {
            "content_hash": fitted.posterior_mean_twin.content_hash,
            "source": fitted.bayesian.content_hash,
            "process_noise_used": False,
        },
        "side_hazard": {
            "content_hash": fitted.side_hazard.content_hash,
            "feature_center": fitted.side_hazard.feature_center.tolist(),
            "feature_scale": fitted.side_hazard.feature_scale.tolist(),
        },
    }
    if fitted.calibration_tuned_dynamics is not None:
        output[CALIBRATION_TUNED_DETERMINISTIC_METHOD] = {
            "content_hash": fitted.calibration_tuned_dynamics.content_hash,
            "selected_alpha": fitted.calibration_tuned_dynamics.selected_alpha,
            "validation_mse_by_alpha": {
                str(alpha): score
                for alpha, score in (
                    fitted.calibration_tuned_dynamics.validation_mse_by_alpha.items()
                )
            },
            "inner_model_content_hash": (fitted.calibration_tuned_dynamics.model.content_hash),
            "primary_comparator": False,
        }
    return output


def _run_identity(
    data: ExperimentData,
    config: Mapping[str, object],
    *,
    outer_fold: str,
) -> dict[str, object]:
    return {
        "data_content_hash": data.input_content_hash,
        "source_manifest_hash": data.source_manifest_hash,
        "build_manifest_hash": data.build_manifest_hash,
        "config_content_hash": _canonical_json_hash(config),
        "code_content_hash": _code_content_hash(),
        "runtime_environment": _runtime_environment_identity(),
        "fold_test_circuit": str(outer_fold),
        "random_seed_base": int(config["model"]["random_seed_base"]),  # type: ignore[index]
        "registered_methods": list(REGISTERED_METHODS),
    }


def _threshold_fold_identity(
    data: ExperimentData,
    config: Mapping[str, object],
    *,
    output_dir: Path,
    circuit: str,
    allow_heldout_complete: bool,
) -> dict[str, object]:
    fold_dir = Path(output_dir).resolve() / f"fold={circuit}"
    manifest_path = fold_dir / "manifest.json"
    expected_run_identity = _run_identity(data, config, outer_fold=circuit)
    expected_input_hash = _canonical_json_hash(expected_run_identity)
    payload = validate_resume_manifest(
        manifest_path,
        expected_input_content_hash=expected_input_hash,
        expected_run_identity=expected_run_identity,
    )
    allowed_statuses = {"thresholds_complete"}
    if allow_heldout_complete:
        allowed_statuses.add("heldout_complete")
    if payload.get("status") not in allowed_statuses:
        raise DataValidationError(
            f"fold {circuit} must be thresholds_complete before held-out unsealing"
        )
    if set(str(value) for value in payload.get("completed_methods", [])) != set(REGISTERED_METHODS):
        raise DataValidationError(f"fold {circuit} does not have the complete method set")
    method_identities = payload.get("method_identities")
    if not isinstance(method_identities, dict) or set(method_identities) != set(REGISTERED_METHODS):
        raise DataValidationError(f"fold {circuit} has incomplete method identities")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataValidationError(f"fold {circuit} has no artifact registry")
    frozen_names = [
        "model-metadata.json",
        "compact-calibrated-scores.parquet",
        "calibrators.json",
        "calibration-thresholds.csv",
        *(f"raw-scores-{method}.parquet" for method in REGISTERED_METHODS),
    ]
    frozen_artifacts: dict[str, str] = {}
    for name in frozen_names:
        path = (fold_dir / name).resolve()
        expected_hash = artifacts.get(str(path))
        if expected_hash is None or not path.is_file() or _sha256(path) != expected_hash:
            raise DataValidationError(f"fold {circuit} has stale frozen artifact {name}")
        frozen_artifacts[name] = str(expected_hash)
    thresholds = pd.read_csv(fold_dir / "calibration-thresholds.csv")
    required_columns = {
        "fold_test_circuit",
        "method",
        "required_lead_s",
        "false_budget_per_hour",
        "threshold",
        "threshold_source_partition",
        "operating_point_status",
        "estimability_reason",
        "calibration_exposure_hours",
        "calibration_false_count_capacity",
        "minimum_false_count_capacity",
    }
    _require_columns(thresholds, required_columns, "sealed threshold table")
    budgets = tuple(
        sorted(
            {
                2.0,
                *(
                    float(value)
                    for value in config["secondary_false_alert_budgets_per_hour"]  # type: ignore[index]
                ),
            }
        )
    )
    expected_keys = {
        (method, float(lead), float(budget))
        for method in REGISTERED_METHODS
        for lead in HORIZONS_S
        for budget in budgets
    }
    actual_keys = {
        (str(row.method), float(row.required_lead_s), float(row.false_budget_per_hour))
        for row in thresholds.itertuples(index=False)
    }
    if len(thresholds) != len(expected_keys) or actual_keys != expected_keys:
        raise DataValidationError(f"fold {circuit} threshold grid is incomplete or duplicated")
    if (
        set(thresholds["fold_test_circuit"].astype(str)) != {circuit}
        or not (thresholds["threshold_source_partition"].astype(str) == "calibration").all()
    ):
        raise DataValidationError(f"fold {circuit} threshold provenance is invalid")
    allowed_statuses = {"estimable", "not_estimable_insufficient_exposure"}
    statuses = thresholds["operating_point_status"].astype(str)
    if not set(statuses).issubset(allowed_statuses):
        raise DataValidationError(f"fold {circuit} has an unknown estimability status")
    exposure_values = pd.to_numeric(
        thresholds["calibration_exposure_hours"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    budget_values = pd.to_numeric(thresholds["false_budget_per_hour"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    capacity_values = pd.to_numeric(
        thresholds["calibration_false_count_capacity"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    minimum_values = pd.to_numeric(
        thresholds["minimum_false_count_capacity"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    frozen_minimum = float(
        config["primary_endpoint"]["minimum_calibration_false_count_capacity"]  # type: ignore[index]
    )
    if (
        not np.isfinite(exposure_values).all()
        or np.any(exposure_values <= 0.0)
        or not np.isfinite(budget_values).all()
        or np.any(budget_values < 0.0)
        or not np.isfinite(capacity_values).all()
        or not np.allclose(
            capacity_values,
            budget_values * exposure_values,
            atol=1e-12,
            rtol=0.0,
        )
        or not np.allclose(
            minimum_values,
            frozen_minimum,
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise DataValidationError(f"fold {circuit} has invalid exposure-capacity accounting")
    expected_estimable = capacity_values + 1e-12 >= frozen_minimum
    estimable = statuses.to_numpy() == "estimable"
    if not np.array_equal(estimable, expected_estimable):
        raise DataValidationError(f"fold {circuit} violates the frozen estimability rule")
    numeric_thresholds = pd.to_numeric(
        thresholds.loc[estimable, "threshold"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_thresholds).all() or np.any(
        (numeric_thresholds < 0.0) | (numeric_thresholds > 1.0)
    ):
        raise DataValidationError(f"fold {circuit} has an invalid estimable threshold")
    if not (thresholds.loc[~estimable, "threshold"].astype(str) == "NOT_ESTIMABLE").all():
        raise DataValidationError(f"fold {circuit} encodes an invalid N/E threshold")
    expected_reason = "false_budget_per_hour_times_calibration_exposure_hours_below_1"
    if not (thresholds.loc[~estimable, "estimability_reason"].astype(str) == expected_reason).all():
        raise DataValidationError(f"fold {circuit} has an invalid N/E reason")
    estimable_reasons = thresholds.loc[estimable, "estimability_reason"]
    if not (estimable_reasons.isna() | estimable_reasons.astype(str).str.strip().eq("")).all():
        raise DataValidationError(f"fold {circuit} has a reason on an estimable row")
    primary = thresholds.loc[
        np.isclose(
            thresholds["false_budget_per_hour"].to_numpy(dtype=np.float64),
            2.0,
            atol=1e-12,
            rtol=0.0,
        )
    ]
    if (
        len(primary) != len(REGISTERED_METHODS) * len(HORIZONS_S)
        or not (primary["operating_point_status"].astype(str) == "estimable").all()
    ):
        raise DataValidationError(f"fold {circuit} has a non-estimable primary point")
    return {
        "fold_test_circuit": circuit,
        "run_identity": expected_run_identity,
        "model_identity_hash": payload.get("model_identity_hash"),
        "method_identities": method_identities,
        "frozen_artifacts": frozen_artifacts,
    }


def create_threshold_freeze_seal(
    data: ExperimentData,
    *,
    config: Mapping[str, object],
    output_dir: Path,
) -> Path:
    """Seal all circuit thresholds before any held-out outcome can be joined."""

    _validate_frozen_config(config)
    if data.build_manifest_hash is None:
        raise DataValidationError("threshold sealing requires an authoritative build manifest")
    registered_circuits = sorted(
        str(value)
        for value in config["cohort"]["circuits"]  # type: ignore[index]
    )
    if len(registered_circuits) != 4 or sorted(data.corridors) != registered_circuits:
        raise DataValidationError("threshold sealing requires exactly the four registered circuits")
    root = Path(output_dir).resolve()
    seal_path = root / "threshold-freeze-seal.json"
    if seal_path.exists():
        raise DataValidationError(f"threshold freeze seal already exists: {seal_path}")
    entries = [
        _threshold_fold_identity(
            data,
            config,
            output_dir=root,
            circuit=circuit,
            allow_heldout_complete=False,
        )
        for circuit in registered_circuits
    ]
    _atomic_json(
        seal_path,
        {
            "schema_version": 1,
            "status": "thresholds_frozen_before_heldout",
            "data_content_hash": data.input_content_hash,
            "source_manifest_hash": data.source_manifest_hash,
            "build_manifest_hash": data.build_manifest_hash,
            "config_content_hash": _canonical_json_hash(config),
            "code_content_hash": _code_content_hash(),
            "registered_circuits": registered_circuits,
            "registered_methods": list(REGISTERED_METHODS),
            "fold_threshold_identities": entries,
        },
    )
    return seal_path


def validate_threshold_freeze_seal(
    data: ExperimentData,
    *,
    config: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    """Validate the all-fold pre-unsealing seal against current frozen artifacts."""

    root = Path(output_dir).resolve()
    registered_circuits = sorted(
        str(value)
        for value in config["cohort"]["circuits"]  # type: ignore[index]
    )
    if len(registered_circuits) != 4 or sorted(data.corridors) != registered_circuits:
        raise DataValidationError(
            "threshold freeze seal requires exactly the four registered circuits"
        )
    seal_path = root / "threshold-freeze-seal.json"
    try:
        payload = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(
            "held-out evaluation requires a valid all-fold threshold freeze seal"
        ) from exc
    expected_header = {
        "status": "thresholds_frozen_before_heldout",
        "data_content_hash": data.input_content_hash,
        "source_manifest_hash": data.source_manifest_hash,
        "build_manifest_hash": data.build_manifest_hash,
        "config_content_hash": _canonical_json_hash(config),
        "code_content_hash": _code_content_hash(),
        "registered_circuits": registered_circuits,
        "registered_methods": list(REGISTERED_METHODS),
    }
    if any(payload.get(key) != value for key, value in expected_header.items()):
        raise DataValidationError("threshold freeze seal identity differs from current run")
    current_entries = [
        _threshold_fold_identity(
            data,
            config,
            output_dir=root,
            circuit=circuit,
            allow_heldout_complete=True,
        )
        for circuit in registered_circuits
    ]
    if payload.get("fold_threshold_identities") != current_entries:
        raise DataValidationError("threshold freeze seal no longer matches frozen fold artifacts")
    return payload


def _evaluate_heldout_from_sealed_artifacts(
    data: ExperimentData,
    *,
    fold: str,
    fold_dir: Path,
    manifest_path: Path,
    manifest: dict[str, object],
    runtime_path: Path,
) -> FoldRunResult:
    """Evaluate one test fold without changing any pre-unsealing artifact bytes."""

    frozen_paths = (
        fold_dir / "model-metadata.json",
        fold_dir / "compact-calibrated-scores.parquet",
        fold_dir / "calibrators.json",
        fold_dir / "calibration-thresholds.csv",
        *(fold_dir / f"raw-scores-{method}.parquet" for method in REGISTERED_METHODS),
    )
    frozen_hashes = {path: _sha256(path) for path in frozen_paths}
    calibrated_scores = pd.read_parquet(fold_dir / "compact-calibrated-scores.parquet")
    thresholds = pd.read_csv(fold_dir / "calibration-thresholds.csv")
    try:
        calibrator_payload = json.loads((fold_dir / "calibrators.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError("cannot read sealed calibrator metadata") from exc
    if calibrator_payload.get("fit_partition") != "calibration":
        raise DataValidationError("sealed prevalence was not fit on calibration")
    prevalence_records = calibrator_payload.get("reference_prevalence")
    if not isinstance(prevalence_records, list) or not prevalence_records:
        raise DataValidationError("sealed calibrator metadata lacks reference prevalence")
    prevalence = pd.DataFrame(prevalence_records)
    exposure = join_exposure_to_split(data.timing, data.splits, fold_test_circuit=fold)
    heldout_start = time.perf_counter()
    heldout = evaluate_heldout_operating_points(
        calibrated_scores,
        data.events,
        exposure,
        thresholds,
        outer_fold=fold,
    )
    probability = evaluate_probability_performance(
        calibrated_scores,
        data.targets,
        prevalence,
        include_pooled=False,
    )
    reliability = reliability_bin_source_table(
        calibrated_scores,
        data.targets,
        include_pooled=False,
    )
    reliability_clusters = reliability_cluster_source_table(
        calibrated_scores,
        data.targets,
    )
    monotonicity = horizon_monotonicity_diagnostics(
        calibrated_scores,
        include_pooled=False,
    )
    output_tables = {
        fold_dir / "heldout-operating-metrics.csv": heldout.metrics,
        fold_dir / "heldout-car-contributions.parquet": heldout.car_contributions,
        fold_dir / "heldout-proposal-labels.parquet": heldout.proposal_labels,
        fold_dir / "heldout-probability-metrics.csv": probability,
        fold_dir / "heldout-reliability-bins.csv": reliability,
        fold_dir / "heldout-reliability-cluster-contributions.parquet": (reliability_clusters),
        fold_dir / "heldout-horizon-monotonicity.csv": monotonicity,
        fold_dir / "heldout-synthetic-delay-metrics.csv": heldout.delay_metrics,
        fold_dir / "heldout-synthetic-delay-car-contributions.parquet": (
            heldout.delay_car_contributions
        ),
    }
    artifacts = dict(manifest.get("artifacts", {}))
    for stale_staging in fold_dir.glob(".heldout-staging-*"):
        if stale_staging.is_dir():
            shutil.rmtree(stale_staging)
    for path in output_tables:
        if not path.exists():
            continue
        if str(path.resolve()) in artifacts:
            raise DataValidationError(
                f"threshold-stage manifest unexpectedly registered held-out output: {path}"
            )
        if not path.is_file():
            raise DataValidationError(f"held-out output target is not a file: {path}")
        path.unlink()
    staging_dir = Path(tempfile.mkdtemp(prefix=".heldout-staging-", dir=fold_dir))
    try:
        staged_outputs: list[tuple[Path, Path]] = []
        for path, table in output_tables.items():
            staged = staging_dir / path.name
            if path.suffix == ".csv":
                table.to_csv(staged, index=False, lineterminator="\n")
            else:
                table.to_parquet(staged, index=False, compression="zstd")
            staged_outputs.append((staged, path))
        staged_runtime = staging_dir / runtime_path.name
        _runtime_event(
            staged_runtime,
            step="heldout_evaluation_from_sealed_artifacts",
            elapsed_seconds=time.perf_counter() - heldout_start,
            fold=fold,
            models_refit=False,
            calibrators_refit=False,
            thresholds_refit=False,
        )
        staged_outputs.append((staged_runtime, runtime_path))
        if any(_sha256(path) != digest for path, digest in frozen_hashes.items()):
            raise DataValidationError("held-out transition changed a sealed artifact")
        for staged, final in staged_outputs:
            artifacts[str(final.resolve())] = _sha256(staged)
            staged.replace(final)
        staging_dir.rmdir()
        manifest.update(
            {
                "status": "heldout_complete",
                "heldout_consumed_sealed_artifacts": True,
                "models_refit_for_heldout": False,
                "calibrators_refit_for_heldout": False,
                "thresholds_refit_for_heldout": False,
                "artifacts": artifacts,
            }
        )
        _atomic_json(manifest_path, manifest)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return FoldRunResult(
        fold,
        "heldout_complete",
        manifest_path,
        tuple(Path(path) for path in artifacts),
    )


def run_outer_fold(
    data: ExperimentData,
    *,
    outer_fold: str,
    config: Mapping[str, object],
    output_dir: Path,
    resume: bool = False,
    evaluate_heldout: bool = False,
    batch_size: int = 128,
) -> FoldRunResult:
    """Run one fold in its own directory, checkpointing every scored method.

    ``evaluate_heldout=False`` stops after calibration-only thresholds.  This is
    the safe pre-unsealing stage.  Passing ``True`` is an explicit request to
    join test outcomes after all thresholds have been frozen.
    """

    _validate_frozen_config(config)
    if data.build_manifest_hash is None:
        raise DataValidationError("fold execution requires an authoritative build manifest")
    if str(outer_fold) not in data.corridors:
        raise DataValidationError(f"unknown outer fold: {outer_fold}")
    safe_fold = str(outer_fold)
    if not safe_fold or any(
        character not in "-_.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        for character in safe_fold
    ):
        raise DataValidationError("outer fold contains unsupported path characters")
    if evaluate_heldout:
        validate_threshold_freeze_seal(
            data,
            config=config,
            output_dir=output_dir,
        )
    fold_dir = Path(output_dir) / f"fold={safe_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fold_dir / "manifest.json"
    runtime_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    runtime_path = fold_dir / f"runtime-{runtime_stamp}.jsonl"
    code_hash = _code_content_hash()
    config_hash = _canonical_json_hash(config)
    random_seed_base = int(config["model"]["random_seed_base"])  # type: ignore[index]
    run_identity = _run_identity(data, config, outer_fold=safe_fold)
    run_input_hash = _canonical_json_hash(run_identity)
    manifest: dict[str, object]
    if manifest_path.exists():
        if not resume:
            raise DataValidationError(
                f"fold output already exists; pass --resume after verifying target: {fold_dir}"
            )
        manifest = validate_resume_manifest(
            manifest_path,
            expected_input_content_hash=run_input_hash,
            expected_run_identity=run_identity,
        )
    else:
        manifest = {
            "schema_version": 1,
            "fold_test_circuit": safe_fold,
            "input_content_hash": run_input_hash,
            "data_content_hash": data.input_content_hash,
            "source_manifest_hash": data.source_manifest_hash,
            "build_manifest_hash": data.build_manifest_hash,
            "config_content_hash": config_hash,
            "code_content_hash": code_hash,
            "random_seed_base": random_seed_base,
            "run_identity": run_identity,
            "status": "initialized",
            "completed_methods": [],
            "artifacts": {},
            "warnings": [
                "Track exits in simulation are not crashes, impacts, or Formula 1 "
                "deployment evidence."
            ],
        }
        _atomic_json(manifest_path, manifest)

    if not evaluate_heldout and manifest.get("status") in {
        "thresholds_complete",
        "heldout_complete",
    }:
        return FoldRunResult(
            safe_fold,
            str(manifest["status"]),
            manifest_path,
            tuple(Path(path) for path in dict(manifest.get("artifacts", {}))),
        )
    if evaluate_heldout:
        status = str(manifest.get("status"))
        if status == "heldout_complete":
            return FoldRunResult(
                safe_fold,
                status,
                manifest_path,
                tuple(Path(path) for path in dict(manifest.get("artifacts", {}))),
            )
        if status != "thresholds_complete":
            raise DataValidationError(
                "held-out evaluation requires this fold to be thresholds_complete"
            )
        return _evaluate_heldout_from_sealed_artifacts(
            data,
            fold=safe_fold,
            fold_dir=fold_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            runtime_path=runtime_path,
        )

    start = time.perf_counter()
    prepared = prepare_outer_fold(data, outer_fold=safe_fold)
    fitted = fit_fold_models(
        prepared.fit_model_table(),
        seed=random_seed_base,
        alpha_grid=tuple(config["model"]["ridge_alpha_grid"]),  # type: ignore[index]
        posterior_draws=256,
    )
    metadata = _model_metadata(fitted)
    model_identity_hash = _canonical_json_hash(metadata)
    previous_model_identity = manifest.get("model_identity_hash")
    if previous_model_identity is not None and previous_model_identity != model_identity_hash:
        raise DataValidationError("resume model identity differs from current fitted models")
    model_metadata_path = fold_dir / "model-metadata.json"
    _atomic_json(model_metadata_path, metadata)
    _runtime_event(
        runtime_path,
        step="fit_models",
        elapsed_seconds=time.perf_counter() - start,
        fold=safe_fold,
    )
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts[str(model_metadata_path.resolve())] = _sha256(model_metadata_path)
    methods = list(REGISTERED_METHODS)
    if fitted.calibration_tuned_dynamics is None:
        raise DataValidationError("registered calibration-MSE comparator was not fitted")
    method_identities = {
        method: (
            str(fitted.models[method].content_hash)
            if method in fitted.models
            else _canonical_json_hash(
                {"method": method, "code_content_hash": code_hash, "config": config_hash}
            )
        )
        for method in methods
    }
    manifest.update(
        {
            "model_identity_hash": model_identity_hash,
            "method_identities": method_identities,
            "artifacts": artifacts,
        }
    )
    _atomic_json(manifest_path, manifest)
    if manifest.get("status") == "heldout_complete" or (
        manifest.get("status") == "thresholds_complete" and not evaluate_heldout
    ):
        artifacts[str(runtime_path.resolve())] = _sha256(runtime_path)
        manifest["artifacts"] = artifacts
        _atomic_json(manifest_path, manifest)
        return FoldRunResult(
            safe_fold,
            str(manifest["status"]),
            manifest_path,
            tuple(Path(path) for path in artifacts),
        )
    completed = set(str(value) for value in manifest.get("completed_methods", []))
    score_tables: list[pd.DataFrame] = []
    for method in methods:
        score_path = fold_dir / f"raw-scores-{method}.parquet"
        if method in completed:
            if not score_path.is_file():
                raise DataValidationError(
                    f"resume manifest is missing method artifact: {score_path}"
                )
            score_tables.append(pd.read_parquet(score_path))
            continue
        method_start = time.perf_counter()
        score = score_partition_methods(
            prepared.causal_features,
            corridors=data.corridors,
            models=fitted.models,
            methods=(method,),
            outer_fold=safe_fold,
            base_seed=random_seed_base,
            batch_size=batch_size,
        )
        score.to_parquet(score_path, index=False, compression="zstd")
        score_tables.append(score)
        completed.add(method)
        artifacts[str(score_path.resolve())] = _sha256(score_path)
        _runtime_event(
            runtime_path,
            step="score_method",
            elapsed_seconds=time.perf_counter() - method_start,
            fold=safe_fold,
            method=method,
            rows=len(score),
        )
        manifest.update(
            {
                "status": "scoring",
                "completed_methods": sorted(completed),
                "artifacts": artifacts,
            }
        )
        _atomic_json(manifest_path, manifest)
    raw_scores = pd.concat(score_tables, ignore_index=True)
    calibrators = fit_horizon_calibrators(raw_scores, prepared.targets)
    prevalence = calibration_reference_prevalence(raw_scores, prepared.targets)
    calibrated_scores = apply_horizon_calibrators(raw_scores, calibrators)
    calibrated_path = fold_dir / "compact-calibrated-scores.parquet"
    calibrated_scores.to_parquet(calibrated_path, index=False, compression="zstd")
    calibrator_path = fold_dir / "calibrators.json"
    _atomic_json(
        calibrator_path,
        {
            "calibrators": [
                {
                    "method": method,
                    "horizon_s": horizon,
                    "slope": calibrator.slope,
                    "intercept": calibrator.intercept,
                }
                for (method, horizon), calibrator in sorted(calibrators.items())
            ],
            "reference_prevalence": prevalence.to_dict(orient="records"),
            "fit_partition": "calibration",
        },
    )
    exposure = join_exposure_to_split(data.timing, data.splits, fold_test_circuit=safe_fold)
    budgets = tuple(
        sorted(
            {
                2.0,
                *(
                    float(value)
                    for value in config["secondary_false_alert_budgets_per_hour"]  # type: ignore[index]
                ),
            }
        )
    )
    thresholds = select_calibration_operating_points(
        calibrated_scores,
        data.events,
        exposure,
        outer_fold=safe_fold,
        required_leads_s=HORIZONS_S,
        false_budgets_per_hour=budgets,
    )
    primary_rows = thresholds.loc[
        np.isclose(
            thresholds["false_budget_per_hour"].to_numpy(dtype=np.float64),
            2.0,
            atol=1e-12,
            rtol=0.0,
        )
    ]
    expected_primary_rows = len(REGISTERED_METHODS) * len(HORIZONS_S)
    if (
        len(primary_rows) != expected_primary_rows
        or not (primary_rows["operating_point_status"].astype(str) == "estimable").all()
    ):
        raise DataValidationError(
            "primary 2-per-hour operating points must be estimable for every method and lead"
        )
    threshold_path = fold_dir / "calibration-thresholds.csv"
    thresholds.to_csv(threshold_path, index=False, lineterminator="\n")
    for path in (calibrated_path, calibrator_path, threshold_path, runtime_path):
        artifacts[str(path.resolve())] = _sha256(path)
    manifest.update(
        {
            "status": "thresholds_complete",
            "completed_methods": sorted(completed),
            "artifacts": artifacts,
        }
    )
    _atomic_json(manifest_path, manifest)
    return FoldRunResult(
        safe_fold,
        "thresholds_complete",
        manifest_path,
        tuple(Path(path) for path in artifacts),
    )


def aggregate_completed_heldout_folds(
    data: ExperimentData,
    *,
    config: Mapping[str, object],
    output_dir: Path,
    destination: Path | None = None,
) -> HeldoutAggregateResult:
    """Pool four immutable held-out fold artifacts without refitting or reselection."""

    aggregate_started = time.perf_counter()
    _validate_frozen_config(config)
    if data.build_manifest_hash is None:
        raise DataValidationError("pooled aggregation requires an authoritative build manifest")
    root = Path(output_dir).resolve()
    registered_circuits = tuple(
        sorted(str(value) for value in config["cohort"]["circuits"])  # type: ignore[index]
    )
    if tuple(sorted(data.corridors)) != registered_circuits or len(registered_circuits) != 4:
        raise DataValidationError(
            "pooled held-out aggregation requires exactly four registered circuits"
        )
    seal_payload = validate_threshold_freeze_seal(
        data,
        config=config,
        output_dir=root,
    )
    seal_path = root / "threshold-freeze-seal.json"
    seal_record = {
        "path": str(seal_path),
        "sha256": _sha256(seal_path),
        "status": seal_payload.get("status"),
    }
    pooled_dir = root / "pooled-heldout" if destination is None else Path(destination).resolve()
    if pooled_dir.exists():
        raise DataValidationError(f"pooled held-out output already exists: {pooled_dir}")
    score_tables: list[pd.DataFrame] = []
    contribution_tables: list[pd.DataFrame] = []
    delay_contribution_tables: list[pd.DataFrame] = []
    operating_metric_tables: list[pd.DataFrame] = []
    prevalence_tables: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, object]] = []
    for circuit in registered_circuits:
        fold_dir = root / f"fold={circuit}"
        manifest_path = fold_dir / "manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"cannot read completed fold manifest: {exc}") from exc
        if payload.get("status") != "heldout_complete":
            raise DataValidationError(f"fold {circuit} is not heldout_complete")
        if payload.get("data_content_hash") != data.input_content_hash:
            raise DataValidationError(f"fold {circuit} was scored from different data")
        if payload.get("build_manifest_hash") != data.build_manifest_hash:
            raise DataValidationError(f"fold {circuit} has a different build manifest")
        if payload.get("source_manifest_hash") != data.source_manifest_hash:
            raise DataValidationError(f"fold {circuit} has a different source manifest")
        if payload.get("config_content_hash") != _canonical_json_hash(config):
            raise DataValidationError(f"fold {circuit} has a different study config")
        if payload.get("code_content_hash") != _code_content_hash():
            raise DataValidationError(f"fold {circuit} was produced by different code")
        validate_resume_manifest(
            manifest_path,
            expected_input_content_hash=str(payload.get("input_content_hash")),
            expected_run_identity=payload.get("run_identity"),  # type: ignore[arg-type]
            expected_model_identity_hash=str(payload.get("model_identity_hash")),
        )
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            raise DataValidationError(f"fold {circuit} has no artifact registry")
        consumed_names = (
            "compact-calibrated-scores.parquet",
            "calibrators.json",
            "heldout-car-contributions.parquet",
            "heldout-operating-metrics.csv",
            "heldout-synthetic-delay-car-contributions.parquet",
        )
        for name in consumed_names:
            path = (fold_dir / name).resolve()
            expected_hash = artifacts.get(str(path))
            if expected_hash is None or not path.is_file() or _sha256(path) != expected_hash:
                raise DataValidationError(
                    f"fold {circuit} consumed artifact is missing or stale: {name}"
                )
        score_columns = [
            "fold_test_circuit",
            "partition",
            "circuit",
            "source_session_id",
            "car_id",
            "frame_index",
            "time_seconds",
            "input_valid_causal",
            "in_corridor",
            "hard_break",
            "method",
            *(f"exit_probability_{_horizon_suffix(horizon)}s" for horizon in HORIZONS_S),
        ]
        fold_scores = pd.read_parquet(
            fold_dir / "compact-calibrated-scores.parquet",
            columns=score_columns,
            filters=[("partition", "==", "test")],
        )
        score_tables.append(fold_scores)
        try:
            calibrator_payload = json.loads(
                (fold_dir / "calibrators.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise DataValidationError(
                f"cannot read sealed calibrator metadata for {circuit}"
            ) from exc
        if calibrator_payload.get("fit_partition") != "calibration":
            raise DataValidationError(f"fold {circuit} prevalence is not calibration-only")
        prevalence_records = calibrator_payload.get("reference_prevalence")
        if not isinstance(prevalence_records, list):
            raise DataValidationError(f"fold {circuit} has no sealed reference prevalence")
        prevalence_tables.append(pd.DataFrame(prevalence_records))
        contribution_tables.append(pd.read_parquet(fold_dir / "heldout-car-contributions.parquet"))
        operating_metric_tables.append(pd.read_csv(fold_dir / "heldout-operating-metrics.csv"))
        delay_contribution_tables.append(
            pd.read_parquet(fold_dir / "heldout-synthetic-delay-car-contributions.parquet")
        )
        fold_manifests.append(
            {
                "fold_test_circuit": circuit,
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "model_identity_hash": payload.get("model_identity_hash"),
            }
        )
    scores = pd.concat(score_tables, ignore_index=True)
    contributions = pd.concat(contribution_tables, ignore_index=True)
    delay_contributions = pd.concat(delay_contribution_tables, ignore_index=True)
    fold_operating = pd.concat(operating_metric_tables, ignore_index=True)
    prevalence = pd.concat(prevalence_tables, ignore_index=True)
    test_scores = scores.loc[scores["partition"].astype(str) == "test"].copy()
    if (
        test_scores.empty
        or not (
            test_scores["fold_test_circuit"].astype(str) == test_scores["circuit"].astype(str)
        ).all()
    ):
        raise DataValidationError("pooled test scores must come only from their held-out circuit")
    physical_score_key = [
        "fold_test_circuit",
        "circuit",
        "source_session_id",
        "car_id",
        "frame_index",
        "time_seconds",
    ]
    score_key = ["method", *physical_score_key]
    if test_scores.duplicated(score_key).any():
        raise DataValidationError("pooled held-out scores duplicate a physical method/frame row")
    expected_heldout_units: set[tuple[str, str, str]] = set()
    for circuit, group in test_scores.groupby("circuit", sort=True):
        if set(group["method"].astype(str)) != set(REGISTERED_METHODS):
            raise DataValidationError(f"held-out score methods are incomplete for {circuit}")
        split_units = data.splits.loc[
            (data.splits["fold_test_circuit"].astype(str) == str(circuit))
            & (data.splits["partition"].astype(str) == "test"),
            list(UNIT_KEYS),
        ].copy()
        expected_heldout_units.update(
            tuple(str(row[column]) for column in ("circuit", "source_session_id", "car_id"))
            for _, row in split_units.iterrows()
        )
        expected_frames = data.frames.merge(
            split_units,
            on=list(UNIT_KEYS),
            how="inner",
            validate="many_to_one",
        )
        expected_keys = expected_frames.loc[
            :, ["circuit", "source_session_id", "car_id", "frame_index", "time_seconds"]
        ].copy()
        expected_keys.insert(0, "fold_test_circuit", str(circuit))
        expected_keys = expected_keys.sort_values(physical_score_key, kind="stable").reset_index(
            drop=True
        )
        if expected_keys.duplicated(physical_score_key).any():
            raise DataValidationError(f"authoritative held-out frames duplicate keys for {circuit}")
        for method, method_rows in group.groupby("method", sort=True):
            actual_keys = (
                method_rows.loc[:, physical_score_key]
                .sort_values(physical_score_key, kind="stable")
                .reset_index(drop=True)
            )
            if not actual_keys.equals(expected_keys):
                raise DataValidationError(
                    f"held-out score frame keys are incomplete for {circuit}/{method}"
                )
    if len(expected_heldout_units) != int(config["cohort"]["car_sessions"]):  # type: ignore[index]
        raise DataValidationError("authoritative held-out car-session count differs from config")
    budgets = tuple(
        sorted(
            {
                2.0,
                *(
                    float(value)
                    for value in config["secondary_false_alert_budgets_per_hour"]  # type: ignore[index]
                ),
            }
        )
    )
    expected_metric_keys = {
        (circuit, method, float(lead), float(budget))
        for circuit in registered_circuits
        for method in REGISTERED_METHODS
        for lead in HORIZONS_S
        for budget in budgets
    }
    actual_metric_keys = {
        (
            str(row.fold_test_circuit),
            str(row.method),
            float(row.required_lead_s),
            float(row.false_budget_per_hour),
        )
        for row in fold_operating.itertuples(index=False)
    }
    if (
        len(fold_operating) != len(expected_metric_keys)
        or actual_metric_keys != expected_metric_keys
    ):
        raise DataValidationError("held-out operating grid is incomplete or duplicated")
    grid_columns = ["method", "required_lead_s", "false_budget_per_hour"]
    globally_estimable: set[tuple[str, float, float]] = set()
    not_estimable_rows: list[dict[str, object]] = []
    for key, group in fold_operating.groupby(grid_columns, sort=True):
        statuses = group["operating_point_status"].astype(str)
        normalized_key = (str(key[0]), float(key[1]), float(key[2]))
        if (statuses == "estimable").all():
            globally_estimable.add(normalized_key)
        else:
            reasons = {
                str(row.fold_test_circuit): {
                    "status": str(row.operating_point_status),
                    "reason": str(row.estimability_reason),
                }
                for row in group.itertuples(index=False)
            }
            not_estimable_rows.append(
                {
                    "method": normalized_key[0],
                    "required_lead_s": normalized_key[1],
                    "false_budget_per_hour": normalized_key[2],
                    "scope": "pooled_four_circuit_heldout",
                    "circuit_count": 4,
                    "operating_point_status": "not_estimable_insufficient_exposure",
                    "estimability_reason": json.dumps(
                        reasons, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    primary_keys = {
        (method, float(lead), 2.0) for method in REGISTERED_METHODS for lead in HORIZONS_S
    }
    if not primary_keys.issubset(globally_estimable):
        raise DataValidationError("pooled primary grid is not estimable in every fold")

    def globally_estimable_mask(table: pd.DataFrame) -> NDArray[np.bool_]:
        return np.asarray(
            [
                (str(row.method), float(row.required_lead_s), float(row.false_budget_per_hour))
                in globally_estimable
                for row in table.itertuples(index=False)
            ],
            dtype=bool,
        )

    contributions = contributions.loc[globally_estimable_mask(contributions)].copy()
    delay_contributions = delay_contributions.loc[
        globally_estimable_mask(delay_contributions)
    ].copy()
    for name, table in (
        ("car contributions", contributions),
        ("delay contributions", delay_contributions),
    ):
        if not (table["fold_test_circuit"].astype(str) == table["circuit"].astype(str)).all():
            raise DataValidationError(f"pooled {name} are not held-out-only")
    contribution_key = [
        "circuit",
        "source_session_id",
        "car_id",
        *grid_columns,
    ]
    if contributions.duplicated(contribution_key).any():
        raise DataValidationError("pooled car contributions contain duplicate operating rows")
    contribution_grid = {
        (str(row.method), float(row.required_lead_s), float(row.false_budget_per_hour))
        for row in contributions.loc[:, grid_columns].drop_duplicates().itertuples(index=False)
    }
    if contribution_grid != globally_estimable:
        raise DataValidationError("pooled car contribution operating grid is incomplete")
    reference_units = expected_heldout_units
    for _, group in contributions.groupby(grid_columns, sort=True):
        units = {
            (str(row.circuit), str(row.source_session_id), str(row.car_id))
            for row in group.itertuples(index=False)
        }
        if units != reference_units:
            raise DataValidationError("pooled operating points do not share complete car units")
    denominator_check = contributions.groupby(
        ["circuit", "source_session_id", "car_id"], sort=False
    )[["qualified_events", "exposure_hours"]].nunique(dropna=False)
    if (denominator_check > 1).any(axis=None):
        raise DataValidationError("pooled method/lead/budget denominators are inconsistent")
    delay_key = [*contribution_key, "synthetic_delay_ms"]
    if delay_contributions.duplicated(delay_key).any():
        raise DataValidationError("pooled delay contributions contain duplicate operating rows")
    if set(delay_contributions["synthetic_delay_ms"].astype(int)) != {0, 40, 80, 160}:
        raise DataValidationError("pooled synthetic-delay grid is incomplete")
    delay_grid = {
        (
            str(row.method),
            float(row.required_lead_s),
            float(row.false_budget_per_hour),
            int(row.synthetic_delay_ms),
        )
        for row in delay_contributions.loc[:, [*grid_columns, "synthetic_delay_ms"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    expected_delay_grid = {
        (*key, delay_ms) for key in globally_estimable for delay_ms in (0, 40, 80, 160)
    }
    if delay_grid != expected_delay_grid:
        raise DataValidationError("pooled delay contribution operating grid is incomplete")
    for _, group in delay_contributions.groupby([*grid_columns, "synthetic_delay_ms"], sort=True):
        delay_units = {
            (str(row.circuit), str(row.source_session_id), str(row.car_id))
            for row in group.itertuples(index=False)
        }
        if delay_units != reference_units:
            raise DataValidationError(
                "pooled delay operating points do not share authoritative car units"
            )
    prevalence_key = ["fold_test_circuit", "method", "horizon_s"]
    expected_prevalence_keys = {
        (circuit, method, float(horizon))
        for circuit in registered_circuits
        for method in REGISTERED_METHODS
        for horizon in HORIZONS_S
    }
    actual_prevalence_keys = {
        (str(row.fold_test_circuit), str(row.method), float(row.horizon_s))
        for row in prevalence.itertuples(index=False)
    }
    if (
        prevalence.duplicated(prevalence_key).any()
        or actual_prevalence_keys != expected_prevalence_keys
    ):
        raise DataValidationError("sealed calibration prevalence grid is incomplete")
    probability = evaluate_probability_performance(scores, data.targets, prevalence)
    reliability = reliability_bin_source_table(scores, data.targets)
    reliability_clusters = reliability_cluster_source_table(scores, data.targets)
    monotonicity = horizon_monotonicity_diagnostics(scores)
    cluster_grid = ["method", "horizon_s", "bin_index"]
    expected_cluster_grid = {
        (method, float(horizon), bin_index)
        for method in REGISTERED_METHODS
        for horizon in HORIZONS_S
        for bin_index in range(10)
    }
    actual_cluster_grid = {
        (str(row.method), float(row.horizon_s), int(row.bin_index))
        for row in reliability_clusters.loc[:, cluster_grid]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if actual_cluster_grid != expected_cluster_grid:
        raise DataValidationError("reliability-bin contribution grid is incomplete")
    for _, group in reliability_clusters.groupby(cluster_grid, sort=True):
        cluster_units = {
            (str(row.circuit), str(row.source_session_id), str(row.car_id))
            for row in group.itertuples(index=False)
        }
        if cluster_units != reference_units:
            raise DataValidationError(
                "reliability-bin contributions do not share authoritative car units"
            )
    operating = pooled_operating_metrics_from_contributions(contributions)
    if not_estimable_rows:
        operating = pd.concat(
            [operating, pd.DataFrame(not_estimable_rows)], ignore_index=True, sort=False
        )
    delay_metrics = pooled_operating_metrics_from_contributions(
        delay_contributions,
        synthetic_delay=True,
    )
    if not_estimable_rows:
        delayed_ne = pd.DataFrame(
            [
                {**row, "synthetic_delay_ms": delay_ms}
                for row in not_estimable_rows
                for delay_ms in (0, 40, 80, 160)
            ]
        )
        delay_metrics = pd.concat([delay_metrics, delayed_ne], ignore_index=True, sort=False)
    delay_l_star = synthetic_delay_l_star_table(delay_contributions)
    l_star_sensitivity = paired_l_star_sensitivity_table(contributions)
    pooled_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{pooled_dir.name}-staging-", dir=pooled_dir.parent)
    )
    output_tables = {
        staging_dir / "pooled-heldout-probability-metrics.csv": probability,
        staging_dir / "pooled-heldout-reliability-bins.csv": reliability,
        staging_dir / "pooled-heldout-reliability-cluster-contributions.parquet": (
            reliability_clusters
        ),
        staging_dir / "pooled-heldout-horizon-monotonicity.csv": monotonicity,
        staging_dir / "pooled-heldout-operating-metrics.csv": operating,
        staging_dir / "pooled-heldout-car-contributions.parquet": contributions,
        staging_dir / "pooled-heldout-synthetic-delay-metrics.csv": delay_metrics,
        staging_dir / "pooled-heldout-synthetic-delay-car-contributions.parquet": (
            delay_contributions
        ),
        staging_dir / "pooled-heldout-synthetic-delay-l-star.csv": delay_l_star,
        staging_dir / "pooled-heldout-l-star-sensitivity.csv": l_star_sensitivity,
    }
    artifacts: dict[str, str] = {}
    try:
        for path, table in output_tables.items():
            if path.suffix == ".csv":
                table.to_csv(path, index=False, lineterminator="\n")
            else:
                table.to_parquet(path, index=False, compression="zstd")
            artifacts[str(pooled_dir / path.name)] = _sha256(path)
        runtime_path = staging_dir / "runtime.jsonl"
        _runtime_event(
            runtime_path,
            step="pooled_heldout_aggregation",
            elapsed_seconds=time.perf_counter() - aggregate_started,
            circuit_count=len(data.corridors),
            models_refit=False,
            thresholds_refit=False,
        )
        artifacts[str(pooled_dir / runtime_path.name)] = _sha256(runtime_path)
        staging_manifest_path = staging_dir / "manifest.json"
        _atomic_json(
            staging_manifest_path,
            {
                "schema_version": 1,
                "status": "pooled_heldout_complete",
                "data_content_hash": data.input_content_hash,
                "source_manifest_hash": data.source_manifest_hash,
                "build_manifest_hash": data.build_manifest_hash,
                "config_content_hash": _canonical_json_hash(config),
                "code_content_hash": _code_content_hash(),
                "threshold_freeze_seal": seal_record,
                "fold_manifests": fold_manifests,
                "models_refit": False,
                "thresholds_refit": False,
                "artifacts": artifacts,
            },
        )
        staging_dir.replace(pooled_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    manifest_path = pooled_dir / "manifest.json"
    return HeldoutAggregateResult(
        "pooled_heldout_complete",
        manifest_path,
        tuple(Path(path) for path in artifacts),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the leakage-safe BRACE LOCO experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--frames", type=Path, default=Path("data/processed/deepracing_frames.parquet")
        )
        command.add_argument(
            "--targets", type=Path, default=Path("data/processed/deepracing_targets.parquet")
        )
        command.add_argument(
            "--events", type=Path, default=Path("data/processed/deepracing_events.parquet")
        )
        command.add_argument(
            "--timing",
            type=Path,
            default=Path("data/processed/deepracing_timing_diagnostics.csv"),
        )
        command.add_argument(
            "--splits", type=Path, default=Path("data/processed/deepracing_loco_splits.csv")
        )
        command.add_argument(
            "--source-manifest",
            type=Path,
            default=Path("data/manifests/deepracing-files.csv"),
        )
        command.add_argument(
            "--build-manifest",
            type=Path,
            default=Path("data/manifests/deepracing-build.json"),
        )
        command.add_argument("--base-dir", type=Path, default=None)

    validate = subparsers.add_parser("validate", help="validate inputs without fitting or scoring")
    add_inputs(validate)
    run = subparsers.add_parser(
        "run-fold",
        help="run one independently resumable outer fold (safe for four parallel processes)",
    )
    add_inputs(run)
    run.add_argument("--fold", required=True)
    run.add_argument("--config", type=Path, default=Path("configs/study.json"))
    run.add_argument("--output-dir", type=Path, default=Path("output/experiment"))
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--stage", choices=("thresholds", "heldout"), default="thresholds")
    run.add_argument(
        "--approve-heldout-evaluation",
        action="store_true",
        help="required to unseal and join held-out outcomes after thresholds are frozen",
    )
    seal = subparsers.add_parser(
        "seal-thresholds",
        help="validate and seal all four threshold-complete folds before unsealing",
    )
    add_inputs(seal)
    seal.add_argument("--config", type=Path, default=Path("configs/study.json"))
    seal.add_argument("--output-dir", type=Path, default=Path("output/experiment"))
    aggregate = subparsers.add_parser(
        "aggregate-heldout",
        help="pool four sealed held-out folds without refitting models or thresholds",
    )
    add_inputs(aggregate)
    aggregate.add_argument("--config", type=Path, default=Path("configs/study.json"))
    aggregate.add_argument("--output-dir", type=Path, default=Path("output/experiment"))
    aggregate.add_argument("--destination", type=Path, default=None)
    aggregate.add_argument("--approve-heldout-aggregation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for input validation or one fold-specific resumable run."""

    args = _parser().parse_args(argv)
    cwd = Path.cwd()
    base_dir = args.base_dir or (cwd.parent if cwd.name == "brace-f1-ssac27" else cwd)
    paths = ExperimentPaths(
        frames=args.frames,
        targets=args.targets,
        events=args.events,
        timing=args.timing,
        splits=args.splits,
        source_manifest=args.source_manifest,
        build_manifest=args.build_manifest,
    )
    data = load_experiment_data(
        paths,
        base_dir=base_dir,
    )
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "input_content_hash": data.input_content_hash,
                    "circuits": sorted(data.corridors),
                    "frame_rows": len(data.frames),
                    "validated_only": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if (
        args.command == "run-fold"
        and args.stage == "heldout"
        and not args.approve_heldout_evaluation
    ):
        raise DataValidationError(
            "heldout stage requires --approve-heldout-evaluation after review authorization"
        )
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read study config {args.config}: {exc}") from exc
    if args.command == "seal-thresholds":
        seal_path = create_threshold_freeze_seal(
            data,
            config=config,
            output_dir=args.output_dir,
        )
        print(json.dumps({"status": "thresholds_frozen", "seal": str(seal_path)}))
        return 0
    if args.command == "aggregate-heldout":
        if not args.approve_heldout_aggregation:
            raise DataValidationError("aggregate-heldout requires --approve-heldout-aggregation")
        validate_threshold_freeze_seal(
            data,
            config=config,
            output_dir=args.output_dir,
        )
        aggregate_result = aggregate_completed_heldout_folds(
            data,
            config=config,
            output_dir=args.output_dir,
            destination=args.destination,
        )
        print(
            json.dumps(
                {
                    "status": aggregate_result.status,
                    "manifest": str(aggregate_result.manifest_path),
                    "artifacts": [str(path) for path in aggregate_result.artifact_paths],
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_outer_fold(
        data,
        outer_fold=args.fold,
        config=config,
        output_dir=args.output_dir,
        resume=args.resume,
        evaluate_heldout=args.stage == "heldout",
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "fold": result.fold,
                "status": result.status,
                "manifest": str(result.manifest_path),
                "artifacts": [str(path) for path in result.artifact_paths],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
