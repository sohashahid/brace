"""Deterministic whole-car/session leave-one-circuit-out split primitives."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from brace_f1.io import DataValidationError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UNIT_COLUMNS = [
    "circuit",
    "source_session_id",
    "car_id",
    "source_revision",
    "source_unit_sha256",
]


def _canonical_unit_id(circuit: str, source_session_id: str, car_id: str) -> str:
    return json.dumps(
        [circuit, source_session_id, car_id],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _split_score(
    salt: str,
    seed: int,
    test_circuit: str,
    circuit: str,
    source_session_id: str,
    car_id: str,
) -> str:
    payload = json.dumps(
        [salt, seed, test_circuit, circuit, source_session_id, car_id],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_hash(value: str, name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise DataValidationError(f"{name} must be a lowercase SHA256 digest")


def _unit_table(frames: pd.DataFrame) -> pd.DataFrame:
    values = frames.loc[:, _UNIT_COLUMNS].copy()
    for column in _UNIT_COLUMNS:
        if values[column].isna().any() or (values[column].astype(str).str.len() == 0).any():
            raise DataValidationError(f"split input contains an empty {column}")
        values[column] = values[column].astype("string")
    source_ids = values["source_session_id"].astype(str)
    machine_specific = source_ids.str.startswith(("/", "~")) | source_ids.str.match(
        r"^[A-Za-z]:[\\/]"
    )
    if machine_specific.any():
        raise DataValidationError("source_session_id must not contain an absolute machine path")
    invalid_hash = ~values["source_unit_sha256"].astype(str).str.fullmatch(_SHA256_PATTERN)
    if invalid_hash.any():
        raise DataValidationError("source_unit_sha256 must contain lowercase SHA256 digests")

    identity_columns = ["circuit", "source_session_id", "car_id"]
    consistency = values.groupby(identity_columns, sort=False, dropna=False).agg(
        source_revision_count=("source_revision", "nunique"),
        source_hash_count=("source_unit_sha256", "nunique"),
    )
    if (consistency.to_numpy() != 1).any():
        raise DataValidationError("split-unit source provenance is inconsistent across frames")
    units = (
        values.groupby(identity_columns, sort=True, dropna=False)
        .agg(
            source_revision=("source_revision", "first"),
            source_unit_sha256=("source_unit_sha256", "first"),
            unit_frame_row_count=("car_id", "size"),
        )
        .reset_index()
    )
    units["split_unit_id"] = [
        _canonical_unit_id(str(circuit), str(source_session_id), str(car_id))
        for circuit, source_session_id, car_id in zip(
            units["circuit"],
            units["source_session_id"],
            units["car_id"],
            strict=True,
        )
    ]
    return units


def make_loco_assignments(
    frames: pd.DataFrame,
    *,
    calibration_fraction: float = 0.20,
    seed: int = 2027,
    salt: str,
    source_manifest_sha256: str,
    assignment_code_version: str,
) -> pd.DataFrame:
    """Assign whole source-session/car units to fit, calibration, or circuit test."""

    missing = set(_UNIT_COLUMNS).difference(frames.columns)
    if missing:
        raise DataValidationError(f"split input missing columns: {sorted(missing)}")
    if frames.empty:
        raise DataValidationError("cannot split an empty frame table")
    if not np.isfinite(calibration_fraction) or not 0 <= calibration_fraction < 1:
        raise DataValidationError("calibration_fraction must be in [0, 1)")
    if not salt:
        raise DataValidationError("split salt must be non-empty")
    _validate_hash(source_manifest_sha256, "source_manifest_sha256")
    if not assignment_code_version:
        raise DataValidationError("assignment_code_version must be non-empty")

    units = _unit_table(frames)
    circuits = sorted(units["circuit"].astype(str).unique())
    if len(circuits) < 2:
        raise DataValidationError("leave-one-circuit-out splitting needs at least two circuits")

    folds: list[pd.DataFrame] = []
    for test_circuit in circuits:
        fold = units.copy()
        fold.insert(0, "fold_test_circuit", test_circuit)
        fold["partition"] = "fit"
        test_mask = fold["circuit"] == test_circuit
        fold.loc[test_mask, "partition"] = "test"
        train_indices = fold.index[~test_mask].tolist()
        ranked = sorted(
            train_indices,
            key=lambda index: _split_score(
                salt,
                seed,
                test_circuit,
                str(fold.loc[index, "circuit"]),
                str(fold.loc[index, "source_session_id"]),
                str(fold.loc[index, "car_id"]),
            ),
        )
        calibration_count = int(np.floor(len(ranked) * calibration_fraction + 0.5))
        if calibration_fraction > 0 and len(ranked) > 1:
            calibration_count = min(max(calibration_count, 1), len(ranked) - 1)
        for index in ranked[:calibration_count]:
            fold.loc[index, "partition"] = "calibration"
        fold["split_key"] = [
            _split_score(
                salt,
                seed,
                test_circuit,
                str(circuit),
                str(source_session_id),
                str(car_id),
            )
            for circuit, source_session_id, car_id in zip(
                fold["circuit"],
                fold["source_session_id"],
                fold["car_id"],
                strict=True,
            )
        ]
        fold["seed"] = seed
        fold["calibration_fraction"] = calibration_fraction
        fold["source_manifest_sha256"] = source_manifest_sha256
        fold["split_salt"] = salt
        fold["assignment_code_version"] = assignment_code_version
        folds.append(fold)
    return (
        pd.concat(folds, ignore_index=True)
        .sort_values(
            ["fold_test_circuit", "circuit", "source_session_id", "car_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def attach_loco_partitions(frames: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Expand each frame once per LOCO fold and attach its unit-level partition."""

    merge_columns = _UNIT_COLUMNS
    required_assignments = {"fold_test_circuit", "partition", *merge_columns}
    missing_assignments = required_assignments.difference(assignments.columns)
    if missing_assignments:
        raise DataValidationError(
            f"assignment table missing columns: {sorted(missing_assignments)}"
        )
    missing_frames = set(merge_columns).difference(frames.columns)
    if missing_frames:
        raise DataValidationError(f"frame table missing split columns: {sorted(missing_frames)}")
    duplicate_key = ["fold_test_circuit", "circuit", "source_session_id", "car_id"]
    if assignments.duplicated(duplicate_key).any():
        raise DataValidationError(
            "a source-session/car unit has multiple assignments within a fold"
        )
    attached = frames.merge(
        assignments,
        on=merge_columns,
        how="left",
        validate="many_to_many",
    )
    if attached["partition"].isna().any():
        raise DataValidationError("some frames have no split assignment")
    expected_folds = assignments["fold_test_circuit"].nunique()
    if len(attached) != len(frames) * expected_folds:
        raise DataValidationError("frame-to-fold expansion has missing or duplicate assignments")
    return attached
