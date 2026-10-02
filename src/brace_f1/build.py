"""Command-line cohort builder for the pinned compact DeepRacing release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brace_f1 import __version__
from brace_f1.events import ExcursionConfig, label_excursions
from brace_f1.exposure import compute_interval_exposure
from brace_f1.geometry import CircuitCorridor, transform_world_kinematics
from brace_f1.io import (
    DataValidationError,
    load_compact_car,
    parse_metadata,
    read_ascii_pcd,
    validate_file_manifest,
)
from brace_f1.splits import make_loco_assignments
from brace_f1.support import add_causal_support_columns
from brace_f1.targets import project_raw_events_to_model_grid
from brace_f1.timebase import CausalResample, TimeAxis, causal_resample, reconstruct_time_axis


@dataclass(frozen=True)
class BuildConfig:
    """Frozen cohort-build choices declared before held-out scoring."""

    expected_manifest_rows: int = 244
    resample_frequency_hz: float = 20.0
    calibration_fraction: float = 0.20
    split_seed: int = 2027
    split_salt: str = "brace-ssac27-loco-v1"
    exposure_segment_edge_buffer_seconds: float = 0.50
    exposure_artifact_buffer_seconds: float = 0.25
    excursion: ExcursionConfig = field(default_factory=ExcursionConfig)

    def __post_init__(self) -> None:
        if self.expected_manifest_rows <= 0:
            raise DataValidationError("expected_manifest_rows must be positive")
        if not np.isclose(self.resample_frequency_hz, 20.0):
            raise DataValidationError("the registered derived grid is fixed at 20 Hz")
        if not 0 <= self.calibration_fraction < 1:
            raise DataValidationError("calibration_fraction must be in [0, 1)")
        if not self.split_salt:
            raise DataValidationError("split_salt must be non-empty")
        if self.exposure_segment_edge_buffer_seconds < 0:
            raise DataValidationError("exposure segment-edge buffer must be non-negative")
        if self.exposure_artifact_buffer_seconds < 0:
            raise DataValidationError("exposure artifact buffer must be non-negative")


@dataclass(frozen=True)
class BuildResult:
    """Paths and descriptive counts from a completed cohort build."""

    outputs: tuple[Path, ...]
    build_manifest_path: Path
    counts: dict[str, int | float]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _read_manifest(manifest_path: Path, base_dir: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "dataset",
        "revision",
        "circuit",
        "circuit_directory",
        "car",
        "role",
        "filename",
        "upstream_path",
        "local_path",
        "bytes",
        "sha256",
    }
    if not rows:
        raise DataValidationError("input manifest has no rows")
    missing = required.difference(rows[0])
    if missing:
        raise DataValidationError(f"input manifest missing columns: {sorted(missing)}")
    for row in rows:
        row["resolved_path"] = str(_resolved_path(row["local_path"], base_dir))
    return rows


def _natural_car_key(value: str) -> tuple[str, int | str]:
    prefix, separator, suffix = value.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return value, value


def _single_file(
    rows: list[dict[str, str]], circuit: str, filename: str
) -> Path:
    matches = [
        Path(row["resolved_path"])
        for row in rows
        if row["circuit"] == circuit and row["role"] == "boundary" and row["filename"] == filename
    ]
    if len(matches) != 1:
        raise DataValidationError(
            f"expected one {filename} boundary for {circuit}, found {len(matches)}"
        )
    return matches[0]


def _build_corridors(rows: list[dict[str, str]]) -> dict[str, CircuitCorridor]:
    circuits = sorted({row["circuit"] for row in rows if row["role"] == "compact_car_data"})
    corridors: dict[str, CircuitCorridor] = {}
    for circuit in circuits:
        center_path = _single_file(rows, circuit, "center_line.pcd")
        boundary_a_path = _single_file(rows, circuit, "inner_boundary.pcd")
        boundary_b_path = _single_file(rows, circuit, "outer_boundary.pcd")
        corridors[circuit] = CircuitCorridor.from_point_clouds(
            centerline=read_ascii_pcd(center_path),
            boundary_a=read_ascii_pcd(boundary_a_path),
            boundary_b=read_ascii_pcd(boundary_b_path),
            boundary_a_id=boundary_a_path.name,
            boundary_b_id=boundary_b_path.name,
        )
    return corridors


def _aggregate_source_flags(
    axis: TimeAxis, sampled: CausalResample, source_flags: np.ndarray
) -> np.ndarray:
    """Map instantaneous source flags to the first causal grid point in their segment."""

    output = np.zeros(sampled.target_time_seconds.shape[0], dtype=bool)
    source_segments = axis.segment_id
    for segment in np.unique(sampled.segment_id):
        target_indices = np.flatnonzero(sampled.segment_id == segment)
        source_indices = np.flatnonzero((source_segments == segment) & source_flags)
        if source_indices.size == 0:
            continue
        targets = sampled.target_time_seconds[target_indices]
        source_times = axis.time_seconds[source_indices]
        locations = np.searchsorted(targets, source_times, side="left")
        supported = locations < target_indices.size
        if not np.any(supported):
            continue
        locations = locations[supported]
        source_times = source_times[supported]
        mapped_times = targets[locations]
        if np.any(mapped_times + 1e-12 < source_times):
            raise DataValidationError("source flag was mapped backward in reconstructed time")
        output[target_indices[locations]] = True
    return output


def _source_provenance(rows: list[dict[str, str]]) -> tuple[str, str, str]:
    """Derive a machine-independent session ID and car-input digest from manifest rows."""

    fields = {
        name: {row[name] for row in rows}
        for name in ("dataset", "revision", "circuit_directory")
    }
    inconsistent = {name: values for name, values in fields.items() if len(values) != 1}
    if inconsistent:
        raise DataValidationError(f"source provenance is inconsistent: {inconsistent}")
    dataset = next(iter(fields["dataset"]))
    revision = next(iter(fields["revision"]))
    circuit_directory = next(iter(fields["circuit_directory"]))
    source_session_id = f"{dataset}@{revision}:{circuit_directory}"
    unit_records = sorted(
        (
            row["upstream_path"],
            row["filename"],
            int(row["bytes"]),
            row["sha256"],
        )
        for row in rows
    )
    encoded = json.dumps(unit_records, ensure_ascii=True, separators=(",", ":")).encode()
    source_unit_sha256 = hashlib.sha256(encoded).hexdigest()
    return source_session_id, revision, source_unit_sha256


def _car_rows(
    *,
    circuit: str,
    car_id: str,
    files: dict[str, Path],
    corridor: CircuitCorridor,
    source_session_id: str,
    source_revision: str,
    source_unit_sha256: str,
    config: BuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    required = {"metadata.yaml", "motion_data.npz", "lap_data.npz", "session_data.npz"}
    missing = required.difference(files)
    extra = set(files).difference(required)
    if missing or extra:
        raise DataValidationError(
            f"{circuit}/{car_id} compact file set invalid; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    parents = {path.parent for path in files.values()}
    if len(parents) != 1:
        raise DataValidationError(f"{circuit}/{car_id} compact files do not share a directory")
    car_directory = parents.pop()
    metadata = parse_metadata(files["metadata.yaml"])
    if metadata.trackname != circuit:
        raise DataValidationError(
            f"metadata track mismatch for {circuit}/{car_id}: {metadata.trackname}"
        )
    if not np.isclose(metadata.frequency_hz, config.resample_frequency_hz):
        raise DataValidationError(
            f"metadata grid for {circuit}/{car_id} is {metadata.frequency_hz}, expected 20 Hz"
        )
    compact = load_compact_car(car_directory)
    transformed = transform_world_kinematics(
        compact.motion["positions"],
        compact.motion["velocities"],
        compact.motion["accelerations"],
        compact.motion["quaternions"],
        metadata,
    )
    location = corridor.locate(transformed.positions)
    axis = reconstruct_time_axis(compact.lap, transformed.positions)
    exposure = compute_interval_exposure(
        axis,
        active=np.asarray(compact.lap["result_status"]) == 2,
        pit_status=compact.lap["pit_status"],
        segment_edge_buffer_seconds=config.exposure_segment_edge_buffer_seconds,
        artifact_buffer_seconds=config.exposure_artifact_buffer_seconds,
    )

    speed = np.linalg.norm(transformed.velocities, axis=1)
    raw_count = compact.sample_count
    raw_frame_data: dict[str, Any] = {
        "circuit": np.full(raw_count, circuit),
        "car_id": np.full(raw_count, car_id),
        "frame_index": np.arange(raw_count, dtype=np.int64),
        "time_seconds": axis.time_seconds,
        "map_x_m": transformed.positions[:, 0],
        "map_y_m": transformed.positions[:, 1],
        "map_z_m": transformed.positions[:, 2],
        "velocity_x_mps": transformed.velocities[:, 0],
        "velocity_y_mps": transformed.velocities[:, 1],
        "velocity_z_mps": transformed.velocities[:, 2],
        "acceleration_x_mps2": transformed.accelerations[:, 0],
        "acceleration_y_mps2": transformed.accelerations[:, 1],
        "acceleration_z_mps2": transformed.accelerations[:, 2],
        "body_acceleration_longitudinal_mps2": transformed.body_accelerations[:, 0],
        "body_acceleration_lateral_mps2": transformed.body_accelerations[:, 1],
        "body_acceleration_vertical_mps2": transformed.body_accelerations[:, 2],
        "quaternion_x": np.asarray(compact.motion["quaternions"])[:, 0],
        "quaternion_y": np.asarray(compact.motion["quaternions"])[:, 1],
        "quaternion_z": np.asarray(compact.motion["quaternions"])[:, 2],
        "quaternion_w": np.asarray(compact.motion["quaternions"])[:, 3],
        "forward_x_map": transformed.forward_vectors[:, 0],
        "forward_y_map": transformed.forward_vectors[:, 1],
        "forward_z_map": transformed.forward_vectors[:, 2],
        "yaw_map_rad": transformed.yaw_map,
        "speed_mps": speed,
        "lap_number": np.asarray(compact.lap["lap_numbers"]),
        "lap_distance_m": np.asarray(compact.lap["lap_distances"]),
        "total_distance_m": np.asarray(compact.lap["total_distances"]),
        "result_status": np.asarray(compact.lap["result_status"]),
        "driver_status": np.asarray(compact.lap["driver_status"]),
        "pit_status": np.asarray(compact.lap["pit_status"]),
        "pit_lane_timer_active": np.asarray(compact.lap["pit_lane_timer_active"]),
        "in_corridor": location.in_corridor,
        "nearest_boundary_id": location.nearest_boundary_id,
        "nearest_boundary_role": location.nearest_boundary_role,
        "clearance_m": location.clearance_m,
        "outside_depth_m": location.outside_depth_m,
        "local_side": location.local_side,
        "centerline_segment_index_pcd": location.centerline_segment,
        "centerline_arclength_wrapped_m": location.centerline_arclength_wrapped_m,
        "track_length_m": np.full(raw_count, corridor.track_length_m),
        "hard_break": axis.hard_break,
        "artifact_seed": axis.artifact_proximity_seed,
    }
    raw_frame_data["active"] = np.asarray(raw_frame_data["result_status"]) == 2
    raw_labeled = label_excursions(pd.DataFrame(raw_frame_data), config.excursion)

    source_columns = [
        "map_x_m",
        "map_y_m",
        "map_z_m",
        "velocity_x_mps",
        "velocity_y_mps",
        "velocity_z_mps",
        "acceleration_x_mps2",
        "acceleration_y_mps2",
        "acceleration_z_mps2",
        "body_acceleration_longitudinal_mps2",
        "body_acceleration_lateral_mps2",
        "body_acceleration_vertical_mps2",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
        "forward_x_map",
        "forward_y_map",
        "forward_z_map",
        "yaw_map_rad",
        "speed_mps",
        "lap_number",
        "lap_distance_m",
        "total_distance_m",
        "result_status",
        "driver_status",
        "pit_status",
        "pit_lane_timer_active",
        "in_corridor",
        "nearest_boundary_id",
        "nearest_boundary_role",
        "clearance_m",
        "outside_depth_m",
        "local_side",
        "centerline_segment_index_pcd",
        "centerline_arclength_wrapped_m",
        "track_length_m",
        "segment_bin_25m",
        "raw_outside",
        "artifact_excluded",
        "eligible_exposure",
        "candidate_event_id",
        "qualifying_event_id",
    ]
    source_arrays = {
        column: raw_labeled.frames[column].to_numpy() for column in source_columns
    }
    sampled = causal_resample(
        axis,
        source_arrays,
        frequency_hz=config.resample_frequency_hz,
    )
    sample_count = sampled.target_time_seconds.shape[0]
    frame_data: dict[str, Any] = {
        "circuit": np.full(sample_count, circuit),
        "car_id": np.full(sample_count, car_id),
        "source_session_id": np.full(sample_count, source_session_id),
        "source_revision": np.full(sample_count, source_revision),
        "source_unit_sha256": np.full(sample_count, source_unit_sha256),
        "frame_index": np.arange(sample_count, dtype=np.int64),
        "source_frame_index": sampled.source_indices,
        "time_seconds": sampled.target_time_seconds,
        "source_age_seconds": sampled.source_age_seconds,
        "timestamp_method": np.full(sample_count, axis.diagnostics.timestamp_method),
        "native_frame_timestamps_available": np.full(sample_count, False),
        **sampled.arrays,
    }
    frame_data["raw_outside_at_source"] = frame_data.pop("raw_outside")
    frame_data["offline_artifact_excluded"] = frame_data.pop("artifact_excluded")
    frame_data["offline_eligible_grid"] = frame_data.pop("eligible_exposure")
    frame_data["candidate_event_id_at_source"] = frame_data.pop("candidate_event_id")
    frame_data["qualifying_event_id_at_source"] = frame_data.pop("qualifying_event_id")
    frame_data["active"] = np.asarray(frame_data["result_status"]) == 2
    hard_break = np.zeros(sample_count, dtype=bool)
    hard_break[0] = True
    hard_break[1:] = sampled.segment_id[1:] != sampled.segment_id[:-1]
    frame_data["hard_break"] = hard_break
    frame_data["lap_boundary"] = _aggregate_source_flags(axis, sampled, axis.lap_boundary)
    frame_data["clock_reset"] = _aggregate_source_flags(axis, sampled, axis.clock_reset)
    frame_data["data_gap"] = _aggregate_source_flags(axis, sampled, axis.data_gap)
    frame_data["offline_artifact_seed"] = _aggregate_source_flags(
        axis, sampled, axis.artifact_proximity_seed
    )
    frames = pd.DataFrame(frame_data)
    velocities = frames[["velocity_x_mps", "velocity_y_mps", "velocity_z_mps"]].to_numpy()
    derived_acceleration = np.full_like(velocities, np.nan, dtype=float)
    dt = np.diff(frames["time_seconds"].to_numpy(dtype=float))
    same_segment = np.diff(sampled.segment_id) == 0
    valid_difference = same_segment & (dt > 0)
    derived_acceleration[1:][valid_difference] = (
        np.diff(velocities, axis=0)[valid_difference] / dt[valid_difference, None]
    )
    frames["derived_acceleration_x_mps2"] = derived_acceleration[:, 0]
    frames["derived_acceleration_y_mps2"] = derived_acceleration[:, 1]
    frames["derived_acceleration_z_mps2"] = derived_acceleration[:, 2]
    frames["continuous_segment_id"] = sampled.segment_id
    frames = add_causal_support_columns(
        frames,
        mandatory_input_columns=(
            "map_x_m",
            "map_y_m",
            "velocity_x_mps",
            "velocity_y_mps",
            "acceleration_x_mps2",
            "acceleration_y_mps2",
            "yaw_map_rad",
            "clearance_m",
        ),
    )
    projected = project_raw_events_to_model_grid(frames, raw_labeled.events)
    target_columns = {
        "offline_artifact_excluded",
        "offline_eligible_grid",
        "raw_outside_at_source",
        "candidate_event_id_at_source",
        "qualifying_event_id_at_source",
        "offline_artifact_seed",
        "continuous_segment_end_time_seconds",
        "continuous_segment_remaining_seconds",
        "qualifying_event_onset_projected",
        "projected_onset_event_id",
    }
    target_columns.update(
        column
        for column in projected.frames.columns
        if column.startswith(("next_", "time_to_next_", "outcome_evaluable_"))
    )
    target_keys = ["circuit", "car_id", "frame_index", "time_seconds"]
    targets = projected.frames[
        target_keys + sorted(target_columns.difference(target_keys))
    ].copy()
    causal_frames = projected.frames.drop(columns=sorted(target_columns))
    key_columns = ["circuit", "car_id", "frame_index"]
    if (
        causal_frames.duplicated(key_columns).any()
        or targets.duplicated(key_columns).any()
        or not causal_frames[key_columns].equals(targets[key_columns])
    ):
        raise DataValidationError("causal-frame and target keys are not one-to-one and ordered")
    forbidden_prefixes = (
        "next_",
        "time_to_next_",
        "qualifying_event_",
        "projected_onset_",
        "outcome_evaluable_",
    )
    forbidden_names = {
        "candidate_event_id_at_source",
        "offline_eligible_grid",
        "offline_artifact_excluded",
        "offline_artifact_seed",
        "raw_outside_at_source",
        "continuous_segment_end_time_seconds",
        "continuous_segment_remaining_seconds",
    }
    leaked = {
        column
        for column in causal_frames.columns
        if column in forbidden_names or column.startswith(forbidden_prefixes)
    }
    if leaked:
        raise DataValidationError(
            f"future/offline fields leaked into causal frames: {sorted(leaked)}"
        )
    timing = {
        "circuit": circuit,
        "car_id": car_id,
        "source_session_id": source_session_id,
        "source_revision": source_revision,
        "source_unit_sha256": source_unit_sha256,
        "raw_frame_count": compact.sample_count,
        "resampled_frame_count": sample_count,
        "prebuffer_exposure_seconds": exposure.prebuffer_seconds,
        "buffered_exposure_seconds": exposure.buffered_seconds,
        "prebuffer_exposure_interval_count": exposure.prebuffer_interval_count,
        "buffered_exposure_interval_count": exposure.buffered_interval_count,
        **asdict(axis.diagnostics),
    }
    return causal_frames, targets, projected.events, timing, compact.sample_count


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.9g")


def _output_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _reproducibility_record(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _code_records() -> list[dict[str, Any]]:
    """Hash the declared code, test, configuration, and environment lock set."""

    project_root = Path(__file__).resolve().parents[2]
    required = [
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "configs" / "study.json",
    ]
    missing = [path.relative_to(project_root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise DataValidationError(f"reproducibility files are missing: {missing}")
    discovered = [
        *project_root.glob("src/brace_f1/**/*.py"),
        *project_root.glob("tests/**/*.py"),
        *project_root.glob("configs/**/*"),
    ]
    paths = sorted({*required, *(path for path in discovered if path.is_file())})
    return [_reproducibility_record(path, project_root) for path in paths]


def _runtime_environment() -> dict[str, Any]:
    """Record the interpreter and installed distributions used for this build."""

    project_root = Path(__file__).resolve().parents[2]
    lock_path = project_root / "uv.lock"
    package_versions = {
        str(distribution.metadata["Name"]).lower().replace("_", "-"): distribution.version
        for distribution in distributions()
        if distribution.metadata["Name"]
    }
    return {
        "execution_contract": "uv run --locked python -m brace_f1.build",
        "lock_environment_active": Path(sys.prefix).resolve()
        == (project_root / ".venv").resolve(),
        "lockfile": _reproducibility_record(lock_path, project_root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": platform.platform(),
        "packages": dict(sorted(package_versions.items())),
    }


def _travel_direction_validation(
    frames: pd.DataFrame,
    corridor: CircuitCorridor,
) -> dict[str, Any]:
    """Check source centerline point order against observed vehicle headings."""

    segment_index = frames["centerline_segment_index_pcd"].to_numpy(dtype=np.int64)
    forward = frames[["forward_x_map", "forward_y_map"]].to_numpy(dtype=np.float64)
    speed = frames["speed_mps"].to_numpy(dtype=np.float64)
    segments = np.roll(corridor.centerline, -1, axis=0) - corridor.centerline
    segment_norm = np.linalg.norm(segments, axis=1)
    forward_norm = np.linalg.norm(forward, axis=1)
    valid = (
        np.isfinite(forward).all(axis=1)
        & np.isfinite(speed)
        & (speed > 10.0)
        & (forward_norm > 0.0)
        & (segment_index >= 0)
        & (segment_index < len(segments))
        & (segment_norm[np.clip(segment_index, 0, len(segments) - 1)] > 0.0)
    )
    count = int(valid.sum())
    result: dict[str, Any] = {
        "travel_direction_validation_basis": (
            "dot(vehicle_forward_xy, source_point_order_centerline_tangent) "
            "for finite frames with speed_mps > 10"
        ),
        "travel_direction_validation_count": count,
        "travel_direction_validation_status": "insufficient_speed_gt_10_observations",
        "travel_direction_same_as_source_order": None,
        "travel_direction_dot_median": None,
        "travel_direction_dot_p01": None,
        "travel_direction_dot_fraction_positive": None,
    }
    if count == 0:
        return result
    valid_index = segment_index[valid]
    tangent = segments[valid_index] / segment_norm[valid_index, None]
    unit_forward = forward[valid] / forward_norm[valid, None]
    alignment = np.einsum("ij,ij->i", unit_forward, tangent)
    fraction_positive = float(np.mean(alignment > 0.0))
    same_direction = bool(float(np.median(alignment)) > 0.0 and fraction_positive >= 0.95)
    result.update(
        {
            "travel_direction_validation_status": (
                "validated_same_as_observed_travel"
                if same_direction
                else "source_order_not_validated_as_observed_travel"
            ),
            "travel_direction_same_as_source_order": same_direction,
            "travel_direction_dot_median": float(np.median(alignment)),
            "travel_direction_dot_p01": float(np.quantile(alignment, 0.01)),
            "travel_direction_dot_fraction_positive": fraction_positive,
        }
    )
    return result


def _map_manifest(
    corridors: dict[str, CircuitCorridor],
    frames: pd.DataFrame,
) -> dict[str, Any]:
    circuits: list[dict[str, Any]] = []
    for circuit, corridor in sorted(corridors.items()):
        length = corridor.track_length_m
        edges = list(np.arange(0.0, length, 25.0, dtype=float))
        if not edges or not np.isclose(edges[-1], length):
            edges.append(length)
        circuit_record = {
                "circuit": circuit,
                "track_length_m": length,
                "centerline_segment_count_pcd": int(corridor.centerline.shape[0]),
                "segment_bin_count_25m": len(edges) - 1,
                "segment_bin_edges_m": edges,
                "geometric_outer_loop_source": corridor.outer_loop.source_id,
                "geometric_inner_hole_source": corridor.inner_hole.source_id,
        }
        circuit_record.update(
            _travel_direction_validation(
                frames.loc[frames["circuit"] == circuit],
                corridor,
            )
        )
        circuits.append(circuit_record)
    return {
        "schema_version": 2,
        "arclength_origin": "first point in source center_line.pcd",
        "arclength_direction": "source_center_line_pcd_point_order",
        "arclength_direction_status": "frozen_from_source_map",
        "arclength_wrap": True,
        "segment_bin_origin_m": 0.0,
        "segment_bin_width_m": 25.0,
        "segment_bin_rule": "floor(wrapped_centerline_arclength_m / 25.0)",
        "pcd_segment_index_is_not_segment_bin_25m": True,
        "circuits": circuits,
    }


def _kinematic_diagnostics(frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for circuit, group in frames.groupby("circuit", sort=True):
        velocity = group[["velocity_x_mps", "velocity_y_mps"]].to_numpy(dtype=float)
        forward = group[["forward_x_map", "forward_y_map"]].to_numpy(dtype=float)
        velocity_norm = np.linalg.norm(velocity, axis=1)
        forward_norm = np.linalg.norm(forward, axis=1)
        alignment_mask = (
            np.isfinite(velocity).all(axis=1)
            & np.isfinite(forward).all(axis=1)
            & (velocity_norm > 10.0)
            & (forward_norm > 0)
        )
        alignment = np.asarray([], dtype=float)
        if np.any(alignment_mask):
            alignment = np.einsum(
                "ij,ij->i", velocity[alignment_mask], forward[alignment_mask]
            ) / (velocity_norm[alignment_mask] * forward_norm[alignment_mask])

        acceleration = group[
            ["acceleration_x_mps2", "acceleration_y_mps2", "acceleration_z_mps2"]
        ].to_numpy(dtype=float)
        derived = group[
            [
                "derived_acceleration_x_mps2",
                "derived_acceleration_y_mps2",
                "derived_acceleration_z_mps2",
            ]
        ].to_numpy(dtype=float)
        finite_acceleration = np.isfinite(acceleration).all(axis=1) & np.isfinite(derived).all(
            axis=1
        )
        error = np.linalg.norm(
            acceleration[finite_acceleration] - derived[finite_acceleration], axis=1
        )
        acceleration_norm = np.linalg.norm(acceleration[finite_acceleration], axis=1)
        derived_norm = np.linalg.norm(derived[finite_acceleration], axis=1)
        cosine_mask = (acceleration_norm > 0.5) & (derived_norm > 0.5)
        acceleration_cosine = np.asarray([], dtype=float)
        if np.any(cosine_mask):
            acceleration_cosine = np.einsum(
                "ij,ij->i",
                acceleration[finite_acceleration][cosine_mask],
                derived[finite_acceleration][cosine_mask],
            ) / (acceleration_norm[cosine_mask] * derived_norm[cosine_mask])
        rows.append(
            {
                "circuit": circuit,
                "forward_velocity_alignment_count_speed_gt_10": int(alignment.size),
                "forward_velocity_alignment_median": float(np.median(alignment))
                if alignment.size
                else np.nan,
                "forward_velocity_alignment_p01": float(np.quantile(alignment, 0.01))
                if alignment.size
                else np.nan,
                "map_acceleration_fd_count": int(error.size),
                "map_acceleration_fd_error_median_mps2": float(np.median(error))
                if error.size
                else np.nan,
                "map_acceleration_fd_error_p95_mps2": float(np.quantile(error, 0.95))
                if error.size
                else np.nan,
                "map_acceleration_fd_cosine_count": int(acceleration_cosine.size),
                "map_acceleration_fd_cosine_median": float(np.median(acceleration_cosine))
                if acceleration_cosine.size
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_deepracing_cohort(
    manifest_path: Path,
    *,
    output_dir: Path,
    build_manifest_path: Path,
    base_dir: Path,
    config: BuildConfig | None = None,
) -> BuildResult:
    """Validate the pinned cohort, derive frame/event tables, and record provenance."""

    config = BuildConfig() if config is None else config
    manifest_path = manifest_path.resolve()
    base_dir = base_dir.resolve()
    validation = validate_file_manifest(
        manifest_path,
        expected_rows=config.expected_manifest_rows,
        base_dir=base_dir,
    )
    rows = _read_manifest(manifest_path, base_dir)
    corridors = _build_corridors(rows)

    groups: dict[tuple[str, str], dict[str, Path]] = {}
    group_manifest_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["role"] != "compact_car_data":
            continue
        key = (row["circuit"], row["car"])
        group_manifest_rows.setdefault(key, []).append(row)
        files = groups.setdefault(key, {})
        if row["filename"] in files:
            raise DataValidationError(f"duplicate {row['filename']} for {key}")
        files[row["filename"]] = Path(row["resolved_path"])

    frame_tables: list[pd.DataFrame] = []
    target_tables: list[pd.DataFrame] = []
    event_tables: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []
    raw_frame_count = 0
    for circuit, car_id in sorted(groups, key=lambda key: (key[0], _natural_car_key(key[1]))):
        source_session_id, source_revision, source_unit_sha256 = _source_provenance(
            group_manifest_rows[(circuit, car_id)]
        )
        frames, targets, events, timing, raw_count = _car_rows(
            circuit=circuit,
            car_id=car_id,
            files=groups[(circuit, car_id)],
            corridor=corridors[circuit],
            source_session_id=source_session_id,
            source_revision=source_revision,
            source_unit_sha256=source_unit_sha256,
            config=config,
        )
        frame_tables.append(frames)
        target_tables.append(targets)
        event_tables.append(events)
        timing_rows.append(timing)
        raw_frame_count += raw_count

    frames = pd.concat(frame_tables, ignore_index=True)
    targets = pd.concat(target_tables, ignore_index=True)
    events = pd.concat(event_tables, ignore_index=True)
    timing = pd.DataFrame(timing_rows)
    kinematic_diagnostics = _kinematic_diagnostics(frames)
    map_manifest_payload = _map_manifest(corridors, frames)
    source_manifest_sha256 = _sha256(manifest_path)
    splits = make_loco_assignments(
        frames,
        calibration_fraction=config.calibration_fraction,
        seed=config.split_seed,
        salt=config.split_salt,
        source_manifest_sha256=source_manifest_sha256,
        assignment_code_version=__version__,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = output_dir / "deepracing_frames.parquet"
    targets_path = output_dir / "deepracing_targets.parquet"
    events_csv_path = output_dir / "deepracing_events.csv"
    events_parquet_path = output_dir / "deepracing_events.parquet"
    timing_path = output_dir / "deepracing_timing_diagnostics.csv"
    splits_path = output_dir / "deepracing_loco_splits.csv"
    map_manifest_path = output_dir / "deepracing_map_manifest.json"
    kinematic_diagnostics_path = output_dir / "deepracing_kinematic_diagnostics.csv"
    frames.to_parquet(frames_path, index=False, compression="zstd")
    targets.to_parquet(targets_path, index=False, compression="zstd")
    _write_csv(events, events_csv_path)
    events.to_parquet(events_parquet_path, index=False, compression="zstd")
    _write_csv(timing, timing_path)
    _write_csv(splits, splits_path)
    map_manifest_path.write_text(
        json.dumps(map_manifest_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(kinematic_diagnostics, kinematic_diagnostics_path)
    output_paths = (
        frames_path,
        targets_path,
        events_csv_path,
        events_parquet_path,
        timing_path,
        splits_path,
        map_manifest_path,
        kinematic_diagnostics_path,
    )

    qualified = events["qualified"].astype(bool) if not events.empty else pd.Series(dtype=bool)
    counts: dict[str, int | float] = {
        "input_files": validation.validated_files,
        "input_bytes": validation.total_bytes,
        "circuits": len(corridors),
        "cars": len(groups),
        "raw_frames": raw_frame_count,
        "resampled_frames": len(frames),
        "candidate_excursions": len(events),
        "qualified_excursions": int(qualified.sum()),
        "prebuffer_exposure_seconds": float(timing["prebuffer_exposure_seconds"].sum()),
        "buffered_exposure_seconds": float(timing["buffered_exposure_seconds"].sum()),
        "prebuffer_exposure_hours": float(
            timing["prebuffer_exposure_seconds"].sum() / 3600.0
        ),
        "buffered_exposure_hours": float(
            timing["buffered_exposure_seconds"].sum() / 3600.0
        ),
        "offline_grid_eligible_seconds": float(
            targets["offline_eligible_grid"].sum() / config.resample_frequency_hz
        ),
        "offline_grid_eligible_in_corridor_seconds": float(
            (
                targets["offline_eligible_grid"]
                & ~targets["raw_outside_at_source"]
            ).sum()
            / config.resample_frequency_hz
        ),
        "causal_input_valid_grid_seconds": float(
            frames["input_valid_causal"].sum() / config.resample_frequency_hz
        ),
    }
    per_circuit = []
    for circuit, circuit_frames in frames.groupby("circuit", sort=True):
        circuit_targets = targets.loc[targets["circuit"] == circuit]
        circuit_events = events.loc[events["circuit"] == circuit]
        per_circuit.append(
            {
                "circuit": circuit,
                "cars": int(circuit_frames["car_id"].nunique()),
                "resampled_frames": int(len(circuit_frames)),
                "offline_grid_eligible_seconds": float(
                    circuit_targets["offline_eligible_grid"].sum()
                    / config.resample_frequency_hz
                ),
                "causal_input_valid_grid_seconds": float(
                    circuit_frames["input_valid_causal"].sum()
                    / config.resample_frequency_hz
                ),
                "prebuffer_exposure_seconds": float(
                    timing.loc[
                        timing["circuit"] == circuit, "prebuffer_exposure_seconds"
                    ].sum()
                ),
                "buffered_exposure_seconds": float(
                    timing.loc[
                        timing["circuit"] == circuit, "buffered_exposure_seconds"
                    ].sum()
                ),
                "candidate_excursions": int(len(circuit_events)),
                "qualified_excursions": int(
                    circuit_events["qualified"].astype(bool).sum()
                    if not circuit_events.empty
                    else 0
                ),
            }
        )

    input_records = [
        {
            "path": str(Path(row["resolved_path"]).resolve()),
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "circuit": row["circuit"],
            "car": row["car"],
            "role": row["role"],
            "filename": row["filename"],
        }
        for row in rows
    ]
    config_payload = asdict(config)
    config_payload["event_definition_status"] = "frozen_predeclared"
    config_payload["event_label_stream"] = "raw_reconstructed_before_resampling"
    config_payload["event_gap_rule"] = "intervening_in_corridor_duration"
    config_payload["exposure_buffer_rule"] = "symmetric_interval_overlap"
    config_payload["exposure_artifact_seed_rule"] = "lap_boundary_or_any_hard_break"
    payload = {
        "schema_version": 1,
        "builder": {"name": "brace", "version": __version__},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_environment": _runtime_environment(),
        "input_manifest": _output_record(manifest_path),
        "inputs": input_records,
        "source_code": _code_records(),
        "config": config_payload,
        "counts": counts,
        "per_circuit_counts": per_circuit,
        "timing_diagnostics": timing.to_dict(orient="records"),
        "outputs": [_output_record(path) for path in output_paths],
        "warnings": [
            "These labels are mapped car-center track-boundary excursions, not crashes or impacts.",
            "The compact exports lack native per-frame timestamps; time is reconstructed "
            "from lap clocks and resampled causally to 20 Hz.",
            "The estimated source cadence must not be described as an exact 100 Hz "
            "measurement clock.",
            "The PCD loops represent mapped track edges, not FIA barrier geometry.",
            "The numerical source is a commercial racing-game simulation domain, not live "
            "Formula 1 telemetry.",
            "Event thresholds are frozen predeclared values and were not selected to maximize "
            "these cohort counts.",
            "Symmetric artifact buffers are offline label/exposure rules only; "
            "input_valid_causal does not use future information.",
            "Outcome-evaluable horizon flags identify resolution windows that remain "
            "within continuous observed support.",
            "The production exposure denominator uses symmetric interval overlap and "
            "supersedes the exploratory sample-mask audit denominator.",
        ],
    }
    build_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    build_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BuildResult(
        outputs=output_paths,
        build_manifest_path=build_manifest_path,
        counts=counts,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and build the BRACE DeepRacing mapped-excursion cohort."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/deepracing-files.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=Path("data/manifests/deepracing-build.json"),
    )
    parser.add_argument("--base-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints only paths and descriptive build counts."""

    args = _parser().parse_args(argv)
    cwd = Path.cwd()
    base_dir = args.base_dir or (cwd.parent if cwd.name == "brace-f1-ssac27" else cwd)
    result = build_deepracing_cohort(
        args.manifest,
        output_dir=args.output_dir,
        build_manifest_path=args.build_manifest,
        base_dir=base_dir,
    )
    print(json.dumps({"build_manifest": str(result.build_manifest_path), "counts": result.counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
