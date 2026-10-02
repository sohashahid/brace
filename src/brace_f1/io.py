"""Strict readers and provenance checks for the compact DeepRacing release."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class DataValidationError(ValueError):
    """Raised when an input violates the declared public-study data contract."""


MOTION_KEYS = frozenset(
    {
        "session_times",
        "frame_identifiers",
        "overall_frame_identifiers",
        "positions",
        "quaternions",
        "velocities",
        "accelerations",
    }
)
LAP_KEYS = frozenset(
    {
        "session_times",
        "frame_identifiers",
        "overall_frame_identifiers",
        "lap_distances",
        "total_distances",
        "last_lap_times",
        "current_lap_times",
        "lap_numbers",
        "result_status",
        "driver_status",
        "pit_status",
        "pit_lane_timer_active",
    }
)
SESSION_KEYS = frozenset(
    {
        "session_times",
        "frame_identifiers",
        "overall_frame_identifiers",
        "track_ids",
        "game_modes",
        "session_types",
        "game_paused",
    }
)
_MOTION_VECTOR_WIDTHS = {
    "positions": 3,
    "quaternions": 4,
    "velocities": 3,
    "accelerations": 3,
}
_LAP_VECTOR_KEYS = LAP_KEYS - {
    "session_times",
    "frame_identifiers",
    "overall_frame_identifiers",
}


@dataclass(frozen=True)
class CompactCarData:
    """Validated arrays from one compact per-car export."""

    motion: Mapping[str, NDArray[np.generic]]
    lap: Mapping[str, NDArray[np.generic]]
    session: Mapping[str, NDArray[np.generic]]
    sample_count: int
    native_frame_timestamps_available: bool = False


@dataclass(frozen=True)
class ManifestValidation:
    """Integrity summary for a validated file manifest."""

    validated_files: int
    total_bytes: int


@dataclass(frozen=True)
class DeepRacingMetadata:
    """Only metadata fields used by the cohort builder."""

    trackname: str
    clockwise: bool
    frequency_hz: float
    timestep_seconds: float
    origin: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PointCloud:
    """A validated, unorganized ASCII PCD v0.7 cloud."""

    fields: tuple[str, ...]
    values: NDArray[np.float64]

    @property
    def xyz(self) -> NDArray[np.float64]:
        indices = [self.fields.index(name) for name in ("x", "y", "z")]
        return self.values[:, indices]


def _load_npz(path: Path, required: frozenset[str]) -> dict[str, NDArray[np.generic]]:
    if not path.is_file():
        raise DataValidationError(f"missing compact file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required.difference(archive.files))
            if missing:
                raise DataValidationError(f"{path.name} missing required keys: {missing}")
            arrays: dict[str, NDArray[np.generic]] = {}
            for key in archive.files:
                try:
                    array = np.asarray(archive[key])
                except ValueError as exc:
                    if "Object arrays" in str(exc) or "allow_pickle=False" in str(exc):
                        raise DataValidationError(
                            f"{path.name}:{key} is an object array; pickle loading is forbidden"
                        ) from exc
                    raise
                if array.dtype.hasobject:
                    raise DataValidationError(
                        f"{path.name}:{key} is an object array; pickle loading is forbidden"
                    )
                arrays[key] = array
    except (OSError, ValueError) as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(f"cannot read {path}: {exc}") from exc
    return arrays


def load_compact_car(car_directory: Path) -> CompactCarData:
    """Load and validate one car's compact numerical release without pickle support."""

    motion = _load_npz(car_directory / "motion_data.npz", MOTION_KEYS)
    lap = _load_npz(car_directory / "lap_data.npz", LAP_KEYS)
    session = _load_npz(car_directory / "session_data.npz", SESSION_KEYS)

    lengths: dict[str, int] = {}
    for key, width in _MOTION_VECTOR_WIDTHS.items():
        array = motion[key]
        if array.ndim != 2 or array.shape[1] != width:
            raise DataValidationError(
                f"motion_data.npz:{key} must have shape (N, {width}), got {array.shape}"
            )
        lengths[f"motion.{key}"] = int(array.shape[0])
        if not np.isfinite(array).all():
            raise DataValidationError(f"non-finite values in mandatory motion array {key}")

    for key in sorted(_LAP_VECTOR_KEYS):
        array = lap[key]
        if array.ndim != 1:
            raise DataValidationError(f"lap_data.npz:{key} must be one-dimensional")
        lengths[f"lap.{key}"] = int(array.shape[0])

    if len(set(lengths.values())) != 1:
        raise DataValidationError(f"motion/lap length mismatch: {lengths}")
    sample_count = next(iter(lengths.values()))
    if sample_count == 0:
        raise DataValidationError("compact car export has zero motion samples")

    session_lengths: dict[str, int] = {}
    for key in sorted(SESSION_KEYS):
        array = session[key]
        if array.ndim != 1:
            raise DataValidationError(f"session_data.npz:{key} must be one-dimensional")
        session_lengths[key] = int(array.shape[0])
    if len(set(session_lengths.values())) != 1 or next(iter(session_lengths.values())) == 0:
        raise DataValidationError(f"session_data length mismatch: {session_lengths}")

    return CompactCarData(
        motion=motion,
        lap=lap,
        session=session,
        sample_count=sample_count,
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    direct = base_dir / path
    if direct.is_file():
        return direct
    return path


def validate_file_manifest(
    manifest_path: Path,
    *,
    expected_rows: int | None = None,
    base_dir: Path | None = None,
) -> ManifestValidation:
    """Validate every declared local file against byte count and SHA-256."""

    base = Path.cwd() if base_dir is None else base_dir
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"local_path", "bytes", "sha256"}
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise DataValidationError(f"manifest missing columns: {sorted(missing_columns)}")
        rows = list(reader)

    if expected_rows is not None and len(rows) != expected_rows:
        raise DataValidationError(
            f"manifest row count mismatch: expected {expected_rows}, found {len(rows)}"
        )

    seen: set[Path] = set()
    total_bytes = 0
    for row_number, row in enumerate(rows, start=2):
        path = _resolve_manifest_path(row["local_path"], base)
        if path in seen:
            raise DataValidationError(f"duplicate manifest path at row {row_number}: {path}")
        seen.add(path)
        if not path.is_file():
            raise DataValidationError(f"manifest file missing at row {row_number}: {path}")
        try:
            expected_size = int(row["bytes"])
        except ValueError as exc:
            raise DataValidationError(f"invalid byte count at row {row_number}") from exc
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise DataValidationError(
                f"size mismatch for {path}: expected {expected_size}, found {actual_size}"
            )
        expected_hash = row["sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise DataValidationError(f"invalid SHA-256 at row {row_number}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise DataValidationError(
                f"SHA-256 mismatch for {path}: expected {expected_hash}, found {actual_hash}"
            )
        total_bytes += actual_size
    return ManifestValidation(validated_files=len(rows), total_bytes=total_bytes)


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _metadata_scalar(lines: list[str], key: str) -> str:
    prefix = f"{key}:"
    matches = [line[len(prefix) :].strip() for line in lines if line.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise DataValidationError(f"metadata must contain exactly one scalar {key}")
    return matches[0]


def _metadata_numeric_list(lines: list[str], key: str, length: int) -> tuple[float, ...]:
    marker = f"{key}:"
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise DataValidationError(f"metadata missing {key}") from exc
    values: list[float] = []
    for line in lines[start:]:
        match = re.fullmatch(rf"-\s*({_FLOAT_PATTERN})", line)
        if not match:
            break
        values.append(float(match.group(1)))
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise DataValidationError(f"metadata {key} must contain {length} finite numbers")
    return tuple(values)


def parse_metadata(path: Path) -> DeepRacingMetadata:
    """Parse the fixed DeepRacing metadata subset without a general YAML evaluator."""

    text = path.read_text(encoding="utf-8")
    if re.search(r"(?:!!|!<|(^|\s)[&*][A-Za-z_])", text, flags=re.MULTILINE):
        raise DataValidationError("unsafe YAML token in metadata")
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    trackname = _metadata_scalar(lines, "trackname")
    if not re.fullmatch(r"[A-Za-z0-9 _.-]+", trackname):
        raise DataValidationError("metadata trackname contains unsupported characters")
    clockwise_raw = _metadata_scalar(lines, "clockwise")
    if clockwise_raw not in {"true", "false"}:
        raise DataValidationError("metadata clockwise must be true or false")
    try:
        frequency_hz = float(_metadata_scalar(lines, "frequency_history"))
        timestep_seconds = float(_metadata_scalar(lines, "timestep_history"))
    except ValueError as exc:
        raise DataValidationError("metadata frequency/timestep must be numeric") from exc
    if not (math.isfinite(frequency_hz) and frequency_hz > 0):
        raise DataValidationError("metadata frequency_history must be positive and finite")
    if not (math.isfinite(timestep_seconds) and timestep_seconds > 0):
        raise DataValidationError("metadata timestep_history must be positive and finite")
    if not math.isclose(frequency_hz * timestep_seconds, 1.0, rel_tol=1e-5, abs_tol=1e-8):
        raise DataValidationError("metadata frequency and timestep are inconsistent")
    origin = _metadata_numeric_list(lines, "starting_pose_origin", 3)
    quaternion = _metadata_numeric_list(lines, "starting_pose_quaternion", 4)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise DataValidationError("metadata quaternion must have unit norm")
    return DeepRacingMetadata(
        trackname=trackname,
        clockwise=clockwise_raw == "true",
        frequency_hz=frequency_hz,
        timestep_seconds=timestep_seconds,
        origin=(origin[0], origin[1], origin[2]),
        quaternion_xyzw=(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
    )


def read_ascii_pcd(path: Path) -> PointCloud:
    """Read an unorganized PCD v0.7 ASCII cloud with scalar float fields."""

    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DataValidationError(f"cannot read ASCII PCD {path}: {exc}") from exc
    header: dict[str, list[str]] = {}
    data_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        key = parts[0].upper()
        if key in header:
            raise DataValidationError(f"duplicate PCD header field {key}")
        header[key] = parts[1:]
        if key == "DATA":
            data_index = index + 1
            break
    if data_index is None:
        raise DataValidationError("PCD missing DATA header")
    required_headers = {
        "VERSION",
        "FIELDS",
        "SIZE",
        "TYPE",
        "COUNT",
        "WIDTH",
        "HEIGHT",
        "POINTS",
        "DATA",
    }
    missing = required_headers.difference(header)
    if missing:
        raise DataValidationError(f"PCD missing headers: {sorted(missing)}")
    if header["VERSION"] != ["0.7"]:
        raise DataValidationError("PCD VERSION must be 0.7")
    if [value.lower() for value in header["DATA"]] != ["ascii"]:
        raise DataValidationError("only ASCII PCD data are accepted")

    fields = tuple(header["FIELDS"])
    if not {"x", "y", "z"}.issubset(fields):
        raise DataValidationError("PCD fields must include x, y, z")
    if len(fields) != len(set(fields)):
        raise DataValidationError("PCD field names must be unique")
    field_count = len(fields)
    if header["TYPE"] != ["F"] * field_count:
        raise DataValidationError("PCD fields must all be floating-point")
    if header["SIZE"] != ["4"] * field_count:
        raise DataValidationError("PCD floating-point fields must be 4-byte values")
    if header["COUNT"] != ["1"] * field_count:
        raise DataValidationError("PCD fields must be scalar")
    try:
        width = int(header["WIDTH"][0])
        height = int(header["HEIGHT"][0])
        points = int(header["POINTS"][0])
    except (ValueError, IndexError) as exc:
        raise DataValidationError("PCD dimensions must be integers") from exc
    if len(header["WIDTH"]) != 1 or len(header["HEIGHT"]) != 1 or len(header["POINTS"]) != 1:
        raise DataValidationError("PCD dimensions must be scalar")
    if height != 1 or width <= 0 or points != width * height:
        raise DataValidationError("PCD point count is inconsistent with WIDTH and HEIGHT")

    body = "\n".join(lines[data_index:])
    values = np.fromstring(body, sep=" ", dtype=np.float64)
    if values.size != points * field_count:
        raise DataValidationError(
            f"PCD point count mismatch: expected {points * field_count} values, found {values.size}"
        )
    values = values.reshape(points, field_count)
    if not np.isfinite(values).all():
        raise DataValidationError("PCD contains non-finite values")
    return PointCloud(fields=fields, values=values)
