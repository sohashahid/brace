from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from brace_f1.io import (
    DataValidationError,
    load_compact_car,
    parse_metadata,
    read_ascii_pcd,
    validate_file_manifest,
)


def _valid_arrays(n: int = 3) -> dict[str, dict[str, np.ndarray]]:
    motion = {
        "session_times": np.asarray(1.25, dtype=np.float32),
        "frame_identifiers": np.asarray(8, dtype=np.int32),
        "overall_frame_identifiers": np.asarray(9, dtype=np.int32),
        "positions": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        "quaternions": np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (n, 1)),
        "velocities": np.ones((n, 3), dtype=np.float32),
        "accelerations": np.zeros((n, 3), dtype=np.float32),
    }
    lap = {
        "session_times": np.asarray(1.25, dtype=np.float32),
        "frame_identifiers": np.asarray(8, dtype=np.int32),
        "overall_frame_identifiers": np.asarray(9, dtype=np.int32),
        "lap_distances": np.arange(n, dtype=np.float32),
        "total_distances": np.arange(n, dtype=np.float32),
        "last_lap_times": np.full(n, np.inf, dtype=np.float32),
        "current_lap_times": np.arange(n, dtype=np.float32),
        "lap_numbers": np.ones(n, dtype=np.int32),
        "result_status": np.full(n, 2, dtype=np.int32),
        "driver_status": np.full(n, 4, dtype=np.int32),
        "pit_status": np.zeros(n, dtype=np.int32),
        "pit_lane_timer_active": np.zeros(n, dtype=np.bool_),
    }
    session = {
        "session_times": np.asarray([1.5, 2.0], dtype=np.float32),
        "frame_identifiers": np.asarray([10, 60], dtype=np.int32),
        "overall_frame_identifiers": np.asarray([10, 60], dtype=np.int32),
        "track_ids": np.asarray([3, 3], dtype=np.int32),
        "game_modes": np.asarray([7, 7], dtype=np.int32),
        "session_types": np.asarray([10, 10], dtype=np.int32),
        "game_paused": np.asarray([0, 0], dtype=np.int32),
    }
    return {"motion": motion, "lap": lap, "session": session}


def _write_compact(root: Path, arrays: dict[str, dict[str, np.ndarray]]) -> None:
    for group in ("motion", "lap", "session"):
        np.savez(root / f"{group}_data.npz", **arrays[group])


def test_load_compact_car_accepts_valid_numeric_arrays(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    _write_compact(tmp_path, arrays)

    compact = load_compact_car(tmp_path)

    assert compact.sample_count == 3
    np.testing.assert_array_equal(compact.motion["positions"], arrays["motion"]["positions"])
    assert compact.native_frame_timestamps_available is False


@pytest.mark.parametrize(
    "group,key",
    [("motion", "positions"), ("lap", "pit_status"), ("session", "game_paused")],
)
def test_load_compact_car_rejects_missing_required_key(
    tmp_path: Path, group: str, key: str
) -> None:
    arrays = _valid_arrays()
    del arrays[group][key]
    _write_compact(tmp_path, arrays)

    with pytest.raises(DataValidationError, match=f"missing required keys.*{key}"):
        load_compact_car(tmp_path)


def test_load_compact_car_rejects_object_array(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["motion"]["positions"] = np.asarray([{"unsafe": "pickle"}], dtype=object)
    _write_compact(tmp_path, arrays)

    with pytest.raises(DataValidationError, match="object array"):
        load_compact_car(tmp_path)


def test_load_compact_car_rejects_motion_lap_length_mismatch(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["lap"]["pit_status"] = np.zeros(2, dtype=np.int32)
    _write_compact(tmp_path, arrays)

    with pytest.raises(DataValidationError, match="length mismatch"):
        load_compact_car(tmp_path)


def test_load_compact_car_rejects_nonfinite_mandatory_motion_array(tmp_path: Path) -> None:
    arrays = _valid_arrays()
    arrays["motion"]["velocities"][1, 2] = np.nan
    _write_compact(tmp_path, arrays)

    with pytest.raises(DataValidationError, match="non-finite.*velocities"):
        load_compact_car(tmp_path)


def test_manifest_validation_checks_size_and_sha256(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"verified bytes")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "files.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["local_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerow({"local_path": str(payload), "bytes": 14, "sha256": digest})

    result = validate_file_manifest(manifest, expected_rows=1, base_dir=tmp_path)

    assert result.validated_files == 1
    assert result.total_bytes == 14

    bad_size = manifest.read_text(encoding="utf-8").replace(",14,", ",13,")
    manifest.write_text(bad_size, encoding="utf-8")
    with pytest.raises(DataValidationError, match="size mismatch"):
        validate_file_manifest(manifest, expected_rows=1, base_dir=tmp_path)

    manifest.write_text(
        bad_size.replace(",13,", ",14,").replace(digest, "0" * 64), encoding="utf-8"
    )
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        validate_file_manifest(manifest, expected_rows=1, base_dir=tmp_path)


def test_parse_metadata_reads_only_required_simple_fields(tmp_path: Path) -> None:
    path = tmp_path / "metadata.yaml"
    path.write_text(
        "trackname: Test Ring\n"
        "clockwise: false\n"
        "frequency_history: 20.0\n"
        "timestep_history: 0.05\n"
        "starting_pose_origin:\n"
        "- 1.0\n- -2.0\n- 3.5\n"
        "starting_pose_quaternion:\n"
        "- 0.0\n- 0.0\n- 0.7071067811865475\n- 0.7071067811865476\n",
        encoding="utf-8",
    )

    metadata = parse_metadata(path)

    assert metadata.trackname == "Test Ring"
    assert metadata.clockwise is False
    assert metadata.frequency_hz == 20.0
    assert metadata.timestep_seconds == 0.05
    assert metadata.origin == (1.0, -2.0, 3.5)
    assert metadata.quaternion_xyzw == (0.0, 0.0, 0.7071067811865475, 0.7071067811865476)


def test_parse_metadata_rejects_yaml_tags(tmp_path: Path) -> None:
    path = tmp_path / "metadata.yaml"
    path.write_text(
        "trackname: !!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="unsafe YAML token"):
        parse_metadata(path)


def test_read_ascii_pcd_validates_schema_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "line.pcd"
    path.write_text(
        "VERSION 0.7\n"
        "FIELDS x y z arclength\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n"
        "0 1 2 0\n3 4 5 3\n",
        encoding="ascii",
    )

    cloud = read_ascii_pcd(path)

    assert cloud.fields == ("x", "y", "z", "arclength")
    np.testing.assert_allclose(cloud.xyz, [[0, 1, 2], [3, 4, 5]])


@pytest.mark.parametrize(
    "replacement,match",
    [
        (("DATA ascii", "DATA binary"), "ASCII"),
        (("TYPE F F F F", "TYPE F I F F"), "floating-point"),
        (("POINTS 2", "POINTS 3"), "point count"),
        (("FIELDS x y z arclength", "FIELDS y z arclength speed"), "x, y, z"),
    ],
)
def test_read_ascii_pcd_rejects_invalid_schema(
    tmp_path: Path, replacement: tuple[str, str], match: str
) -> None:
    text = (
        "VERSION 0.7\nFIELDS x y z arclength\nSIZE 4 4 4 4\nTYPE F F F F\n"
        "COUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n"
        "0 1 2 0\n3 4 5 3\n"
    )
    path = tmp_path / "bad.pcd"
    path.write_text(text.replace(*replacement), encoding="ascii")

    with pytest.raises(DataValidationError, match=match):
        read_ascii_pcd(path)
