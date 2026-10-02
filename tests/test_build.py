from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from brace_f1.build import BuildConfig, _aggregate_source_flags, build_deepracing_cohort
from brace_f1.events import ExcursionConfig
from brace_f1.timebase import CausalResample, TimeAxis, TimingDiagnostics


def _write_pcd(path: Path, points: list[tuple[float, float]]) -> None:
    rows = "\n".join(f"{x} {y} 0 0" for x, y in points)
    path.write_text(
        "VERSION 0.7\nFIELDS x y z arclength\nSIZE 4 4 4 4\nTYPE F F F F\n"
        f"COUNT 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\n"
        f"DATA ascii\n{rows}\n",
        encoding="ascii",
    )


def _write_car(car_dir: Path, circuit: str, x_offset: float) -> None:
    car_dir.mkdir(parents=True)
    car_dir.joinpath("metadata.yaml").write_text(
        f"trackname: {circuit}\nclockwise: true\nfrequency_history: 20.0\n"
        "timestep_history: 0.05\nstarting_pose_origin:\n"
        f"- {x_offset}\n- 0.0\n- 0.0\n"
        "starting_pose_quaternion:\n- 0.0\n- 0.0\n- 0.0\n- 1.0\n",
        encoding="utf-8",
    )
    n = 41
    positions = np.column_stack(
        (x_offset + np.linspace(-2, 2, n), np.full(n, -3.0), np.zeros(n))
    ).astype(np.float32)
    motion = {
        "session_times": np.asarray(0.0, dtype=np.float32),
        "frame_identifiers": np.asarray(0, dtype=np.int32),
        "overall_frame_identifiers": np.asarray(0, dtype=np.int32),
        "positions": positions,
        "quaternions": np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (n, 1)),
        "velocities": np.tile(np.asarray([4, 0, 0], dtype=np.float32), (n, 1)),
        "accelerations": np.zeros((n, 3), dtype=np.float32),
    }
    times = np.arange(n, dtype=np.float32) * np.float32(0.05)
    lap = {
        "session_times": np.asarray(0.0, dtype=np.float32),
        "frame_identifiers": np.asarray(0, dtype=np.int32),
        "overall_frame_identifiers": np.asarray(0, dtype=np.int32),
        "lap_distances": np.arange(n, dtype=np.float32),
        "total_distances": np.arange(n, dtype=np.float32),
        "last_lap_times": np.full(n, np.inf, dtype=np.float32),
        "current_lap_times": times,
        "lap_numbers": np.ones(n, dtype=np.int32),
        "result_status": np.full(n, 2, dtype=np.int32),
        "driver_status": np.full(n, 4, dtype=np.int32),
        "pit_status": np.zeros(n, dtype=np.int32),
        "pit_lane_timer_active": np.zeros(n, dtype=np.bool_),
    }
    session = {
        "session_times": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        "frame_identifiers": np.asarray([0, 20, 40], dtype=np.int32),
        "overall_frame_identifiers": np.asarray([0, 20, 40], dtype=np.int32),
        "track_ids": np.ones(3, dtype=np.int32),
        "game_modes": np.full(3, 7, dtype=np.int32),
        "session_types": np.full(3, 10, dtype=np.int32),
        "game_paused": np.zeros(3, dtype=np.int32),
    }
    np.savez(car_dir / "motion_data.npz", **motion)
    np.savez(car_dir / "lap_data.npz", **lap)
    np.savez(car_dir / "session_data.npz", **session)
    _write_pcd(car_dir / "center_line.pcd", [(-3, -3), (3, -3), (3, 3), (-3, 3)])
    _write_pcd(car_dir / "inner_boundary.pcd", [(-1, -1), (-1, 1), (1, 1), (1, -1)])
    _write_pcd(car_dir / "outer_boundary.pcd", [(-5, -5), (5, -5), (5, 5), (-5, 5)])


def _make_manifest(root: Path) -> Path:
    rows: list[dict[str, str | int]] = []
    for circuit, car_id, offset in (("Alpha", "car_0", 10.0), ("Beta", "car_1", -7.0)):
        car_dir = root / circuit / car_id
        _write_car(car_dir, circuit, offset)
        for filename in (
            "metadata.yaml",
            "motion_data.npz",
            "lap_data.npz",
            "session_data.npz",
            "center_line.pcd",
            "inner_boundary.pcd",
            "outer_boundary.pcd",
        ):
            path = car_dir / filename
            rows.append(
                {
                    "dataset": "fixture",
                    "revision": "abc123",
                    "circuit": circuit,
                    "circuit_directory": circuit,
                    "car": car_id,
                    "role": "boundary" if filename.endswith(".pcd") else "compact_car_data",
                    "filename": filename,
                    "upstream_path": f"Trajectory_Prediction/{circuit}/{car_id}/{filename}",
                    "local_path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "license": "CC-BY-4.0",
                }
            )
    manifest = root / "files.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_build_cohort_writes_hashed_derived_tables_and_diagnostics(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    output_dir = tmp_path / "processed"
    build_manifest = tmp_path / "manifests" / "build.json"
    config = BuildConfig(
        expected_manifest_rows=14,
        resample_frequency_hz=20.0,
        calibration_fraction=0.25,
        split_seed=2027,
        excursion=ExcursionConfig(
            gap_merge_seconds=0.25,
            min_max_depth_m=0.5,
            min_duration_seconds=0.25,
            min_lead_in_seconds=0.25,
            artifact_exclusion_seconds=0.25,
        ),
    )

    result = build_deepracing_cohort(
        manifest,
        output_dir=output_dir,
        build_manifest_path=build_manifest,
        base_dir=tmp_path,
        config=config,
    )

    assert result.counts["input_files"] == 14
    assert result.counts["cars"] == 2
    assert result.counts["circuits"] == 2
    assert result.counts["raw_frames"] == 82
    assert result.counts["resampled_frames"] == 82
    assert result.counts["qualified_excursions"] == 0
    assert np.isclose(result.counts["prebuffer_exposure_seconds"], 4.0)
    assert np.isclose(result.counts["buffered_exposure_seconds"], 2.0)
    assert (output_dir / "deepracing_frames.parquet").is_file()
    assert (output_dir / "deepracing_targets.parquet").is_file()
    assert (output_dir / "deepracing_events.csv").is_file()
    assert (output_dir / "deepracing_events.parquet").is_file()
    assert (output_dir / "deepracing_timing_diagnostics.csv").is_file()
    assert (output_dir / "deepracing_loco_splits.csv").is_file()
    assert (output_dir / "deepracing_map_manifest.json").is_file()
    assert (output_dir / "deepracing_kinematic_diagnostics.csv").is_file()

    frames = pd.read_parquet(output_dir / "deepracing_frames.parquet")
    assert len(frames) == 82
    assert not frames["native_frame_timestamps_available"].any()
    assert set(frames["timestamp_method"]) == {"reconstructed_from_lap_clock"}
    required_frame_columns = {
        "body_acceleration_longitudinal_mps2",
        "body_acceleration_lateral_mps2",
        "body_acceleration_vertical_mps2",
        "acceleration_x_mps2",
        "derived_acceleration_x_mps2",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
        "yaw_map_rad",
        "forward_x_map",
        "forward_y_map",
        "centerline_segment_index_pcd",
        "centerline_arclength_wrapped_m",
        "segment_bin_25m",
        "input_valid_causal",
    }
    assert required_frame_columns.issubset(frames.columns)
    assert {
        "source_session_id",
        "source_revision",
        "source_unit_sha256",
    }.issubset(frames.columns)
    forbidden_truth_prefixes = (
        "next_",
        "time_to_next_",
        "qualifying_event_",
        "projected_onset_",
        "outcome_evaluable_",
    )
    assert not {
        column
        for column in frames.columns
        if column.startswith(forbidden_truth_prefixes)
    }
    assert "candidate_event_id_at_source" not in frames.columns
    assert "offline_eligible_grid" not in frames.columns
    assert "offline_artifact_excluded" not in frames.columns
    assert "offline_artifact_seed" not in frames.columns
    assert "raw_outside_at_source" not in frames.columns
    assert "continuous_segment_end_time_seconds" not in frames.columns
    assert "continuous_segment_remaining_seconds" not in frames.columns
    assert not frames.groupby(["circuit", "car_id"]).head(1)["input_valid_causal"].any()
    targets = pd.read_parquet(output_dir / "deepracing_targets.parquet")
    assert len(targets) == len(frames)
    assert not targets.duplicated(["circuit", "car_id", "frame_index"]).any()
    pd.testing.assert_frame_equal(
        frames[["circuit", "car_id", "frame_index"]],
        targets[["circuit", "car_id", "frame_index"]],
    )
    assert {
        "outcome_evaluable_0p25s",
        "outcome_evaluable_0p50s",
        "outcome_evaluable_1p00s",
        "outcome_evaluable_1p50s",
        "next_qualifying_event_id",
        "time_to_next_excursion_seconds",
        "offline_eligible_grid",
        "offline_artifact_excluded",
    }.issubset(targets.columns)
    splits = pd.read_csv(output_dir / "deepracing_loco_splits.csv")
    assert len(splits) == 4
    assert {
        "source_session_id",
        "source_revision",
        "source_unit_sha256",
        "source_manifest_sha256",
        "split_salt",
        "assignment_code_version",
        "unit_frame_row_count",
    }.issubset(splits.columns)
    assert not splits["source_session_id"].str.startswith(("/", "~")).any()
    timing = pd.read_csv(output_dir / "deepracing_timing_diagnostics.csv")
    assert np.isclose(timing["prebuffer_exposure_seconds"].sum(), 4.0)
    assert np.isclose(timing["buffered_exposure_seconds"].sum(), 2.0)
    map_manifest = json.loads(
        (output_dir / "deepracing_map_manifest.json").read_text(encoding="utf-8")
    )
    assert map_manifest["segment_bin_width_m"] == 25.0
    assert map_manifest["segment_bin_origin_m"] == 0.0
    assert map_manifest["arclength_direction"] == "source_center_line_pcd_point_order"
    assert map_manifest["arclength_direction_status"] == "frozen_from_source_map"
    assert map_manifest["arclength_wrap"] is True
    assert {entry["track_length_m"] for entry in map_manifest["circuits"]} == {24.0}
    assert {
        entry["travel_direction_validation_status"] for entry in map_manifest["circuits"]
    } == {"insufficient_speed_gt_10_observations"}
    assert all(
        entry["travel_direction_validation_count"] == 0
        for entry in map_manifest["circuits"]
    )

    payload = json.loads(build_manifest.read_text(encoding="utf-8"))
    assert payload["input_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len(payload["inputs"]) == 14
    assert len(payload["timing_diagnostics"]) == 2
    assert payload["config"]["event_definition_status"] == "frozen_predeclared"
    assert payload["config"]["event_label_stream"] == "raw_reconstructed_before_resampling"
    assert payload["config"]["exposure_buffer_rule"] == "symmetric_interval_overlap"
    assert payload["config"]["event_gap_rule"] == "intervening_in_corridor_duration"
    reproducibility_paths = {record["path"] for record in payload["source_code"]}
    assert {
        "pyproject.toml",
        "uv.lock",
        "configs/study.json",
        "src/brace_f1/build.py",
        "tests/test_build.py",
    }.issubset(reproducibility_paths)
    runtime = payload["runtime_environment"]
    assert runtime["execution_contract"] == "uv run --locked python -m brace_f1.build"
    assert runtime["lock_environment_active"] is True
    assert runtime["python"]["version"] == platform.python_version()
    assert Path(runtime["python"]["executable"]).resolve() == Path(sys.executable).resolve()
    assert runtime["lockfile"]["path"] == "uv.lock"
    lock_path = Path(__file__).parents[1] / "uv.lock"
    assert runtime["lockfile"]["sha256"] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    for package in ("numpy", "pandas", "pyarrow", "scipy"):
        assert runtime["packages"][package] == version(package)
    assert any("not crash" in warning.lower() for warning in payload["warnings"])
    for output in payload["outputs"]:
        path = Path(output["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]


def test_aggregate_source_flags_discards_flags_after_last_target() -> None:
    diagnostics = TimingDiagnostics(
        timestamp_method="fixture",
        native_frame_timestamps_available=False,
        sample_count=4,
        median_source_dt_seconds=0.4,
        estimated_source_hz=2.5,
        lap_boundary_count=0,
        clock_reset_count=0,
        data_gap_count=0,
        position_jump_count=0,
        distance_reset_count=0,
        status_transition_count=0,
        hard_break_count=0,
    )
    zeros = np.zeros(4, dtype=bool)
    axis = TimeAxis(
        time_seconds=np.asarray([0.0, 0.4, 0.8, 1.2]),
        hard_break=np.asarray([True, False, False, False]),
        lap_boundary=zeros.copy(),
        clock_reset=zeros.copy(),
        data_gap=zeros.copy(),
        artifact_proximity_seed=zeros.copy(),
        diagnostics=diagnostics,
    )
    sampled = CausalResample(
        target_time_seconds=np.asarray([0.0, 0.5, 1.0]),
        source_indices=np.asarray([0, 1, 2]),
        source_age_seconds=np.asarray([0.0, 0.1, 0.2]),
        segment_id=np.zeros(3, dtype=np.int64),
        arrays={},
        frequency_hz=2.0,
    )
    after_grid = np.asarray([False, False, False, True])
    within_grid = np.asarray([False, False, True, False])

    assert not _aggregate_source_flags(axis, sampled, after_grid).any()
    mapped = _aggregate_source_flags(axis, sampled, within_grid)
    assert mapped.tolist() == [False, False, True]
    assert sampled.target_time_seconds[mapped][0] >= axis.time_seconds[within_grid][0]
