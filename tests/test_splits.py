from __future__ import annotations

import hashlib

import pandas as pd

from brace_f1.splits import attach_loco_partitions, make_loco_assignments


def _frames() -> pd.DataFrame:
    rows = []
    for circuit in ("Alpha", "Beta", "Gamma"):
        for car_id in ("car_1", "car_2", "car_3"):
            for frame_index in range(4):
                rows.append(
                    {
                        "circuit": circuit,
                        "car_id": car_id,
                        "source_session_id": f"fixture@rev-a:{circuit.lower()}-session",
                        "source_revision": "rev-a",
                        "source_unit_sha256": hashlib.sha256(
                            f"{circuit}/{car_id}".encode()
                        ).hexdigest(),
                        "frame_index": frame_index,
                        "value": len(rows),
                    }
                )
    return pd.DataFrame(rows)


def _assign(
    frames: pd.DataFrame, *, calibration_fraction: float, seed: int, salt: str = "test-salt"
) -> pd.DataFrame:
    return make_loco_assignments(
        frames,
        calibration_fraction=calibration_fraction,
        seed=seed,
        salt=salt,
        source_manifest_sha256="a" * 64,
        assignment_code_version="0.1.0",
    )


def test_loco_assignments_hold_out_every_car_from_test_circuit() -> None:
    assignments = _assign(_frames(), calibration_fraction=1 / 3, seed=2027)

    assert set(assignments["fold_test_circuit"]) == {"Alpha", "Beta", "Gamma"}
    for test_circuit, fold in assignments.groupby("fold_test_circuit"):
        assert (fold.loc[fold["circuit"] == test_circuit, "partition"] == "test").all()
        assert not (fold.loc[fold["circuit"] != test_circuit, "partition"] == "test").any()
        assert set(fold.loc[fold["circuit"] != test_circuit, "partition"]) == {
            "fit",
            "calibration",
        }


def test_loco_assignments_are_deterministic_under_input_reordering() -> None:
    frames = _frames()

    first = _assign(frames, calibration_fraction=0.25, seed=42)
    second = _assign(
        frames.sample(frac=1.0, random_state=99), calibration_fraction=0.25, seed=42
    )

    pd.testing.assert_frame_equal(first, second)


def test_attach_partitions_has_no_car_group_or_frame_leakage() -> None:
    frames = _frames()
    assignments = _assign(frames, calibration_fraction=1 / 3, seed=7)

    attached = attach_loco_partitions(frames, assignments)

    per_group = attached.groupby(
        ["fold_test_circuit", "circuit", "car_id"]
    )["partition"].nunique()
    assert (per_group == 1).all()
    per_frame = attached.groupby(
        ["fold_test_circuit", "circuit", "car_id", "frame_index"]
    )["partition"].nunique()
    assert (per_frame == 1).all()
    assert len(attached) == len(frames) * frames["circuit"].nunique()


def test_split_identity_includes_source_session_and_records_provenance() -> None:
    frames = _frames()

    first = _assign(frames, calibration_fraction=0.25, seed=42, salt="salt-a")
    second = _assign(frames, calibration_fraction=0.25, seed=42, salt="salt-b")

    required = {
        "source_session_id",
        "split_unit_id",
        "source_revision",
        "source_unit_sha256",
        "source_manifest_sha256",
        "split_salt",
        "assignment_code_version",
        "unit_frame_row_count",
    }
    assert required.issubset(first.columns)
    assert set(first["source_revision"]) == {"rev-a"}
    assert set(first["source_manifest_sha256"]) == {"a" * 64}
    assert set(first["split_salt"]) == {"salt-a"}
    assert set(first["assignment_code_version"]) == {"0.1.0"}
    assert set(first["unit_frame_row_count"]) == {4}
    assert first["split_unit_id"].str.contains("fixture@rev-a").all()
    assert (first["split_key"] != second["split_key"]).all()


def test_same_circuit_and_car_in_distinct_source_sessions_do_not_collapse() -> None:
    frames = _frames()
    duplicate = frames.loc[
        (frames["circuit"] == "Alpha") & (frames["car_id"] == "car_1")
    ].copy()
    duplicate["source_session_id"] = "fixture@rev-a:alpha-second-session"
    duplicate["source_unit_sha256"] = "b" * 64
    combined = pd.concat([frames, duplicate], ignore_index=True)

    assignments = _assign(combined, calibration_fraction=0.25, seed=42)
    alpha_car = assignments.loc[
        (assignments["fold_test_circuit"] == "Beta")
        & (assignments["circuit"] == "Alpha")
        & (assignments["car_id"] == "car_1")
    ]

    assert alpha_car["source_session_id"].nunique() == 2
    attached = attach_loco_partitions(combined, assignments)
    assert len(attached) == len(combined) * combined["circuit"].nunique()
