#!/usr/bin/env python3
"""Build the calibration-only operating-cliff diagnostic for BRACE.

The renderer authenticates the frozen threshold seal, fold manifests, build manifest,
and processed inputs. It then recomputes every plotted point from calibration rows with
the frozen threshold grid, proposal state machine, event labels, and exposure accounting.
It refuses to write anywhere inside the frozen experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

os.environ["MPLBACKEND"] = "Agg"
os.environ["SOURCE_DATE_EPOCH"] = "1790812800"

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pubfig as pf  # noqa: E402

from brace_f1.experiment import (  # noqa: E402
    _code_content_hash,
    _events_for_units,
    calibration_threshold_grid,
    join_exposure_to_split,
)
from brace_f1.policy import label_proposals, run_proposal_state_machine  # noqa: E402

DEFAULT_INPUT: Final = (
    ROOT / "paper" / "figure-source" / "figure-03-calibration-operating-cliff.csv"
)
DEFAULT_OUTPUT_DIR: Final = ROOT / "paper" / "figures"
FROZEN_EXPERIMENT_DIR: Final = (ROOT / "output" / "experiment").resolve()
THRESHOLD_SEAL_PATH: Final = ROOT / "output" / "experiment" / "threshold-freeze-seal.json"
BUILD_MANIFEST_PATH: Final = ROOT / "data" / "manifests" / "deepracing-build.json"
EVENTS_PATH: Final = ROOT / "data" / "processed" / "deepracing_events.parquet"
TIMING_PATH: Final = ROOT / "data" / "processed" / "deepracing_timing_diagnostics.csv"
SPLITS_PATH: Final = ROOT / "data" / "processed" / "deepracing_loco_splits.csv"
OUTPUT_STEM: Final = "figure-03-calibration-operating-cliff"
PUBFIG_VERSION: Final = "0.3.0"
METHOD_ORDER: Final = ("brace_bayesian", "posterior_mean_twin")
CIRCUIT_ORDER: Final = ("Bahrain", "Britain", "Jeddah", "Monza")
RATE_COLUMN: Final = "false_proposals_per_eligible_simulated_car_hour"
EXPECTED_SEAL_SHA256: Final = (
    "ad26f9505a8e00fbdf000bfbbaacf278792eabfb21e937c723418032c234667d"
)
EXPECTED_FOLD_MANIFEST_SHA256: Final = {
    "Bahrain": "f550bc56e03f398421b2af2b03ad2a1dc709018241e1a6bc76835ddaa142b43d",
    "Britain": "0fdbc0307584c6f69d75645fbb2b80b3b38e5fbddae005c17b46bdaa370375b5",
    "Jeddah": "5fd126b1299d477ca0011443ee8419b8de29cd6497cf956798bbb70c8def6bcc",
    "Monza": "3aa16d9d32d8ccc007732fe161447f1f1421da5d8ac22aa62fd1784b8043dc1e",
}
HASH_SOURCES: Final = (
    ("source_calibrated_scores_path", "source_calibrated_scores_sha256"),
    ("source_calibration_thresholds_path", "source_calibration_thresholds_sha256"),
    ("fold_manifest_path", "fold_manifest_sha256"),
    ("threshold_freeze_seal_path", "threshold_freeze_seal_sha256"),
)
REQUIRED_COLUMNS: Final = {
    "fold_test_circuit",
    "method",
    "method_label",
    "threshold",
    "calibration_false_proposals",
    "calibration_exposure_hours",
    RATE_COLUMN,
    "selection_rule",
    *(item for pair in HASH_SOURCES for item in pair),
}


class FigureDataError(ValueError):
    """Raised when the checked-in diagnostic table fails its evidence contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_source(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise FigureDataError(f"source path must be repository-relative: {relative_path!r}")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise FigureDataError(f"source path escapes repository root: {relative_path!r}")
    return resolved


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FigureDataError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise FigureDataError(f"{label} must contain a JSON object")
    return payload


def _registered_artifact_hash(
    artifacts: object,
    *,
    filename: str,
    label: str,
) -> str:
    if not isinstance(artifacts, dict):
        raise FigureDataError(f"{label} lacks an artifact registry")
    matches = [str(value) for key, value in artifacts.items() if Path(str(key)).name == filename]
    if len(matches) != 1:
        raise FigureDataError(f"{label} must register {filename!r} exactly once")
    return matches[0]


def _validate_source_hashes(data: pd.DataFrame) -> set[Path]:
    checked: dict[Path, str] = {}
    for path_column, hash_column in HASH_SOURCES:
        for row in data[[path_column, hash_column]].drop_duplicates().itertuples(index=False):
            path = _resolve_source(str(row[0]))
            expected = str(row[1]).lower()
            if path in checked:
                if checked[path] != expected:
                    raise FigureDataError(f"conflicting source hashes recorded for {path}")
                continue
            if not path.is_file():
                raise FigureDataError(f"referenced source artifact is missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise FigureDataError(
                    f"source hash mismatch for {path}: expected {expected}, got {actual}"
                )
            checked[path] = expected
    return set(checked)


def _validate_table_structure(data: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise FigureDataError(f"diagnostic table is missing columns: {missing}")
    if len(data) != 8:
        raise FigureDataError(f"expected exactly 8 diagnostic rows, found {len(data)}")

    expected_pairs = {(circuit, method) for circuit in CIRCUIT_ORDER for method in METHOD_ORDER}
    actual_pairs = set(zip(data["fold_test_circuit"], data["method"], strict=True))
    if actual_pairs != expected_pairs:
        raise FigureDataError("diagnostic table must contain both methods for all four circuits")
    if set(data["selection_rule"].astype(str)) != {"closest_active_below_silence"}:
        raise FigureDataError("unexpected threshold-selection rule")

    numeric_columns = (
        "threshold",
        "calibration_false_proposals",
        "calibration_exposure_hours",
        RATE_COLUMN,
    )
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FigureDataError(f"column {column!r} contains a non-finite value")

    thresholds = data["threshold"].to_numpy(dtype=float)
    counts = data["calibration_false_proposals"].to_numpy(dtype=float)
    exposure = data["calibration_exposure_hours"].to_numpy(dtype=float)
    rates = data[RATE_COLUMN].to_numpy(dtype=float)
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise FigureDataError("active thresholds must lie strictly between zero and one")
    if np.any(counts < 0.0) or not np.allclose(counts, np.round(counts), atol=0.0, rtol=0.0):
        raise FigureDataError("false-proposal counts must be non-negative integers")
    if np.any(exposure <= 0.0):
        raise FigureDataError("calibration exposures must be positive")
    if not np.allclose(rates, counts / exposure, atol=1e-9, rtol=0.0):
        raise FigureDataError("recorded rates do not equal false proposals divided by exposure")
    if np.any(rates <= 10.0):
        raise FigureDataError("operating-cliff diagnostic expects every active point above 10/h")

    for _, group in data.groupby("fold_test_circuit", sort=False):
        if not np.allclose(
            group["calibration_exposure_hours"],
            group["calibration_exposure_hours"].iloc[0],
            atol=0.0,
            rtol=0.0,
        ):
            raise FigureDataError("paired methods must use the same exposure within each circuit")


def _validate_build_output(build_manifest: dict[str, object], *, path: Path) -> None:
    outputs = build_manifest.get("outputs")
    if not isinstance(outputs, list):
        raise FigureDataError("build manifest lacks an output registry")
    matches = [
        item
        for item in outputs
        if isinstance(item, dict) and Path(str(item.get("path", ""))).name == path.name
    ]
    if len(matches) != 1:
        raise FigureDataError(f"build manifest must register {path.name!r} exactly once")
    entry = matches[0]
    if not path.is_file():
        raise FigureDataError(f"processed evidence file is missing: {path}")
    if entry.get("sha256") != _sha256(path) or entry.get("bytes") != path.stat().st_size:
        raise FigureDataError(f"processed evidence does not match build manifest: {path}")


def _validate_silent_threshold_table(
    path: Path,
    *,
    candidate_counts: dict[str, int] | None,
) -> None:
    table = pd.read_csv(path)
    methods = table.loc[table["method"].astype(str).isin(METHOD_ORDER)].copy()
    if methods.empty or set(methods["threshold_source_partition"].astype(str)) != {
        "calibration"
    }:
        raise FigureDataError(f"threshold table is not calibration-only: {path}")
    budget = pd.to_numeric(methods["false_budget_per_hour"], errors="coerce")
    selected = methods.loc[
        budget.isin((2.0, 5.0, 10.0))
        & (methods["operating_point_status"].astype(str) == "estimable")
    ].copy()
    expected_rows = len(METHOD_ORDER) * 4 * 3
    if len(selected) != expected_rows:
        raise FigureDataError(f"threshold table lacks all estimable 2/5/10-h gate rows: {path}")
    for column in (
        "threshold",
        "calibration_false_proposals_per_hour",
        "calibration_false_proposals",
        "calibration_proposal_count",
    ):
        values = pd.to_numeric(selected[column], errors="coerce").to_numpy(dtype=float)
        expected = 1.0 if column == "threshold" else 0.0
        if not np.isfinite(values).all() or not np.allclose(
            values,
            expected,
            atol=1e-12,
            rtol=0.0,
        ):
            raise FigureDataError(f"frozen gate rows are not all-silent in {path}")
    if candidate_counts is not None:
        for method, expected_count in candidate_counts.items():
            counts = pd.to_numeric(
                methods.loc[
                    methods["method"].astype(str) == method,
                    "threshold_candidate_count",
                ],
                errors="coerce",
            ).to_numpy(dtype=float)
            if counts.size == 0 or not np.all(counts == expected_count):
                raise FigureDataError(
                    f"threshold-grid count disagrees with frozen table for {path.name}/{method}"
                )


def _authenticate_evidence(data: pd.DataFrame) -> int:
    checked = _validate_source_hashes(data)
    if _sha256(THRESHOLD_SEAL_PATH) != EXPECTED_SEAL_SHA256:
        raise FigureDataError("threshold-freeze seal does not match its fixed reviewed digest")
    checked.add(THRESHOLD_SEAL_PATH.resolve())
    seal = _load_json(THRESHOLD_SEAL_PATH, "threshold-freeze seal")
    if seal.get("schema_version") != 1 or seal.get("status") != "thresholds_frozen_before_heldout":
        raise FigureDataError("threshold-freeze seal has an unexpected schema or status")
    registered_circuits = seal.get("registered_circuits")
    if not isinstance(registered_circuits, list) or tuple(map(str, registered_circuits)) != (
        CIRCUIT_ORDER
    ):
        raise FigureDataError("threshold-freeze seal circuit registry is unexpected")
    registered_methods = seal.get("registered_methods")
    if not isinstance(registered_methods, list) or not set(METHOD_ORDER).issubset(
        set(map(str, registered_methods))
    ):
        raise FigureDataError("threshold-freeze seal lacks a required method")
    if _code_content_hash() != seal.get("code_content_hash"):
        raise FigureDataError("current producer source tree does not match the threshold seal")

    build_hash = _sha256(BUILD_MANIFEST_PATH)
    if build_hash != seal.get("build_manifest_hash"):
        raise FigureDataError("build manifest does not match the threshold seal")
    checked.add(BUILD_MANIFEST_PATH.resolve())
    build_manifest = _load_json(BUILD_MANIFEST_PATH, "DeepRacing build manifest")
    for path in (EVENTS_PATH, TIMING_PATH, SPLITS_PATH):
        _validate_build_output(build_manifest, path=path)
        checked.add(path.resolve())

    fold_entries = seal.get("fold_threshold_identities")
    if not isinstance(fold_entries, list):
        raise FigureDataError("threshold-freeze seal lacks fold identities")
    by_fold = {
        str(entry.get("fold_test_circuit")): entry
        for entry in fold_entries
        if isinstance(entry, dict)
    }
    if set(by_fold) != set(CIRCUIT_ORDER):
        raise FigureDataError("threshold-freeze seal fold identities are incomplete")

    identity_keys = (
        "build_manifest_hash",
        "code_content_hash",
        "config_content_hash",
        "data_content_hash",
        "source_manifest_hash",
    )
    expected_seal_relative = THRESHOLD_SEAL_PATH.relative_to(ROOT).as_posix()
    for circuit in CIRCUIT_ORDER:
        rows = data.loc[data["fold_test_circuit"].astype(str) == circuit]
        compact_path = ROOT / f"output/experiment/fold={circuit}/compact-calibrated-scores.parquet"
        threshold_path = ROOT / f"output/experiment/fold={circuit}/calibration-thresholds.csv"
        manifest_path = ROOT / f"output/experiment/fold={circuit}/manifest.json"
        expected_paths = {
            "source_calibrated_scores_path": compact_path.relative_to(ROOT).as_posix(),
            "source_calibration_thresholds_path": threshold_path.relative_to(ROOT).as_posix(),
            "fold_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "threshold_freeze_seal_path": expected_seal_relative,
        }
        for column, expected in expected_paths.items():
            if set(rows[column].astype(str)) != {expected}:
                raise FigureDataError(f"{circuit} CSV source path is not bound to {expected}")

        if _sha256(manifest_path) != EXPECTED_FOLD_MANIFEST_SHA256[circuit]:
            raise FigureDataError(f"{circuit} held-out manifest digest is not the reviewed value")
        checked.add(manifest_path.resolve())
        manifest = _load_json(manifest_path, f"{circuit} held-out manifest")
        entry = by_fold[circuit]
        run_identity = entry.get("run_identity")
        if not isinstance(run_identity, dict):
            raise FigureDataError(f"{circuit} seal entry lacks a run identity")
        for key in identity_keys:
            if run_identity.get(key) != seal.get(key) or manifest.get(key) != seal.get(key):
                raise FigureDataError(f"{circuit} identity field {key!r} is not seal-bound")
        if manifest.get("run_identity") != run_identity:
            raise FigureDataError(f"{circuit} fold run identity differs from the threshold seal")
        if manifest.get("model_identity_hash") != entry.get("model_identity_hash"):
            raise FigureDataError(f"{circuit} model identity differs from the threshold seal")
        if manifest.get("method_identities") != entry.get("method_identities"):
            raise FigureDataError(f"{circuit} method identities differ from the threshold seal")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "heldout_complete"
            or manifest.get("fold_test_circuit") != circuit
            or manifest.get("heldout_consumed_sealed_artifacts") is not True
            or manifest.get("models_refit_for_heldout") is not False
            or manifest.get("calibrators_refit_for_heldout") is not False
            or manifest.get("thresholds_refit_for_heldout") is not False
        ):
            raise FigureDataError(f"{circuit} held-out manifest violates the sealed-run contract")
        if set(map(str, manifest.get("completed_methods", ()))) != set(
            map(str, registered_methods)
        ):
            raise FigureDataError(f"{circuit} held-out method registry is incomplete")

        frozen = entry.get("frozen_artifacts")
        compact_hash = _registered_artifact_hash(
            frozen,
            filename=compact_path.name,
            label=f"{circuit} threshold seal",
        )
        threshold_hash = _registered_artifact_hash(
            frozen,
            filename=threshold_path.name,
            label=f"{circuit} threshold seal",
        )
        if (
            set(rows["source_calibrated_scores_sha256"].astype(str)) != {compact_hash}
            or set(rows["source_calibration_thresholds_sha256"].astype(str))
            != {threshold_hash}
            or set(rows["fold_manifest_sha256"].astype(str))
            != {EXPECTED_FOLD_MANIFEST_SHA256[circuit]}
            or set(rows["threshold_freeze_seal_sha256"].astype(str))
            != {EXPECTED_SEAL_SHA256}
        ):
            raise FigureDataError(f"{circuit} CSV digests are not cross-bound to the seal")
        if _registered_artifact_hash(
            manifest.get("artifacts"),
            filename=compact_path.name,
            label=f"{circuit} manifest",
        ) != compact_hash or _registered_artifact_hash(
            manifest.get("artifacts"),
            filename=threshold_path.name,
            label=f"{circuit} manifest",
        ) != threshold_hash:
            raise FigureDataError(f"{circuit} held-out manifest differs from frozen artifacts")
        _validate_silent_threshold_table(threshold_path, candidate_counts=None)
    return len(checked)


def _derive_closest_active_diagnostic(data: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_parquet(EVENTS_PATH)
    timing = pd.read_csv(TIMING_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    rows: list[dict[str, object]] = []
    for circuit in CIRCUIT_ORDER:
        exposure = join_exposure_to_split(timing, splits, fold_test_circuit=circuit)
        calibration_exposure = exposure.loc[
            (exposure["fold_test_circuit"].astype(str) == circuit)
            & (exposure["partition"].astype(str) == "calibration")
        ].copy()
        exposure_hours = float(calibration_exposure["exposure_hours"].sum())
        calibration_events = _events_for_units(events, calibration_exposure)
        source_path = str(
            data.loc[
                data["fold_test_circuit"].astype(str) == circuit,
                "source_calibrated_scores_path",
            ].iloc[0]
        )
        calibrated = pd.read_parquet(_resolve_source(source_path))
        for method in METHOD_ORDER:
            method_scores = calibrated.loc[
                (calibrated["fold_test_circuit"].astype(str) == circuit)
                & (calibrated["partition"].astype(str) == "calibration")
                & (calibrated["method"].astype(str) == method)
            ].copy()
            if method_scores.empty:
                raise FigureDataError(f"missing calibration scores for {circuit}/{method}")
            candidate_thresholds = calibration_threshold_grid(
                method_scores["proposal_score"].to_numpy(dtype=float)
            )
            proposals = None
            active_threshold = None
            for threshold in candidate_thresholds[::-1]:
                candidate_proposals = run_proposal_state_machine(
                    method_scores,
                    threshold=float(threshold),
                )
                if not candidate_proposals.empty:
                    active_threshold = float(threshold)
                    proposals = candidate_proposals
                    break
            if active_threshold is None or proposals is None:
                raise FigureDataError(f"no active calibration threshold for {circuit}/{method}")
            labels = label_proposals(
                proposals,
                calibration_events,
                segment_tolerance_bins=1,
            )
            false_proposals = int(labels["false_proposal"].astype(bool).sum())
            rows.append(
                {
                    "fold_test_circuit": circuit,
                    "method": method,
                    "source_partition": "calibration",
                    "threshold": active_threshold,
                    "calibration_false_proposals": false_proposals,
                    "calibration_exposure_hours": exposure_hours,
                    RATE_COLUMN: false_proposals / exposure_hours,
                    "threshold_candidate_count": int(candidate_thresholds.size),
                    "calibration_proposal_count": len(proposals),
                }
            )
    return pd.DataFrame(rows)


def _compare_derived(data: pd.DataFrame, derived: pd.DataFrame) -> None:
    observed = data.merge(
        derived,
        on=["fold_test_circuit", "method"],
        how="left",
        validate="one_to_one",
        suffixes=("_recorded", "_derived"),
    )
    if observed["source_partition"].isna().any() or set(observed["source_partition"]) != {
        "calibration"
    }:
        raise FigureDataError("diagnostic derivation did not remain calibration-only")
    if not np.array_equal(
        observed["calibration_false_proposals_recorded"].to_numpy(dtype=int),
        observed["calibration_false_proposals_derived"].to_numpy(dtype=int),
    ):
        raise FigureDataError(
            "recorded calibration_false_proposals does not match sealed recomputation"
        )
    tolerances = {
        "threshold": 1e-12,
        "calibration_exposure_hours": 1e-12,
        RATE_COLUMN: 1e-9,
    }
    for column, tolerance in tolerances.items():
        if not np.allclose(
            observed[f"{column}_recorded"].to_numpy(dtype=float),
            observed[f"{column}_derived"].to_numpy(dtype=float),
            atol=tolerance,
            rtol=0.0,
        ):
            raise FigureDataError(f"recorded {column} does not match sealed recomputation")

    for circuit in CIRCUIT_ORDER:
        threshold_path = ROOT / f"output/experiment/fold={circuit}/calibration-thresholds.csv"
        counts = {
            str(row.method): int(row.threshold_candidate_count)
            for row in derived.loc[
                derived["fold_test_circuit"].astype(str) == circuit
            ].itertuples(index=False)
        }
        _validate_silent_threshold_table(threshold_path, candidate_counts=counts)


def _validate(data: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    _validate_table_structure(data)
    checked_sources = _authenticate_evidence(data)
    derived = _derive_closest_active_diagnostic(data)
    _compare_derived(data, derived)
    return checked_sources, derived


def _validated_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == FROZEN_EXPERIMENT_DIR or FROZEN_EXPERIMENT_DIR in resolved.parents:
        raise FigureDataError(
            f"refusing to write inside the frozen experiment directory: {resolved}"
        )
    return resolved


def _ordered_values(data: pd.DataFrame, method: str) -> np.ndarray:
    indexed = data.loc[data["method"] == method].set_index("fold_test_circuit")
    return indexed.loc[list(CIRCUIT_ORDER), RATE_COLUMN].to_numpy(dtype=float)


def _build_figure(data: pd.DataFrame):
    brace = _ordered_values(data, METHOD_ORDER[0])
    twin = _ordered_values(data, METHOD_ORDER[1])
    figure = pf.dumbbell(
        brace,
        twin,
        category_names=CIRCUIT_ORDER,
        left_label="BRACE",
        right_label="Posterior-mean twin",
        x_label="False proposals per eligible simulated car-hour (log scale)",
        title=None,
        connector_color="0.72",
        connector_line_width=1.0,
        left_color="#0072B2",
        right_color="#D55E00",
        left_marker="o",
        right_marker="s",
        marker_size=5.8,
        row_band_alpha=0.025,
        show_x_grid=False,
        show_y_grid=False,
    )
    axis = figure.axes[0]
    axis.set_xscale("log")
    axis.set_xlim(1.0, 1200.0)
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(value) for value in ticks])
    axis.grid(axis="x", which="major", color="0.88", linewidth=0.55, zorder=-20)

    axis.axvspan(1.0, 10.0, color="#009E73", alpha=0.10, linewidth=0.0, zorder=-30)
    for budget in (2.0, 5.0, 10.0):
        axis.axvline(budget, color="#167A63", linewidth=0.75, linestyle=(0, (2, 2)), zorder=-10)
        axis.text(
            budget,
            0.965,
            f"{budget:g}/h",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.4,
            color="#145C4D",
        )
    axis.text(
        1.12,
        0.055,
        "examined gate region (≤10/h)",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#145C4D",
    )

    for row_index, (brace_rate, twin_rate) in enumerate(zip(brace, twin, strict=True)):
        axis.annotate(
            f"{brace_rate:.1f}",
            xy=(brace_rate, row_index),
            xytext=(4, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#005A8C",
        )
        axis.annotate(
            f"{twin_rate:.1f}",
            xy=(twin_rate, row_index),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#A74300",
        )

    axis.set_title("Calibration-only operating cliff", loc="left", pad=29, fontweight="bold")
    axis.text(
        0.0,
        1.045,
        "Exploratory mechanism diagnostic · primary held-out result unchanged",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="0.34",
    )
    legend = axis.get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((1.0, 1.17))
        legend.set_loc("upper right")
        legend.set_frame_on(False)
        legend.set_ncols(2)
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = _validated_output_dir(args.output_dir)
    if getattr(pf, "__version__", None) != PUBFIG_VERSION:
        raise FigureDataError(
            f"pubfig=={PUBFIG_VERSION} is required; found {getattr(pf, '__version__', 'unknown')}"
        )
    data = pd.read_csv(args.input)
    checked_sources, derived = _validate(data)
    figure = _build_figure(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / OUTPUT_STEM
    outputs = pf.batch_export(
        figure,
        base_path,
        formats=("pdf", "png"),
        spec="nature",
        width="double",
        height_mm=92,
        dpi=300,
        trim=True,
    )
    plt.close(figure)

    report = {
        "input": str(args.input.resolve()),
        "pubfig_version": pf.__version__,
        "rows": len(data),
        "recomputed_calibration_only_rows": len(derived),
        "source_artifacts_hash_checked": checked_sources,
        "outputs": [
            {
                "path": str(Path(path).resolve()),
                "bytes": Path(path).stat().st_size,
                "sha256": _sha256(Path(path)),
            }
            for path in outputs
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
