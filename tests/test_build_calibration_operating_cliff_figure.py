from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_calibration_operating_cliff_figure",
    ROOT / "scripts" / "build_calibration_operating_cliff_figure.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
_PREVIOUS_SOURCE_DATE_EPOCH = os.environ.get("SOURCE_DATE_EPOCH")
SPEC.loader.exec_module(MODULE)
if _PREVIOUS_SOURCE_DATE_EPOCH is None:
    os.environ.pop("SOURCE_DATE_EPOCH", None)
else:
    os.environ["SOURCE_DATE_EPOCH"] = _PREVIOUS_SOURCE_DATE_EPOCH


@pytest.fixture(scope="module")
def validated_evidence() -> tuple[pd.DataFrame, int, pd.DataFrame]:
    data = pd.read_csv(MODULE.DEFAULT_INPUT)
    checked, derived = MODULE._validate(data)
    return data, checked, derived


def test_sealed_calibration_derivation_matches_all_eight_rows(
    validated_evidence: tuple[pd.DataFrame, int, pd.DataFrame],
) -> None:
    _, checked, derived = validated_evidence
    expected_false = {
        ("Bahrain", "brace_bayesian"): 139,
        ("Bahrain", "posterior_mean_twin"): 749,
        ("Britain", "brace_bayesian"): 59,
        ("Britain", "posterior_mean_twin"): 801,
        ("Jeddah", "brace_bayesian"): 34,
        ("Jeddah", "posterior_mean_twin"): 363,
        ("Monza", "brace_bayesian"): 155,
        ("Monza", "posterior_mean_twin"): 889,
    }
    actual_false = {
        (str(row.fold_test_circuit), str(row.method)): int(row.calibration_false_proposals)
        for row in derived.itertuples(index=False)
    }
    assert checked == 17
    assert len(derived) == 8
    assert set(derived["source_partition"]) == {"calibration"}
    assert actual_false == expected_false


def test_internally_consistent_csv_count_tampering_fails_recomputation(
    validated_evidence: tuple[pd.DataFrame, int, pd.DataFrame],
) -> None:
    data, _, derived = validated_evidence
    tampered = data.copy()
    row = (tampered["fold_test_circuit"] == "Bahrain") & (
        tampered["method"] == "brace_bayesian"
    )
    tampered.loc[row, "calibration_false_proposals"] = 140
    tampered.loc[row, MODULE.RATE_COLUMN] = (
        140 / tampered.loc[row, "calibration_exposure_hours"]
    )
    MODULE._validate_table_structure(tampered)
    with pytest.raises(MODULE.FigureDataError, match="does not match sealed recomputation"):
        MODULE._compare_derived(tampered, derived)


def test_csv_digest_must_be_cross_bound_to_fixed_seal(monkeypatch: pytest.MonkeyPatch) -> None:
    data = pd.read_csv(MODULE.DEFAULT_INPUT)
    data.loc[data["fold_test_circuit"] == "Bahrain", "fold_manifest_sha256"] = "0" * 64
    monkeypatch.setattr(MODULE, "_validate_source_hashes", lambda _: set())
    with pytest.raises(MODULE.FigureDataError, match="not cross-bound to the seal"):
        MODULE._authenticate_evidence(data)


def test_output_directory_rejects_frozen_tree_and_symlink_descendant(tmp_path: Path) -> None:
    with pytest.raises(MODULE.FigureDataError, match="refusing to write"):
        MODULE._validated_output_dir(MODULE.FROZEN_EXPERIMENT_DIR / "figures")

    alias = tmp_path / "frozen-experiment-alias"
    alias.symlink_to(MODULE.FROZEN_EXPERIMENT_DIR, target_is_directory=True)
    with pytest.raises(MODULE.FigureDataError, match="refusing to write"):
        MODULE._validated_output_dir(alias / "figures")


def test_safe_paper_output_directory_is_accepted() -> None:
    assert MODULE._validated_output_dir(MODULE.DEFAULT_OUTPUT_DIR) == (
        MODULE.DEFAULT_OUTPUT_DIR.resolve()
    )
