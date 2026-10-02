from __future__ import annotations

import numpy as np
import pytest

from brace_f1.calibration import (
    MonotonePlattCalibrator,
    calibrate_outcome_distribution,
)
from brace_f1.io import DataValidationError


def test_monotone_platt_transform_is_bounded_and_non_decreasing() -> None:
    raw = np.asarray([0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.95])
    labels = np.asarray([0, 0, 0, 1, 0, 1, 1])

    calibrator = MonotonePlattCalibrator.fit(raw, labels)
    transformed = calibrator.transform(raw)

    assert calibrator.slope >= 0.0
    assert np.all((transformed > 0.0) & (transformed < 1.0))
    assert np.all(np.diff(transformed) >= -1e-12)


@pytest.mark.parametrize("label", [0, 1])
def test_degenerate_calibration_labels_produce_a_finite_constant(label: int) -> None:
    raw = np.asarray([0.05, 0.20, 0.80, 0.95])
    labels = np.full(raw.shape, label, dtype=int)

    calibrator = MonotonePlattCalibrator.fit(raw, labels)
    transformed = calibrator.transform(raw)

    assert calibrator.slope == 0.0
    assert np.isfinite(calibrator.intercept)
    assert np.all((transformed > 0.0) & (transformed < 1.0))
    np.testing.assert_allclose(transformed, transformed[0])


def test_distribution_calibration_preserves_spatial_conditionals_and_mass() -> None:
    no_exit = np.asarray([[0.80, 0.40], [1.00, 0.25]])
    outcome = np.zeros((2, 2, 3, 4), dtype=float)
    outcome[0, 0, 0, 0] = 0.05
    outcome[0, 0, 0, 1] = 0.15
    outcome[0, 1, 1, 2] = 0.60
    outcome[1, 1, 0, 0] = 0.25
    outcome[1, 1, 1, 3] = 0.50
    calibrator = MonotonePlattCalibrator(slope=0.5, intercept=-0.2)

    calibrated_no_exit, calibrated_outcome = calibrate_outcome_distribution(
        no_exit, outcome, calibrator
    )

    np.testing.assert_allclose(
        calibrated_no_exit + calibrated_outcome.sum(axis=(-2, -1)), 1.0, atol=1e-12
    )
    original_positive = outcome[0, 0] / outcome[0, 0].sum()
    calibrated_positive = calibrated_outcome[0, 0] / calibrated_outcome[0, 0].sum()
    np.testing.assert_allclose(calibrated_positive, original_positive, atol=1e-12)
    assert calibrated_outcome[1, 0].sum() == 0.0


def test_calibrator_rejects_non_probability_or_mismatched_inputs() -> None:
    with pytest.raises(DataValidationError, match="same one-dimensional shape"):
        MonotonePlattCalibrator.fit(np.asarray([0.2, 0.3]), np.asarray([0]))
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        MonotonePlattCalibrator.fit(np.asarray([1.2]), np.asarray([1]))
    with pytest.raises(DataValidationError, match="binary"):
        MonotonePlattCalibrator.fit(np.asarray([0.2]), np.asarray([2]))
