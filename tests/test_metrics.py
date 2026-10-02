from __future__ import annotations

import numpy as np
import pytest

from brace_f1.io import DataValidationError
from brace_f1.metrics import (
    calibration_intercept_slope,
    expected_calibration_error,
    false_proposal_rate_upper,
    probability_metrics,
)


def test_probability_metrics_match_hand_calculation_and_external_reference() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    probability = np.asarray([0.10, 0.20, 0.70, 0.90])

    got = probability_metrics(labels, probability, reference_prevalence=0.25)

    expected_brier = float(np.mean((probability - labels) ** 2))
    expected_reference = float(np.mean((0.25 - labels) ** 2))
    assert got["brier_score"] == pytest.approx(expected_brier)
    assert got["brier_skill_score"] == pytest.approx(1.0 - expected_brier / expected_reference)
    assert got["log_score"] == pytest.approx(
        -np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
    )
    assert got["n"] == 4
    assert got["event_count"] == 2


def test_expected_calibration_error_uses_declared_equal_width_bins() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    probability = np.asarray([0.10, 0.20, 0.70, 0.90])

    result = expected_calibration_error(labels, probability, n_bins=2)

    assert result.ece == pytest.approx(0.175)
    assert result.counts.tolist() == [2, 2]
    np.testing.assert_allclose(result.mean_probability, [0.15, 0.80])
    np.testing.assert_allclose(result.observed_frequency, [0.0, 1.0])


def test_calibration_regression_recovers_a_positive_finite_slope() -> None:
    probability = np.asarray([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    intercept, slope = calibration_intercept_slope(labels, probability)

    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert slope > 0.0


def test_false_proposal_rate_upper_is_exact_poisson_and_monotone() -> None:
    zero = false_proposal_rate_upper(0, exposure_hours=10.0, confidence=0.95)
    one = false_proposal_rate_upper(1, exposure_hours=10.0, confidence=0.95)

    assert zero == pytest.approx(-np.log(0.05) / 10.0)
    assert one > zero


def test_metrics_reject_invalid_shapes_probabilities_and_exposure() -> None:
    with pytest.raises(DataValidationError, match="same one-dimensional shape"):
        probability_metrics(np.asarray([0, 1]), np.asarray([0.2]), 0.5)
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        probability_metrics(np.asarray([0]), np.asarray([-0.1]), 0.5)
    with pytest.raises(DataValidationError, match="positive"):
        false_proposal_rate_upper(1, exposure_hours=0.0)
