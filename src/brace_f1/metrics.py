"""Predeclared probability and false-proposal metrics for BRACE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import chi2

from brace_f1.io import DataValidationError

_EPS = 1e-12


def _validated_probability_inputs(
    labels: ArrayLike, probability: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    outcome = np.asarray(labels)
    forecast = np.asarray(probability, dtype=np.float64)
    if (
        outcome.ndim != 1
        or forecast.ndim != 1
        or outcome.shape != forecast.shape
        or outcome.size == 0
    ):
        raise DataValidationError(
            "labels and probabilities must have the same one-dimensional shape"
        )
    if not np.isin(outcome, [0, 1]).all():
        raise DataValidationError("labels must be binary")
    if not np.isfinite(forecast).all() or np.any((forecast < 0.0) | (forecast > 1.0)):
        raise DataValidationError("probabilities must be finite and in [0, 1]")
    return outcome.astype(np.float64), forecast


def probability_metrics(
    labels: ArrayLike, probability: ArrayLike, reference_prevalence: float
) -> dict[str, float | int]:
    """Return proper scores using a calibration-prevalence constant reference."""

    outcome, forecast = _validated_probability_inputs(labels, probability)
    if not np.isfinite(reference_prevalence) or not 0.0 <= reference_prevalence <= 1.0:
        raise DataValidationError("reference prevalence must be finite and in [0, 1]")
    brier = float(np.mean((forecast - outcome) ** 2))
    reference_brier = float(np.mean((reference_prevalence - outcome) ** 2))
    brier_skill = float("nan") if reference_brier <= 0.0 else 1.0 - brier / reference_brier
    clipped = np.clip(forecast, _EPS, 1.0 - _EPS)
    log_score = float(-np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped)))
    return {
        "n": int(outcome.size),
        "event_count": int(outcome.sum()),
        "brier_score": brier,
        "reference_brier_score": reference_brier,
        "brier_skill_score": brier_skill,
        "log_score": log_score,
    }


@dataclass(frozen=True)
class ReliabilityTable:
    """Equal-width reliability-bin summary."""

    ece: float
    counts: NDArray[np.int64]
    mean_probability: NDArray[np.float64]
    observed_frequency: NDArray[np.float64]
    bin_edges: NDArray[np.float64]


def expected_calibration_error(
    labels: ArrayLike, probability: ArrayLike, *, n_bins: int = 10
) -> ReliabilityTable:
    outcome, forecast = _validated_probability_inputs(labels, probability)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise DataValidationError("n_bins must be an integer of at least two")
    indices = np.minimum((forecast * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    mean_probability = np.full(n_bins, np.nan, dtype=np.float64)
    observed_frequency = np.full(n_bins, np.nan, dtype=np.float64)
    for bin_index in range(n_bins):
        selected = indices == bin_index
        if np.any(selected):
            mean_probability[bin_index] = float(forecast[selected].mean())
            observed_frequency[bin_index] = float(outcome[selected].mean())
    occupied = counts > 0
    ece = float(
        np.sum(
            counts[occupied]
            / outcome.size
            * np.abs(mean_probability[occupied] - observed_frequency[occupied])
        )
    )
    return ReliabilityTable(
        ece=ece,
        counts=counts,
        mean_probability=mean_probability,
        observed_frequency=observed_frequency,
        bin_edges=np.linspace(0.0, 1.0, n_bins + 1),
    )


def calibration_intercept_slope(labels: ArrayLike, probability: ArrayLike) -> tuple[float, float]:
    """Fit the standard logistic calibration intercept and slope."""

    outcome, forecast = _validated_probability_inputs(labels, probability)
    if np.all(outcome == outcome[0]):
        return float("nan"), float("nan")
    predictor = logit(np.clip(forecast, 1e-9, 1.0 - 1e-9))

    def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        intercept, slope = parameters
        linear = intercept + slope * predictor
        fitted = expit(linear)
        loss = float(np.logaddexp(0.0, linear).sum() - np.dot(outcome, linear))
        residual = fitted - outcome
        return loss, np.asarray([residual.sum(), np.dot(residual, predictor)])

    result = minimize(
        objective,
        x0=np.asarray([0.0, 1.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=((-50.0, 50.0), (-50.0, 50.0)),
    )
    if not result.success or not np.isfinite(result.x).all():
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


def false_proposal_rate_upper(
    false_proposals: int, *, exposure_hours: float, confidence: float = 0.95
) -> float:
    """Exact one-sided Poisson upper confidence limit per exposure hour."""

    if not isinstance(false_proposals, (int, np.integer)) or false_proposals < 0:
        raise DataValidationError("false_proposals must be a non-negative integer")
    if not np.isfinite(exposure_hours) or exposure_hours <= 0.0:
        raise DataValidationError("exposure_hours must be finite and positive")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise DataValidationError("confidence must lie strictly between zero and one")
    upper_count = 0.5 * chi2.ppf(confidence, 2.0 * (false_proposals + 1))
    return float(upper_count / exposure_hours)
