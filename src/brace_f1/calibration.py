"""Calibration fitted only on declared calibration car-sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

from brace_f1.io import DataValidationError

_EPS = 1e-9


def _validated_binary_inputs(
    raw_probability: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    probability = np.asarray(raw_probability, dtype=np.float64)
    outcome = np.asarray(labels)
    if (
        probability.ndim != 1
        or outcome.ndim != 1
        or probability.shape != outcome.shape
        or probability.size == 0
    ):
        raise DataValidationError(
            "probabilities and labels must have the same one-dimensional shape"
        )
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise DataValidationError("probabilities must be finite and in [0, 1]")
    if not np.isin(outcome, [0, 1]).all():
        raise DataValidationError("calibration labels must be binary")
    return probability, outcome.astype(np.float64)


@dataclass(frozen=True)
class MonotonePlattCalibrator:
    """Non-decreasing logistic calibration on the clipped raw log-odds."""

    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or self.slope < 0.0:
            raise DataValidationError("calibration slope must be finite and non-negative")
        if not np.isfinite(self.intercept):
            raise DataValidationError("calibration intercept must be finite")

    @classmethod
    def fit(cls, raw_probability: ArrayLike, labels: ArrayLike) -> MonotonePlattCalibrator:
        probability, outcome = _validated_binary_inputs(raw_probability, labels)
        prevalence = float((outcome.sum() + 0.5) / (outcome.size + 1.0))
        initial_intercept = float(logit(prevalence))
        if np.all(outcome == outcome[0]):
            return cls(slope=0.0, intercept=initial_intercept)

        predictor = logit(np.clip(probability, _EPS, 1.0 - _EPS))

        def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            slope, intercept = parameters
            linear = slope * predictor + intercept
            fitted = expit(linear)
            loss = float(np.logaddexp(0.0, linear).sum() - np.dot(outcome, linear))
            residual = fitted - outcome
            gradient = np.asarray([np.dot(residual, predictor), residual.sum()])
            return loss, gradient

        result = minimize(
            objective,
            x0=np.asarray([1.0, initial_intercept]),
            jac=True,
            method="L-BFGS-B",
            bounds=((0.0, 50.0), (-50.0, 50.0)),
        )
        if not result.success or not np.isfinite(result.x).all():
            raise DataValidationError(f"monotone Platt fit failed: {result.message}")
        return cls(slope=float(result.x[0]), intercept=float(result.x[1]))

    def transform(self, raw_probability: ArrayLike) -> NDArray[np.float64]:
        probability = np.asarray(raw_probability, dtype=np.float64)
        if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
            raise DataValidationError("probabilities must be finite and in [0, 1]")
        predictor = logit(np.clip(probability, _EPS, 1.0 - _EPS))
        return np.clip(expit(self.slope * predictor + self.intercept), _EPS, 1.0 - _EPS)


def calibrate_outcome_distribution(
    no_exit_probability: ArrayLike,
    outcome_probability: ArrayLike,
    calibrator: MonotonePlattCalibrator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calibrate total exit mass while retaining each non-zero spatial conditional."""

    no_exit = np.asarray(no_exit_probability, dtype=np.float64)
    outcome = np.asarray(outcome_probability, dtype=np.float64)
    if no_exit.ndim != 2 or outcome.ndim != 4 or outcome.shape[:2] != no_exit.shape:
        raise DataValidationError("forecast arrays must have shapes (N,H) and (N,H,S,K)")
    if (
        not np.isfinite(no_exit).all()
        or not np.isfinite(outcome).all()
        or np.any(no_exit < 0.0)
        or np.any(outcome < 0.0)
    ):
        raise DataValidationError("forecast probabilities must be finite and non-negative")
    raw_total = outcome.sum(axis=(-2, -1))
    if not np.allclose(no_exit + raw_total, 1.0, atol=1e-8, rtol=0.0):
        raise DataValidationError("forecast probability mass must sum to one")

    calibrated_total = calibrator.transform(raw_total)
    has_spatial_mass = raw_total > 0.0
    calibrated_total = np.where(has_spatial_mass, calibrated_total, 0.0)
    scale = np.divide(
        calibrated_total,
        raw_total,
        out=np.zeros_like(calibrated_total),
        where=has_spatial_mass,
    )
    calibrated_outcome = outcome * scale[..., None, None]
    calibrated_no_exit = 1.0 - calibrated_total
    return calibrated_no_exit, calibrated_outcome
