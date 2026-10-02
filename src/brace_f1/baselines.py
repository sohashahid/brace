"""Registered kinematic, deterministic-residual, and side-hazard baselines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from brace_f1.forecast import ForecastProbabilities, forecast_ctra_particles
from brace_f1.geometry import CircuitCorridor
from brace_f1.io import DataValidationError
from brace_f1.residual import DeterministicResidualDynamics

HAZARD_CLASS_ORDER: tuple[str, ...] = ("no_exit", "left", "right")


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    output = np.asarray(values, dtype=np.float64)
    if output.ndim == 1:
        output = output[:, None]
    if output.ndim != 2 or output.shape[0] == 0 or output.shape[1] == 0:
        raise DataValidationError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(output).all():
        raise DataValidationError(f"{name} contains non-finite values")
    return output


def _content_hash(metadata: dict[str, object], *arrays: NDArray[np.float64]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def forecast_constant_velocity(
    initial_states: ArrayLike,
    corridor: CircuitCorridor,
    **forecast_kwargs: object,
) -> ForecastProbabilities:
    """Forecast a fixed global velocity (zero turn rate and acceleration)."""

    state = np.asarray(initial_states, dtype=np.float64).copy()
    if state.ndim != 2 or state.shape[1] != 8:
        raise DataValidationError("initial_states must have shape (N, 8)")
    state[:, 5:] = 0.0
    return forecast_ctra_particles(state, corridor, **forecast_kwargs)


def forecast_constant_turn_rate(
    initial_states: ArrayLike,
    corridor: CircuitCorridor,
    **forecast_kwargs: object,
) -> ForecastProbabilities:
    """Forecast constant body velocity and yaw rate (CTRV)."""

    state = np.asarray(initial_states, dtype=np.float64).copy()
    if state.ndim != 2 or state.shape[1] != 8:
        raise DataValidationError("initial_states must have shape (N, 8)")
    # These body accelerations exactly cancel the rotating-frame terms in the
    # CTRA integrator, keeping v_long and v_lat constant while yaw advances.
    state[:, 6] = -state[:, 5] * state[:, 3]
    state[:, 7] = state[:, 5] * state[:, 2]
    return forecast_ctra_particles(state, corridor, **forecast_kwargs)


@dataclass(frozen=True)
class TunedDeterministicResidualDynamics:
    """Ridge residual dynamics with alpha selected on calibration transitions."""

    model: DeterministicResidualDynamics
    selected_alpha: float
    validation_mse_by_alpha: Mapping[float, float]
    content_hash: str

    def __post_init__(self) -> None:
        scores = {
            float(alpha): float(score) for alpha, score in self.validation_mse_by_alpha.items()
        }
        if (
            not scores
            or self.selected_alpha not in scores
            or any(
                not np.isfinite(alpha) or not np.isfinite(score) for alpha, score in scores.items()
            )
        ):
            raise DataValidationError("validation tuning scores are invalid")
        object.__setattr__(self, "validation_mse_by_alpha", MappingProxyType(scores))

    @classmethod
    def fit(
        cls,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        X_validation: ArrayLike,
        Y_validation: ArrayLike,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        alpha_grid: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0),
    ) -> TunedDeterministicResidualDynamics:
        design = _matrix(X, "X")
        target = _matrix(Y, "Y")
        validation_design = _matrix(X_validation, "X_validation")
        validation_target = _matrix(Y_validation, "Y_validation")
        if target.shape[0] != design.shape[0]:
            raise DataValidationError("X and Y row counts differ")
        if validation_target.shape[0] != validation_design.shape[0]:
            raise DataValidationError("validation row counts differ")
        if (
            validation_design.shape[1] != design.shape[1]
            or validation_target.shape[1] != target.shape[1]
        ):
            raise DataValidationError("fit and validation matrix dimensions differ")
        grid = tuple(float(alpha) for alpha in alpha_grid)
        if (
            not grid
            or len(set(grid)) != len(grid)
            or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in grid)
        ):
            raise DataValidationError("alpha_grid must contain unique finite non-negative values")
        scores: dict[float, float] = {}
        candidates: dict[float, DeterministicResidualDynamics] = {}
        for alpha in grid:
            candidate = DeterministicResidualDynamics.fit(
                design,
                target,
                feature_names=feature_names,
                target_names=target_names,
                ridge_alpha=alpha,
            )
            error = validation_target - candidate.predict(validation_design)
            scores[alpha] = float(np.mean(error**2))
            candidates[alpha] = candidate
        selected = min(grid, key=lambda alpha: (scores[alpha], alpha))
        model = candidates[selected]
        metadata = {
            "kind": "validation_tuned_deterministic_residual",
            "selected_alpha": selected,
            "alpha_grid": grid,
            "validation_mse_by_alpha": scores,
            "model_hash": model.content_hash,
        }
        return cls(model, selected, scores, _content_hash(metadata))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.model.feature_names

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.model.target_names

    @property
    def process_covariance(self) -> NDArray[np.float64]:
        return self.model.process_covariance

    @property
    def coefficients(self) -> NDArray[np.float64]:
        return self.model.coefficients

    @property
    def feature_center(self) -> NDArray[np.float64]:
        return self.model.feature_center

    @property
    def feature_scale(self) -> NDArray[np.float64]:
        return self.model.feature_scale

    def transform_design(self, X: ArrayLike) -> NDArray[np.float64]:
        return self.model.transform_design(X)

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        return self.model.predict(X)


@dataclass(frozen=True)
class RegularizedSideHazard:
    """L2-regularized discrete-time competing left/right exit hazard."""

    feature_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    regularization: float
    content_hash: str

    def __post_init__(self) -> None:
        coefficients = np.array(self.coefficients, dtype=np.float64, copy=True, order="C")
        if (
            coefficients.shape != (len(self.feature_names) + 1, 2)
            or not np.isfinite(coefficients).all()
        ):
            raise DataValidationError("hazard coefficients have invalid shape or values")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)

    @classmethod
    def fit(
        cls,
        X: ArrayLike,
        labels: ArrayLike,
        *,
        feature_names: Sequence[str],
        regularization: float = 1.0,
        max_iterations: int = 500,
    ) -> RegularizedSideHazard:
        design = _matrix(X, "X")
        names = tuple(str(name) for name in feature_names)
        if len(names) != design.shape[1] or len(set(names)) != len(names):
            raise DataValidationError("feature_names do not match hazard design")
        label_array = np.asarray(labels, dtype=np.str_)
        if label_array.shape != (design.shape[0],):
            raise DataValidationError("hazard labels and design rows differ")
        invalid = sorted(set(label_array).difference(HAZARD_CLASS_ORDER))
        if invalid:
            raise DataValidationError(f"hazard labels contain unregistered values: {invalid}")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise DataValidationError("regularization must be finite and non-negative")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise DataValidationError("max_iterations must be a positive integer")
        augmented = np.column_stack((np.ones(design.shape[0]), design))
        class_index = np.asarray(
            [HAZARD_CLASS_ORDER.index(label) for label in label_array], dtype=np.int64
        )

        def objective(flat: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            coefficients = flat.reshape(augmented.shape[1], 2)
            logits = np.column_stack((np.zeros(design.shape[0]), augmented @ coefficients))
            logits -= logits.max(axis=1, keepdims=True)
            exponentiated = np.exp(logits)
            probability = exponentiated / exponentiated.sum(axis=1, keepdims=True)
            loss = -float(np.log(probability[np.arange(design.shape[0]), class_index]).mean())
            loss += 0.5 * regularization * float(np.sum(coefficients[1:] ** 2))
            indicator = np.zeros_like(probability)
            indicator[np.arange(design.shape[0]), class_index] = 1.0
            gradient = augmented.T @ (probability[:, 1:] - indicator[:, 1:]) / design.shape[0]
            gradient[1:] += regularization * coefficients[1:]
            return loss, gradient.ravel()

        result = minimize(
            objective,
            np.zeros(augmented.shape[1] * 2, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-9},
        )
        if not result.success:
            raise DataValidationError(f"hazard optimization failed: {result.message}")
        coefficients = np.asarray(result.x, dtype=np.float64).reshape(augmented.shape[1], 2)
        metadata = {
            "kind": "regularized_multinomial_side_hazard",
            "feature_names": names,
            "class_order": HAZARD_CLASS_ORDER,
            "regularization": float(regularization),
            "max_iterations": max_iterations,
        }
        return cls(
            names,
            coefficients,
            float(regularization),
            _content_hash(metadata, coefficients),
        )

    def predict_step_probabilities(self, X: ArrayLike) -> NDArray[np.float64]:
        design = _matrix(X, "X")
        if design.shape[1] != len(self.feature_names):
            raise DataValidationError("hazard X feature count does not match model")
        augmented = np.column_stack((np.ones(design.shape[0]), design))
        logits = np.column_stack((np.zeros(design.shape[0]), augmented @ self.coefficients))
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def predict_horizon_probabilities(
        self,
        X: ArrayLike,
        *,
        horizons_s: Sequence[float],
        dt_s: float,
    ) -> NDArray[np.float64]:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise DataValidationError("dt_s must be positive and finite")
        horizons = np.asarray(tuple(horizons_s), dtype=np.float64)
        if horizons.ndim != 1 or horizons.size == 0 or np.any(horizons <= 0.0):
            raise DataValidationError("hazard horizons must be positive")
        steps = np.rint(horizons / dt_s).astype(np.int64)
        if not np.allclose(steps * dt_s, horizons, rtol=0.0, atol=1e-10):
            raise DataValidationError("hazard horizons must be integer multiples of dt_s")
        step_probability = self.predict_step_probabilities(X)
        output = np.empty((step_probability.shape[0], horizons.size, 3), dtype=np.float64)
        step_exit = step_probability[:, 1:].sum(axis=1)
        side_share = np.divide(
            step_probability[:, 1:],
            step_exit[:, None],
            out=np.zeros_like(step_probability[:, 1:]),
            where=step_exit[:, None] > 0.0,
        )
        for index, step_count in enumerate(steps):
            no_exit = step_probability[:, 0] ** step_count
            output[:, index, 0] = no_exit
            output[:, index, 1:] = (1.0 - no_exit)[:, None] * side_share
        return output
