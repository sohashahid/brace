"""Cluster-aware Bayesian-bootstrap residual dynamics for BRACE.

The model learns one-step transition residuals only.  It never imports or
inspects the future-excursion target table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from brace_f1.io import DataValidationError

RESIDUAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "body_speed_longitudinal_mps",
    "body_speed_lateral_mps",
    "yaw_rate_radps",
    "body_acceleration_longitudinal_mps2",
    "body_acceleration_lateral_mps2",
    "heading_error_rad",
    "track_offset_m",
    "track_curvature_per_m",
    "transition_dt_s",
)

RESIDUAL_TARGET_NAMES: tuple[str, ...] = (
    "delta_v_longitudinal_mps",
    "delta_v_lateral_mps",
    "delta_yaw_rate_radps",
    "delta_acceleration_longitudinal_mps2",
    "delta_acceleration_lateral_mps2",
)

_PAIR_INPUT_COLUMNS = {
    "circuit",
    "car_id",
    "frame_index",
    "time_seconds",
    "continuous_segment_id",
    "input_valid_causal",
    "in_corridor",
    "body_speed_longitudinal_mps",
    "body_speed_lateral_mps",
    "yaw_rate_radps",
    "body_acceleration_longitudinal_mps2",
    "body_acceleration_lateral_mps2",
    "heading_error_rad",
    "track_offset_m",
    "track_curvature_per_m",
}


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise DataValidationError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise DataValidationError(f"{name} contains non-finite values")
    return array


def _names(values: Sequence[str], count: int, name: str) -> tuple[str, ...]:
    output = tuple(str(value) for value in values)
    if len(output) != count or len(set(output)) != count:
        raise DataValidationError(f"{name} must contain {count} unique names")
    return output


def _hash_arrays(metadata: dict[str, object], *arrays: NDArray[np.float64]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _readonly_float_copy(values: ArrayLike) -> NDArray[np.float64]:
    output = np.array(values, dtype=np.float64, copy=True, order="C")
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class ResidualTrainingSet:
    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    circuits: NDArray[np.str_]
    cars: NDArray[np.str_]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    row_keys: pd.DataFrame


def residual_design_from_features(
    features: pd.DataFrame, *, transition_dt_s: float | ArrayLike
) -> NDArray[np.float64]:
    """Select the registered residual-design order without touching outcome columns."""

    dynamic_columns = RESIDUAL_FEATURE_COLUMNS[:-1]
    missing = set(dynamic_columns).difference(features.columns)
    if missing:
        raise DataValidationError(f"residual design input missing columns: {sorted(missing)}")
    design = features.loc[:, dynamic_columns].to_numpy(dtype=np.float64)
    dt = np.asarray(transition_dt_s, dtype=np.float64)
    if dt.ndim == 0:
        dt = np.full(features.shape[0], float(dt), dtype=np.float64)
    if dt.shape != (features.shape[0],) or not np.isfinite(dt).all() or np.any(dt <= 0.0):
        raise DataValidationError("transition_dt_s must be positive and aligned with feature rows")
    if not np.isfinite(design).all():
        raise DataValidationError("residual design features contain non-finite values")
    return np.column_stack((design, dt))


def body_frame_ctra_midpoint_velocity_step(
    v_longitudinal: ArrayLike,
    v_lateral: ArrayLike,
    yaw_rate: ArrayLike,
    acceleration_longitudinal: ArrayLike,
    acceleration_lateral: ArrayLike,
    dt_s: float | ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Advance body velocity with the explicit midpoint CTRA transition."""

    arrays = np.broadcast_arrays(
        np.asarray(v_longitudinal, dtype=np.float64),
        np.asarray(v_lateral, dtype=np.float64),
        np.asarray(yaw_rate, dtype=np.float64),
        np.asarray(acceleration_longitudinal, dtype=np.float64),
        np.asarray(acceleration_lateral, dtype=np.float64),
        np.asarray(dt_s, dtype=np.float64),
    )
    v_long, v_lat, rate, a_long, a_lat, dt = arrays
    if not all(np.isfinite(array).all() for array in arrays) or np.any(dt <= 0.0):
        raise DataValidationError("midpoint transition inputs must be finite with positive dt")
    derivative_long = a_long + rate * v_lat
    derivative_lateral = a_lat - rate * v_long
    midpoint_long = v_long + 0.5 * derivative_long * dt
    midpoint_lateral = v_lat + 0.5 * derivative_lateral * dt
    next_long = v_long + (a_long + rate * midpoint_lateral) * dt
    next_lateral = v_lat + (a_lat - rate * midpoint_long) * dt
    return next_long, next_lateral, midpoint_long, midpoint_lateral


def _fit_robust_scaling(
    design: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = np.median(design, axis=0)
    quartiles = np.percentile(design, [25.0, 75.0], axis=0)
    scale = quartiles[1] - quartiles[0]
    scale = np.where(scale > 0.0, scale, 1.0)
    return np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def _transform_with_scaling(
    values: ArrayLike,
    *,
    feature_count: int,
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    design = np.asarray(values, dtype=np.float64)
    if design.ndim < 2 or design.shape[-1] != feature_count or design.size == 0:
        raise DataValidationError(f"X feature count must match model feature count {feature_count}")
    if not np.isfinite(design).all():
        raise DataValidationError("X contains non-finite values")
    return (design - center) / scale


def build_residual_training_pairs(features: pd.DataFrame) -> ResidualTrainingSet:
    """Build valid consecutive one-step residual labels within causal segments."""

    missing = _PAIR_INPUT_COLUMNS.difference(features.columns)
    if missing:
        raise DataValidationError(f"residual pair input missing columns: {sorted(missing)}")
    if features.empty:
        raise DataValidationError("residual pair input is empty")
    rows: list[list[float]] = []
    targets: list[list[float]] = []
    circuits: list[str] = []
    cars: list[str] = []
    keys: list[dict[str, object]] = []
    for (circuit, car, segment), group in features.groupby(
        ["circuit", "car_id", "continuous_segment_id"], sort=False, dropna=False
    ):
        group = group.sort_values("time_seconds", kind="stable")
        if len(group) < 2:
            continue
        source = group.iloc[:-1]
        target = group.iloc[1:]
        valid = source["input_valid_causal"].to_numpy(dtype=bool) & target[
            "input_valid_causal"
        ].to_numpy(dtype=bool)
        valid &= source["in_corridor"].to_numpy(dtype=bool) & target["in_corridor"].to_numpy(
            dtype=bool
        )
        dt = target["time_seconds"].to_numpy(dtype=np.float64) - source["time_seconds"].to_numpy(
            dtype=np.float64
        )
        valid &= np.isfinite(dt) & (dt > 0.0)
        for offset in np.flatnonzero(valid):
            current = source.iloc[int(offset)]
            following = target.iloc[int(offset)]
            step = float(dt[int(offset)])
            v_long = float(current["body_speed_longitudinal_mps"])
            v_lat = float(current["body_speed_lateral_mps"])
            yaw_rate = float(current["yaw_rate_radps"])
            a_long = float(current["body_acceleration_longitudinal_mps2"])
            a_lat = float(current["body_acceleration_lateral_mps2"])
            predicted_long, predicted_lateral, _, _ = body_frame_ctra_midpoint_velocity_step(
                v_long,
                v_lat,
                yaw_rate,
                a_long,
                a_lat,
                step,
            )
            predicted = np.asarray(
                [
                    float(predicted_long),
                    float(predicted_lateral),
                    yaw_rate,
                    a_long,
                    a_lat,
                ],
                dtype=np.float64,
            )
            observed = np.asarray(
                [
                    following["body_speed_longitudinal_mps"],
                    following["body_speed_lateral_mps"],
                    following["yaw_rate_radps"],
                    following["body_acceleration_longitudinal_mps2"],
                    following["body_acceleration_lateral_mps2"],
                ],
                dtype=np.float64,
            )
            design = [float(current[column]) for column in RESIDUAL_FEATURE_COLUMNS[:-1]]
            design.append(step)
            if not np.isfinite(design).all() or not np.isfinite(observed).all():
                continue
            rows.append(design)
            targets.append((observed - predicted).tolist())
            circuits.append(str(circuit))
            cars.append(str(car))
            keys.append(
                {
                    "circuit": str(circuit),
                    "car_id": str(car),
                    "continuous_segment_id": int(segment),
                    "source_frame_index": int(current["frame_index"]),
                    "target_frame_index": int(following["frame_index"]),
                    "transition_dt_s": step,
                }
            )
    if not rows:
        raise DataValidationError("residual pair input contains no valid consecutive transitions")
    return ResidualTrainingSet(
        X=np.asarray(rows, dtype=np.float64),
        Y=np.asarray(targets, dtype=np.float64),
        circuits=np.asarray(circuits, dtype=np.str_),
        cars=np.asarray(cars, dtype=np.str_),
        feature_names=RESIDUAL_FEATURE_COLUMNS,
        target_names=RESIDUAL_TARGET_NAMES,
        row_keys=pd.DataFrame(keys),
    )


def cluster_bayesian_bootstrap_weights(
    circuits: ArrayLike, cars: ArrayLike, *, n_draws: int, seed: int
) -> NDArray[np.float64]:
    """Draw two-stage circuit/car Bayesian-bootstrap row weights.

    A car receives a draw weight independent of its frame count.  Each row then
    receives an equal share of that car's weight, preventing long streams from
    silently becoming independent replicates.
    """

    circuit_array = np.asarray(circuits, dtype=np.str_)
    car_array = np.asarray(cars, dtype=np.str_)
    if circuit_array.ndim != 1 or car_array.shape != circuit_array.shape or not circuit_array.size:
        raise DataValidationError("circuits and cars must be non-empty aligned vectors")
    if not isinstance(n_draws, int) or n_draws <= 0:
        raise DataValidationError("n_draws must be a positive integer")
    circuit_names = np.unique(circuit_array)
    cluster_keys = np.asarray(
        [f"{circuit}\x1f{car}" for circuit, car in zip(circuit_array, car_array, strict=True)],
        dtype=np.str_,
    )
    unique_clusters = np.unique(cluster_keys)
    cluster_count = unique_clusters.size
    rng = np.random.default_rng(seed)
    output = np.empty((n_draws, circuit_array.size), dtype=np.float64)
    for draw in range(n_draws):
        circuit_weight = rng.dirichlet(np.ones(circuit_names.size))
        output[draw] = 0.0
        for circuit_index, circuit in enumerate(circuit_names):
            circuit_rows = np.flatnonzero(circuit_array == circuit)
            circuit_cars = np.unique(car_array[circuit_rows])
            car_weight = rng.dirichlet(np.ones(circuit_cars.size))
            for car_index, car in enumerate(circuit_cars):
                rows = np.flatnonzero((circuit_array == circuit) & (car_array == car))
                output[draw, rows] = (
                    cluster_count
                    * circuit_weight[circuit_index]
                    * car_weight[car_index]
                    / rows.size
                )
    return output


@dataclass(frozen=True)
class _ClusterSufficientStatistics:
    circuit_names: NDArray[np.str_]
    cluster_circuit: NDArray[np.str_]
    cluster_car: NDArray[np.str_]
    gram: NDArray[np.float64]
    cross: NDArray[np.float64]
    target_second_moment: NDArray[np.float64]
    fit_row_count: int

    @property
    def cluster_count(self) -> int:
        return int(self.cluster_circuit.size)


def _cluster_sufficient_statistics(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    circuits: ArrayLike,
    cars: ArrayLike,
) -> _ClusterSufficientStatistics:
    circuit_array = np.asarray(circuits, dtype=np.str_)
    car_array = np.asarray(cars, dtype=np.str_)
    if (
        circuit_array.shape != (X.shape[0],)
        or car_array.shape != circuit_array.shape
        or circuit_array.size == 0
    ):
        raise DataValidationError("cluster labels and training rows differ")
    augmented = np.column_stack((np.ones(X.shape[0]), X))
    circuit_names = np.unique(circuit_array)
    cluster_circuit: list[str] = []
    cluster_car: list[str] = []
    gram: list[NDArray[np.float64]] = []
    cross: list[NDArray[np.float64]] = []
    target_second: list[NDArray[np.float64]] = []
    for circuit in circuit_names:
        circuit_rows = np.flatnonzero(circuit_array == circuit)
        for car in np.unique(car_array[circuit_rows]):
            rows = np.flatnonzero((circuit_array == circuit) & (car_array == car))
            car_design = augmented[rows]
            car_target = Y[rows]
            divisor = float(rows.size)
            cluster_circuit.append(str(circuit))
            cluster_car.append(str(car))
            gram.append(car_design.T @ car_design / divisor)
            cross.append(car_design.T @ car_target / divisor)
            target_second.append(car_target.T @ car_target / divisor)
    return _ClusterSufficientStatistics(
        circuit_names=np.asarray(circuit_names, dtype=np.str_),
        cluster_circuit=np.asarray(cluster_circuit, dtype=np.str_),
        cluster_car=np.asarray(cluster_car, dtype=np.str_),
        gram=np.stack(gram),
        cross=np.stack(cross),
        target_second_moment=np.stack(target_second),
        fit_row_count=int(X.shape[0]),
    )


def _draw_cluster_weights(
    statistics: _ClusterSufficientStatistics, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return normalized circuit→car Dirichlet weights over car clusters."""

    circuit_weight = rng.dirichlet(np.ones(statistics.circuit_names.size))
    output = np.zeros(statistics.cluster_count, dtype=np.float64)
    for circuit_index, circuit in enumerate(statistics.circuit_names):
        clusters = np.flatnonzero(statistics.cluster_circuit == circuit)
        car_weight = rng.dirichlet(np.ones(clusters.size))
        output[clusters] = circuit_weight[circuit_index] * car_weight
    return output


def _ridge_from_cluster_statistics(
    statistics: _ClusterSufficientStatistics,
    cluster_weight: NDArray[np.float64],
    *,
    ridge_alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    gram = np.einsum("m,mij->ij", cluster_weight, statistics.gram)
    cross = np.einsum("m,mij->ij", cluster_weight, statistics.cross)
    target_second = np.einsum("m,mij->ij", cluster_weight, statistics.target_second_moment)
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    normal = statistics.cluster_count * gram + penalty
    rhs = statistics.cluster_count * cross
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    covariance = (
        target_second
        - cross.T @ coefficients
        - coefficients.T @ cross
        + coefficients.T @ gram @ coefficients
    )
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    covariance += np.eye(covariance.shape[0], dtype=np.float64) * 1e-9
    return coefficients, covariance


def _ridge_solution(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    *,
    ridge_alpha: float,
    sample_weight: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    augmented = np.column_stack((np.ones(X.shape[0]), X))
    weighted = augmented * sample_weight[:, None]
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    normal = augmented.T @ weighted + penalty
    rhs = augmented.T @ (Y * sample_weight[:, None])
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    residual = Y - augmented @ coefficients
    covariance = (residual * sample_weight[:, None]).T @ residual / sample_weight.sum()
    covariance = (covariance + covariance.T) * 0.5
    covariance += np.eye(Y.shape[1], dtype=np.float64) * 1e-9
    return coefficients, covariance


@dataclass(frozen=True)
class DeterministicResidualDynamics:
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    process_covariance: NDArray[np.float64]
    ridge_alpha: float
    feature_center: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    content_hash: str

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        target_count = len(self.target_names)
        arrays = {
            "coefficients": self.coefficients,
            "process_covariance": self.process_covariance,
            "feature_center": self.feature_center,
            "feature_scale": self.feature_scale,
        }
        expected_shapes = {
            "coefficients": (feature_count + 1, target_count),
            "process_covariance": (target_count, target_count),
            "feature_center": (feature_count,),
            "feature_scale": (feature_count,),
        }
        for name, values in arrays.items():
            copied = _readonly_float_copy(values)
            if copied.shape != expected_shapes[name] or not np.isfinite(copied).all():
                raise DataValidationError(f"{name} has invalid shape or values")
            if name == "feature_scale" and np.any(copied <= 0.0):
                raise DataValidationError("feature_scale must be strictly positive")
            object.__setattr__(self, name, copied)

    @classmethod
    def fit(
        cls,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        ridge_alpha: float = 1.0,
        sample_weight: ArrayLike | None = None,
    ) -> DeterministicResidualDynamics:
        design = _matrix(X, "X")
        target = _matrix(Y, "Y")
        if target.shape[0] != design.shape[0]:
            raise DataValidationError("X and Y row counts differ")
        features = _names(feature_names, design.shape[1], "feature_names")
        targets = _names(target_names, target.shape[1], "target_names")
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
            raise DataValidationError("ridge_alpha must be finite and non-negative")
        if sample_weight is None:
            weight = np.ones(design.shape[0], dtype=np.float64)
        else:
            weight = np.asarray(sample_weight, dtype=np.float64)
            if (
                weight.shape != (design.shape[0],)
                or not np.isfinite(weight).all()
                or np.any(weight <= 0)
            ):
                raise DataValidationError("sample weights must be finite and strictly positive")
        center, scale = _fit_robust_scaling(design)
        scaled_design = (design - center) / scale
        coefficients, covariance = _ridge_solution(
            scaled_design, target, ridge_alpha=float(ridge_alpha), sample_weight=weight
        )
        metadata = {
            "kind": "deterministic_residual_ridge",
            "feature_names": features,
            "target_names": targets,
            "ridge_alpha": float(ridge_alpha),
            "feature_scaling": "median_iqr",
        }
        return cls(
            features,
            targets,
            coefficients,
            covariance,
            float(ridge_alpha),
            center,
            scale,
            _hash_arrays(metadata, coefficients, covariance, center, scale),
        )

    def transform_design(self, X: ArrayLike) -> NDArray[np.float64]:
        """Transform raw physical-unit features using fit-bound median/IQR values."""

        return _transform_with_scaling(
            X,
            feature_count=len(self.feature_names),
            center=self.feature_center,
            scale=self.feature_scale,
        )

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        design = self.transform_design(X)
        if design.ndim != 2:
            raise DataValidationError("predict X must be a two-dimensional matrix")
        return np.column_stack((np.ones(design.shape[0]), design)) @ self.coefficients


@dataclass(frozen=True)
class BayesianResidualDynamics:
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    coefficient_draws: NDArray[np.float64]
    process_covariance_draws: NDArray[np.float64]
    ridge_alpha: float
    seed: int
    fit_algorithm: str
    fit_row_count: int
    cluster_count: int
    feature_center: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    content_hash: str

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        target_count = len(self.target_names)
        coefficient_draws = _readonly_float_copy(self.coefficient_draws)
        covariance_draws = _readonly_float_copy(self.process_covariance_draws)
        center = _readonly_float_copy(self.feature_center)
        scale = _readonly_float_copy(self.feature_scale)
        draw_count = coefficient_draws.shape[0] if coefficient_draws.ndim == 3 else -1
        if (
            coefficient_draws.shape != (draw_count, feature_count + 1, target_count)
            or covariance_draws.shape != (draw_count, target_count, target_count)
            or center.shape != (feature_count,)
            or scale.shape != (feature_count,)
            or draw_count <= 0
            or not all(
                np.isfinite(array).all()
                for array in (coefficient_draws, covariance_draws, center, scale)
            )
            or np.any(scale <= 0.0)
        ):
            raise DataValidationError(
                "Bayesian residual model arrays have invalid shapes or values"
            )
        object.__setattr__(self, "coefficient_draws", coefficient_draws)
        object.__setattr__(self, "process_covariance_draws", covariance_draws)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)

    @classmethod
    def fit(
        cls,
        X: ArrayLike,
        Y: ArrayLike,
        *,
        circuits: ArrayLike,
        cars: ArrayLike,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        n_draws: int = 256,
        ridge_alpha: float = 1.0,
        seed: int = 20270927,
    ) -> BayesianResidualDynamics:
        design = _matrix(X, "X")
        target = _matrix(Y, "Y")
        if target.shape[0] != design.shape[0]:
            raise DataValidationError("X and Y row counts differ")
        features = _names(feature_names, design.shape[1], "feature_names")
        targets = _names(target_names, target.shape[1], "target_names")
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
            raise DataValidationError("ridge_alpha must be finite and non-negative")
        if not isinstance(n_draws, int) or n_draws <= 0:
            raise DataValidationError("n_draws must be a positive integer")
        center, scale = _fit_robust_scaling(design)
        scaled_design = (design - center) / scale
        statistics = _cluster_sufficient_statistics(scaled_design, target, circuits, cars)
        coefficient_draws = np.empty(
            (n_draws, design.shape[1] + 1, target.shape[1]), dtype=np.float64
        )
        covariance_draws = np.empty((n_draws, target.shape[1], target.shape[1]), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for draw in range(n_draws):
            cluster_weight = _draw_cluster_weights(statistics, rng)
            coefficient_draws[draw], covariance_draws[draw] = _ridge_from_cluster_statistics(
                statistics,
                cluster_weight,
                ridge_alpha=float(ridge_alpha),
            )
        fit_algorithm = "per_car_normalized_sufficient_statistics_v1"
        metadata = {
            "kind": "two_stage_cluster_bayesian_bootstrap_residual_ridge",
            "fit_algorithm": fit_algorithm,
            "fit_row_count": statistics.fit_row_count,
            "cluster_count": statistics.cluster_count,
            "feature_names": features,
            "target_names": targets,
            "ridge_alpha": float(ridge_alpha),
            "n_draws": n_draws,
            "seed": int(seed),
            "feature_scaling": "median_iqr",
        }
        content_hash = _hash_arrays(metadata, coefficient_draws, covariance_draws, center, scale)
        return cls(
            features,
            targets,
            coefficient_draws,
            covariance_draws,
            float(ridge_alpha),
            int(seed),
            fit_algorithm,
            statistics.fit_row_count,
            statistics.cluster_count,
            center,
            scale,
            content_hash,
        )

    def transform_design(self, X: ArrayLike) -> NDArray[np.float64]:
        """Transform raw physical-unit features using fit-bound median/IQR values."""

        return _transform_with_scaling(
            X,
            feature_count=len(self.feature_names),
            center=self.feature_center,
            scale=self.feature_scale,
        )

    def _validated_design(self, X: ArrayLike) -> NDArray[np.float64]:
        design = self.transform_design(X)
        if design.ndim != 2:
            raise DataValidationError("prediction X must be a two-dimensional matrix")
        return design

    def predict_mean(self, X: ArrayLike) -> NDArray[np.float64]:
        design = self._validated_design(X)
        augmented = np.column_stack((np.ones(design.shape[0]), design))
        return augmented @ self.coefficient_draws.mean(axis=0)

    def deterministic_twin(self) -> DeterministicResidualDynamics:
        """Return posterior-mean transitions for noise-free deterministic propagation.

        The covariance is retained for provenance and diagnostics, but
        :func:`brace_f1.forecast.forecast_ctra_particles` deliberately omits
        both coefficient sampling and process noise for this deterministic
        comparator.
        """

        coefficients = self.coefficient_draws.mean(axis=0)
        covariance = self.process_covariance_draws.mean(axis=0)
        metadata = {
            "kind": "posterior_mean_twin",
            "source_hash": self.content_hash,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feature_scaling": "median_iqr",
        }
        return DeterministicResidualDynamics(
            self.feature_names,
            self.target_names,
            coefficients,
            covariance,
            self.ridge_alpha,
            self.feature_center.copy(),
            self.feature_scale.copy(),
            _hash_arrays(
                metadata,
                coefficients,
                covariance,
                self.feature_center,
                self.feature_scale,
            ),
        )

    def sample_residuals(self, X: ArrayLike, *, n_particles: int, seed: int) -> NDArray[np.float64]:
        design = self._validated_design(X)
        if not isinstance(n_particles, int) or n_particles <= 0:
            raise DataValidationError("n_particles must be a positive integer")
        rng = np.random.default_rng(seed)
        draw_index = rng.integers(
            0, self.coefficient_draws.shape[0], size=(design.shape[0], n_particles)
        )
        augmented = np.column_stack((np.ones(design.shape[0]), design))
        selected_coefficients = self.coefficient_draws[draw_index]
        means = np.einsum("ni,npio->npo", augmented, selected_coefficients)
        standard_normal = rng.standard_normal(means.shape)
        cholesky = np.linalg.cholesky(self.process_covariance_draws)
        selected_cholesky = cholesky[draw_index]
        noise = np.einsum("npij,npj->npi", selected_cholesky, standard_normal)
        return means + noise
