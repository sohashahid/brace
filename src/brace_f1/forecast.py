"""Vectorized short-horizon particle forecasts and mapped first crossings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from brace_f1.geometry import CircuitCorridor
from brace_f1.io import DataValidationError
from brace_f1.residual import (
    RESIDUAL_TARGET_NAMES,
    BayesianResidualDynamics,
    DeterministicResidualDynamics,
    body_frame_ctra_midpoint_velocity_step,
)

SIDE_ORDER: tuple[str, ...] = ("left", "right", "unknown")
_ROW_SEED_PREFIX = "brace-row-seed-v1:"
_ROW_SEED_KEYS = {"base_seed", "outer_fold", "method", "circuit", "car", "frame_index"}
STATE_ORDER: tuple[str, ...] = (
    "x_m",
    "y_m",
    "v_longitudinal_mps",
    "v_lateral_mps",
    "yaw_rad",
    "yaw_rate_radps",
    "acceleration_longitudinal_mps2",
    "acceleration_lateral_mps2",
)


def circular_bin_distance(bins: ArrayLike, *, center_bin: int, n_bins: int) -> NDArray[np.int64]:
    """Return unsigned shortest distances on a circular integer-bin lattice."""

    if not isinstance(n_bins, int) or n_bins <= 0:
        raise DataValidationError("n_bins must be a positive integer")
    if not isinstance(center_bin, (int, np.integer)):
        raise DataValidationError("center_bin must be an integer")
    values = np.asarray(bins, dtype=np.int64)
    wrapped = np.mod(values, n_bins)
    center = int(center_bin) % n_bins
    direct = np.abs(wrapped - center)
    return np.minimum(direct, n_bins - direct).astype(np.int64)


def circular_neighborhood_mask(
    bins: ArrayLike, *, center_bin: int, tolerance_bins: int, n_bins: int
) -> NDArray[np.bool_]:
    if not isinstance(tolerance_bins, int) or tolerance_bins < 0:
        raise DataValidationError("tolerance_bins must be a non-negative integer")
    return circular_bin_distance(bins, center_bin=center_bin, n_bins=n_bins) <= tolerance_bins


@dataclass(frozen=True)
class ForecastProbabilities:
    """Cumulative competing outcomes for one circuit and a batch of score rows."""

    horizons_s: NDArray[np.float64]
    no_exit_probability: NDArray[np.float64]
    outcome_probability: NDArray[np.float64]
    side_order: tuple[str, ...]
    segment_bin_count: int
    first_crossing_time_s: NDArray[np.float64]
    first_crossing_side_index: NDArray[np.int8]
    first_crossing_bin: NDArray[np.int64]
    seed: int
    model_hash: str
    config_hash: str


@dataclass(frozen=True)
class ForecastBatch:
    """One bounded-memory forecast chunk and its source row positions."""

    row_indices: NDArray[np.int64]
    forecast: ForecastProbabilities


def planar_states_from_features(features: pd.DataFrame) -> NDArray[np.float64]:
    """Select :data:`STATE_ORDER` from causal feature rows in its registered order."""

    columns = (
        "map_x_m",
        "map_y_m",
        "body_speed_longitudinal_mps",
        "body_speed_lateral_mps",
        "yaw_rad",
        "yaw_rate_radps",
        "body_acceleration_longitudinal_mps2",
        "body_acceleration_lateral_mps2",
    )
    missing = set(columns).difference(features.columns)
    if missing:
        raise DataValidationError(f"forecast state input missing columns: {sorted(missing)}")
    state = features.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(state).all():
        raise DataValidationError("forecast state features contain non-finite values")
    return state


def _configuration_hash(
    *,
    horizons: NDArray[np.float64],
    dt_s: float,
    n_particles: int,
    seed: int,
    segment_length_m: float,
    model_hash: str,
    row_ids: NDArray[np.str_],
) -> str:
    payload = {
        "state_order": STATE_ORDER,
        "mean_transition": "body_frame_ctra_midpoint",
        "side_order": SIDE_ORDER,
        "horizons_s": horizons.tolist(),
        "dt_s": dt_s,
        "n_particles": n_particles,
        "seed": seed,
        "segment_length_m": segment_length_m,
        "model_hash": model_hash,
        "row_ids_sha256": hashlib.sha256("\x1f".join(row_ids).encode()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_forecast_row_ids(
    *,
    base_seed: int,
    outer_fold: str,
    method: str,
    circuits: ArrayLike,
    cars: ArrayLike,
    frame_indices: ArrayLike,
) -> NDArray[np.str_]:
    """Encode the frozen row RNG identity without delimiter ambiguity."""

    if not isinstance(base_seed, (int, np.integer)):
        raise DataValidationError("base_seed must be an integer")
    fold = str(outer_fold)
    method_name = str(method)
    if not fold or not method_name:
        raise DataValidationError("outer_fold and method must be non-empty")
    circuit_array = np.asarray(circuits, dtype=np.str_)
    car_array = np.asarray(cars, dtype=np.str_)
    raw_frames = np.asarray(frame_indices)
    if (
        circuit_array.ndim != 1
        or circuit_array.size == 0
        or car_array.shape != circuit_array.shape
        or raw_frames.shape != circuit_array.shape
        or np.any(circuit_array == "")
        or np.any(car_array == "")
    ):
        raise DataValidationError(
            "circuits, cars, and frame_indices must be aligned non-empty vectors"
        )
    try:
        frames = raw_frames.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("frame_indices must contain integers") from exc
    try:
        frame_numeric = raw_frames.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("frame_indices must contain integers") from exc
    if not np.isfinite(frame_numeric).all() or not np.array_equal(frame_numeric, frames):
        raise DataValidationError("frame_indices must contain finite integers")
    output: list[str] = []
    for circuit, car, frame_index in zip(circuit_array, car_array, frames, strict=True):
        payload = {
            "base_seed": int(base_seed),
            "outer_fold": fold,
            "method": method_name,
            "circuit": str(circuit),
            "car": str(car),
            "frame_index": int(frame_index),
        }
        output.append(
            _ROW_SEED_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
    return np.asarray(output, dtype=np.str_)


def _validate_canonical_row_id(value: str, *, base_seed: int) -> None:
    if not value.startswith(_ROW_SEED_PREFIX):
        raise DataValidationError("Bayesian forecasts require canonical row_ids")
    try:
        payload = json.loads(value[len(_ROW_SEED_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise DataValidationError("canonical row_ids contain invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _ROW_SEED_KEYS:
        raise DataValidationError("canonical row_ids have an invalid field set")
    if payload["base_seed"] != base_seed:
        raise DataValidationError("canonical row_ids base_seed does not match forecast seed")
    if (
        not isinstance(payload["outer_fold"], str)
        or not payload["outer_fold"]
        or not isinstance(payload["method"], str)
        or not payload["method"]
        or not isinstance(payload["circuit"], str)
        or not payload["circuit"]
        or not isinstance(payload["car"], str)
        or not payload["car"]
        or not isinstance(payload["frame_index"], int)
        or isinstance(payload["frame_index"], bool)
    ):
        raise DataValidationError("canonical row_ids contain invalid field values")
    canonical = _ROW_SEED_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if value != canonical:
        raise DataValidationError("canonical row_ids are not canonically serialized")


def _validated_row_ids(
    row_ids: ArrayLike | None,
    row_count: int,
    *,
    base_seed: int,
    require_canonical: bool,
) -> NDArray[np.str_]:
    if row_ids is None:
        if require_canonical:
            raise DataValidationError("Bayesian forecasts require canonical row_ids")
        output = np.arange(row_count).astype(np.str_)
    else:
        output = np.asarray(row_ids).astype(np.str_)
    if output.shape != (row_count,):
        raise DataValidationError(f"row_ids must have shape ({row_count},)")
    if len(set(output.tolist())) != row_count:
        raise DataValidationError("row_ids must be unique within a forecast call")
    if require_canonical or np.any(np.char.startswith(output, _ROW_SEED_PREFIX)):
        for value in output:
            _validate_canonical_row_id(str(value), base_seed=base_seed)
    return output


def canonical_forecast_row_seed(row_id: str) -> int:
    """Return the frozen 64-bit RNG seed from one canonical row identity."""

    value = str(row_id)
    if not value.startswith(_ROW_SEED_PREFIX):
        raise DataValidationError("row_id is not a canonical forecast row identity")
    try:
        payload = json.loads(value[len(_ROW_SEED_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise DataValidationError("canonical row_id contains invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("base_seed"), int):
        raise DataValidationError("canonical row_id does not contain an integer base_seed")
    _validate_canonical_row_id(value, base_seed=int(payload["base_seed"]))
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _row_random_generator(seed: int, row_id: str) -> np.random.Generator:
    if row_id.startswith(_ROW_SEED_PREFIX):
        _validate_canonical_row_id(row_id, base_seed=seed)
        return np.random.default_rng(canonical_forecast_row_seed(row_id))
    digest = hashlib.sha256(f"{seed}\x1f{row_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))


def _validated_horizons(
    horizons_s: Sequence[float], dt_s: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise DataValidationError("dt_s must be positive and finite")
    horizons = np.asarray(tuple(horizons_s), dtype=np.float64)
    if horizons.ndim != 1 or horizons.size == 0 or not np.isfinite(horizons).all():
        raise DataValidationError("horizons must be a non-empty finite vector")
    if np.any(horizons <= 0.0) or np.any(np.diff(horizons) <= 0.0):
        raise DataValidationError("horizons must be positive and strictly increasing")
    steps = np.rint(horizons / dt_s).astype(np.int64)
    if not np.allclose(steps * dt_s, horizons, rtol=0.0, atol=1e-10):
        raise DataValidationError("horizons must be integer multiples of dt_s")
    return horizons, steps


def _side_indices(local_side: NDArray[np.str_]) -> NDArray[np.int8]:
    return np.where(local_side == "left", 0, np.where(local_side == "right", 1, 2)).astype(np.int8)


def _evolving_residual_design(
    particles: NDArray[np.float64],
    initial_design: NDArray[np.float64],
    feature_names: tuple[str, ...],
    corridor: CircuitCorridor,
    dt_s: float,
) -> NDArray[np.float64]:
    """Update registered physical-unit features from the current particle state.

    Model-specific columns outside :data:`RESIDUAL_FEATURE_COLUMNS` remain at
    their score-time values.  The BRACE registered design is fully recomputed.
    """

    row_count, particle_count, _ = particles.shape
    design = np.broadcast_to(
        initial_design[:, None, :], (row_count, particle_count, initial_design.shape[1])
    ).copy()
    feature_index = {name: index for index, name in enumerate(feature_names)}
    state_features = {
        "body_speed_longitudinal_mps": particles[..., 2],
        "body_speed_lateral_mps": particles[..., 3],
        "yaw_rate_radps": particles[..., 5],
        "body_acceleration_longitudinal_mps2": particles[..., 6],
        "body_acceleration_lateral_mps2": particles[..., 7],
    }
    for name, values in state_features.items():
        if name in feature_index:
            design[..., feature_index[name]] = values
    if "transition_dt_s" in feature_index:
        design[..., feature_index["transition_dt_s"]] = dt_s

    track_names = {"heading_error_rad", "track_offset_m", "track_curvature_per_m"}
    if track_names.intersection(feature_index):
        track = corridor.track_features_planar(particles[..., :2].reshape(-1, 2))
        shape = (row_count, particle_count)
        if "heading_error_rad" in feature_index:
            heading = track.heading_rad.reshape(shape)
            design[..., feature_index["heading_error_rad"]] = (
                particles[..., 4] - heading + np.pi
            ) % (2.0 * np.pi) - np.pi
        if "track_offset_m" in feature_index:
            design[..., feature_index["track_offset_m"]] = track.offset_m.reshape(shape)
        if "track_curvature_per_m" in feature_index:
            design[..., feature_index["track_curvature_per_m"]] = track.curvature_per_m.reshape(
                shape
            )
    return design


def forecast_ctra_particles(
    initial_states: ArrayLike,
    corridor: CircuitCorridor,
    *,
    residual_model: BayesianResidualDynamics | DeterministicResidualDynamics | None = None,
    residual_features: ArrayLike | None = None,
    horizons_s: Sequence[float] = (0.25, 0.50, 1.00, 1.50),
    dt_s: float = 0.05,
    n_particles: int = 256,
    seed: int = 20270927,
    segment_length_m: float = 25.0,
    row_ids: ArrayLike | None = None,
) -> ForecastProbabilities:
    """Propagate body-frame CTRA particles to their first mapped corridor exit.

    The eight state entries follow :data:`STATE_ORDER`.  Bayesian coefficient
    draws are fixed per particle over the forecast; process residuals are drawn
    independently at each 20 Hz transition.
    """

    state = np.asarray(initial_states, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != len(STATE_ORDER):
        raise DataValidationError("initial_states must have shape (N, 8)")
    if state.shape[0] == 0 or not np.isfinite(state).all():
        raise DataValidationError("initial_states must be non-empty and finite")
    if not isinstance(n_particles, int) or n_particles <= 0:
        raise DataValidationError("n_particles must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise DataValidationError("seed must be an integer")
    if not np.isfinite(segment_length_m) or segment_length_m <= 0.0:
        raise DataValidationError("segment_length_m must be positive and finite")
    horizons, horizon_steps = _validated_horizons(horizons_s, float(dt_s))
    row_count = state.shape[0]
    stable_row_ids = _validated_row_ids(
        row_ids,
        row_count,
        base_seed=int(seed),
        require_canonical=isinstance(residual_model, BayesianResidualDynamics),
    )
    row_generators = [_row_random_generator(int(seed), str(row_id)) for row_id in stable_row_ids]
    particles = np.repeat(state[:, None, :], n_particles, axis=1)
    model_hash = "physics_only"
    residual_design: NDArray[np.float64] | None = None
    selected_coefficients: NDArray[np.float64] | None = None
    deterministic_coefficients: NDArray[np.float64] | None = None
    residual_cholesky: NDArray[np.float64] | None = None

    if residual_model is not None:
        if tuple(residual_model.target_names) != RESIDUAL_TARGET_NAMES:
            raise DataValidationError("residual model target order does not match BRACE dynamics")
        if residual_features is None:
            raise DataValidationError("residual_features are required with a residual model")
        design = np.asarray(residual_features, dtype=np.float64)
        expected_shape = (row_count, len(residual_model.feature_names))
        if design.shape != expected_shape or not np.isfinite(design).all():
            raise DataValidationError(
                f"residual_features must be finite with shape {expected_shape}"
            )
        residual_design = design
        model_hash = residual_model.content_hash
        if isinstance(residual_model, BayesianResidualDynamics):
            draw_index = np.stack(
                [
                    generator.integers(
                        0, residual_model.coefficient_draws.shape[0], size=n_particles
                    )
                    for generator in row_generators
                ]
            )
            selected_coefficients = residual_model.coefficient_draws[draw_index]
            all_cholesky = np.linalg.cholesky(residual_model.process_covariance_draws)
            residual_cholesky = all_cholesky[draw_index]
        else:
            deterministic_coefficients = residual_model.coefficients
    elif residual_features is not None:
        raise DataValidationError("residual_features were supplied without a residual model")

    initial_inside = corridor.contains_planar(state[:, :2])
    was_inside = np.repeat(initial_inside[:, None], n_particles, axis=1)
    crossed = np.zeros((row_count, n_particles), dtype=bool)
    first_time = np.full((row_count, n_particles), np.nan, dtype=np.float64)
    first_side = np.full((row_count, n_particles), -1, dtype=np.int8)
    first_bin = np.full((row_count, n_particles), -1, dtype=np.int64)
    segment_bin_count = int(np.ceil(corridor.track_length_m / segment_length_m))

    for step_index in range(1, int(horizon_steps[-1]) + 1):
        previous_position = particles[..., :2].copy()
        x = particles[..., 0]
        y = particles[..., 1]
        v_long = particles[..., 2]
        v_lat = particles[..., 3]
        yaw = particles[..., 4]
        yaw_rate = particles[..., 5]
        a_long = particles[..., 6]
        a_lat = particles[..., 7]
        next_v_long, next_v_lat, midpoint_v_long, midpoint_v_lat = (
            body_frame_ctra_midpoint_velocity_step(
                v_long,
                v_lat,
                yaw_rate,
                a_long,
                a_lat,
                dt_s,
            )
        )
        midpoint_yaw = yaw + 0.5 * yaw_rate * dt_s
        cos_midpoint_yaw = np.cos(midpoint_yaw)
        sin_midpoint_yaw = np.sin(midpoint_yaw)
        midpoint_velocity_x = midpoint_v_long * cos_midpoint_yaw - midpoint_v_lat * sin_midpoint_yaw
        midpoint_velocity_y = midpoint_v_long * sin_midpoint_yaw + midpoint_v_lat * cos_midpoint_yaw
        next_yaw_rate = yaw_rate.copy()
        next_a_long = a_long.copy()
        next_a_lat = a_lat.copy()

        if residual_design is not None:
            current_design = _evolving_residual_design(
                particles,
                residual_design,
                tuple(residual_model.feature_names),
                corridor,
                float(dt_s),
            )
            current_design = residual_model.transform_design(current_design)
            augmented = np.concatenate(
                (np.ones((*current_design.shape[:2], 1), dtype=np.float64), current_design),
                axis=2,
            )
            if selected_coefficients is not None:
                residual_mean = np.einsum("npi,npio->npo", augmented, selected_coefficients)
            else:
                residual_mean = np.einsum("npi,io->npo", augmented, deterministic_coefficients)
            residual = residual_mean
            if residual_cholesky is not None:
                standard_normal = np.stack(
                    [
                        generator.standard_normal((n_particles, residual_mean.shape[-1]))
                        for generator in row_generators
                    ]
                )
                residual = residual_mean + np.einsum(
                    "npij,npj->npi", residual_cholesky, standard_normal
                )
            next_v_long = next_v_long + residual[..., 0]
            next_v_lat = next_v_lat + residual[..., 1]
            next_yaw_rate = next_yaw_rate + residual[..., 2]
            next_a_long = next_a_long + residual[..., 3]
            next_a_lat = next_a_lat + residual[..., 4]

        particles[..., 0] = x + midpoint_velocity_x * dt_s
        particles[..., 1] = y + midpoint_velocity_y * dt_s
        particles[..., 2] = next_v_long
        particles[..., 3] = next_v_lat
        particles[..., 4] = (yaw + yaw_rate * dt_s + np.pi) % (2.0 * np.pi) - np.pi
        particles[..., 5] = next_yaw_rate
        particles[..., 6] = next_a_long
        particles[..., 7] = next_a_lat

        flat_position = particles[..., :2].reshape(-1, 2)
        inside = corridor.contains_planar(flat_position).reshape(row_count, n_particles)
        active = ~crossed & was_inside
        flat_active = np.flatnonzero(active.ravel())
        if flat_active.size:
            segment_exit = corridor.first_exit_along_segments(
                previous_position.reshape(-1, 2)[flat_active], flat_position[flat_active]
            )
            flat_crossing = flat_active[segment_exit.has_exit]
        else:
            flat_crossing = np.empty(0, dtype=np.int64)
        if flat_crossing.size:
            crossing_fraction = segment_exit.fraction[segment_exit.has_exit]
            crossing_point = segment_exit.point[segment_exit.has_exit]
            crossing_location = corridor.locate(crossing_point)
            side_index = _side_indices(crossing_location.local_side)
            segment_bin = np.floor(
                crossing_location.centerline_arclength_wrapped_m / segment_length_m
            ).astype(np.int64)
            segment_bin = np.mod(segment_bin, segment_bin_count)
            first_time.ravel()[flat_crossing] = (step_index - 1 + crossing_fraction) * dt_s
            first_side.ravel()[flat_crossing] = side_index
            first_bin.ravel()[flat_crossing] = segment_bin
            crossed.ravel()[flat_crossing] = True
        was_inside = inside

    no_exit_probability = np.empty((row_count, horizons.size), dtype=np.float64)
    outcome_probability = np.zeros(
        (row_count, horizons.size, len(SIDE_ORDER), segment_bin_count), dtype=np.float64
    )
    for horizon_index, horizon in enumerate(horizons):
        observed = np.isfinite(first_time) & (first_time <= horizon + 1e-12)
        no_exit_probability[:, horizon_index] = 1.0 - observed.mean(axis=1)
        for row in range(row_count):
            particle_indices = np.flatnonzero(observed[row])
            if not particle_indices.size:
                continue
            np.add.at(
                outcome_probability[row, horizon_index],
                (first_side[row, particle_indices], first_bin[row, particle_indices]),
                1.0 / n_particles,
            )
    config_hash = _configuration_hash(
        horizons=horizons,
        dt_s=float(dt_s),
        n_particles=n_particles,
        seed=int(seed),
        segment_length_m=float(segment_length_m),
        model_hash=model_hash,
        row_ids=stable_row_ids,
    )
    return ForecastProbabilities(
        horizons,
        no_exit_probability,
        outcome_probability,
        SIDE_ORDER,
        segment_bin_count,
        first_time,
        first_side,
        first_bin,
        int(seed),
        model_hash,
        config_hash,
    )


def iter_forecast_ctra_batches(
    initial_states: ArrayLike,
    corridor: CircuitCorridor,
    *,
    residual_model: BayesianResidualDynamics | DeterministicResidualDynamics | None = None,
    residual_features: ArrayLike | None = None,
    horizons_s: Sequence[float] = (0.25, 0.50, 1.00, 1.50),
    dt_s: float = 0.05,
    n_particles: int = 256,
    seed: int = 20270927,
    segment_length_m: float = 25.0,
    row_ids: ArrayLike | None = None,
    batch_size: int = 128,
) -> Iterator[ForecastBatch]:
    """Yield bounded-memory forecast chunks with batch-invariant row RNG.

    The dense joint side/segment tensor exists only for ``batch_size`` rows at
    a time.  No distance-based shortcut is applied: Gaussian residual support
    is unbounded, so excluding apparently distant rows would not be
    mathematically lossless.  Consumers should aggregate policy outputs and
    release per-score probabilities incrementally.
    """

    states = np.asarray(initial_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != len(STATE_ORDER) or states.shape[0] == 0:
        raise DataValidationError("initial_states must have shape (N, 8)")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise DataValidationError("batch_size must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise DataValidationError("seed must be an integer")
    stable_row_ids = _validated_row_ids(
        row_ids,
        states.shape[0],
        base_seed=int(seed),
        require_canonical=isinstance(residual_model, BayesianResidualDynamics),
    )
    design: NDArray[np.float64] | None
    if residual_features is None:
        design = None
    else:
        design = np.asarray(residual_features, dtype=np.float64)
        if design.ndim != 2 or design.shape[0] != states.shape[0]:
            raise DataValidationError("residual_features must align with all state rows")
    for start in range(0, states.shape[0], batch_size):
        stop = min(start + batch_size, states.shape[0])
        forecast = forecast_ctra_particles(
            states[start:stop],
            corridor,
            residual_model=residual_model,
            residual_features=None if design is None else design[start:stop],
            horizons_s=horizons_s,
            dt_s=dt_s,
            n_particles=n_particles,
            seed=seed,
            segment_length_m=segment_length_m,
            row_ids=stable_row_ids[start:stop],
        )
        yield ForecastBatch(np.arange(start, stop, dtype=np.int64), forecast)
