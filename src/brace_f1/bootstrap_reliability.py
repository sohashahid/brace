"""Car-session cluster bootstrap for fixed reliability-bin summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from brace_f1.experiment import HORIZONS_S
from brace_f1.io import DataValidationError

_UNIT_KEYS = ("circuit", "source_session_id", "car_id")


@dataclass(frozen=True)
class ReliabilityBootstrapResult:
    """Cluster-bootstrap intervals for fixed reliability-bin sufficient statistics."""

    summary: pd.DataFrame
    n_resamples: int
    seed: int
    cluster_level: str = "car_session_within_circuit"


@dataclass(frozen=True)
class _ReliabilityArrays:
    units: pd.DataFrame
    circuits: tuple[str, ...]
    circuit_indices: tuple[NDArray[np.int64], ...]
    horizons: NDArray[np.float64]
    cells: tuple[tuple[str, int, int], ...]
    values: NDArray[np.float64]
    n_bins: int


def _require_columns(table: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise DataValidationError(
            f"reliability contribution table missing columns: {sorted(missing)}"
        )


def _validate_cluster_keys(table: pd.DataFrame) -> None:
    invalid = table.loc[:, _UNIT_KEYS].isna()
    for column in _UNIT_KEYS:
        invalid[column] |= table[column].astype("string").str.strip().eq("").fillna(True)
    if invalid.any(axis=None):
        raise DataValidationError("reliability cluster keys must be non-null and non-blank")


def _validated_axes(
    contributions: pd.DataFrame,
    *,
    horizons_s: Sequence[float],
    n_bins: int,
    expected_methods: Sequence[str] | None,
) -> tuple[pd.DataFrame, NDArray[np.float64], tuple[str, ...]]:
    if not isinstance(n_bins, int) or n_bins < 2:
        raise DataValidationError("reliability n_bins must be an integer of at least two")
    required = {
        "fold_test_circuit",
        *_UNIT_KEYS,
        "method",
        "horizon_s",
        "bin_index",
        "count",
        "probability_sum",
        "event_count",
    }
    _require_columns(contributions, required)
    if contributions.empty:
        raise DataValidationError("reliability contribution table is empty")
    _validate_cluster_keys(contributions)
    if not (
        contributions["fold_test_circuit"].astype(str) == contributions["circuit"].astype(str)
    ).all():
        raise DataValidationError(
            "reliability contributions must come from their declared held-out circuit"
        )
    horizons = np.asarray(tuple(float(value) for value in horizons_s), dtype=np.float64)
    if (
        horizons.ndim != 1
        or horizons.size == 0
        or not np.isfinite(horizons).all()
        or np.any(horizons <= 0.0)
        or np.any(np.diff(horizons) <= 0.0)
    ):
        raise DataValidationError(
            "reliability horizon grid must be finite, positive, and increasing"
        )
    method_source = (
        contributions["method"].astype(str).unique()
        if expected_methods is None
        else expected_methods
    )
    methods = tuple(sorted(str(value) for value in method_source))
    if (
        not methods
        or len(methods) != len(set(methods))
        or any(not value.strip() for value in methods)
    ):
        raise DataValidationError("reliability methods must be unique and non-empty")
    return contributions.copy(), horizons, methods


def _assign_grid_slots(
    table: pd.DataFrame,
    *,
    horizons: NDArray[np.float64],
    methods: tuple[str, ...],
    n_bins: int,
) -> pd.DataFrame:
    horizon_values = pd.to_numeric(table["horizon_s"], errors="coerce").to_numpy(dtype=np.float64)
    matches = np.isclose(horizon_values[:, None], horizons[None, :], atol=1e-12, rtol=0.0)
    if not (matches.sum(axis=1) == 1).all():
        raise DataValidationError("every reliability row must match one registered horizon")
    bins = pd.to_numeric(table["bin_index"], errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(bins).all()
        or not np.equal(bins, np.floor(bins)).all()
        or np.any((bins < 0.0) | (bins >= n_bins))
    ):
        raise DataValidationError("reliability bin indices must be integers in range")
    table["horizon_slot"] = np.argmax(matches, axis=1)
    table["bin_slot"] = bins.astype(np.int64)
    if set(table["method"].astype(str)) != set(methods):
        raise DataValidationError("reliability method grid is incomplete")
    key = [*_UNIT_KEYS, "method", "horizon_slot", "bin_slot"]
    if table.duplicated(key).any():
        raise DataValidationError(
            "reliability bootstrap needs one row per car-session/method/horizon/bin"
        )
    return table


def _validate_statistics(table: pd.DataFrame, *, n_bins: int) -> None:
    numeric = table.loc[:, ["count", "probability_sum", "event_count"]].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise DataValidationError(
            "reliability sufficient statistics must be finite and non-negative"
        )
    counts, probability_sums, events = values.T
    if not np.equal(counts, np.floor(counts)).all() or not np.equal(events, np.floor(events)).all():
        raise DataValidationError("reliability counts must be integer counts")
    if np.any(events > counts):
        raise DataValidationError("reliability event counts cannot exceed bin counts")
    lower = table["bin_slot"].to_numpy(dtype=np.float64) / n_bins
    upper = (table["bin_slot"].to_numpy(dtype=np.float64) + 1.0) / n_bins
    if np.any(probability_sums < counts * lower - 1e-12) or np.any(
        probability_sums > counts * upper + 1e-12
    ):
        raise DataValidationError(
            "reliability probability sums are inconsistent with assigned bin bounds"
        )
    table[["count", "probability_sum", "event_count"]] = numeric
    support = table.groupby([*_UNIT_KEYS, "horizon_slot", "method"], sort=False)[
        ["count", "event_count"]
    ].sum()
    support_variation = support.groupby([*_UNIT_KEYS, "horizon_slot"], sort=False).nunique(
        dropna=False
    )
    if (support_variation > 1).any(axis=None):
        raise DataValidationError(
            "reliability methods must share per-unit/horizon frame and event support"
        )


def _build_arrays(
    table: pd.DataFrame,
    *,
    horizons: NDArray[np.float64],
    methods: tuple[str, ...],
    n_bins: int,
) -> _ReliabilityArrays:
    units = (
        table.loc[:, _UNIT_KEYS]
        .drop_duplicates()
        .sort_values(list(_UNIT_KEYS), kind="stable")
        .reset_index(drop=True)
    )
    unit_lookup = {
        tuple(str(row[column]) for column in _UNIT_KEYS): index for index, row in units.iterrows()
    }
    cells = tuple(
        (method, horizon_index, bin_index)
        for method in methods
        for horizon_index in range(horizons.size)
        for bin_index in range(n_bins)
    )
    cell_lookup = {cell: index for index, cell in enumerate(cells)}
    values = np.full((len(cells), len(units), 3), np.nan, dtype=np.float64)
    for row in table.itertuples(index=False):
        unit = tuple(str(getattr(row, column)) for column in _UNIT_KEYS)
        cell = (str(row.method), int(row.horizon_slot), int(row.bin_slot))
        values[cell_lookup[cell], unit_lookup[unit]] = (
            float(row.count),
            float(row.probability_sum),
            float(row.event_count),
        )
    if np.isnan(values).any():
        raise DataValidationError(
            "reliability bootstrap requires every car-session in every fixed bin cell"
        )
    circuits = tuple(units["circuit"].astype(str).drop_duplicates())
    circuit_indices = tuple(
        np.flatnonzero(units["circuit"].astype(str).to_numpy() == circuit) for circuit in circuits
    )
    return _ReliabilityArrays(units, circuits, circuit_indices, horizons, cells, values, n_bins)


def _draw_weights(
    unit_count: int,
    circuit_indices: Sequence[NDArray[np.int64]],
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    weights = np.zeros(unit_count, dtype=np.int64)
    for positions in circuit_indices:
        selected = rng.choice(positions, size=positions.size, replace=True)
        weights += np.bincount(selected, minlength=unit_count).astype(np.int64)
    return weights


def _bootstrap_means(
    arrays: _ReliabilityArrays,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    probability = np.full((n_resamples, len(arrays.cells)), np.nan, dtype=np.float64)
    frequency = np.full_like(probability, np.nan)
    for draw in range(n_resamples):
        weights = _draw_weights(len(arrays.units), arrays.circuit_indices, rng=rng)
        totals = np.einsum("cuk,u->ck", arrays.values, weights.astype(np.float64))
        nonempty = totals[:, 0] > 0.0
        probability[draw, nonempty] = totals[nonempty, 1] / totals[nonempty, 0]
        frequency[draw, nonempty] = totals[nonempty, 2] / totals[nonempty, 0]
    return probability, frequency


def _interval(values: NDArray[np.float64]) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return float(lower), float(upper), int(finite.size)


def _summary_table(
    arrays: _ReliabilityArrays,
    probability_draws: NDArray[np.float64],
    frequency_draws: NDArray[np.float64],
) -> pd.DataFrame:
    point = arrays.values.sum(axis=1)
    rows: list[dict[str, object]] = []
    for index, (method, horizon_index, bin_index) in enumerate(arrays.cells):
        count, probability_sum, event_count = point[index]
        probability_interval = _interval(probability_draws[:, index])
        frequency_interval = _interval(frequency_draws[:, index])
        rows.append(
            {
                "method": method,
                "horizon_s": float(arrays.horizons[horizon_index]),
                "bin_index": bin_index,
                "bin_left": bin_index / arrays.n_bins,
                "bin_right": (bin_index + 1) / arrays.n_bins,
                "right_edge_inclusive": bin_index == arrays.n_bins - 1,
                "count": int(count),
                "mean_probability": float(probability_sum / count) if count else np.nan,
                "mean_probability_lower_95": probability_interval[0],
                "mean_probability_upper_95": probability_interval[1],
                "observed_frequency": float(event_count / count) if count else np.nan,
                "observed_frequency_lower_95": frequency_interval[0],
                "observed_frequency_upper_95": frequency_interval[1],
                "valid_probability_resamples": probability_interval[2],
                "valid_frequency_resamples": frequency_interval[2],
                "circuit_count": len(arrays.circuits),
                "car_session_count": len(arrays.units),
                "cluster_level": "car_session_within_circuit",
                "transport_uncertainty_included": False,
            }
        )
    return pd.DataFrame(rows)


def reliability_bin_cluster_bootstrap(
    contributions: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 20270927,
    horizons_s: Sequence[float] = HORIZONS_S,
    n_bins: int = 10,
    expected_methods: Sequence[str] | None = None,
) -> ReliabilityBootstrapResult:
    """Bootstrap fixed reliability bins with weights paired across every cell."""

    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise DataValidationError("n_resamples must be a positive integer")
    table, horizons, methods = _validated_axes(
        contributions,
        horizons_s=horizons_s,
        n_bins=n_bins,
        expected_methods=expected_methods,
    )
    table = _assign_grid_slots(table, horizons=horizons, methods=methods, n_bins=n_bins)
    _validate_statistics(table, n_bins=n_bins)
    arrays = _build_arrays(table, horizons=horizons, methods=methods, n_bins=n_bins)
    probability, frequency = _bootstrap_means(arrays, n_resamples=n_resamples, seed=seed)
    return ReliabilityBootstrapResult(
        summary=_summary_table(arrays, probability, frequency),
        n_resamples=n_resamples,
        seed=seed,
    )
