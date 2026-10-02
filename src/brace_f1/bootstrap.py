"""Paired cluster bootstrap for the BRACE held-out L-star contrast.

Models, calibrators, and thresholds remain fixed.  Resampling acts on complete
car-sessions nested in circuit; frame-IID resampling is deliberately absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from brace_f1.bootstrap_reliability import (
    ReliabilityBootstrapResult as ReliabilityBootstrapResult,
)
from brace_f1.bootstrap_reliability import (
    reliability_bin_cluster_bootstrap as reliability_bin_cluster_bootstrap,
)
from brace_f1.experiment import (
    HORIZONS_S,
    PRIMARY_BAYESIAN_METHOD,
    PRIMARY_DETERMINISTIC_METHOD,
)
from brace_f1.io import DataValidationError

_UNIT_KEYS = ("circuit", "source_session_id", "car_id")
_LEAST_FAVORABLE_FALSE_COUNT_COLUMN = "least_favorable_false_proposals"


@dataclass(frozen=True)
class BootstrapResult:
    """Point contrast, paired draws, and percentile interval."""

    l_star_brace: float
    l_star_comparator: float
    delta_l_star: float
    l_star_brace_draws: NDArray[np.float64]
    l_star_comparator_draws: NDArray[np.float64]
    delta_l_star_draws: NDArray[np.float64]
    l_star_brace_percentile_95: tuple[float, float]
    l_star_comparator_percentile_95: tuple[float, float]
    percentile_95: tuple[float, float]
    n_resamples: int
    seed: int
    cluster_level: str
    models_and_thresholds_refit: bool = False


def _require_columns(table: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise DataValidationError(
            f"bootstrap contribution table missing columns: {sorted(missing)}"
        )


def _validate_cluster_keys(table: pd.DataFrame) -> None:
    _require_columns(table, set(_UNIT_KEYS))
    invalid = table.loc[:, _UNIT_KEYS].isna()
    for column in _UNIT_KEYS:
        invalid[column] |= table[column].astype("string").str.strip().eq("").fillna(True)
    if invalid.any(axis=None):
        raise DataValidationError("bootstrap cluster keys must be non-null and non-blank")


def _validate_operating_parameters(
    *,
    false_budget_per_hour: float,
    minimum_recall: float,
    lead_grid_s: Sequence[float],
) -> NDArray[np.float64]:
    if not np.isfinite(false_budget_per_hour) or false_budget_per_hour < 0.0:
        raise DataValidationError("false budget must be finite and non-negative")
    if not np.isfinite(minimum_recall) or not 0.0 <= minimum_recall <= 1.0:
        raise DataValidationError("minimum recall must be finite and lie in [0, 1]")
    leads = np.asarray(tuple(float(value) for value in lead_grid_s), dtype=np.float64)
    if (
        leads.ndim != 1
        or leads.size == 0
        or not np.isfinite(leads).all()
        or np.any(leads <= 0.0)
        or np.any(np.diff(leads) <= 0.0)
    ):
        raise DataValidationError("bootstrap lead grid must be finite, positive, and increasing")
    return leads


def within_circuit_car_weights(
    units: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Draw integer duplicate-cluster weights within each fixed circuit."""

    _validate_cluster_keys(units)
    if units.empty or units.duplicated(list(_UNIT_KEYS)).any():
        raise DataValidationError("bootstrap units must be unique complete car-sessions")
    positional_units = units.reset_index(drop=True)
    circuit_indices = tuple(
        np.asarray(indices, dtype=np.int64)
        for indices in positional_units.groupby("circuit", sort=False).groups.values()
    )
    return _within_circuit_weights_from_indices(
        len(positional_units),
        circuit_indices,
        rng=rng,
    )


def _within_circuit_weights_from_indices(
    unit_count: int,
    circuit_indices: Sequence[NDArray[np.int64]],
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    weights = np.zeros(unit_count, dtype=np.int64)
    for positions in circuit_indices:
        draws = rng.choice(positions, size=positions.size, replace=True)
        weights += np.bincount(draws, minlength=unit_count).astype(np.int64)
    return weights


@dataclass(frozen=True)
class _BootstrapArrays:
    units: pd.DataFrame
    circuits: tuple[str, ...]
    circuit_unit_indices: tuple[NDArray[np.int64], ...]
    leads: NDArray[np.float64]
    values: NDArray[np.float64]


def _validate_contribution_contract(
    contributions: pd.DataFrame,
    *,
    false_count_column: str,
) -> None:
    required = {
        *_UNIT_KEYS,
        "fold_test_circuit",
        "method",
        "required_lead_s",
        "false_budget_per_hour",
        "localized_event_hits",
        "qualified_events",
        false_count_column,
        "exposure_hours",
        "fixed_model_and_threshold",
        "operating_point_status",
    }
    if false_count_column == _LEAST_FAVORABLE_FALSE_COUNT_COLUMN:
        required.update({"false_proposals", "unresolved_proposals"})
    _require_columns(contributions, required)
    if contributions.empty:
        raise DataValidationError("bootstrap contribution table is empty")
    if false_count_column == _LEAST_FAVORABLE_FALSE_COUNT_COLUMN:
        count_columns = (
            "false_proposals",
            "unresolved_proposals",
            _LEAST_FAVORABLE_FALSE_COUNT_COLUMN,
        )
        raw_counts = contributions.loc[:, count_columns]
        contains_boolean = raw_counts.apply(
            lambda column: column.map(lambda value: isinstance(value, (bool, np.bool_)))
        ).to_numpy(dtype=bool).any()
        counts = raw_counts.apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if (
            contains_boolean
            or not np.isfinite(counts).all()
            or np.any(counts < 0.0)
            or not np.equal(counts, np.floor(counts)).all()
        ):
            raise DataValidationError(
                "least-favorable false, unresolved, and total counts must be "
                "non-negative integer counts"
            )
        if not np.array_equal(counts[:, 2], counts[:, 0] + counts[:, 1]):
            raise DataValidationError(
                "least_favorable_false_proposals must equal false_proposals + "
                "unresolved_proposals rowwise"
            )
    _validate_cluster_keys(contributions)
    heldout_matches = contributions["fold_test_circuit"].astype(str) == contributions[
        "circuit"
    ].astype(str)
    if not heldout_matches.all():
        raise DataValidationError(
            "bootstrap contributions must come from their declared held-out circuit"
        )
    fixed = contributions["fixed_model_and_threshold"]
    if not fixed.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataValidationError("fixed_model_and_threshold must contain literal boolean values")
    if not fixed.all():
        raise DataValidationError("bootstrap requires fixed models and thresholds")
    status = contributions["operating_point_status"].astype(str)
    if not status.eq("estimable").all():
        raise DataValidationError("bootstrap excludes not-estimable operating points")


def _select_registered_grid(
    contributions: pd.DataFrame,
    *,
    false_budget_per_hour: float,
    methods: tuple[str, str],
    leads: NDArray[np.float64],
) -> pd.DataFrame:
    if len(methods) != 2 or methods[0] == methods[1] or any(not name for name in methods):
        raise DataValidationError("paired bootstrap requires two distinct method names")
    registered_budget = pd.to_numeric(
        contributions["false_budget_per_hour"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    budget_match = np.isclose(
        registered_budget,
        false_budget_per_hour,
        atol=1e-12,
        rtol=0.0,
    )
    selected = contributions.loc[
        contributions["method"].astype(str).isin(methods) & budget_match
    ].copy()
    if selected.empty or set(selected["method"].astype(str)) != set(methods):
        raise DataValidationError("no complete rows exist for the requested methods and budget")
    lead_values = pd.to_numeric(selected["required_lead_s"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    matches = np.isclose(
        lead_values[:, None],
        leads[None, :],
        atol=1e-12,
        rtol=0.0,
    )
    if not (matches.sum(axis=1) == 1).all():
        raise DataValidationError("every selected row must match one registered lead")
    selected["lead_slot"] = np.argmax(matches, axis=1)
    normalized_key = [*_UNIT_KEYS, "method", "lead_slot"]
    if selected.duplicated(normalized_key).any():
        raise DataValidationError(
            "bootstrap needs one contribution per car-session/method/registered lead "
            "and exactly one source row"
        )
    return selected


def _assemble_bootstrap_values(
    selected: pd.DataFrame,
    *,
    methods: tuple[str, str],
    leads: NDArray[np.float64],
    false_count_column: str,
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    units = (
        selected.loc[:, _UNIT_KEYS]
        .drop_duplicates()
        .sort_values(list(_UNIT_KEYS), kind="stable")
        .reset_index(drop=True)
    )
    unit_lookup = {
        tuple(str(row[column]) for column in _UNIT_KEYS): index for index, row in units.iterrows()
    }
    columns = (
        "localized_event_hits",
        "qualified_events",
        false_count_column,
        "exposure_hours",
    )
    values = np.full((2, leads.size, len(units), len(columns)), np.nan)
    method_index = {name: index for index, name in enumerate(methods)}
    for row in selected.itertuples(index=False):
        unit_key = tuple(str(getattr(row, column)) for column in _UNIT_KEYS)
        values[
            method_index[str(row.method)],
            int(row.lead_slot),
            unit_lookup[unit_key],
        ] = [float(getattr(row, column)) for column in columns]
    return units, values


def _validate_bootstrap_values(values: NDArray[np.float64]) -> None:
    if np.isnan(values).any():
        raise DataValidationError(
            "paired bootstrap requires both methods and every registered lead for every car-session"
        )
    if not np.isfinite(values).all():
        raise DataValidationError("bootstrap contribution counts and exposure must be finite")
    counts = values[..., :3]
    if not np.equal(counts, np.floor(counts)).all():
        raise DataValidationError("bootstrap contribution count fields must be integer counts")
    if np.any(counts < 0.0) or np.any(values[..., 3] <= 0.0):
        raise DataValidationError(
            "bootstrap contribution counts must be non-negative and exposure positive"
        )
    if np.any(values[..., 0] > values[..., 1]):
        raise DataValidationError("localized event hits cannot exceed qualified events")
    same_across_methods = np.all(values[0, :, :, (1, 3)] == values[1, :, :, (1, 3)])
    if not same_across_methods:
        raise DataValidationError("paired methods must share event and exposure denominators")
    same_across_leads = np.all(values[:, :, :, (1, 3)] == values[:, :1, :, (1, 3)])
    if not same_across_leads:
        raise DataValidationError("event and exposure denominators must be constant across leads")


def _validated_arrays(
    contributions: pd.DataFrame,
    *,
    false_budget_per_hour: float,
    methods: tuple[str, str],
    lead_grid_s: Sequence[float],
    false_count_column: str = "false_proposals",
) -> _BootstrapArrays:
    _validate_contribution_contract(
        contributions,
        false_count_column=false_count_column,
    )
    leads = _validate_operating_parameters(
        false_budget_per_hour=false_budget_per_hour,
        minimum_recall=0.0,
        lead_grid_s=lead_grid_s,
    )
    selected = _select_registered_grid(
        contributions,
        false_budget_per_hour=false_budget_per_hour,
        methods=methods,
        leads=leads,
    )
    units, values = _assemble_bootstrap_values(
        selected,
        methods=methods,
        leads=leads,
        false_count_column=false_count_column,
    )
    _validate_bootstrap_values(values)
    circuits = tuple(units["circuit"].astype(str).drop_duplicates())
    circuit_indices = tuple(
        np.flatnonzero(units["circuit"].astype(str).to_numpy() == circuit) for circuit in circuits
    )
    return _BootstrapArrays(units, circuits, circuit_indices, leads, values)


def _l_stars(
    arrays: _BootstrapArrays,
    weights: NDArray[np.int64],
    *,
    false_budget_per_hour: float,
    minimum_recall: float,
) -> tuple[float, float]:
    totals = np.einsum("mluc,u->mlc", arrays.values, weights.astype(np.float64))
    output: list[float] = []
    for method_index in range(2):
        passing: list[float] = []
        for lead_index, lead in enumerate(arrays.leads):
            hits, events, false, exposure = totals[method_index, lead_index]
            if events <= 0.0 or exposure <= 0.0:
                continue
            if hits / events >= minimum_recall - 1e-12 and false / exposure <= (
                false_budget_per_hour + 1e-12
            ):
                passing.append(float(lead))
        output.append(0.0 if not passing else max(passing))
    return output[0], output[1]


def _result(
    arrays: _BootstrapArrays,
    brace_draws: NDArray[np.float64],
    comparator_draws: NDArray[np.float64],
    *,
    n_resamples: int,
    seed: int,
    cluster_level: str,
    false_budget_per_hour: float,
    minimum_recall: float,
) -> BootstrapResult:
    brace, comparator = _l_stars(
        arrays,
        np.ones(len(arrays.units), dtype=np.int64),
        false_budget_per_hour=false_budget_per_hour,
        minimum_recall=minimum_recall,
    )
    delta_draws = brace_draws - comparator_draws
    brace_interval = np.quantile(brace_draws, [0.025, 0.975])
    comparator_interval = np.quantile(comparator_draws, [0.025, 0.975])
    delta_interval = np.quantile(delta_draws, [0.025, 0.975])
    return BootstrapResult(
        l_star_brace=brace,
        l_star_comparator=comparator,
        delta_l_star=brace - comparator,
        l_star_brace_draws=brace_draws,
        l_star_comparator_draws=comparator_draws,
        delta_l_star_draws=delta_draws,
        l_star_brace_percentile_95=(
            float(brace_interval[0]),
            float(brace_interval[1]),
        ),
        l_star_comparator_percentile_95=(
            float(comparator_interval[0]),
            float(comparator_interval[1]),
        ),
        percentile_95=(float(delta_interval[0]), float(delta_interval[1])),
        n_resamples=n_resamples,
        seed=seed,
        cluster_level=cluster_level,
    )


def paired_car_session_bootstrap(
    contributions: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 20270927,
    false_budget_per_hour: float = 2.0,
    minimum_recall: float = 0.50,
    lead_grid_s: Sequence[float] = HORIZONS_S,
    brace_method: str = PRIMARY_BAYESIAN_METHOD,
    comparator_method: str = PRIMARY_DETERMINISTIC_METHOD,
    false_count_column: str = "false_proposals",
) -> BootstrapResult:
    """Paired car-session-within-circuit percentile bootstrap of delta L-star."""

    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise DataValidationError("n_resamples must be a positive integer")
    _validate_operating_parameters(
        false_budget_per_hour=false_budget_per_hour,
        minimum_recall=minimum_recall,
        lead_grid_s=lead_grid_s,
    )
    arrays = _validated_arrays(
        contributions,
        false_budget_per_hour=false_budget_per_hour,
        methods=(brace_method, comparator_method),
        lead_grid_s=lead_grid_s,
        false_count_column=false_count_column,
    )
    rng = np.random.default_rng(seed)
    brace_draws = np.empty(n_resamples, dtype=np.float64)
    comparator_draws = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        weights = _within_circuit_weights_from_indices(
            len(arrays.units), arrays.circuit_unit_indices, rng=rng
        )
        brace, comparator = _l_stars(
            arrays,
            weights,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
        )
        brace_draws[draw] = brace
        comparator_draws[draw] = comparator
    return _result(
        arrays,
        brace_draws,
        comparator_draws,
        n_resamples=n_resamples,
        seed=seed,
        cluster_level="car_session_within_circuit",
        false_budget_per_hour=false_budget_per_hour,
        minimum_recall=minimum_recall,
    )


def paired_two_stage_circuit_car_bootstrap(
    contributions: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 20270927,
    false_budget_per_hour: float = 2.0,
    minimum_recall: float = 0.50,
    lead_grid_s: Sequence[float] = HORIZONS_S,
    brace_method: str = PRIMARY_BAYESIAN_METHOD,
    comparator_method: str = PRIMARY_DETERMINISTIC_METHOD,
    false_count_column: str = "false_proposals",
) -> BootstrapResult:
    """Paired circuit-then-car sensitivity bootstrap for transport uncertainty."""

    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise DataValidationError("n_resamples must be a positive integer")
    _validate_operating_parameters(
        false_budget_per_hour=false_budget_per_hour,
        minimum_recall=minimum_recall,
        lead_grid_s=lead_grid_s,
    )
    arrays = _validated_arrays(
        contributions,
        false_budget_per_hour=false_budget_per_hour,
        methods=(brace_method, comparator_method),
        lead_grid_s=lead_grid_s,
        false_count_column=false_count_column,
    )
    if len(arrays.circuits) < 2:
        raise DataValidationError("two-stage bootstrap requires at least two circuits")
    rng = np.random.default_rng(seed)
    brace_draws = np.empty(n_resamples, dtype=np.float64)
    comparator_draws = np.empty(n_resamples, dtype=np.float64)
    circuit_count = len(arrays.circuits)
    for draw in range(n_resamples):
        weights = np.zeros(len(arrays.units), dtype=np.int64)
        selected_circuits = rng.integers(0, circuit_count, size=circuit_count)
        for circuit_index in selected_circuits:
            indices = arrays.circuit_unit_indices[int(circuit_index)]
            selected_cars = rng.choice(indices, size=indices.size, replace=True)
            weights += np.bincount(selected_cars, minlength=len(arrays.units)).astype(np.int64)
        brace, comparator = _l_stars(
            arrays,
            weights,
            false_budget_per_hour=false_budget_per_hour,
            minimum_recall=minimum_recall,
        )
        brace_draws[draw] = brace
        comparator_draws[draw] = comparator
    return _result(
        arrays,
        brace_draws,
        comparator_draws,
        n_resamples=n_resamples,
        seed=seed,
        cluster_level="circuit_then_car_session",
        false_budget_per_hour=false_budget_per_hour,
        minimum_recall=minimum_recall,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compatible bootstrap CLI entry point."""

    from brace_f1.bootstrap_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
