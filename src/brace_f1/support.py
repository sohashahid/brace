"""Causal input validity and future outcome-support indicators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from brace_f1.io import DataValidationError


def horizon_column_name(horizon_seconds: float) -> str:
    """Return a stable column name such as ``outcome_evaluable_0p25s``."""

    if not np.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise DataValidationError("outcome horizon must be positive and finite")
    token = f"{horizon_seconds:.2f}".replace(".", "p")
    return f"outcome_evaluable_{token}s"


def add_causal_support_columns(
    frames: pd.DataFrame,
    *,
    mandatory_input_columns: Sequence[str],
    horizons_seconds: Iterable[float] = (0.25, 0.50, 1.00, 1.50),
) -> pd.DataFrame:
    """Separate present-time input validity from offline outcome evaluability."""

    required = {
        "time_seconds",
        "continuous_segment_id",
        "hard_break",
        "active",
        "pit_status",
        *mandatory_input_columns,
    }
    missing = required.difference(frames.columns)
    if missing:
        raise DataValidationError(f"support table missing columns: {sorted(missing)}")
    output = frames.copy()
    finite = np.ones(len(output), dtype=bool)
    for column in mandatory_input_columns:
        values = output[column].to_numpy()
        try:
            finite &= np.isfinite(values)
        except TypeError as exc:
            raise DataValidationError(f"mandatory input {column} is not numeric") from exc
    output["input_valid_causal"] = (
        finite
        & output["active"].to_numpy(dtype=bool)
        & (output["pit_status"].to_numpy() == 0)
        & ~output["hard_break"].to_numpy(dtype=bool)
    )
    segment_end = output.groupby("continuous_segment_id", sort=False)[
        "time_seconds"
    ].transform("max")
    output["continuous_segment_end_time_seconds"] = segment_end
    remaining = segment_end.to_numpy(dtype=float) - output["time_seconds"].to_numpy(dtype=float)
    remaining[np.abs(remaining) < 1e-12] = 0.0
    output["continuous_segment_remaining_seconds"] = remaining
    for horizon in horizons_seconds:
        name = horizon_column_name(float(horizon))
        output[name] = remaining + 1e-12 >= float(horizon)
    return output
