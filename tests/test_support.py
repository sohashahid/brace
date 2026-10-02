from __future__ import annotations

import numpy as np
import pandas as pd

from brace_f1.support import add_causal_support_columns


def test_causal_input_validity_does_not_use_symmetric_future_artifact_buffer() -> None:
    frames = pd.DataFrame(
        {
            "time_seconds": [0.00, 0.05, 0.10, 0.15],
            "continuous_segment_id": [0, 0, 1, 1],
            "hard_break": [True, False, True, False],
            "active": [True, True, True, True],
            "pit_status": [0, 0, 0, 0],
            "artifact_excluded": [True, True, False, False],
            "speed_mps": [10.0, 11.0, 12.0, 13.0],
        }
    )

    output = add_causal_support_columns(
        frames, mandatory_input_columns=("speed_mps",), horizons_seconds=(0.05, 0.25)
    )

    assert output["input_valid_causal"].tolist() == [False, True, False, True]
    assert output["outcome_evaluable_0p05s"].tolist() == [True, False, True, False]
    assert not output["outcome_evaluable_0p25s"].any()
    np.testing.assert_allclose(
        output["continuous_segment_remaining_seconds"], [0.05, 0.0, 0.05, 0.0]
    )


def test_causal_input_validity_rejects_current_pit_or_nonfinite_input() -> None:
    frames = pd.DataFrame(
        {
            "time_seconds": [0.00, 0.05, 0.10],
            "continuous_segment_id": [0, 0, 0],
            "hard_break": [True, False, False],
            "active": [True, True, True],
            "pit_status": [0, 1, 0],
            "speed_mps": [10.0, 11.0, float("nan")],
        }
    )

    output = add_causal_support_columns(
        frames, mandatory_input_columns=("speed_mps",), horizons_seconds=(0.25,)
    )

    assert output["input_valid_causal"].tolist() == [False, False, False]
