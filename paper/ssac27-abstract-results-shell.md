# {{PAPER_TITLE}}

## Introduction

Frame-level prediction metrics do not show whether alerts arrive early, localize risk, and remain sparse. We test BRACE, the Bayesian Racing-Envelope Alert for Crossing Events, against a posterior-mean, no-process-noise twin. The endpoint is a simulated car-center boundary excursion, not a crash or contact.

## Methods

Four leave-one-circuit-out folds contained 58 complete DeepRacing streams, 460,269 causal 20 Hz frames, 294 excursions, and 6.317055 eligible simulated car-hours. BRACE propagated 256 particles through physics plus cluster-weighted Bayesian-bootstrap residuals; the twin omitted process noise and used posterior-mean coefficients. Calibration cars alone set persistent 1.50 s thresholds. $L^*$ was the greatest realized proposal-to-onset lead in 0.25--1.50 s with at least 0.50 correct-side, same-or-adjacent-25-m-segment recall and at most two false proposals per car-hour. Uncertainty used 10,000 paired car-stream resamples within the fixed circuits.

## Results

{{PROBABILITY_COMPARISON_RESULT}} {{ALL_METHOD_PROBABILITY_RESULT}} {{PRIMARY_POLICY_RESULT}} Endpoint values were $L^*={{PRIMARY_LSTAR_BRACE_S}}$ s for BRACE and ${{PRIMARY_LSTAR_TWIN_S}}$ s for the twin, giving $\Delta L^*={{PRIMARY_DELTA_LSTAR_S}}$ s (conditional 95% paired-bootstrap interval, ${{PRIMARY_DELTA_LSTAR_CI_LOW_S}}$ to ${{PRIMARY_DELTA_LSTAR_CI_HIGH_S}}$ s). {{PRIMARY_BOOTSTRAP_RESULT}} Calibration-only, the highest active frozen-grid thresholds below the silent threshold incurred 54.9--162.9 false proposals/h for BRACE and 645.0--877.6 for the twin, with zero localized recall at at least 1.50 s realized lead.

{{FIGURE_WARNING_FRONTIER_MARKDOWN}}

{{FIGURE_RELIABILITY_MARKDOWN}}

## Conclusion

{{PRIMARY_EVIDENCE_CONCLUSION}} {{METRIC_TO_DECISION_RESULT}} The audit exposes this result before physical testing; current evidence concerns simulated excursions, not crashes, impacts, padding efficacy, or deployment readiness.
