# BRACE Public-Data Study Protocol

**Freeze status:** the DeepRacing cohort, event definition, model hyperparameters, calibration procedure, threshold grid, and operating rule are frozen before model fitting or held-out scoring; fold-specific fitted values and thresholds will be estimated from fit/calibration cars only.  
**Empirical scope:** public open-wheel racing simulation whose dataset card declares CC BY 4.0; branded game and circuit rights still require review before raw redistribution. This is not live Formula 1 and not physical barrier contact.  
**Primary dataset:** the complete compact DeepRacing cohort: 58 cars across Bahrain, Britain, Jeddah, and Monza, with 294 qualifying mapped boundary-excursion car-events under the frozen definition below.  
**External checks:** 250 outcome-enriched AssettoCorsaGym Dallara F317 rollouts across Barcelona, Austria, and Monza, plus MixNet trajectory/boundary data where their labels permit.

## 1. Falsifiable question

At the same complete-stream false-alert budget, does a Bayesian residual-dynamics forecast provide more correctly localized warning time before a qualifying track-boundary excursion than its matched posterior-mean deterministic twin on a circuit not used for fitting or calibration?

The study does **not** ask whether padding prevents injury. It does not relabel a boundary crossing as a crash or claim Formula 1 deployment readiness.

## 2. Unit of observation and event

- A **car-session** is one car's complete compact motion/lap stream for one released circuit session. The primary release contains 58 car-sessions.
- An eligible frame has finite pose, velocity, acceleration, and reconstructed time; `result_status == 2`; and `pit_status == 0`.
- World pose is transformed to the PCD map frame using the car metadata: `p_map = (p_world - starting_pose_origin) @ R(q)` for row vectors, with quaternion order `[x,y,z,w]`.
- The corridor is the area inside the larger-absolute-area boundary loop and outside the smaller loop. Filenames do not determine loop topology because Jeddah's named inner/outer files are reversed geometrically.
- A base outside run is a consecutive set of eligible car-center samples outside that corridor.
- Hard breaks occur at active-state changes; clock resets; position jumps over 5 m; total-distance drops below -5 m; or reconstructed gaps greater than `max(0.05 s, 5 x the car's median positive sub-0.05 s interval)`. Runs never merge across hard breaks.
- Outside runs separated by at most 0.25 seconds of in-corridor time are merged within a hard-break-free segment.
- A qualifying event has maximum planar depth from the nearest boundary of at least 0.50 m and duration of at least 0.25 seconds.
- Exclude an event that touches pit state; begins within 0.50 seconds of an active/file start; ends within 0.50 seconds of an active/file end; or lies within 0.25 seconds of a lap boundary, clock/position reset, or data-gap boundary.
- Lead-in is the contiguous active, non-pit, in-corridor interval immediately before the event, stopping at the previous outside sample or hard break. It is reported but has no primary minimum.
- Primary crossing side and segment are recorded at the first outside sample of the qualifying event, relative to local direction of travel and fixed centerline arclength. Side and segment at maximum depth are retained as diagnostics. Ambiguous geometry remains unknown rather than being forced.
- Every eligible ordinary frame remains in the scoring denominator. Incident-centered negative sampling is prohibited for evaluation.

Two validity flags are kept separate. `input_valid_causal(t)` uses only information observed by score time `t`: finite mandatory fields, currently active status, non-pit state, and no already-observed discontinuity. Current map-derived corridor membership, signed clearance, nearest side, and arclength are causal features because they use only the current pose and the fixed circuit map; future event qualification is not a feature. Offline symmetric artifact buffers are never used to suppress an online score or proposal. `outcome_evaluable_h(t)` states whether the full future resolution interval `(t, t+h]` remains observable without an end, reset, or missing-data break. A causally produced proposal that loses follow-up is marked censored/unresolved, not retrospectively deleted; a least-favorable sensitivity counts every such unresolved proposal as false. Horizon-specific scorable exposure and unresolved counts are reported alongside the 6.317055-hour base cohort denominator.

This frozen definition produces 294 non-overlapping mapped boundary-excursion car-events: Bahrain 36, Britain 113, Jeddah 106, and Monza 39. With continuous-time interval accounting, 0.50-second active/file edge buffers, and symmetric 0.25-second buffers around lap, clock, position/distance-reset, and data-gap boundaries, eligible ordinary exposure is 6.317055 simulated car-hours: 2.542886 in Bahrain, 0.939339 in Britain, 2.488895 in Jeddah, and 0.345935 in Monza. Events are not assumed independent across one car-session; inference clusters by circuit/session/car, with only four independent circuit-session clusters available for circuit-level transport. Prespecified descriptive strata include depth, duration, speed, lead-in, and circuit. These strata do not replace the primary cohort.

## 3. Forecast target

At score time \(t\), using only measurements at or before \(t\), BRACE (Bayesian Racing-Envelope Alert for Crossing Events) estimates

\[
p_{h,s,k}(t)=\Pr\{T_{\mathrm{exit}}\le t+h,\ S=s,\ K=k\mid\mathcal I_t\},
\]

where \(h\in\{0.25,0.50,1.00,1.50\}\) seconds, \(S\) is local left/right/unknown, and \(K\) is a fixed circuit segment. The survival probability is one minus the sum over exit outcomes. All four cumulative forecast windows are probability-calibration targets; they are not themselves seconds of achieved warning. The 1.50-second score drives the primary proposal state machine, and actual proposal-to-onset time defines warning lead.

Segments are generated from a fixed start/finish origin in 25 m centerline-arclength bins before outcomes are scored. Primary localization requires the correct boundary side and a predicted bin within one adjacent bin of the observed bin (a 75 m three-bin neighborhood, with circular wrap at start/finish). Exact-bin performance is secondary. The map manifest records origin, direction, wrap rule, and bin edges.

## 4. Information clock

The compact DeepRacing files do not contain a native timestamp for every motion frame. Time is reconstructed from `current_lap_times` and `lap_numbers`: at a lap increment, elapsed time advances by `last_lap_times + new current_lap_time - prior current_lap_time`; lap decreases or same-lap clock drops greater than 0.05 seconds are resets; smaller negative jitter is monotonized. Reconstruction diagnostics and discrepancies against the approximately 2 Hz session clock are retained. The analysis is resampled causally to the metadata-declared 20 Hz representation and does not claim an exact raw 100 Hz clock.

The files also do not contain live packet-receipt timestamps. The primary public study therefore adds a prespecified synthetic latency stress test only after the undelayed analysis:

- no added delay;
- 40 ms delay;
- 80 ms delay;
- 160 ms delay;
- observed data gaps and hard breaks retained in every replay; no additional synthetic missingness is imposed.

These are sensitivity analyses, not measured Formula 1 communication latencies. Interpolation and smoothing are causal and may not use future frames.

The delay analysis holds the fitted model, calibrator, and threshold fixed. It shifts each proposal issue time by the stated delay and subtracts that delay from measured lead. An originally localized matched proposal that arrives after event onset is reclassified as false; original false and unresolved labels otherwise remain unchanged. The required-lead recall is then recomputed from the reduced lead. Observed missing-frame patterns already enter the primary streams as data gaps or hard breaks and remain present in all four delay replays.

## 5. Data quota and go/no-go rule

The data gate requires:

- at least 100 qualifying exit car-events with dependence handled at circuit/session/car level;
- qualifying events from all four primary circuits;
- complete ordinary exposure from the same source car-sessions;
- at least one entire circuit reserved from every fitting and calibration operation;
- valid borders and coordinate transforms for each scored circuit.

The acquired cohort passes the floor with 294 qualifying events and complete streams from 58 cars. It falls six events below the preferred 300-event capacity gate, so the optional sequence neural network is omitted. If later validation removes events and drops the cohort below 100, the work reverts to a feasibility audit and no detector-performance claim is made.

## 6. Partitions

The outer analysis is leave-one-circuit-out over Bahrain, Britain, Jeddah, and Monza. For each outer fold:

1. the held-out circuit is untouched test data;
2. complete cars from the remaining circuits are deterministically assigned to model-fit and calibration subsets by a salted hash of circuit, source path, and car identifier;
3. no frame, lap fragment, map correction, normalization statistic, or label from the held-out circuit enters fitting, threshold selection, or calibration;
4. all frames and events from one car-session remain in one partition.

The split manifest records source revision, hashes, salt, assignment code version, and row counts.

## 7. Proposed model

The public-data implementation is a **Bayesian residual-dynamics particle forecast**:

1. A causal state estimator represents position, longitudinal and lateral velocity, yaw, yaw rate, and available acceleration state. DeepRacing compact data do not provide driver controls, so the primary model may not use them.
2. A single-track or constant-turn vehicle model supplies the physics mean transition.
3. Bayesian regularized residual models learn only the transition error from the model-fit car-sessions. Posterior coefficient uncertainty and residual process noise are retained.
4. At each score time, posterior particles are propagated without future controls. Each particle is intersected with the fixed circuit polygon to obtain first-crossing time, side, and segment.
5. Particle proportions form the raw competing probabilities. A calibration map fitted only on the calibration subset may adjust probability values but may not change particle trajectories or use test outcomes.
6. Missing mandatory pose or time fields cause abstention. Optional missing channels are marginalized or replaced using training-only rules recorded in the model manifest.

Before held-out scoring, `configs/study.json` freezes the eight-state order; nine residual predictors and five transition-residual targets; fit-only median/IQR scaling; two-stage circuit/car Dirichlet(1) Bayesian-bootstrap priors with equal total car weight; ridge grid and inner leave-one-fit-circuit-out Gaussian negative-log-likelihood selection; 256 coefficient draws; 256 forecast particles; 0.05-second integration; draw-specific Gaussian process noise; swept-segment first-crossing detection; per-row hash-derived seeds; and horizon-specific non-decreasing Platt calibration on calibration cars only. Because the four horizon maps are fitted separately, no post-hoc cross-horizon coherence correction is applied. The analysis reports the number and rate of score rows for which any adjacent calibrated cumulative exit probability decreases by more than (10^{-12}) as horizon increases. The deterministic primary comparator uses the posterior-mean residual coefficients with process noise omitted, yielding a controlled component comparison of the complete uncertainty-propagating pipeline against its deterministic counterpart rather than separate causal effects of coefficient uncertainty and process noise.

## 8. Prespecified comparisons

Every method receives identical causal inputs, maps, eligible exposure, persistence rule, score cadence, and compute accounting.

1. Constant velocity.
2. Constant turn rate and velocity.
3. The matched posterior-mean residual-dynamics twin with process noise omitted, the principal controlled component comparator for the complete uncertainty-propagating pipeline.
4. An independently refit deterministic residual-dynamics model whose ridge penalty is selected by calibration-transition mean-squared error, a secondary comparator.
5. Regularized discrete-time side-specific hazard model using the same current-state and map-distance features.
6. Bayesian residual-dynamics particle forecast, the proposed method.

The side-hazard model does not natively predict a circuit segment. Its frozen causal adapter assigns each left/right horizon probability to the circular 25 m bin containing current centerline arclength plus non-negative body-longitudinal speed multiplied by one-half of that horizon. This deliberately simple adapter permits the same localization scoring without giving the hazard model future geometry or test labels.

A sequence neural network is added only if the frozen qualifying-event quota reaches 300 before any held-out scoring. The audited cohort has 294 events, so this model is omitted under the prespecified capacity rule rather than trained on a marginal event cohort.

## 9. Proposal rule

- Scores are produced on the causally reconstructed, metadata-declared 20 Hz grid. Clock reconstruction uncertainty is evaluated separately.
- A proposal opens only after the same side and segment neighborhood pass the probability threshold for three consecutive completed scores.
- The first proposal in an open decision episode is non-overwritable for the primary analysis.
- Thresholds are selected using calibration car-sessions only.
- A proposal is a correct localized event proposal only when a qualifying excursion begins within its prespecified horizon on the predicted side and within the prespecified segment tolerance. A proposal with no qualifying excursion in the horizon is false; a temporally timely proposal for the wrong side or segment is both a localized event miss and a false protective-zone proposal. This category-specific definition is fixed before held-out scoring because the proposed application must activate the correct zone, not merely anticipate that some exit will occur. Future knowledge is never used to delete a proposal.
- If the prespecified resolution window becomes unobservable because of a later reset, data gap, or stream end, the proposal is unresolved in the primary censoring analysis and false in the prespecified least-favorable sensitivity.
- After a proposal resolves or an observed excursion ends, the detector re-arms only after one continuous in-corridor second. This causal state machine permits multiple proposals and events in one car-session without counting repeated above-threshold frames as independent alerts.
- Repeated above-threshold frames in the same open proposal count once; occupied proposal time is reported separately.

## 10. Outcomes and metrics

### Probability quality

- Brier score and Brier skill score;
- logarithmic score on horizon-evaluable frames only, without a censoring-adjusted claim;
- calibration intercept and slope;
- reliability diagram with car-session-aware uncertainty;
- expected calibration error reported only with its binning rule and bin counts.

### Detection and localization

- average precision, computed as the stepwise precision--recall area after grouping tied scores at each distinct threshold (not trapezoidal PR AUC);
- event recall at fixed false alerts per simulated car-hour;
- false alerts per simulated car-hour with a one-sided interval;
- first-proposal lead time;
- correct-side recall;
- correct-segment and adjacent-segment recall;
- abstention rate and reasons.

### Primary effect

The primary operating point is fixed before any held-out score is examined:

- false-alert budget: at most 2 false proposals per eligible simulated car-hour;
- minimum correctly localized event recall: 0.50;
- correctly localized: correct boundary side and predicted 25 m segment within one adjacent bin;
- required actual-lead grid: \(L\in\{0.25,0.50,1.00,1.50\}\) seconds;
- primary proposal score: probability mass from the 1.50-second forecast window for one side and a circular three-bin segment neighborhood;
- threshold selection: for each outer fold, method, and required lead \(L\), select the calibration-only threshold that maximizes correctly localized event recall with first-proposal lead at least \(L\), subject to the calibration false-alert budget; ties use the higher threshold.

For a method, \(L^*\) is the greatest prespecified required lead whose pooled circuit-held-out proposals achieve both the observed false-alert requirement and localized event recall of at least 0.50 with actual first-proposal lead at least \(L\). If no lead value qualifies, \(L^*=0\). This test-set rule defines the summary metric rather than selecting a deployable threshold: every threshold was already fixed from that fold's calibration cars. The primary effect is

\[
\Delta L^*=
L^*_{\mathrm{BRACE}}-L^*_{\mathrm{dynamics}}.
\]

The operating-budget table reports the full actual-lead frontier at 0.5, 1, 2, 5, and 10 false alerts per simulated car-hour; no single favorable point may be substituted after test scoring. The 0.5, 1, 5, and 10 points are secondary. Cumulative forecast-window metrics are reported separately and never called warning lead. Correct-side, within-one-bin, and exact-bin recall are reported alongside every point. Because there are only four circuit-session clusters, the observed qualification rule is accompanied by paired car-session bootstrap intervals and all four circuit-specific results; an interval containing zero is reported as no demonstrated gain.

Estimability is frozen before held-out scoring as an exposure-only design rule. A fold-specific calibration operating point is estimable only when the requested false-alert budget per hour multiplied by calibration exposure hours is at least 1.0, so the calibration stream could contain one false proposal without automatically exceeding that budget. Otherwise the row is retained with status `not_estimable_insufficient_exposure`; it receives no threshold and is excluded from pooled or bootstrap qualification. The primary 2-per-hour point must be estimable for every fold, method, and required-lead value. This design-support rule is separate from the one-sided Poisson-model upper bound reported for observed false-alert rates.

## 11. Uncertainty

- The primary cluster is the complete car-session nested in circuit/session; individual frames and repeated events within one car are not treated as independent.
- Paired cluster bootstrap resamples compare methods on identical car-sessions.
- Primary intervals are 95% intervals with the number of resamples, random seed, and percentile or studentized construction recorded.
- Sensitivity analyses use circuit-session clusters above the nested car-session level.
- Millions of frames are never described as millions of independent samples.
- Unknown crossing side or corrupt follow-up is reported explicitly. The least-favorable sensitivity adds every unresolved proposal to the false-proposal count and recomputes pooled and paired-bootstrap (L^*) and \(\Delta L^*\) with the fitted models, calibrators, and thresholds fixed.

## 12. External checks

AssettoCorsaGym contributes 178 strict terminal `out_of_track + done` rollouts across three circuits, but the storage-conscious smallest-file rule enriched exits. It may test event sensitivity and transport after safe conversion; it cannot estimate natural prevalence or false alerts per car-hour without an unbiased denominator. MixNet contributes normal high-speed trajectory and map-intersection checks; its audited trajectories contain no center crossings. Neither source supplies adjudicated crash/contact outcomes or supports barrier/contact claims.

## 13. Release contract

The submission repository will contain:

- acquisition scripts and immutable source URLs;
- source revisions, licenses, byte sizes, and SHA-256 hashes;
- safe conversion code and conversion logs;
- data dictionary and map manifest;
- car-session/event/split manifests;
- license-permitted derived tables or repository links;
- frozen model/baseline configurations and seeds;
- per-score and per-car-session held-out predictions;
- metric tables, bootstrap outputs, and figure source tables;
- manuscript, bibliography, figures, build instructions, and environment lock.

Raw legacy pickles and third-party Formula 1-derived files remain outside ordinary Git history unless their redistribution terms are confirmed. The public repository must still enable an independent researcher to reacquire the exact source objects and reproduce every reported number.

## 14. Claim boundary

Permitted after a successful evaluation: a named Bayesian model improved or failed to improve held-out simulated open-wheel **track-exit** forecasting under the exact tested circuits, thresholds, and false-alert budgets.

Not permitted from this study: real Formula 1 crash probability, barrier-module contact probability, FIA readiness, device activation safety, crash prevention, or injury reduction.
