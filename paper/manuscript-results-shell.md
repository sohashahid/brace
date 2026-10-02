---
title: "{{PAPER_TITLE}}"
author: "Anonymous authors"
date: "October 2026"
bibliography: references.bib
link-citations: true
geometry: margin=0.8in
fontsize: 10pt
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - |
    ```{=latex}
    \usepackage{booktabs}
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{xcolor}
    \renewcommand{\arraystretch}{1.12}
    ```
---

# Abstract

**Introduction.** A racing-safety forecast can rank risky frames well yet fail when converted into early, localized alerts at a tolerable false-proposal burden. We introduce BRACE, the Bayesian Racing-Envelope Alert for Crossing Events, and test whether its uncertainty-propagating pipeline adds usable warning over a matched posterior-mean, no-process-noise twin.

**Methods.** We reconstructed 6.317055 eligible simulated car-hours from the complete compact DeepRacing release: 58 open-wheel car streams, 460,269 causal 20 Hz frames, and 294 qualifying mapped car-center boundary excursions across Bahrain, Britain, Jeddah, and Monza. In four leave-one-circuit-out folds, BRACE propagated 256 particles from a constant-turn-rate-and-acceleration mean transition plus a Bayesian-bootstrap residual model. A matched twin used the posterior-mean residual coefficients and no process noise. Calibration cars alone determined horizon-specific marginal recalibration functions and persistent 1.50 s proposal thresholds for each required-lead/budget pair. The primary endpoint, $L^*$, was the greatest prespecified realized proposal-to-onset lead in $\{0.25,0.50,1.00,1.50\}$ seconds that achieved at least 0.50 correct-side, same-or-adjacent-25-m-segment event recall at no more than two false proposals per eligible simulated car-hour.

**Results.** {{PROBABILITY_COMPARISON_RESULT}} {{ALL_METHOD_PROBABILITY_RESULT}} {{PRIMARY_POLICY_RESULT}} The endpoint values were $L^*={{PRIMARY_LSTAR_BRACE_S}}$ s for BRACE and ${{PRIMARY_LSTAR_TWIN_S}}$ s for its twin, giving $\Delta L^*={{PRIMARY_DELTA_LSTAR_S}}$ s (conditional 95% paired-bootstrap interval, ${{PRIMARY_DELTA_LSTAR_CI_LOW_S}}$ to ${{PRIMARY_DELTA_LSTAR_CI_HIGH_S}}$ s; 10,000 car-session resamples within the four fixed circuits). {{PRIMARY_BOOTSTRAP_RESULT}} Calibration-only, the highest active frozen-grid thresholds below the silent threshold incurred 54.9--162.9 false proposals/h for BRACE and 645.0--877.6 for the twin, with zero localized recall at at least 1.50 s realized lead.

**Conclusion.** {{PRIMARY_EVIDENCE_CONCLUSION}} {{METRIC_TO_DECISION_RESULT}} No result from this simulated excursion study alone justifies an active circuit-side response. The contribution is an auditable evaluation rule that exposes probability-to-decision behavior before external testing; the endpoint is a simulated mapped boundary excursion, not a Formula 1 crash, physical barrier contact, or evidence that active padding works.

# 1. Introduction

A probability of “crashing soon” is neither the target of this study nor an operational safety signal. A circuit-side monitoring system first needs to estimate which side and which section of the mapped racing envelope is threatened before sampled event onset while remaining quiet through ordinary running. Whether that warning is early enough for a downstream action depends on separately measured communication, decision, and device latency. A forecaster that recognizes every excursion only after the first eligible outside sample, points to the wrong location, or repeatedly alerts during normal laps does not create useful warning lead. This creates a practical metric-to-decision problem: favorable frame-level discrimination or probability scores can coexist with zero usable alerts. The numerical gates in this paper were prespecified and sealed before held-out outcome access; they are analytical benchmarks, not FIA or device requirements.

Motorsport safety and vehicle-control research supplies several pieces of this problem. Formula 1 accident recorders and the Circuit and Safety Analysis System support reconstruction and simulation-based circuit-safety assessment [@Wright_2000; @Corcelle_2000]. Motorsport studies have examined early loss-of-control detection [@Della_Rossa_2025], while time-to-line-crossing and false-alarm-aware warning methods predate the present work in road vehicles [@Mammar_2006_TimeToLine; @Gao_2020_LimitedFalseAlarm]. Autonomous-racing research has developed predictive safety filters, learned error dynamics, and uncertainty-aware control [@Tearle_2021_PredictiveSafetyFilter; @Hewing_2018_CautiousNMPC; @Kabzan_2019_LearningMPC; @Xue_2024_ErrorDynamics]. Bayesian racing-trajectory inference [@Weiss_2022_DifferentialBayesian], learned vehicle-model correction [@Baader_2026_VehiclePrediction], and probabilistic track-limit reasoning [@Gulisano_2025_ProbabilisticSafety] are therefore not new in isolation.

The unresolved question is narrower and decision-centered: does the complete uncertainty-propagating BRACE pipeline provide additional *correctly localized* warning lead under the same maximum car-level false-proposal budget? Both primary models share the same state, physics mean, residual features, fitted Bayesian residual object, circuit maps, calibration partitions, and proposal policy. BRACE propagates coefficient and process uncertainty. Its deterministic twin uses the posterior-mean residual coefficients and omits process noise. Each method then receives its own calibration-car-only probability map and threshold. Their difference therefore reports the held-out contrast between the two complete pipelines; it does not isolate a causal effect of coefficient uncertainty or process noise alone.

We evaluate that contrast on complete public simulation streams rather than selected incident clips. The endpoint is a qualifying mapped car-center boundary excursion. It is not a collision label and is never relabeled as a crash. Every eligible ordinary frame remains in the false-proposal denominator. Four entire circuits are held out in turn, so each car is scored only in the fold where its circuit was excluded from model fitting, probability calibration, and threshold selection.

The paper makes three contributions.

1. We define a causal competing-location forecast for the first qualifying boundary excursion, including the horizon, boundary side, and fixed 25 m circuit segment.
2. We introduce an actual-lead warning frontier that jointly charges localization errors and false proposals on complete eligible exposure. Its primary contrast, $\Delta L^*$, compares the complete uncertainty-propagating pipeline with its posterior-mean, no-process-noise counterpart.
3. We provide a circuit-held-out falsification test that places frame-level probability metrics beside the operational warning frontier, with event rules, calibration, and threshold selection sealed before held-out outcome access.

The contribution is the joint estimand and matched pipeline comparison, not a claim that each component is individually novel. The study combines calibration-car-fitted marginal recalibration evaluated on held-out circuits, correct side-and-segment localization, realized first-proposal lead, and false proposals per complete eligible simulated car-hour in one circuit-held-out comparison. We make no claim about confidential team or governing-body systems.

# 2. Related work

## 2.1 Racing prediction and safety

DeepRacing established an open framework and public data for autonomous-racing trajectory research [@Weiss_2020_DeepRacing_Framework; @Weiss_2020_DeepRacing]. Subsequent racing-control work has incorporated learned residual or error dynamics into model-based control [@Hewing_2018_CautiousNMPC; @Kabzan_2019_LearningMPC; @Xue_2024_ErrorDynamics; @Baader_2026_VehiclePrediction]. Predictive safety filters constrain learned racing controllers before their proposed action reaches the vehicle [@Tearle_2021_PredictiveSafetyFilter]. These methods primarily target trajectory tracking, control, or constraint satisfaction. BRACE instead evaluates a probabilistic event forecast under an operational false-proposal budget.

Loss-of-control detection is a closer precursor. Della Rossa et al. forecast vehicle instability in a dynamic-driving simulator [@Della_Rossa_2025]. A recent preprint tested rolling split-conformal slip monitoring against timestamped Formula 1 Race Control messages and track-limit incidents and reported no useful early-warning performance at its evaluated alert burden [@Kotla_2026_ConformalSlip]. Loss of control, however, is not equivalent to a mapped boundary excursion: an unstable car can recover, and a car can cross a boundary without a labeled crash. BRACE targets a different estimand—the time, side, and mapped segment of a first car-center boundary excursion—and retains ordinary running in the denominator.

## 2.2 Probabilistic warning and evaluation

Particle methods approximate nonlinear predictive distributions by propagating weighted or sampled states [@Arulampalam_2002]. Proper scoring rules evaluate the probability distribution rather than only a thresholded decision [@Gneiting_2007]. BRACE uses both ideas but centers the decision analysis on warning time. Timeliness-versus-false-alarm evaluation has a long history in activity monitoring and departure warning [@Fawcett_1999_ActivityMonitoring; @Mammar_2006_TimeToLine; @Gao_2020_LimitedFalseAlarm]. Our warning frontier is a motorsport-specific application of that general tradeoff, with localization and complete exposure added to the pass rule.

The event target resembles landmark prediction because BRACE updates the probability of a near-term event at each score time [@Nicolaie_2012]. Calibration remains essential: a displayed 30% must refer to an observed frequency under a specified horizon and population. We therefore report Brier and logarithmic scores [@Gneiting_2007], logistic calibration intercept and slope [@VanCalster_2019_Calibration], ten-bin reliability and expected calibration error [@Guo_2017_Calibration], and stepwise average precision [@Davis_2006_PrecisionRecall]. Fixed-bin expected calibration error is descriptive and depends on the chosen binning. The primary $L^*$ endpoint complements these frame-level scores by measuring whether calibrated probabilities lead to a usable persistent proposal.

## 2.3 From excursion warning to circuit protection

Formula 1 has long used accident reconstruction and simulation-based circuit-safety assessment to inform safety engineering [@Wright_2000; @Corcelle_2000]. Generic hydraulic impact absorbers and laboratory impact-control prototypes predate this work [@Kim_2002; @Kim_2004]. A separate patent describes a movable energy-absorbing roadside barrier with post-impact sensing [@Breed_2003], while current racetrack-barrier work combines simulation with physical evidence [@Arya_2025; @FIA_2018_BarrierStandard]. These sources motivate a future interface in which a correctly localized forecast could nominate a pre-installed protection zone. They do not establish that BRACE predicts physical contact or that a selectable barrier state is beneficial. Such claims would require surveyed barrier geometry, adjudicated first-contact labels, measured end-to-end latency, fault analysis, and full-scale mechanical qualification. None is available in the present dataset.

# 3. Data and endpoint

## 3.1 Source and integrity

The primary source is the complete compact DeepRacing cohort at pinned revision `1e7ec1dfdb6a3cde726d377b0403b51fa51ab2c4` [@DeepRacing_Dataset_2026], interpreted with the accompanying framework papers [@Weiss_2020_DeepRacing_Framework; @Weiss_2020_DeepRacing]. The accompanying immutable acquisition manifest identifies 244 files totaling 86,360,332 bytes. The cohort covers 58 car-session streams from one released session at each of four circuits and includes one centerline plus two circuit-boundary loops per circuit. All 174 NumPy archives opened with object loading disabled; all 1,508 members were non-object arrays. Local byte counts and SHA-256 hashes matched that manifest.

The compact car files contain world pose, velocity, body-frame acceleration, orientation quaternion, lap distance, accumulated distance, lap state, active/result state, and pit state. They do not contain collision, barrier-contact, damage, weather, intervention, or injury labels. The study endpoint is therefore restricted to what the files and maps directly support: a mapped car-center boundary excursion.

Two alternative public sources were audited but excluded from the primary experiment. An outcome-enriched AssettoCorsaGym subset contained terminal off-track rollouts but could not estimate natural prevalence or false proposals per hour [@Remonda_2024_AssettoCorsaGym]. The complete safe MixNet evaluation subset contained no centerline boundary crossing across 27 audited trajectories [@MixNet_Dataset_2022]. Neither source supplied the complete positive-event and ordinary-exposure design required by the prespecified endpoint.

| Held-out circuit | Car sessions | Qualifying excursions | Eligible simulated car-hours | Excursions per car-hour |
|:--|--:|--:|--:|--:|
| Bahrain | 20 | 36 | 2.542886 | 14.16 |
| Britain | 11 | 113 | 0.939339 | 120.30 |
| Jeddah | 20 | 106 | 2.488895 | 42.59 |
| Monza | 7 | 39 | 0.345935 | 112.74 |
| **Total** | **58** | **294** | **6.317055** | **46.54** |

Table: Frozen primary cohort. Each circuit is the untouched test circuit in one outer fold. Events repeat within cars, and cars share one session within a circuit; the 294 rows are not independent crashes. {#tbl:cohort}

The cohort contains 46.54 qualifying excursions per eligible simulated car-hour, or one event per 77.35 seconds of accumulated car exposure. This high and circuit-dependent simulator event rate is part of the benchmark population; it is not an estimate of real-racing crash or off-track prevalence.

## 3.2 Reconstructed causal clock

The compact motion arrays contain 2,473,009 rows but no vector timestamp for each row. We reconstructed elapsed time from lap number, current lap time, and previous lap time. A lap decrease or same-lap clock drop greater than 0.05 s created a hard break; smaller negative jitter was monotonized. Active-state changes, reconstructed gaps, position jumps above 5 m, and total-distance drops below -5 m also created hard breaks. No event, transition pair, or exposure interval crossed a hard break.

We causally resampled the stream to the metadata-declared 20 Hz representation, producing 460,269 model frames. We do not describe the compact source as native 100 Hz telemetry. The reconstructed clock disagreed with an endpoint-matched uniform index clock at some session locations, so clock provenance is a limitation. At the 294 retained events, however, the absolute disagreement between event durations under the two clocks had a median of 0.0034 s and a 95th percentile of 0.0125 s. These diagnostics assess sensitivity to two imperfect clocks; neither creates a measured packet-receipt timestamp.

## 3.3 Circuit geometry and event definition

World positions were transformed to the map coordinate system using the origin and quaternion in each car's metadata. Corridor topology was determined geometrically: the loop with larger absolute signed area was the exterior and the smaller loop was the hole. This rule matters because the Jeddah filenames reverse the geometric inner/outer roles.

An eligible raw sample had finite required state, `result_status == 2`, and `pit_status == 0`. A base outside run comprised consecutive eligible car-center samples outside the mapped corridor. Runs separated by at most 0.25 s of observed in-corridor time were merged when no hard break intervened. A qualifying event reached at least 0.50 m planar depth beyond the nearest boundary and lasted at least 0.25 s. We excluded an event that touched pit state, began within 0.50 s of an active or file start, ended within 0.50 s of an active or file end, or lay within 0.25 s of a lap boundary, reset, or data-gap boundary.

The production builder found 1,616 candidate runs and 294 qualifying events that were non-overlapping within each car. Event counts were stable around the prespecified merge rule: the same depth and duration criteria produced 295 events at a 0.10 s merge gap and 292 at 0.50 s. Primary event onset was the reconstructed timestamp of the first eligible outside sample in a run that subsequently satisfied the depth and duration rules. Side was left or right relative to the directed centerline tangent, whose source order was validated against observed vehicle travel, and segment was the circular 25 m centerline-arclength bin at onset. Thus event onset is a sampled time, not an interpolated physical crossing or contact time. Primary localization required the correct side and the same or an immediately adjacent bin (a circular three-bin candidate neighborhood). Exact-bin localization was secondary.

## 3.4 Causal inputs and future truth

Here *causal* means available at score time without future information; it does not mean causal-effect estimation. Score-time features and future targets were stored in separate tables. The causal validity flag used only information observed by the score time: finite mandatory fields, current active and non-pit state, and discontinuities already observed. Current pose relative to a fixed circuit map was causal. Future event qualification, future artifact buffers, and future observability were not model inputs.

A separate horizon flag recorded whether the full outcome window remained observable. A proposal whose future window later encountered a reset, gap, or stream end was unresolved, not retrospectively deleted. The primary analysis censored unresolved proposals; a least-favorable sensitivity counted every unresolved proposal as false. Every eligible ordinary frame remained available for scoring. We did not perform incident-centered negative sampling.

# 4. BRACE

## 4.1 Forecast target

At score time $t$, BRACE estimates the cumulative probability

$$
p_{h,s,k}(t)=Pr\!\left(t<T_{\mathrm{next}}^{(t)}\le t+h,\ S_{\mathrm{next}}^{(t)}=s,\ K_{\mathrm{next}}^{(t)}=k\mid\mathcal I_t\right),
$$

where $\mathcal I_t$ contains only the reconstructed causal history; $T_{\mathrm{next}}^{(t)}$ is the sampled onset time of the first qualifying excursion after $t$; $S_{\mathrm{next}}^{(t)}$ and $K_{\mathrm{next}}^{(t)}$ are its onset side and circular 25 m segment bin; and $h\in\{0.25,0.50,1.00,1.50\}$ s. Observed qualifying-event sides were left or right. The forecast retained a non-actionable `unknown` side for geometrically indeterminate simulated crossings; that mass contributed to total excursion risk but could not form a localized proposal. The complement of the sum over side and segment outcomes is the probability of no qualifying excursion inside the horizon. The displayed total exit percentage is

$$
p_h(t)=\sum_s\sum_k p_{h,s,k}(t).
$$

This percentage is the conditional probability of a qualifying mapped excursion inside a fixed forecast window. It is not an estimate of physical barrier contact or injury.

## 4.2 Physics mean and residual target

The causal state was

$$
\mathbf x_t=(X_t,Y_t,u_t,v_t,\psi_t,r_t,a^x_t,a^y_t),
$$

with map position, body-frame longitudinal and lateral speed, yaw, yaw rate, and body-frame longitudinal and lateral acceleration. DeepRacing does not provide driver control inputs in the compact release, so no model used steering, brake, or throttle.

A body-frame constant-turn-rate-and-acceleration midpoint step supplied the physics mean. For interval $\delta$,

$$
\dot u_t=a^x_t+r_t v_t, \qquad
\dot v_t=a^y_t-r_t u_t,
$$

Body velocities were advanced by this explicit midpoint step. The midpoint body velocity, rotated by $\psi_t+r_t\delta/2$, advanced $X$ and $Y$; yaw advanced as $\psi_{t+\delta}=\operatorname{wrap}(\psi_t+r_t\delta)$, while the physics mean held $r$, $a^x$, and $a^y$ constant. The model then learned the one-step residual in

$$
\mathbf y_t=
(u_{t+\delta},v_{t+\delta},r_{t+\delta},a^x_{t+\delta},a^y_{t+\delta})-
(\hat u_{t+\delta},\hat v_{t+\delta},r_t,a^x_t,a^y_t).
$$

The nine prespecified predictors were $u_t$, $v_t$, $r_t$, $a^x_t$, $a^y_t$, heading error, track offset, local curvature, and transition duration. Predictors were centered by the fit-partition median and scaled by the fit-partition interquartile range. Only consecutive, causal, in-corridor transition pairs from model-fit cars contributed residual labels. During forecasting, the predicted residual—and, for BRACE, process noise—updated these five dynamic components after each physics-mean step.

## 4.3 Cluster Bayesian residual dynamics

For each outer fold, ridge strength was selected from $\{0.01,0.1,1,10,100\}$ by mean Gaussian negative log likelihood in an inner leave-one-fit-circuit-out procedure. The intercept was not penalized. To avoid treating long frame streams as independent replications, sufficient statistics were normalized within each car. Each of 256 Bayesian-bootstrap draws [@Rubin_1981_BayesianBootstrap] sampled circuit weights from a symmetric Dirichlet distribution and then sampled car weights within circuit from another symmetric Dirichlet distribution. The resulting ridge coefficient matrix and multivariate residual covariance were retained together for each draw.

This procedure represents uncertainty over a deliberately small residual model. It is not a fully Bayesian vehicle model, and the Bayesian bootstrap does not create new independent circuits. Its purpose is to propagate plausible coefficient and residual variation while respecting the data hierarchy.

## 4.4 Particle forecast and mapped crossing

At each valid score time, BRACE initialized 256 identical physical states and assigned each particle one posterior coefficient draw. The coefficient draw remained fixed within that particle's forecast horizon. At each 0.05 s propagation step, the particle received its draw-specific mean residual and Gaussian process noise from the associated covariance. State-dependent residual predictors were recomputed from the evolving particle and fixed circuit geometry.

Swept line segments between successive particle positions were intersected with the corridor boundary. The first intersection determined crossing time, local side, and 25 m segment. Particle fractions formed the raw distribution over no exit and first crossing outcomes. Per-row seeds were derived from a hash of the frozen base seed, outer fold, method, circuit, car, and frame index, making results invariant to scoring batch size.

## 4.5 Matched deterministic twin

The principal comparator used the mean of the same posterior coefficient draws and omitted process noise. It shared the state, physics transition, residual design, robust scaling, fitted Bayesian residual object, forecast grid, geometry, calibration method, and proposal rule with BRACE. The comparison asks a single question: what operational warning-frontier difference was observed between these two complete pipelines?

# 5. Calibration, decisions, and comparators

## 5.1 Circuit-held-out partitions

The outer analysis held out Bahrain, Britain, Jeddah, and Monza in turn. Within each fold, all non-held-out source-session/car units were ranked by a SHA-256 key over the frozen salt, seed, held-out circuit, source circuit, source session, and car identifier. The first $\lfloor0.20n+0.5\rfloor$ units, clipped to leave at least one fit and one calibration unit, formed the calibration partition; the remainder formed the model-fit partition. A source-session/car unit never crossed partitions. The immutable split table records every assignment, key, source hash, and the 0.20 fraction. No frame, event label, map correction, scaling statistic, calibration coefficient, or threshold from the held-out circuit entered fitting or selection.

The held-out outcomes remained sealed until each fold had saved its fitted models, raw calibration and test scores, horizon calibrators, operating thresholds, method identities, source hashes, and an authenticated threshold-freeze seal. Aggregation required all four completed held-out folds.

## 5.2 Probability calibration

For every method, fold, and horizon, a monotone logistic recalibration map, analogous to Platt scaling [@Platt_1999_ProbabilisticOutputs], was fitted on calibration cars only. For raw marginal probability $q_h$, let $\tilde q_h=\operatorname{clip}(q_h,10^{-9},1-10^{-9})$. Then

$$
\operatorname{logit}\!\left(q_h^{\mathrm{cal}}\right)
=a_h+b_h\operatorname{logit}\!\left(\tilde q_h\right),
\qquad b_h\ge0.
$$

Only the marginal total exit probability was calibrated. For $q_h>0$, spatial mass was rescaled as $p^{\mathrm{cal}}_{h,s,k}=q_h^{\mathrm{cal}}p^{\mathrm{raw}}_{h,s,k}/q_h$; a raw forecast with zero exit mass retained zero exit mass. Thus the nonzero raw conditional distribution over side and segment was preserved but was not itself calibrated, and spatial quality was evaluated through localization outcomes. Because horizons were calibrated separately, cross-horizon monotonicity was not imposed after fitting. We report every held-out row in which an adjacent cumulative probability decreased by more than $10^{-12}$.

## 5.3 Persistent localized proposals

The primary proposal score used the 1.50 s total-calibrated, spatially rescaled distribution. For a candidate side and center bin, the score summed probability over that side and the circular three-bin neighborhood centered on the candidate. A proposal opened only after the same side and segment neighborhood exceeded its threshold for three consecutive completed 20 Hz scores. The first proposal in the episode was non-overwritable. The system re-armed only after one continuous in-corridor second.

Each proposal was matched to the first qualifying event in its observable 1.50 s window. A correct-side, within-one-bin match counted as localized; when its actual proposal-to-onset lead was below a tested requirement $L$, it did not contribute to timely recall but was not reclassified as false. A wrong-location match counted both as a localized event miss and as a false proposal. A proposal with no qualifying event in its horizon was false, while an unresolved proposal was censored in the primary analysis. Future information never removed a causal proposal.

For each fold and method, the candidate grid comprised calibrated-score values at 100 equally spaced quantiles from 0 to 0.99, 401 upper-tail quantiles $1-10^{-q}$ for equally spaced $q\in[2,6]$, and endpoints 0 and 1; duplicate values were removed. For each false-proposal budget and required lead, calibration cars alone selected the 1.50 s proposal threshold that maximized localized event recall subject to the budget. Thus $H_M(L)$ and $F_M(L)$ below use the fold-specific threshold selected for method $M$ at that same $L$ and budget. These analytical operating gates were prespecified and sealed before held-out outcome access; they are not claims about a deployable protection system. Ties favored the higher threshold. A calibration operating point was estimable only if

$$
B\,E_{\mathrm{cal}}\ge1,
$$

where $B$ is the permitted false proposals per hour and $E_{\mathrm{cal}}$ is eligible calibration exposure in hours. Thus one false proposal could occur without automatically exceeding the requested budget. Unsupported rows remained explicitly not estimable and received no threshold.

## 5.4 Prespecified comparators

All six methods received the same causal states, maps, forecast horizons, calibration partitions, persistence, segment tolerance, scoring cadence, exposure, and held-out outcomes.

1. Constant velocity.
2. Constant turn rate and velocity.
3. The matched posterior-mean residual-dynamics twin.
4. A separately fitted deterministic residual model whose ridge penalty was selected by calibration-transition mean-squared error.
5. A regularized discrete-time side-specific hazard model.
6. BRACE, the Bayesian residual-dynamics particle forecast.

The side-hazard model had no native spatial trajectory. Its frozen adapter assigned left/right horizon mass to the circular bin reached by current centerline arclength plus nonnegative longitudinal speed times half the horizon. This intentionally simple rule permitted the common localization metric without using future geometry or test labels. A sequence neural network was omitted under a capacity rule fixed before held-out scoring because the cohort contained 294 events, below the prespecified 300-event gate.

# 6. Evaluation

## 6.1 Probability quality

Probability metrics used only causal in-corridor rows whose full outcome horizon was observable. We report the Brier score, a Brier skill score against the corresponding fold's calibration prevalence, and logarithmic score as proper-probability summaries [@Gneiting_2007]. Logistic calibration intercept and slope summarize calibration-in-the-large and spread [@VanCalster_2019_Calibration]. Ten-bin equal-width reliability and expected calibration error provide a descriptive, bin-dependent summary [@Guo_2017_Calibration]. We also report stepwise average precision grouped at distinct score thresholds [@Davis_2006_PrecisionRecall] and do not call it “trapezoidal PR AUC.” Reliability intervals resampled whole car sessions within their fixed circuits.

## 6.2 Warning frontier

Let $H_M(L)$ be the number of held-out qualifying events receiving a correct-side, within-one-bin first proposal at least $L$ seconds before onset under method $M$. Let $N$ be the held-out qualifying-event count, $F_M(L)$ the number of false proposals, and $E$ eligible simulated car-hours. At the primary budget $B=2$ and recall requirement $R=0.50$, define

$$
L_M^*=\max\left(
\{L\in\{0.25,0.50,1.00,1.50\}: H_M(L)/N\ge R,\ F_M(L)/E\le B\}\cup\{0\}
\right).
$$

The primary operational contrast is

$$
\Delta L^*=L_{\mathrm{BRACE}}^*-L_{\mathrm{twin}}^*.
$$

This grid-valued metric is actual proposal-to-onset lead, not the forecast horizon. A 1.50 s cumulative forecast can yield less than 1.50 s of actual persistent warning. We also report the entire operating table at 0.5, 1, 2, 5, and 10 false proposals per eligible simulated car-hour, together with correct-side, within-one-bin, exact-bin, and localized event recall. The primary analytical budget of two car-level false proposals per car-hour would scale arithmetically to 40 proposals per 20-car field-hour before any cross-car spatial or temporal deduplication. It is not a deployable field-level requirement.

## 6.3 Uncertainty and sensitivities

The primary interval used 10,000 paired bootstrap resamples of complete car sessions within each of the four fixed circuit sessions. Models, calibrators, thresholds, and their selection uncertainty were not refitted inside a bootstrap draw. The bootstrap paired BRACE and the deterministic twin on the same resampled cars; its interval is conditional on the observed circuits and fitted analysis pipeline. A two-stage circuit-then-car bootstrap assessed transport sensitivity, but only four circuit-session clusters were available; that interval is descriptive rather than a substitute for broader circuit sampling.

We computed one-sided 95% Poisson-model upper limits for false-proposal rates and reported the underlying counts and exposure. Persistence and re-arming can induce dependence, so these bounds are model-based rather than distribution-free. A least-favorable analysis added every unresolved proposal to the false count and recomputed $L^*$ and $\Delta L^*$ with fixed models and thresholds. Synthetic-delay analyses shifted proposal issue time by 40, 80, and 160 ms, subtracted that delay from actual lead, and reclassified a formerly matched proposal as false if it arrived after event onset. No delay analysis refitted a model, calibrator, or threshold.

# 7. Results

## 7.1 Cohort flow and estimability

{{COHORT_FLOW_RESULT}}

The primary two-false-proposals-per-hour grid was {{PRIMARY_GRID_ESTIMABILITY_STATUS}} across all four folds, six methods, and four required-lead values. {{ESTIMABILITY_DETAIL}}

## 7.2 Primary warning-time contrast

{{FIGURE_WARNING_FRONTIER_MARKDOWN}}

The prespecified endpoint values were $L^*={{PRIMARY_LSTAR_BRACE_S}}$ s for BRACE and $L^*={{PRIMARY_LSTAR_TWIN_S}}$ s for the matched posterior-mean twin. The resulting difference was ${{PRIMARY_DELTA_LSTAR_S}}$ s, with a paired car-session bootstrap 95% percentile interval from ${{PRIMARY_DELTA_LSTAR_CI_LOW_S}}$ to ${{PRIMARY_DELTA_LSTAR_CI_HIGH_S}}$ s. {{PRIMARY_RESULT_INTERPRETATION}}

{{PRIMARY_OPERATING_POINT_NARRATIVE}}

{{PRIMARY_POLICY_RESULT}} False-proposal counts must be interpreted together with the selected threshold and proposal count; abstention is not demonstrated specificity.

| Outer fold (held-out circuit) | Calibration exposure (h) | Calibration excursions | Highest active frozen-grid threshold below silence, BRACE / twin | False proposals/h at that threshold, BRACE / twin | Localized recall at least 1.50 s realized lead, BRACE / twin |
|:--|--:|--:|--:|--:|--:|
| Bahrain | 0.853466 | 46 | 0.17450 / 0.07650 | 162.87 / 877.60 | 0.000 / 0.000 |
| Britain | 1.075031 | 31 | 0.23100 / 0.02824 | 54.88 / 745.09 | 0.000 / 0.000 |
| Jeddah | 0.562759 | 52 | 0.50793 / 0.10263 | 60.42 / 645.04 | 0.000 / 0.000 |
| Monza | 1.056322 | 100 | 0.31944 / 0.08071 | 146.74 / 841.60 | 0.000 / 0.000 |

Table: Exploratory calibration-only operating-cliff diagnostic. Calibration data in each row come only from the three non-held-out circuits assigned to that outer fold. The displayed point is the highest active frozen-grid threshold below the all-silent threshold and was not selected for held-out use. Every displayed point exceeded 10 false proposals/h and produced no correctly localized proposal at least 1.50 s before onset. These are calibration diagnostics, not held-out estimates, and they do not exclude unexamined off-grid thresholds. {#tbl:threshold-abstention}

{{TABLE_PRIMARY_FRONTIER_MARKDOWN}}

The full budget frontier showed {{OPERATING_FRONTIER_SUMMARY}}. Wrong-location proposals were charged as specified; removing that cost was not part of the primary analysis.

## 7.3 Probability quality and localization

{{FIGURE_RELIABILITY_MARKDOWN}}

At the 1.50 s forecast horizon, the held-out BRACE Brier score was {{BRACE_BRIER_1P50}}, compared with {{TWIN_BRIER_1P50}} for the matched twin. BRACE's calibration-prevalence Brier skill score was {{BRACE_BRIER_SKILL_1P50}}, logarithmic score was {{BRACE_LOG_SCORE_1P50}}, and stepwise average precision was {{BRACE_AP_1P50}}. Its calibration intercept and slope were {{BRACE_CAL_INTERCEPT_1P50}} and {{BRACE_CAL_SLOPE_1P50}}, respectively, with ten-bin equal-width expected calibration error {{BRACE_ECE_1P50}}.

{{TABLE_PROBABILITY_METRICS_MARKDOWN}}

{{PROBABILITY_COMPARISON_RESULT}} {{METRIC_TO_DECISION_RESULT}}

{{ALL_METHOD_PROBABILITY_RESULT}} These probability metrics are reported alongside the operational warning frontier rather than treated as evidence of a feasible policy.

Across {{BRACE_MONOTONICITY_SCORE_ROWS}} held-out BRACE score rows, {{BRACE_MONOTONICITY_VIOLATION_COUNT}} had at least one adjacent calibrated cumulative probability decrease larger than $10^{-12}$, a rate of {{BRACE_MONOTONICITY_VIOLATION_RATE}}. Because the analysis froze separate horizon calibrators, these rows were reported rather than repaired after seeing held-out outcomes.

{{LOCALIZATION_COMPONENT_SUMMARY}}

## 7.4 Circuit transport and secondary comparators

{{TABLE_CIRCUIT_RESULTS_MARKDOWN}}

{{CIRCUIT_HETEROGENEITY_SUMMARY}}

The prespecified kinematic, independently refitted deterministic, and side-hazard comparators produced {{SECONDARY_BASELINE_SUMMARY}}. These models contextualize the difficulty of the task. The posterior-mean twin is the primary matched comparator, but the contrast reports the held-out difference between the complete uncertainty-propagating pipeline and its deterministic counterpart; it does not separately identify coefficient-uncertainty and process-noise effects. Uniform circuit ties, if observed, would not demonstrate transportability.

## 7.5 Unresolved follow-up and synthetic delay

Under the primary censoring rule, {{UNRESOLVED_PROPOSAL_SUMMARY}}. Counting every unresolved proposal as false left BRACE $L^*$ at {{LF_LSTAR_BRACE_S}} s and the twin $L^*$ at {{LF_LSTAR_TWIN_S}} s, giving $\Delta L^*={{LF_DELTA_LSTAR_S}}$ s (95% interval, {{LF_DELTA_LSTAR_CI_LOW_S}} to {{LF_DELTA_LSTAR_CI_HIGH_S}} s).

{{TABLE_LEAST_FAVORABLE_MARKDOWN}}

{{TABLE_DELAY_RESULTS_MARKDOWN}}

With 160 ms of added issue delay and no refitting, the primary contrast was ${{DELAY_160_DELTA_LSTAR_S}}$ s. {{DELAY_SENSITIVITY_SUMMARY}} {{DELAY_RESULT_INTERPRETATION}}

## 7.6 Reproducibility and computation

Across the four held-out folds, all recorded stages summed to {{TOTAL_EXPERIMENT_RUNTIME_HOURS}} aggregate compute-hours on {{COMPUTE_ENVIRONMENT}}. This sum is not end-to-end elapsed time. {{INFERENCE_THROUGHPUT_SUMMARY}} The offline batch throughput is not measured real-time deployment latency. All reported tables were generated from authenticated held-out artifacts after the threshold-freeze seal. The pooled held-out manifest SHA-256 was `{{POOLED_MANIFEST_SHA256}}`; the primary bootstrap manifest SHA-256 was `{{BOOTSTRAP_MANIFEST_SHA256}}`. The analysis used frozen seed 20270927 and 10,000 bootstrap resamples.

# 8. Discussion

## 8.1 Did the complete BRACE pipeline add warning time?

{{PRIMARY_DISCUSSION_PARAGRAPH}}

The matched design is central to that interpretation. BRACE and the twin use the same training rows, state and residual features, physics mean, posterior residual object, held-out circuits, and prespecified evaluation rule. They differ in coefficient propagation, process noise, Monte Carlo support, and the downstream method-specific calibrators and thresholds. The study therefore reports the held-out difference between those two complete pipelines; it does not identify separate causal effects for coefficient uncertainty or process noise. The observed contrast does not prove that uncertainty is useful or useless in other datasets, models, or operating regimes.

$L^*$ forces probability performance to confront the intended decision. {{METRIC_TO_DECISION_RESULT}} Reporting probability scores, localization components, selected thresholds, and the full burden frontier alongside $L^*$ prevents a favorable ranking metric from being mistaken for a decision result.

## 8.2 Operational interpretation

BRACE outputs a marginal excursion probability passed through a calibration-car-fitted monotone map plus localized forecast mass over future boundary side and segment; it does not produce calibrated zone probabilities or a binary crash alarm. That structure could support a future shadow-mode workflow that ranks which pre-installed circuit section deserves attention before a crossing. The evidence-derived policy result is reported in Section 7; no result in this paper authorizes that operational role.

The primary analytical budget also illustrates the deployment gap. At two false proposals per car-hour, a 20-car field could generate 40 car-level proposals per field-hour before cross-car deduplication; the observed held-out rate must therefore be interpreted alongside a field-scale equivalent. No human or automated response is validated within the maximum 1.50 s horizon. The study measures forecast behavior, not intervention actionability.

The results do not establish that an active mat, padding system, or selectable barrier would help. A real protective system would require the predicted trajectory to reach a surveyed physical barrier after leaving the track. It would also require a qualified device state to become ready before contact without introducing new hazards to the incident car, other cars, marshals, or spectators. Those links are unmeasured here. Prediction performance is therefore one research layer, not a deployment authorization.

## 8.3 Circuit transport

Holding out entire circuits is more demanding than splitting frames from the same circuit, but four circuits are still a small transport sample. Bahrain and Jeddah contribute most exposure, while Monza contributes only 0.345935 eligible hours. A pooled result can consequently mask circuit-specific success or failure. We report every circuit separately and treat the circuit-then-car bootstrap as a sensitivity analysis because resampling four top-level clusters cannot approximate a broad population of circuits.

## 8.4 Why a compact statistical model?

We chose a compact residual model because it permits a closely matched posterior-mean twin, explicit posterior propagation, deterministic batch-invariant replay, and a direct audit of what changed. This subtraction strengthens the experiment: the paper compares two tightly matched pipelines rather than presenting a collection of loosely connected models. A sequence model was not part of the prespecified comparison, so the results should not be read as a claim that the compact model dominates richer architectures.

# 9. Limitations

First, the dataset is simulated and derived from a commercial Formula 1 game. It contains 46.54 qualifying excursions per eligible car-hour, with circuit rates from 14.16 to 120.30 per hour. These values are benchmark prevalence, not estimates of real-racing off-track events or crashes. Vehicle behavior, circuit representation, control policies, and excursion frequency can differ from real racing, and probability calibration need not survive that prevalence shift. The public license and numerical provenance do not remove all game, brand, circuit, or redistribution considerations. We therefore describe the data as open-wheel simulation and do not claim Formula 1 validation.

Second, a mapped car-center boundary crossing is not a crash or physical contact. The source has no adjudicated loss-of-control onset, first-contact time, contacted object, impact speed, impact angle, damage, or injury outcome. Some qualifying excursions can be benign, and some physical incidents can occur without the car center satisfying our event rule. The endpoint is useful for a falsifiable public warning study but cannot support claims about crash prevention or padding efficacy.

Third, the information clock is reconstructed. The compact release has no native per-frame packet source and receipt timestamps. Synthetic 40--160 ms delays test arithmetic sensitivity to later issue times; they are not measured communication or actuator latencies. A deployable system would need packet-age tracking, end-to-end timing trials, command acknowledgement, readiness confirmation, and failure-state evidence.

Fourth, the model lacks driver inputs, tire state, grip, weather, elevation dynamics, other-car state, damage, and surface contamination. These omissions limit both predictive accuracy and the interpretation of process uncertainty. The Bayesian bootstrap reflects sampling variation in a small residual model; it is not a complete physical uncertainty model.

Fifth, repeated frames and excursions are strongly dependent. We resample whole cars and retain circuits in the primary bootstrap, but only four circuit sessions exist. The scheme targets car-session variation conditional on these circuits and the fixed fitted pipeline. {{BOOTSTRAP_VARIATION_LIMIT}}It does not include model-, calibration-, or threshold-selection uncertainty and does not establish transport to unseen circuit populations.

Sixth, threshold selection is constrained by short calibration exposure. The exposure-only estimability rule prevents unsupported low-budget thresholds, but it does not make a few observed false proposals precise. Poisson-model upper limits, raw counts, exposure, and the full operating table should be read alongside point rates.

Seventh, Gaussian residual propagation is unbounded. The production artifacts do not retain particle-level speed, acceleration, or yaw-rate plausibility diagnostics, so early tail crossings could reflect both uncertainty and implausible simulated states. External validation should retain these diagnostics and include covariance-scale sensitivity before the forecast is used for engineering decisions.

Finally, the lead grid is coarse. $L^*$ can take only 0, 0.25, 0.50, 1.00, or 1.50 s. The resulting bootstrap distribution is discrete, and a small change in recall around 0.50 can move the endpoint by an entire grid step. The full recall and false-proposal frontier contains more information than $L^*$ alone.

# 10. Safety, ethics, and governance

The study uses vehicle simulation streams and circuit geometry; it does not use medical records or infer individual injury risk. Raw third-party data should be redistributed only when license and underlying rights permit. The release should otherwise provide immutable acquisition instructions, hashes, derived tables permitted for redistribution, environment locks, and exact build commands.

Any future physical intervention must remain outside the authority of this paper. Surveyed barrier geometry, zone-to-device mapping, passive comparators, communication security, fault states, unintended entrants, mixed device states, and full-scale impacts are separate requirements [@FIA_2018_BarrierStandard; @FIA_2025_Homologation]. Any proposed FIA-sanctioned deployment would require the applicable governing-body approval pathway; equipment classified under FIA Standard 3501-2017 would require its specified homologation and approved-test-house assessment. No worker should be placed in a sub-second response loop, and the model should never be interpreted as a command to deploy loose material onto a live run-off area.

# 11. Conclusion

We introduced BRACE and a circuit-held-out comparison of a complete uncertainty-propagating pipeline with the posterior-mean, no-process-noise twin of the same residual model. The study uses 58 complete car streams, 294 frozen excursion events, and 6.317055 eligible simulated car-hours.

The endpoint difference was $\Delta L^*={{PRIMARY_DELTA_LSTAR_S}}$ s with 95% interval ${{PRIMARY_DELTA_LSTAR_CI_LOW_S}}$ to ${{PRIMARY_DELTA_LSTAR_CI_HIGH_S}}$ s. {{CONCLUSION_RESULT_SENTENCE}} {{METRIC_TO_DECISION_RESULT}}

The current simulated evidence does not justify an active protection response. Whether additional uncertainty-propagation complexity is warranted under this operating rule must follow the evidence-derived comparison above, not a static model preference. The next scientific task is not to tune against the held-out circuits, but to test the proposal mechanism prospectively with native clocks, richer vehicle and driver inputs, field-level alert accounting, and surveyed barrier-contact outcomes. Until then, the auditable finding is narrower but useful: frame-level probability metrics and the operational endpoint must be evaluated separately. Real crash, barrier-contact, and protection claims require a different evidence layer.

# Data and code availability

A reproducibility package has been prepared locally but is not yet publicly released. It contains the pinned acquisition manifest, source code, frozen configuration, derived manifests, fitted calibration and threshold artifacts, pooled metrics, bootstrap outputs, figure-source tables, tests, bibliography, manuscript, and environment lock. Before submission, the permanent repository URL, archived commit, software license, and exact rights-reviewed artifact inventory must be added. Exact reacquisition instructions and hashes should replace redistribution of third-party raw files when rights are unresolved; unsafe legacy pickles, caches, secrets, and temporary renders must remain excluded.

# Appendix A. Frozen analysis constants

| Component | Frozen value |
|:--|:--|
| Forecast horizons | 0.25, 0.50, 1.00, 1.50 s |
| Primary proposal horizon | 1.50 s |
| Required actual-lead grid | 0.25, 0.50, 1.00, 1.50 s |
| Primary false-proposal budget | 2 per eligible simulated car-hour |
| Primary localized-recall requirement | 0.50 |
| Segment length / tolerance | 25 m / one adjacent bin |
| Persistence / re-arm | 3 completed scores / 1 continuous in-corridor s |
| Physics integration step | 0.05 s |
| Bayesian coefficient draws / forecast particles | 256 / 256 |
| Ridge grid | 0.01, 0.1, 1, 10, 100 |
| Probability calibration | Horizon-specific monotone logistic recalibration |
| Bootstrap | 10,000 paired car-session-within-circuit resamples |
| Frozen seed | 20270927 |

# References
