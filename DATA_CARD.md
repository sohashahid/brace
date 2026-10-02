# Data Card: BRACE DeepRacing Boundary-Excursion Cohort

## Summary

The BRACE primary cohort is a derived research dataset for forecasting **mapped car-center track-boundary excursions in open-wheel racing simulation**. It contains complete ordinary-running exposure and event labels reconstructed from compact DeepRacing trajectory files and fixed circuit maps.

It is not a crash, impact, barrier-contact, injury, or active-padding dataset. It contains no real Formula 1 telemetry and cannot validate FIA deployment or a physical protection device.

## Source and version

- Upstream dataset: [DeepRacing racing trajectory prediction](https://huggingface.co/datasets/deepracing/racing_trajectory_prediction).
- Pinned revision: `1e7ec1dfdb6a3cde726d377b0403b51fa51ab2c4`.
- Field-semantics reference: `linklab-uva/deepracing` commit `c4127ae1c5ce3f6d9b8b8c7e87872aabce01a781`.
- Acquisition manifest: `data/manifests/deepracing-files.csv`.
- Cohort build manifest: `data/manifests/deepracing-build.json`.
- Detailed source audit: `data/deepracing-audit.md`.
- Freeze date: 27 September 2026.

The acquisition manifest is authoritative for each source object's pinned URL, remote object ID, byte count, SHA-256 digest, local path, role, and declared license.

## Composition

| Circuit session | Car sessions | Qualifying events |
|:--|--:|--:|
| Bahrain | 20 | 36 |
| Britain | 11 | 113 |
| Jeddah | 20 | 106 |
| Monza | 7 | 39 |
| **Total** | **58** | **294** |

The pinned compact acquisition contains:

- 244 retained files totaling 86,360,332 bytes;
- 232 per-car files: `metadata.yaml`, `motion_data.npz`, `lap_data.npz`, and `session_data.npz` for each of 58 cars;
- 12 PCD map files: one centerline and two boundary loops for each circuit;
- 2,473,009 raw motion/lap rows;
- 460,269 reconstructed causal 20 Hz frames;
- 1,616 candidate excursion runs before the frozen qualification filters;
- 294 qualifying events over 6.3170549663 eligible simulated car-hours.

The four circuit sessions are the top-level environmental clusters. The 294 events are not 294 independent crashes, and millions of frames are not millions of independent experimental units.

## Source fields

The compact source provides world position and velocity, body-frame acceleration, orientation quaternion, lap distance, total distance, lap number and lap time, result/driver/pit state, and lower-rate session metadata. The maps provide centerline and corridor-edge geometry.

The compact motion files do not provide a native timestamp or frame identifier per motion row. Time is reconstructed from lap clocks, then causally resampled to the metadata-declared 20 Hz grid. The data do not provide collision state, first-contact time, contacted object, barrier geometry, damage, injury, weather, race-control intervention, or protection-device state.

## Derivation and labeling

1. Transform world-frame car-center coordinates to each circuit's map frame using the metadata origin and quaternion.
2. Construct the drivable corridor geometrically as the area inside the larger boundary loop and outside the smaller loop. Filenames do not determine topology; Jeddah's named inner and outer files are reversed geometrically.
3. Reconstruct elapsed time from lap clocks and place hard breaks at state changes, clock resets, large position/distance discontinuities, and large data gaps.
4. Define an outside run from consecutive eligible car-center samples outside the mapped corridor. Merge runs only across at most 0.25 seconds of observed in-corridor time and never across a hard break.
5. Qualify an event when maximum boundary depth is at least 0.50 m and duration is at least 0.25 seconds.
6. Exclude pit-intersecting events and events within the frozen edge, lap, reset, or data-gap artifact buffers.
7. Record onset side and centerline arclength segment at the first outside sample. Primary localization uses the correct side and the same or an adjacent circular 25 m segment.

Offline symmetric artifact buffers determine label and exposure eligibility only. They do not suppress online model inputs or proposals.

## Derived artifacts

| Path | Contents | Release note |
|:--|:--|:--|
| `data/processed/deepracing_frames.parquet` | Causal 20 Hz features and current map geometry | 96,194,642 bytes; use Git LFS or a release asset if redistribution is approved. |
| `data/processed/deepracing_targets.parquet` | Separately keyed future outcomes and horizon evaluability | Keep physically separate from causal features. |
| `data/processed/deepracing_events.csv` / `.parquet` | Candidate events, qualification, rejection reasons, and localization | Not crash labels. |
| `data/processed/deepracing_loco_splits.csv` | Whole-car leave-one-circuit-out fit/calibration/test assignments | Frozen before held-out scoring. |
| `data/processed/deepracing_map_manifest.json` | Map hashes, transforms, loop roles, lengths, and segment definitions | Map geometry may require a separate rights review. |
| `data/processed/deepracing_timing_diagnostics.csv` | Reconstructed-clock and gap/reset diagnostics | Does not establish a native high-rate clock. |
| `data/processed/deepracing_kinematic_diagnostics.csv` | Frame and acceleration consistency checks | Diagnostic only. |

Fold predictions, proposal logs, metrics, and bootstrap inputs are produced under `output/experiment/` and `output/bootstrap-*`. The authenticated publication bundle is produced under `output/submission-ready/analysis-output/`. Their manifests authenticate exact file hashes.

## Partitions and leakage controls

The outer split leaves one entire circuit out. Within the other three circuits, whole car sessions are assigned to model-fit or calibration partitions using the frozen salted-hash rule in `configs/study.json`. No frame, outcome, normalization statistic, calibration fit, or threshold from the held-out circuit enters fitting or threshold selection.

Held-out outcomes cannot be joined until all four calibration threshold bundles are complete and `output/experiment/threshold-freeze-seal.json` has been created. This procedural gate is enforced by the experiment CLI.

## Intended uses

- Reproduce the BRACE simulated boundary-excursion forecasting benchmark.
- Study circuit-held-out probability calibration, warning lead, false-proposal rates, and coarse crossing localization.
- Compare the uncertainty-propagating model with prespecified deterministic and hazard baselines under identical causal inputs.
- Audit causal timing reconstruction, map geometry, event labeling, and cluster-aware uncertainty analysis.

## Out-of-scope uses and claims

Do not use this cohort to claim or estimate:

- real Formula 1 crash or barrier-contact probability;
- driver injury or crash-severity reduction;
- an impact location on a real circuit barrier;
- FIA readiness, homologation, or operational authorization;
- safe activation, latency, fault tolerance, or effectiveness of active mats or padding;
- prevalence from the outcome-enriched AssettoCorsaGym subset;
- broad circuit transport from only four top-level circuit sessions.

A future active-protection study needs native packet source/receipt timestamps, adjudicated loss-of-control and contact truth, surveyed barrier geometry, environment and race-control state, actuator command/acknowledgement/fault logs, unintended-entrant outcomes, validated simulation, and full-scale impact tests.

## Known limitations

- All primary observations come from a commercial racing-game simulation domain.
- Time is reconstructed rather than observed at every raw motion sample.
- The endpoint is a car-center corridor crossing, not full-vehicle envelope contact.
- The maps are track edges, not surveyed FIA barrier polygons.
- There are only four top-level circuit-session clusters.
- Side and segment probabilities inherit the forecast's conditional spatial allocation; only marginal total exit probability is calibrated.
- Primary intervals condition on the observed four circuits and fixed fitted models, calibrators, and thresholds; they do not include model-selection uncertainty or establish population-wide transport.
- The public source lacks controls, tire state, damage, weather, grip, traffic semantics, and contact severity.

## Licensing, attribution, and redistribution

The DeepRacing dataset card declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve DeepRacing attribution and the pinned revision in any permitted redistribution or derivative use.

The source derives from a commercial Formula 1 video game. The dataset-card license is not a warranty that the publisher controls all game, Formula 1, circuit, team, sponsor, or branded rights. A separate rights review is required before redistributing raw source objects, mapped geometry, screenshots, or other branded material.

Raw third-party data are intentionally excluded from ordinary Git. Release the acquisition manifest and checksums so an independent researcher can reacquire exact upstream objects. Publish only derived data that have passed the rights review, using Git LFS or immutable release assets when appropriate.

## Security and privacy

The retained primary files are numerical simulation arrays, YAML metadata, and PCD geometry; they contain no known personal identifiers. Third-party legacy pickle files are not part of the primary pipeline and must not be loaded with unrestricted pickle deserialization. Credentials and confidential partner material are prohibited from the public release.

## Validation performed

- All 244 retained source sizes and SHA-256 digests match the acquisition manifest.
- All 174 retained NPZ archives were opened with `allow_pickle=False`.
- Their 1,508 members contain no object arrays.
- All 12 PCD files passed header and row-count checks.
- The complete derived build is bound to `data/manifests/deepracing-build.json`.

## Maintenance and issue reporting

The public repository URL and maintainer contact have not yet been assigned. Until they are added, report data defects to the study authors through the conference submission channel. Any correction that changes source hashes, event counts, exposure, splits, or held-out outputs requires a new version and a documented rerun; do not silently overwrite the frozen cohort.
