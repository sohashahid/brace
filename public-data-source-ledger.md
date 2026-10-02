# BRACE-F1 Public Data Source Ledger

**Audit date:** 27 September 2026  
**Decision:** No public source contains FIA-grade Formula 1 telemetry, exact first-contact time, contacted barrier segment, complete non-incident exposure, weather, and original receipt timestamps in one rights-clean dataset. The executable study must combine a public Formula 1 context layer with an open high-frequency racing/simulation layer and must narrow its claim to the observed endpoint.

## A. Formula 1 sources

| Source | Useful data | Coverage / cadence | License or rights status | BRACE use | Hard limit |
|---|---|---|---|---|---|
| [F1DB](https://github.com/f1db/f1db) | Races, sessions, results, grids, pits, circuits, entrants, chassis, tyres | 1950–present | CC BY 4.0 | Cohort and incident-candidate backbone | No high-rate telemetry or contact time/location |
| [Lap Ledger](https://lapledger.org/data) | Results metadata, classifications, pits, circuits, provenance, some OSM geometry | Historical through current season | CC BY-SA 4.0; OSM subset ODbL | Rights-clean metadata cross-check | Deliberately excludes telemetry, lap timing, race control, and stints |
| [Jolpica-F1](https://api.jolpi.ca/ergast/f1/) | Results, status, qualifying, sprints, laps, pits | Historical; lap timing mainly from 1996 | CC BY-NC-SA 4.0 | Retirement/incident candidates and race-lap denominator | No first-contact time or location; not complete car-second exposure |
| [OpenF1](https://openf1.org/docs/) | Car data, approximate `x/y/z`, laps, positions, pits, stints, race control, weather, results | Historical from 2023; car/location about 3.7 Hz | Repository CC BY-NC-SA 4.0; underlying Formula 1 rights remain unresolved | Public F1 feasibility, complete-session replay, weak incident labels | No exact barrier, lateral placement, high-rate state, impact clock, or original receipt time |
| [FastF1](https://github.com/theOehrly/Fast-F1) | Similar F1 timing and telemetry | Mainly 2018–present; car about 240 ms and position about 220 ms | MIT covers software, not the underlying F1 data | Cross-checking only unless data rights are resolved | Maintainers caution against raw/bulk redistribution |
| [FIA event documents](https://www.fia.com/documents) | Classifications, timing reports, race-control messages, steward decisions | Per event | Official sources; no general open data license | Manual verification and source-linked annotations | Do not redistribute PDFs/media in the data package without permission |
| [Formula 1 content guidelines](https://www.formula1.com/en/information/guidelines.4EOKE9RRqevL4niTK9kWyt) | Rights boundary | Current | F1 asserts rights over timing/results, audiovisual material, graphics, circuit outlines, and data-mining uses | Determines release constraints | Open-source wrappers do not automatically clear upstream F1 rights |

**Release rule for F1-derived data:** until written clarification is obtained, publish acquisition code, immutable query manifests, hashes, schemas, source URLs, and independently created interval labels. Do not mirror raw timing or audiovisual content merely because an API wrapper is open source.

## B. Maps, circuit geometry, and weather

| Source | Useful data | License | Use | Limitation |
|---|---|---|---|---|
| [OpenStreetMap](https://www.openstreetmap.org/copyright), [Geofabrik](https://download.geofabrik.de/), [ohsome](https://api.ohsome.org/) | Dated WGS84 raceway centerlines, some widths/surfaces/barriers, historical snapshots | ODbL 1.0 | Fixed-date public circuit context | Accuracy and barrier coverage vary; not FIA survey geometry |
| [TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database) | Local-metre centerlines, left/right widths, racelines for more than 20 circuits | LGPL-3.0 repository; imagery-derived widths require provenance review | Development envelope | Not authoritative barriers; quality varies |
| [track-atlas](https://github.com/tobi/track-atlas) | Centerlines, corners, pit points, selected surface edges | Layer-specific: MIT/ODbL/other terms | Audited geometry layers only | Must preserve per-layer provenance |
| [Open-Meteo historical API](https://open-meteo.com/en/docs/historical-weather-api) | Temperature, humidity, precipitation, pressure, wind, radiation | CC BY 4.0 | Weather sensitivity and missing OpenF1 weather checks | Reanalysis/grid weather, not local track-surface grip |
| [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Observed station weather | US government distribution; inspect source notices | Event weather cross-check | Station distance and sampling vary |

Do not use `bacinger/f1-circuits` as the release geometry: the repository states that initial geometry came from Google despite an MIT file.

## C. High-frequency racing datasets

| Rank | Dataset | Key contents | Size and access | License | Best use |
|---:|---|---|---|---|---|
| 1 | [AssettoCorsaGym](https://huggingface.co/datasets/dasgringuen/assettoCorsaGym) | 50 Hz, 64M steps, 900+ laps, controls, full motion state, wheel/tyre state, dense track borders, tires-out/out-of-track/termination fields | 136.86 GiB, ungated; select shards/stream because the local disk cannot hold the full release | CC BY 4.0 | Primary calibrated first-boundary-crossing experiment; track exit, not impact |
| 2 | [DeepRacing trajectory prediction](https://huggingface.co/datasets/deepracing/racing_trajectory_prediction) | F1-game data at 20 Hz, 3 s history/future, pose, velocity, acceleration, angular velocity, boundaries, centerline, optimal line | 56.23 GiB, ungated; selective test bundles only | CC BY 4.0 | F1-style external trajectory/boundary test; no crash labels |
| 3 | [RACECAR/IAC](https://registry.opendata.aws/racecar-dataset/) | Real AV-21 racing to 170 mph; GNSS 20 Hz, raw IMU 125 Hz, lidar, radar, cameras, opponents | 2.167 TiB anonymous S3; download only selected state/log subsets | CC BY-NC 4.0 | Real high-speed trajectory and uncertainty transport; no impact labels |
| 4 | [MixNet data](https://doi.org/10.5281/zenodo.6954020) | 10 Hz history/future trajectories, left/right boundaries, HIL IAC data | 1.046 GiB direct ZIP | CC BY 4.0 | Immediately feasible trajectory baseline and boundary-conditioned test |
| 5 | [A2RL Vmax](https://huggingface.co/datasets/a2rl-vmax/a2rl-vmax) | Real Super Formula cars on Yas Marina; raw GNSS/INS/wheel, lidar/radar, multi-car scenes | About 67–72 GB; gated | CC BY-NC 4.0 | Multi-car transfer after access; no localization product or crash labels |
| 6 | [F1TENTH real-car driving](https://doi.org/10.5281/zenodo.12536536) | Nine physical runs with Vicon twist and commanded velocity | 0.993 GiB | CC BY 4.0 | Small dynamics/system-identification check |
| 7 | [DTU Roadrunners](https://data.dtu.dk/articles/dataset/14229995) | Physical test-track lidar, stereo/IMU, RTK GPS, pose, speed, acceleration | 4.293 GiB | CC BY-NC-SA 4.0 | Mapping/state-estimation check; no incidents |

`BETTY` is not counted as available: its download still resolves to “COMING SOON.”

## D. Generic incident data

| Source | Contents | License | Use | Limitation |
|---|---|---|---|---|
| [VZCrash](https://huggingface.co/datasets/vzc-research-chapter/VZCrash) | About 190k 16-second road events, more than 31k verified crashes, 100 Hz accelerometer/gyroscope, 1 Hz GPS, three-reviewer labels | CC BY-NC 4.0 | Generic inertial pretraining and hard negatives | Road vehicles, not racecars; no first-contact object/zone |
| FIA public accident investigations | Selected facts on major accidents, including speed, angle, sequence, or location | Citation use only; no open media/data license | Small external plausibility case series | Too few and selected for calibration or prevalence |

FIA WADB is confidential and is not a public dataset.

## E. Open simulation and geometry generation

| Source | Capability | License | Use |
|---|---|---|---|
| [Project Chrono / Chrono::Vehicle](https://www.projectchrono.org/) | Vehicle dynamics, rigid/flexible bodies, collision shapes, contact callbacks, exact contact point/normal/force/time | BSD-3-Clause | Generate exact, reproducible first-contact labels |
| [Open-Car-Dynamics](https://github.com/TUMFTM/Open-Car-Dynamics) | Open multibody race-car model validated against Dallara AV-21 data | Apache-2.0 | High-fidelity race dynamics before contact; couple to Chrono or swept geometry |
| [CARLA](https://github.com/carla-simulator/carla) | OpenDRIVE roads, collision sensor, cameras/lidar/radar | Code MIT; assets CC BY | Sensor pipeline experiments, not F1-quality mechanics |
| [MapZoo](https://github.com/zhouhengli/MapZoo) | 32 racing occupancy maps, boundaries, centerlines, racelines | MIT | Lightweight simulated geometry |
| [F1TENTH racetracks](https://github.com/f1tenth/f1tenth_racetracks) | Downscaled real-circuit occupancy maps, boundaries, widths, racing lines | GPL-3.0 | Additional simulated circuit holdouts |

## F. Frozen empirical scope decision

The executable primary endpoint is:

> At a fixed false-alert budget, how much earlier can a calibrated Bayesian detector correctly forecast the first track-boundary crossing and its side/segment than tuned kinematic baselines?

Planned evidence layers:

1. **Primary:** selectively acquired AssettoCorsaGym sessions, using complete runs and real `out_of_track`/termination labels.
2. **External F1-style test:** selected DeepRacing circuit bundles for trajectory and boundary transport.
3. **Real high-speed transfer:** MixNet immediately; selected RACECAR state logs if small downloadable objects can be isolated.
4. **Formula 1 context:** OpenF1/F1DB/Jolpica only at the resolution and under the rights boundaries described above.
5. **Exact contact extension:** Project Chrono/Open-Car-Dynamics simulation, explicitly labeled synthetic.

The manuscript must say `first track-boundary crossing`, not `barrier impact`, for the empirical result. Exact barrier activation remains the intended future application and requires Tier C data from the master data specification.

