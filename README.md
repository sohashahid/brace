# BRACE: Bayesian Racing-Envelope Alert for Crossing Events

BRACE is a reproducible benchmark for forecasting **mapped car-center track-boundary excursions in open-wheel racing simulation**. It asks whether an uncertainty-propagating Bayesian residual-dynamics forecast provides more correctly localized warning lead than a matched posterior-mean deterministic twin when an entire circuit is held out from fitting and calibration.

This repository does **not** contain evidence about real Formula 1 crashes, barrier contact, injury reduction, FIA readiness, or the effectiveness or safety of active padding. Those are future validation questions that require different data and physical testing.

## Repository status

As of 2 October 2026, the authenticated four-circuit experiment, primary bootstrap, transport sensitivity, and empirical manuscript are complete. This directory has not been initialized as a Git repository and nothing has been pushed to a public host. The public URL, software license, rights-reviewed release inventory, and fresh-clone verification remain release blockers.

The MIT Sloan Sports Analytics Conference requires an open-source repository link with the abstract submission. Add the final public repository URL only after the release checks in [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) pass. See the [official SSAC Research Paper Competition page](https://www.sloansportsconference.com/research-paper-competition).

## Frozen study

- Dataset: [DeepRacing racing trajectory prediction](https://huggingface.co/datasets/deepracing/racing_trajectory_prediction/tree/1e7ec1dfdb6a3cde726d377b0403b51fa51ab2c4), pinned revision `1e7ec1dfdb6a3cde726d377b0403b51fa51ab2c4`.
- Domain: commercial racing-game simulation, not live Formula 1 telemetry.
- Cohort: 58 complete car sessions from Bahrain (20), Britain (11), Jeddah (20), and Monza (7).
- Source inventory: 244 retained files, 86,360,332 bytes, and 2,473,009 raw motion/lap rows.
- Causal grid: 460,269 reconstructed 20 Hz frames.
- Endpoint: 294 non-overlapping qualifying mapped car-center boundary excursions—Bahrain 36, Britain 113, Jeddah 106, and Monza 39.
- Eligible exposure: 6.3170549663 simulated car-hours.
- Primary design: four leave-one-circuit-out folds; whole-car fit, calibration, and test assignments.
- Primary endpoint: greatest prespecified actual warning lead in `{0.25, 0.50, 1.00, 1.50}` seconds with localized event recall at least 0.50 and no more than two false proposals per eligible simulated car-hour.
- Localization: correct boundary side and the same or an adjacent circular 25 m segment.
- Uncertainty: 10,000 paired complete-car-session bootstrap resamples within the four fixed circuits; a circuit-then-car bootstrap is descriptive because there are only four top-level circuit sessions.

The complete frozen protocol is in [`public-study-protocol.md`](public-study-protocol.md), and data provenance and limitations are in [`DATA_CARD.md`](DATA_CARD.md).

## Measured result

- BRACE and its matched twin both received the prespecified no-qualifying-lead sentinel, $L^*=0.00$ s, at the primary two-false-proposals-per-car-hour gate.
- All 10,000 paired bootstrap contrasts were zero because both selected policies abstained; the resulting `[0,0]` interval is endpoint degeneracy, not evidence of precise equivalence.
- BRACE improved 1.50-second frame-level ranking and log loss relative to the twin (stepwise average precision `0.064552` versus `0.032225`; log score `0.097476` versus `0.197694`) but had a worse Brier score (`0.020975` versus `0.020069`).
- Calibration selected threshold `1.0` for both methods in every fold at budgets of 2, 5, and 10 false proposals per car-hour. At the highest active frozen-grid threshold below silence, BRACE still produced `54.88` to `162.87` false proposals per hour across folds; the twin produced `645.04` to `877.60`.

The paper's conclusion is therefore **no demonstrated gain under the prespecified gates**. The current pipeline is an operationally failed simulator prototype, not a crash detector or protection-system authorization.

## What BRACE predicts

At each score time, BRACE estimates the probability of crossing the mapped corridor boundary within 0.25, 0.50, 1.00, and 1.50 seconds, together with crossing side and circuit segment. A 256-particle forecast propagates Bayesian coefficient uncertainty and empirical residual process noise. The primary controlled comparator uses the same fitted residual dynamics at the posterior-mean coefficients with process noise omitted.

The calibrated marginal quantity is the total probability of an excursion within each horizon. The side/segment distribution is preserved conditionally from the raw forecast; it is not independently calibrated.

## Project layout

```text
configs/study.json                  frozen study and analysis contract
data/manifests/                     immutable source and build manifests
data/processed/                     derived cohort tables
paper/                              manuscript and SSAC abstract sources
src/brace_f1/                       cohort, model, replay, metric, and bootstrap code
scripts/build_submission_results.py authenticated result-to-paper compiler
tests/                              unit and end-to-end tests
output/experiment/                  fold and pooled held-out artifacts
output/bootstrap-primary/           primary paired-bootstrap bundle
output/bootstrap-two-stage/         descriptive transport-sensitivity bundle
output/submission-ready/            authenticated paper, tables, figures, and report
output/pdf/                         canonical visually inspected paper after verification
```

Downloaded third-party raw files under `data/raw/` are deliberately outside Git. See [Data access and rights](#data-access-and-rights).

## Environment

The experiment project requires Python 3.12 or newer and uses the committed `uv.lock` file.

```bash
uv sync --locked --extra test
uv run --locked pytest -q
uv run --locked ruff check .
```

The commands above verify code. They do not reacquire the third-party dataset or rerun the full experiment.

The authenticated result compiler uses a separate hash-pinned publication environment so figure and PDF packages do not alter the experiment environment:

```bash
uv venv .venv-publication --python 3.12
uv pip sync \
  --python .venv-publication/bin/python \
  publication-requirements.lock \
  --require-hashes
```

Publication rendering also requires `pandoc` and a working TeX PDF engine on `PATH`. The compiler records their checks and the exact publication-lock hash in the submission manifest.

## Data acquisition and cohort build

The authoritative acquisition ledger is `data/manifests/deepracing-files.csv`. Each of its 244 rows records the pinned URL, remote object ID, byte count, SHA-256 digest, local path, role, and declared license. Reacquire those exact objects from the pinned DeepRacing revision and preserve the manifest paths under `data/raw/deepracing/`.

Do not substitute the much larger publisher-generated `train_data.npz`, `test_data.npz`, `val_data.npz`, or pickle caches: they are not part of the frozen cohort. Do not load any third-party pickle with unrestricted `pickle.load`.

From the repository root, validate and rebuild the derived cohort:

```bash
uv run --locked brace-f1-build-deepracing \
  --manifest data/manifests/deepracing-files.csv \
  --output-dir data/processed \
  --build-manifest data/manifests/deepracing-build.json \
  --base-dir ..

uv run --locked brace-f1-experiment validate
```

The manifest records the frozen source identities. Reacquiring the pinned objects and rebuilding under a checkout named `brace-f1-ssac27` should reproduce the recorded content hashes, counts, labels, and numerical tables; host paths and generation timestamps are not expected to be byte-identical.

## Leakage-safe experiment

The two-stage command sequence is intentional. First fit models on fit cars, generate causal forecasts without joining held-out future outcomes, fit probability maps and thresholds from calibration cars only, and seal all four folds. Only then approve joining held-out outcomes and evaluating test metrics.

The four threshold jobs are independent and can run in parallel:

```bash
for fold in Bahrain Britain Jeddah Monza; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  uv run --locked brace-f1-experiment run-fold \
    --fold "$fold" \
    --stage thresholds \
    --batch-size 128 \
    --output-dir output/experiment &
done
wait

uv run --locked brace-f1-experiment seal-thresholds \
  --output-dir output/experiment
```

After reviewing `output/experiment/threshold-freeze-seal.json`, run held-out scoring:

```bash
for fold in Bahrain Britain Jeddah Monza; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  uv run --locked brace-f1-experiment run-fold \
    --fold "$fold" \
    --resume \
    --stage heldout \
    --approve-heldout-evaluation \
    --batch-size 128 \
    --output-dir output/experiment &
done
wait

uv run --locked brace-f1-experiment aggregate-heldout \
  --output-dir output/experiment \
  --approve-heldout-aggregation
```

Every completed fold and pooled bundle contains a manifest with SHA-256 hashes. Do not edit authenticated inputs or analysis code between threshold fitting, held-out scoring, aggregation, and bootstrap analysis.

## Bootstrap analysis and submission artifacts

Run the canonical paired bootstrap and the descriptive circuit-then-car sensitivity:

```bash
uv run --locked brace-f1-bootstrap \
  output/experiment/pooled-heldout/pooled-heldout-car-contributions.parquet \
  --output output/bootstrap-primary \
  --experiment-manifest output/experiment/pooled-heldout/manifest.json \
  --reliability-contributions \
    output/experiment/pooled-heldout/pooled-heldout-reliability-cluster-contributions.parquet

uv run --locked brace-f1-bootstrap \
  output/experiment/pooled-heldout/pooled-heldout-car-contributions.parquet \
  --two-stage \
  --output output/bootstrap-two-stage \
  --experiment-manifest output/experiment/pooled-heldout/manifest.json \
  --reliability-contributions \
    output/experiment/pooled-heldout/pooled-heldout-reliability-cluster-contributions.parquet
```

Compile only authenticated evidence into the result-ready paper artifacts:

```bash
.venv-publication/bin/python scripts/build_submission_results.py \
  --pooled-dir output/experiment/pooled-heldout \
  --bootstrap-dir output/bootstrap-primary \
  --transport-bootstrap-dir output/bootstrap-two-stage \
  --manuscript-shell paper/manuscript-results-shell.md \
  --abstract-shell paper/ssac27-abstract-results-shell.md \
  --output-root output/submission-ready
```

The output directory must not already exist. Use a new path for a reproducibility rerun; the compiler will not overwrite an authenticated bundle. The verified release build contains:

- `output/submission-ready/paper/manuscript-final.md`;
- `output/submission-ready/paper/ssac27-abstract-final.md`;
- `output/submission-ready/paper/manuscript-final.pdf`;
- `output/submission-ready/analysis-output/submission-results-manifest.json`;
- `output/submission-ready/analysis-output/analysis-report.md` and `stats-appendix.md`;
- `output/submission-ready/analysis-output/tables/` and `figures/`.

Two clean compiler runs produced byte-identical 28-file bundles. The 18-page paper passed citation, resource, portability, and page-by-page raster checks. The canonical visually approved PDF is `output/pdf/brace-f1-ssac27-final.pdf`, with SHA-256 `8d40082e70d0f2dcd25674fe9ebc8cf4058c38dd29195d03bf8f015e96e02945`.

## Data access and rights

The DeepRacing dataset card declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), but the numerical release derives from a commercial Formula 1 video game. The dataset-card label is not a warranty that the publisher controls all game, Formula 1, circuit, team, sponsor, or other branded rights.

Do not push the following into ordinary Git history:

- `data/raw/` or any downloaded third-party source object;
- the 1.1 GiB MixNet archive;
- AssettoCorsaGym legacy pickle files;
- OpenF1, Formula 1, FIA, team, circuit, driver, supplier, or partner material without explicit redistribution rights;
- credentials, tokens, private keys, `.env` files, or confidential review records;
- aborted/failed experiment directories, caches, and temporary PDF renders.

Release source manifests, checksums, acquisition instructions, code, tests, frozen configuration, and rights-reviewed derived/result artifacts. If a derived artifact contains geometry or branded source content, complete the separate rights review before publishing it. The 96,194,642-byte processed frame table is close to ordinary Git hosting limits; use a documented release asset or Git LFS if it is approved for redistribution.

## Citation

Repository citation metadata are in [`CITATION.cff`](CITATION.cff). The author entry is intentionally anonymous for conference review and must be replaced with the actual author list only when the submission policy permits de-anonymization. Dataset attribution remains independently required.

## License status

No software license file has been added yet. Until the authors select and add a `LICENSE`, public availability does not grant a general reuse license for repository code. This is a release blocker, not an invitation to infer a license from the DeepRacing data card.
