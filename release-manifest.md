# Later Git Release Manifest

No Git repository has been created or pushed. When the empirical study is complete, the public submission repository should contain the following.

## Paper

- editable manuscript source;
- final anonymous submission PDF;
- verified bibliography;
- final under-500-word SSAC abstract with actual numerical Results;
- figures and tables generated from machine-readable results.

## Reproduction

- source code for ingestion, label validation, model training, calibration, replay, baselines, statistics, and figures;
- locked environment and exact setup instructions;
- one command that regenerates every submitted result;
- unit, leakage, clock, geometry, and deterministic replay tests;
- model and configuration hashes.

## Data

- every rights-cleared data record used for submitted results;
- data dictionary, units, coordinate frames, timestamps, quality flags, and map versions;
- immutable cohort, exposure, exclusion, split, and label manifests;
- label protocol and adjudication records without unnecessary personal identifiers;
- data license and provenance statement.

## Results

- machine-readable metric tables with confidence bounds;
- command and false-activation episode logs;
- calibration predictions and outcome files;
- wall-clock latency logs;
- failure-case index;
- artifact checksums.

## Repository controls

- `README.md`, `LICENSE`, `CITATION.cff`, `CODE_OF_CONDUCT.md`, and a data card;
- `.gitignore` entries for confidential raw data, credentials, local caches, temporary renders, and partner-only files;
- a pre-push secret and restricted-data scan;
- a clean-room reproduction check from a fresh clone.

Never push confidential Formula 1, FIA, team, circuit, driver, or supplier material unless the owner has granted explicit public redistribution rights for those exact files.
