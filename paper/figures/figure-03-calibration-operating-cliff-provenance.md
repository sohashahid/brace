# Figure 3 provenance: calibration-only operating cliff

## Scope

This is an **exploratory, calibration-only mechanism diagnostic**. It does not replace,
retune, or otherwise change the prespecified held-out evaluation. A fold name denotes
the circuit withheld from model fitting and calibration; each plotted point is measured
only on that fold's calibration partition.

The companion source table records the closest active threshold below the all-silent
threshold for BRACE and its posterior-mean twin in each fold. "Closest active" means the
highest score-derived threshold that still produced an active proposal stream. The
false-proposal burden is the integer number of false proposals divided by the exact
eligible calibration exposure. The script checks that identity to an absolute tolerance
of `1e-9` proposals per hour before rendering.

## Evidence and integrity

- Source table: `paper/figure-source/figure-03-calibration-operating-cliff.csv`
- Frozen threshold seal SHA-256:
  `ad26f9505a8e00fbdf000bfbbaacf278792eabfb21e937c723418032c234667d`
- Each CSV row records the repository-relative calibrated-score artifact, selected-threshold
  table, fold manifest, and their SHA-256 digests.
- The renderer verifies 17 distinct artifacts before plotting: four calibrated-score
  files, four fold manifests, four threshold tables, the threshold-freeze seal, the
  sealed build manifest, and three processed inputs (events, timing, and splits).
- The fixed seal digest cross-binds the fold identities, frozen artifact hashes, producer
  code hash, and build manifest. The build manifest in turn authenticates the processed
  event, timing, and split inputs.
- The renderer independently rebuilds the frozen threshold grid, runs the proposal state
  machine from its highest threshold downward, selects the first nonempty proposal
  stream, labels proposals against calibration events, and recomputes calibration
  exposure, false counts, and rates. All eight recorded rows must match this derivation.
- The output path is resolved before writing; the frozen `output/experiment` directory
  and all of its descendants, including paths reached through symlinks, are rejected.

The eight exact diagnostic values are:

| Fold | BRACE threshold | BRACE false proposals/h | Twin threshold | Twin false proposals/h |
|---|---:|---:|---:|---:|
| Bahrain | 0.174497710629967 | 162.865230801 | 0.0765009769357812 | 877.597538632 |
| Britain | 0.231003380517263 | 54.882146508 | 0.0282417423146718 | 745.094904280 |
| Jeddah | 0.507934227932975 | 60.416608753 | 0.102626327847883 | 645.036146387 |
| Monza | 0.31944368129248 | 146.735530271 | 0.0807074957833098 | 841.599267165 |

## Rebuild command

Run from the repository root with the project environment; the script requires and
checks `pubfig==0.3.0`:

```bash
.venv/bin/python scripts/build_calibration_operating_cliff_figure.py
```

Expected outputs:

- `paper/figures/figure-03-calibration-operating-cliff.pdf`
- `paper/figures/figure-03-calibration-operating-cliff.png`

## Manuscript-ready caption

**Calibration-only exploratory mechanism diagnostic; the primary held-out result is
unchanged.** For each leave-one-circuit-out fold, the points show the false-proposal
burden at the closest active threshold below silence for BRACE and its matched
posterior-mean twin. Dumbbells join the two methods within a fold; labels give rates per
eligible simulated car-hour, and the horizontal axis is logarithmic. The shaded region
marks the examined operational budgets at or below 10 false proposals/h, with reference
lines at 2, 5, and 10/h. Every active point lies beyond that region, explaining why the
frozen gate selected an all-silent threshold. These calibration data diagnose the
operating cliff; they are not held-out performance estimates and were not used to revise
the primary analysis.
