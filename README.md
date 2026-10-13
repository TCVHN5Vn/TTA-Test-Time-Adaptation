# TTA: Test Time Adaptation

Test-time adaptation for cross-building human activity recognition and occupancy
estimation on WSU smart-home sensor data. Evaluates DIGA, PETAL and PeTTA under a
deployable online protocol, in both transfer directions, with chance baselines and
paired bootstrap intervals throughout.

## Headline

| | |
|---|---|
| source-domain fit, grouped by day | 0.51–0.53 |
| cross-building gap, both directions | 9–15 pts |
| best target macro-F1 | 0.447 |
| only method with a significant gain | DIGA, +1.7 |
| PETAL / PeTTA | neither adapts; see `src/exp7`, `src/exp12` |
| occupancy | no method beats source-only |

Stride-1 windowing under a random validation split inflates source macro-F1 to 0.998
regardless of how the window is defined; grouping the split by recording day removes the
effect. Per-sensor features score below stratified-random on the target in both transfer
directions. The feature set that validates best in-domain transfers worst.

## Layout

```
src/        experiment scripts (see below)
data/       engineered features, one .npz per windowing variant
results/    one CSV per experiment + SUMMARY.txt
paper/      cas-dc manuscript, TikZ diagrams, generated figures
```

## Reproducing

Everything runs from the raw CSVs bundled in `data/raw/`. No external downloads.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/build_features.py        # 4 windowing variants from raw CSVs
.venv/bin/python src/build_features_v2.py     # extended layout-invariant features
.venv/bin/python src/exp4_windowing.py        # leakage decomposition
.venv/bin/python src/exp5_features_causal.py  # feature sweep, causal
.venv/bin/python src/exp6_tta_table.py --featset global+temporal --transform zscore --seeds 3
.venv/bin/python src/exp7_petta_noop.py       # PeTTA divergence saturation
.venv/bin/python src/exp10_improve.py         # source-side improvements
.venv/bin/python src/exp11_reverse.py         # both transfer directions
.venv/bin/python src/exp12_lambda_bar.py      # PETAL lambda_bar + faithful hyperparams
.venv/bin/python src/exp13_occ_scaler.py      # occupancy, three normalisation protocols
.venv/bin/python src/summary.py               # consolidated table
.venv/bin/python src/make_figures.py          # manuscript figures
```

`build_features.py` reads `data/raw/` by default; set `TTA_RAW` to point elsewhere. The
occupancy data is private (two offices at the Grenoble Institute of Technology, via
Dridi, Amayri & Bouguila) and is not included, so `exp8`/`exp13` will not run without it.

Implementations of the three adaptation methods as originally applied live in a separate
repository: <https://github.com/TCVHN5Vn/Test-Time-Adaptation>.

## Building the manuscript

```bash
cd paper && tectonic -X compile main.tex
```

Requires `cas-dc.cls` (Elsevier CAS), which ships with TeX Live and with Overleaf's
Elsevier CAS gallery template.
