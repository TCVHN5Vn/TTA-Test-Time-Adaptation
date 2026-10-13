"""
EXPERIMENT 1 — How much of the source-domain macro-F1 of 0.991 is window-overlap leakage?

The feature extractor in feature_engineering_wsu.ipynb builds, for every row i, a
window spanning [i - W/2, i + W/2] with stride 1 (W=20). Two consequences:
  (a) features are NON-CAUSAL: row i sees 10 timesteps (5 min) into the future;
  (b) adjacent rows share 19/20 of their window, so a RANDOM train/val split puts
      near-duplicates of every validation sample into training.
This script measures (b) by comparing a random split against a temporally blocked
split and a blocked split with a guard gap of W rows at the boundary.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *

D = os.path.join(os.path.dirname(__file__), "..", "data")
s1 = pd.read_csv(os.path.join(D, "S1_features_engineered.csv"))
cols = [c for c in s1.columns if c != "label"]
y = s1.label.values
mu, sc = np.load(os.path.join(D, "sc_mean.npy")), np.load(os.path.join(D, "sc_scale.npy"))
Xraw = s1[cols].values * sc + mu                      # undo the original source z-scoring
W = 20

def run(split, seeds=(0, 1, 2, 3, 4)):
    out = []
    for sd in seeds:
        tr, va = split(sd)
        tf = fit_zscore(Xraw[tr])
        m, _ = train_mlp(tf(Xraw[tr]), y[tr], tf(Xraw[va]), y[va], len(cols), seed=sd, epochs=60)
        yp, _ = predict(m, tf(Xraw[va]))
        out.append(metrics(y[va], yp))
    return out

n = len(s1)
def random_split(sd):
    rng = np.random.default_rng(sd); idx = rng.permutation(n); k = int(.8 * n)
    return idx[:k], idx[k:]
def blocked_split(sd):
    # 5 contiguous temporal folds; fold `sd` is held out
    b = np.array_split(np.arange(n), 5); va = b[sd % 5]
    tr = np.concatenate([b[i] for i in range(5) if i != sd % 5]); return tr, va
def blocked_gap_split(sd):
    tr, va = blocked_split(sd)
    lo, hi = va.min() - W, va.max() + W                # drop W rows either side of the held-out block
    return tr[(tr < lo) | (tr > hi)], va

print("=" * 92)
print("EXPERIMENT 1 — source-domain validation macro-F1 under three split protocols")
print("  (identical model, identical features, identical 135-d z-scored input)")
print("=" * 92)
rows = []
for name, fn in [("random 80/20 (what the paper did)", random_split),
                 ("temporally blocked 5-fold", blocked_split),
                 (f"blocked + {W}-row guard gap", blocked_gap_split)]:
    r = run(fn)
    f1 = np.array([x["macroF1"] for x in r]); ac = np.array([x["acc"] for x in r]); mc = np.array([x["mcc"] for x in r])
    rows.append((name, ac.mean(), ac.std(), f1.mean(), f1.std(), mc.mean(), mc.std()))
    print(f"  {name:<36} acc={ac.mean():.4f}+/-{ac.std():.4f}  "
          f"macroF1={f1.mean():.4f}+/-{f1.std():.4f}  MCC={mc.mean():.4f}+/-{mc.std():.4f}")
pd.DataFrame(rows, columns=["protocol", "acc", "acc_sd", "macroF1", "macroF1_sd", "mcc", "mcc_sd"]) \
  .to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "exp1_leakage.csv"), index=False)
print("\n  -> results/exp1_leakage.csv")
