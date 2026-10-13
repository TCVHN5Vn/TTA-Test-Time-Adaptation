"""
EXPERIMENT 2 — Which part of the feature design is responsible for near-chance target
performance, and what recovers it?

Diagnosed defects in the original 135-d vector:
  (B) 20 `pair_*` features are the top-20 co-active sensor pairs *in the source*;
      they are near-constant in the target (target std ~ 0.000) -> dead weight + bias.
  (C) 26 features have ZERO variance in S1 (unlearnable at train time) but are live in
      S2 -> pure noise injected at inference. Dropping them needs NO target data.
  (D) z-scoring heavy-tailed sparse activation counts sends target values to +105 source
      sigmas, saturating the first BN layer.

All transforms are fit on SOURCE ONLY, so every configuration remains protocol-legal
for test-time adaptation (no target labels, no target statistics).
Model selection uses a temporally BLOCKED source split (no window-overlap leakage).
"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.dirname(__file__))
from common import *

D = os.path.join(os.path.dirname(__file__), "..", "data")
s1 = pd.read_csv(os.path.join(D, "S1_features_engineered.csv"))
s2 = pd.read_csv(os.path.join(D, "S2_features_engineered.csv"))
cols = [c for c in s1.columns if c != "label"]
mu, sc = np.load(os.path.join(D, "sc_mean.npy")), np.load(os.path.join(D, "sc_scale.npy"))
X1 = s1[cols].values * sc + mu; y1 = s1.label.values
X2 = s2[cols].values * sc + mu; y2 = s2.label.values
fam = np.array([family(c) for c in cols])
srcvar = X1.var(0)
dead = srcvar < 1e-12

# the paper's KS pruning, reproduced (uses unlabeled target statistics)
ks = np.array([ks_2samp(X1[:, j], X2[:, j]).statistic for j in range(len(cols))])
ks_keep = np.ones(len(cols), bool); ks_keep[np.argsort(-ks)[:27]] = False

SETS = {
 "all135 (paper)":                 np.ones(len(cols), bool),
 "KS-prune 20% (paper's fix)":     ks_keep,
 "drop zero-src-var  [fix C]":     ~dead,
 "  + drop pair_*    [fix C+B]":   (~dead) & (fam != "pair"),
 "global+temporal only (9-d)":     np.isin(fam, ["global", "temporal"]),
 "no per-sensor ids (29-d)":       fam != "per_sensor",
 "per-sensor ids only (106-d)":    fam == "per_sensor",
}
SEEDS = (0, 1, 2, 3, 4)

# blocked source split: first 80% train / last 20% val, with a guard gap
n = len(X1); cut = int(.8 * n)
tr_idx, va_idx = np.arange(0, cut - 20), np.arange(cut, n)

print("=" * 108)
print("EXPERIMENT 2 — S1 -> S2 transfer by feature set and source-fit transform")
print(f"  source split: temporally blocked (train n={len(tr_idx)}, val n={len(va_idx)}), {len(SEEDS)} seeds")
tb = trivial_baselines(y2)
print(f"  TARGET TRIVIAL BASELINES:  constant-majority acc={tb['majority_acc']:.4f} "
      f"macroF1={tb['majority_macroF1']:.4f}   |   stratified-random acc={tb['strat_acc']:.4f} "
      f"macroF1={tb['strat_macroF1']:.4f}")
print("=" * 108)
hdr = f"{'feature set':<30}{'transform':<14}{'d':>4} | {'src-val F1':>11} | {'TGT acc':>8}{'TGT macroF1':>12}{'TGT MCC':>9}{'bal.acc':>9}  {'>chance?':>9}"
print(hdr); print("-" * len(hdr))
rows = []
for sname, mask in SETS.items():
    for tname, tfit in TRANSFORMS.items():
        d = int(mask.sum())
        tf = tfit(X1[tr_idx][:, mask])
        Z1tr, Z1va, Z2 = tf(X1[tr_idx][:, mask]), tf(X1[va_idx][:, mask]), tf(X2[:, mask])
        sv, ta, tf1, tm, tb_ = [], [], [], [], []
        for sd in SEEDS:
            m, bv = train_mlp(Z1tr, y1[tr_idx], Z1va, y1[va_idx], d, seed=sd, epochs=60)
            yp2, _ = predict(m, Z2); mm = metrics(y2, yp2)
            sv.append(bv); ta.append(mm["acc"]); tf1.append(mm["macroF1"]); tm.append(mm["mcc"]); tb_.append(mm["bal"])
        gain = np.mean(tf1) - tb["strat_macroF1"]
        flag = f"+{gain*100:.1f}pp" if gain > 0 else f"{gain*100:.1f}pp"
        print(f"{sname:<30}{tname:<14}{d:>4} | {np.mean(sv):>11.4f} | "
              f"{np.mean(ta):>8.4f}{np.mean(tf1):>12.4f}{np.mean(tm):>9.4f}{np.mean(tb_):>9.4f}  {flag:>9}")
        rows.append(dict(feature_set=sname, transform=tname, d=d, src_val_macroF1=np.mean(sv),
                         tgt_acc=np.mean(ta), tgt_acc_sd=np.std(ta), tgt_macroF1=np.mean(tf1),
                         tgt_macroF1_sd=np.std(tf1), tgt_mcc=np.mean(tm), tgt_mcc_sd=np.std(tm),
                         tgt_bal=np.mean(tb_)))
    print("-" * len(hdr))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "exp2_features.csv"), index=False)
print("-> results/exp2_features.csv")
