"""
EXPERIMENT 5 — the feature-design sweep, redone on CAUSAL, non-oracle-segmented features.

Experiment 2 found that a 9-d layout-invariant feature set transfers far better than the
paper's 135-d vector. That was measured on the paper's own non-causal, oracle-segmented
features, so it could not be reported: the numbers were not deployable. This repeats the
sweep on feat_causal_recording_*, which is what an online system can actually compute.

Every transform is fit on the SOURCE TRAINING SPLIT only. No target labels and no target
statistics are used anywhere, so every configuration stays legal as a TTA source model.
Source model selection uses GroupKFold over year_day: no day appears in both halves and
no window can overlap across the split, which removes both leakage channels at once.
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, os.path.dirname(__file__))
from common import *

D = os.path.join(os.path.dirname(__file__), "..", "data")
VAR = os.environ.get("VARIANT", "causal_recording")
SEEDS = (0, 1, 2, 3, 4)


def load(nm):
    z = np.load(os.path.join(D, f"feat_{VAR}_{nm}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


X1, y1, d1, cols = load("S1")
X2, y2, d2, _ = load("S2")
fam = np.array([family(c) for c in cols])
dead = X1.var(0) < 1e-12
ks = np.array([ks_2samp(X1[:, j], X2[:, j]).statistic for j in range(len(cols))])
ks_keep = np.ones(len(cols), bool); ks_keep[np.argsort(-ks)[:27]] = False

SETS = {
    "all135 (paper)":              np.ones(len(cols), bool),
    "KS-prune 20% (paper's fix)":  ks_keep,
    "drop zero-src-var":           ~dead,
    "  + drop pair_*":             (~dead) & (fam != "pair"),
    "no per-sensor ids (29-d)":    fam != "per_sensor",
    "global+temporal only (9-d)":  np.isin(fam, ["global", "temporal"]),
    "global only (5-d)":           fam == "global",
    "per-sensor ids only":         fam == "per_sensor",
}

# grouped source split: held-out DAYS, 5 folds
days = np.unique(d1)
folds = np.array_split(days, 5)

tb = trivial_baselines(y2)
print("=" * 112)
print(f"EXPERIMENT 5 — S1 -> S2 transfer by feature set and source-fit transform   [{VAR} features]")
print(f"  source selection: GroupKFold over {len(days)} days, {len(SEEDS)} seeds   n_S1={len(y1)} n_S2={len(y2)}")
print(f"  TARGET TRIVIAL BASELINES: majority acc={tb['majority_acc']:.4f} F1={tb['majority_macroF1']:.4f}"
      f"  |  stratified acc={tb['strat_acc']:.4f} F1={tb['strat_macroF1']:.4f}")
print("=" * 112)
hdr = (f"{'feature set':<28}{'transform':<13}{'d':>4} | {'src-val F1':>11} | "
       f"{'TGT acc':>9}{'TGT macroF1':>13}{'TGT MCC':>10}{'bal.acc':>9}{'#cls':>6}  {'vs chance':>10}")
print(hdr); print("-" * len(hdr))
rows = []
for sname, mask in SETS.items():
    for tname, tfit in TRANSFORMS.items():
        d = int(mask.sum())
        if d == 0: continue
        sv, ta, tf1, tm, tbal, ncl = [], [], [], [], [], []
        for sd in SEEDS:
            for f in folds:
                va = np.isin(d1, f); tr = ~va
                tf = tfit(X1[tr][:, mask])
                m, bv = train_mlp(tf(X1[tr][:, mask]), y1[tr], tf(X1[va][:, mask]), y1[va],
                                  d, seed=sd, epochs=60)
                sv.append(bv)
                yp, _ = predict(m, tf(X2[:, mask])); mm = metrics(y2, yp)
                ta.append(mm["acc"]); tf1.append(mm["macroF1"]); tm.append(mm["mcc"])
                tbal.append(mm["bal"]); ncl.append(mm["n_pred_classes"])
        g = np.mean(tf1) - tb["strat_macroF1"]
        print(f"{sname:<28}{tname:<13}{d:>4} | {np.mean(sv):>11.4f} | {np.mean(ta):>9.4f}"
              f"{np.mean(tf1):>13.4f}{np.mean(tm):>10.4f}{np.mean(tbal):>9.4f}{np.mean(ncl):>6.1f}"
              f"  {('+' if g>0 else '')+f'{g*100:.1f}pp':>10}")
        rows.append(dict(variant=VAR, feature_set=sname, transform=tname, d=d,
                         src_val_macroF1=np.mean(sv), src_val_sd=np.std(sv),
                         tgt_acc=np.mean(ta), tgt_acc_sd=np.std(ta),
                         tgt_macroF1=np.mean(tf1), tgt_macroF1_sd=np.std(tf1),
                         tgt_mcc=np.mean(tm), tgt_mcc_sd=np.std(tm), tgt_bal=np.mean(tbal)))
    print("-" * len(hdr))
pd.DataFrame(rows).to_csv(os.path.join(D, "..", "results", f"exp5_features_{VAR}.csv"), index=False)
print(f"-> results/exp5_features_{VAR}.csv")
