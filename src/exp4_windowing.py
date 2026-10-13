"""
EXPERIMENT 4 — decomposing the 0.991 source fit.

The paper reports source-domain validation macro-F1 = 0.991 and target macro-F1 = 0.162,
and attributes the whole 83-point drop to "the severity of cross-environment sensor
domain shift". Three distinct mechanisms are tangled together in that number:

  (1) EVALUATION leakage  - stride-1 windows share 19/20 of their content, so a random
                            train/val split puts a 95%-identical twin of every validation
                            sample in training.
  (2) NON-CAUSAL features - the window is centred on the labelled row, so every sample
                            sees 5 minutes into the future. Not deployable online, which
                            is the premise of the entire paper.
  (3) ORACLE segmentation - windows are cut at ground-truth ACTIVITY boundaries
                            (get_blocks), so the feature extractor is handed the label
                            transitions it is supposed to predict.

This script turns each one off independently and measures what is left. The feature
arithmetic is identical throughout (verified bit-exact against the published CSVs);
only the window definition and the validation split change.

Protocols
  random     : 80/20 i.i.d. split            <- what the paper did
  blocked    : first 80% / last 20% in time, 20-row guard gap
  grouped    : 5-fold GroupKFold over year_day  <- strictest; no shared day, no overlap
Target       : trained on the blocked-80% split, evaluated on all of S2.
"""
import os, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, train_mlp, predict, metrics, trivial_baselines, fit_zscore

D = os.path.join(os.path.dirname(__file__), "..", "data")
SEEDS = (0, 1, 2, 3, 4)
GUARD = 20


def load(cz, sg, nm):
    z = np.load(os.path.join(D, f"feat_{cz}_{sg}_{nm}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"]


def splits_random(n, rng):
    p = rng.permutation(n); c = int(.8 * n)
    return [(p[:c], p[c:])]


def splits_blocked(n):
    c = int(.8 * n)
    return [(np.arange(0, c - GUARD), np.arange(c, n))]


def splits_grouped(day, k=5):
    u = np.unique(day); folds = np.array_split(u, k)
    out = []
    for f in folds:
        va = np.isin(day, f)
        out.append((np.where(~va)[0], np.where(va)[0]))
    return out


def run(X1, y1, d1, X2, y2, idx_pairs, seed):
    """Train on each (tr, va) pair; return mean source-val metrics and the model."""
    sv, models = [], []
    for tr, va in idx_pairs:
        tf = fit_zscore(X1[tr])
        m, _ = train_mlp(tf(X1[tr]), y1[tr], tf(X1[va]), y1[va], X1.shape[1], seed=seed, epochs=60)
        yp, _ = predict(m, tf(X1[va]))
        sv.append(metrics(y1[va], yp))
        models.append((m, tf))
    return sv, models


if __name__ == "__main__":
    rows = []
    print("=" * 110)
    print("EXPERIMENT 4 - source fit and target transfer by windowing policy and validation protocol")
    print("              (all135 features, z-score, identical arithmetic; only the window changes)")
    print("=" * 110)
    for cz, sg in itertools.product(("centered", "causal"), ("label", "recording")):
        X1, y1, d1 = load(cz, sg, "S1")
        X2, y2, _ = load(cz, sg, "S2")
        lab = {("centered", "label"): "centred + oracle seg.  <- THE PAPER",
               ("centered", "recording"): "centred + recording seg.",
               ("causal", "label"): "causal  + oracle seg.",
               ("causal", "recording"): "causal  + recording seg. <- DEPLOYABLE"}[(cz, sg)]
        print(f"\n{lab}   (n_S1={len(y1)}, n_S2={len(y2)})")
        print(f"  {'protocol':<26}{'src-val acc':>13}{'src-val F1':>13}{'src-val MCC':>13}"
              f" |{'TGT acc':>10}{'TGT F1':>10}{'TGT MCC':>10}")
        for proto in ("random", "blocked", "grouped"):
            A = {k: [] for k in ("sa", "sf", "sm", "ta", "tf", "tm")}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                sp = {"random": lambda: splits_random(len(y1), rng),
                      "blocked": lambda: splits_blocked(len(y1)),
                      "grouped": lambda: splits_grouped(d1)}[proto]()
                sv, models = run(X1, y1, d1, X2, y2, sp, seed)
                A["sa"].append(np.mean([s["acc"] for s in sv]))
                A["sf"].append(np.mean([s["macroF1"] for s in sv]))
                A["sm"].append(np.mean([s["mcc"] for s in sv]))
                # target: always from the blocked-80% model, so transfer is comparable
                tr, va = splits_blocked(len(y1))[0]
                tf = fit_zscore(X1[tr])
                m, _ = train_mlp(tf(X1[tr]), y1[tr], tf(X1[va]), y1[va], X1.shape[1], seed=seed, epochs=60)
                yp, _ = predict(m, tf(X2))
                mm = metrics(y2, yp)
                A["ta"].append(mm["acc"]); A["tf"].append(mm["macroF1"]); A["tm"].append(mm["mcc"])
            f = lambda k: f"{np.mean(A[k]):.4f}"
            print(f"  {proto:<26}{f('sa'):>13}{f('sf'):>13}{f('sm'):>13}"
                  f" |{f('ta'):>10}{f('tf'):>10}{f('tm'):>10}")
            rows.append(dict(causality=cz, segmenting=sg, protocol=proto, n_s1=len(y1), n_s2=len(y2),
                             **{f"{k}_mean": np.mean(A[k]) for k in A},
                             **{f"{k}_sd": np.std(A[k]) for k in A}))
    tb = trivial_baselines(load("causal", "recording", "S2")[1])
    print(f"\nTarget trivial baselines: majority acc={tb['majority_acc']:.4f} F1={tb['majority_macroF1']:.4f}"
          f" | stratified acc={tb['strat_acc']:.4f} F1={tb['strat_macroF1']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(D, "..", "results", "exp4_windowing.csv"), index=False)
    print("-> results/exp4_windowing.csv")
