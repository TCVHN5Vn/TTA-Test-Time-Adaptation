"""
EXPERIMENT 9 — how much of the "DIGA improves MCC by 13.8 points" headline is selection?

Section 4.6 states the selection rule plainly:

    "For all three methods, the retained configuration was the one that achieved the best
     target-domain macro-F1 for HAR or the best MCC for occupancy estimation on held-out
     target labels."

Section 4.4 gives the DIGA grid: lambda_BN in {0.80,0.90,0.95,1.00}, P0 in {0.50,0.60,
0.65,0.70}, rho_P in {0.02,0.10}, K in {16,32} -- 128 configurations.

So the reported OE number is the MAXIMUM of 128 estimates, each measured on a test set of
150 rows with 34 minority samples. The maximum of many noisy estimates is biased upward
even when every configuration is, in truth, exactly as good as the baseline. This script
runs that exact grid and reports the whole distribution instead of its maximum.

The honest counterpart is a label-free selection rule, so the last block picks the
configuration by an unsupervised criterion (mean prediction entropy on the target stream)
and reports what that configuration scores.
"""
import os, sys, itertools, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, fit_zscore
from tta import SourceOnly, DIGA, source_feature_stats
import tta as _tta
_tta.K = 2
from exp8_occupancy import m2, train_bin, run_stream, OE, BS

R = os.path.join(os.path.dirname(__file__), "..", "results")

GRID = dict(lam_bn=[0.80, 0.90, 0.95, 1.00], P0=[0.50, 0.60, 0.65, 0.70],
            rho_P=[0.02, 0.10], topK=[16, 32])


def run_with_probs(mk, Z, y):
    ad = mk(); pred = np.zeros(len(y), int); pro = np.zeros((len(y), 2))
    for i in range(0, len(Z), BS):
        x = torch.tensor(Z[i:i + BS], dtype=torch.float32)
        p, q = ad(x); pred[i:i + BS] = p; pro[i:i + BS] = q
    return pred, pro


if __name__ == "__main__":
    sp = {n: pd.read_csv(f"{OE}/{n}.csv") for n in
          ["Source_train", "Source_test", "Target_train", "Target_test"]}
    Xs, ys = sp["Source_train"].iloc[:, :9].values, sp["Source_train"].labels.values.astype(int)
    Xsv, ysv = sp["Source_test"].iloc[:, :9].values, sp["Source_test"].labels.values.astype(int)
    Xtt = sp["Target_train"].iloc[:, :9].values
    Xte, yte = sp["Target_test"].iloc[:, :9].values, sp["Target_test"].labels.values.astype(int)

    tf = fit_zscore(Xtt)
    Zs, Zsv, Zte = tf(Xs), tf(Xsv), tf(Xte)

    print("=" * 96)
    print("EXPERIMENT 9 — oracle hyperparameter selection on a 150-row target test set")
    print("=" * 96)

    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    rows = []
    for seed in (0, 1, 2):
        src, _ = train_bin(Zs, ys, Zsv, ysv, 9, seed=seed)
        p0, _ = run_with_probs(lambda: SourceOnly(src), Zte, yte)
        base = m2(yte, p0)
        for c in combos:
            cfg = dict(zip(keys, c))
            pr, pro = run_with_probs(lambda: DIGA(src, cfg=cfg), Zte, yte)
            mm = m2(yte, pr)
            ent = float(-(pro * np.log(pro + 1e-12)).sum(1).mean())   # label-free criterion
            rows.append(dict(seed=seed, **cfg, entropy=ent,
                             acc=mm["acc"], macroF1=mm["macroF1"], mcc=mm["mcc"],
                             d_acc=mm["acc"] - base["acc"],
                             d_macroF1=mm["macroF1"] - base["macroF1"],
                             d_mcc=mm["mcc"] - base["mcc"]))
        print(f"  seed {seed}: source-only MCC={base['mcc']:.4f}, {len(combos)} DIGA configs done")

    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "exp9_oracle_selection.csv"), index=False)

    print(f"\nDistribution of DIGA's MCC gain over source-only, across {len(combos)} configurations:")
    print(f"{'seed':<6}{'min':>9}{'median':>9}{'mean':>9}{'max':>9}{'sd':>9}   "
          f"{'<- what the paper reports':>26}")
    for seed, g in df.groupby("seed"):
        d = g.d_mcc
        print(f"{seed:<6}{d.min():>+9.4f}{d.median():>+9.4f}{d.mean():>+9.4f}"
              f"{d.max():>+9.4f}{d.std():>9.4f}   {'max of the grid':>26}")
    d = df.d_mcc
    print(f"\nPooled: median gain {d.median():+.4f}, mean {d.mean():+.4f}, "
          f"MAX {d.max():+.4f} ({d.max()*100:+.1f} points)")
    print(f"        fraction of configurations that HURT: {(d < 0).mean()*100:.0f}%")

    print("\nLabel-free selection (lowest mean target prediction entropy) vs oracle selection:")
    print(f"{'seed':<6}{'rule':<28}{'dMCC':>9}{'dMacroF1':>11}")
    for seed, g in df.groupby("seed"):
        o = g.loc[g.d_mcc.idxmax()]
        e = g.loc[g.entropy.idxmin()]
        print(f"{seed:<6}{'oracle (best target MCC)':<28}{o.d_mcc:>+9.4f}{o.d_macroF1:>+11.4f}")
        print(f"{'':<6}{'label-free (min entropy)':<28}{e.d_mcc:>+9.4f}{e.d_macroF1:>+11.4f}")
    print("\n-> results/exp9_oracle_selection.csv")
