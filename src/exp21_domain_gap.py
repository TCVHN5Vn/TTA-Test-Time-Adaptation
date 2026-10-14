"""
EXPERIMENT 21 — what is the cross-building penalty, measured against the right ceiling?

The paper so far reports a "gap" of source-validation minus target macro-F1: 8.8 points
forward, 14.7 reverse. That quantity mixes two different things. It compares performance on
building A with performance on building B, so it moves whenever the two buildings differ in
difficulty, whether or not anything transfers badly.

The quantity that actually measures domain shift holds the evaluation set fixed and varies
only the training set:

    oracle   train on the TARGET's own days (GroupKFold), evaluate on its held-out days
    transfer train on the SOURCE building entirely, evaluate on the SAME held-out days

Their difference is the price of not having labels in the target building, which is what a
deployment pays and what TTA is supposed to recover. Identical features, identical folds,
identical smoothing; the only difference between the two numbers is where the labels came
from.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import K, metrics, trivial_baselines, TRANSFORMS, train_mlp, predict
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)
SEG = {t: segments(load_raw(t), "recording") for t in ("S1", "S2")}


def load_inv(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def causal_median(pred, seg, L=41):
    out = np.empty_like(pred)
    for i in range(len(pred)):
        lo = i
        while lo > 0 and i - lo < L - 1 and seg[lo - 1] == seg[i]:
            lo -= 1
        out[i] = np.bincount(pred[lo:i + 1], minlength=K).argmax()
    return out


if __name__ == "__main__":
    X1, y1, d1, cols = load_inv("S1"); X2, y2, d2, _ = load_inv("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols])
    A = {"S1": (X1[:, mask], y1, d1), "S2": (X2[:, mask], y2, d2)}
    d_in = int(mask.sum())
    rows = []

    print("=" * 94)
    print("EXPERIMENT 21 — cross-building penalty on a FIXED evaluation set")
    print("=" * 94)
    print(f"  {'target':<8}{'train on':<26}{'held-out macroF1':>18}{'sd':>8}{'MCC':>9}{'chance':>9}")
    print("  " + "-" * 78)

    for tgt, src in (("S2", "S1"), ("S1", "S2")):
        Xt, yt, dt = A[tgt]; Xs, ys, ds = A[src]
        folds = np.array_split(np.unique(dt), 5)
        ch = trivial_baselines(yt)["strat_macroF1"]
        res = {}
        for mode in ("oracle (target's own days)", "transfer (source building)"):
            f1, mc = [], []
            for sd in SEEDS:
                for f in folds:
                    te = np.isin(dt, f)                       # held-out TARGET days
                    if mode.startswith("oracle"):
                        tr_X, tr_y = Xt[~te], yt[~te]
                        # inner split for early stopping, by day
                        inner = np.unique(dt[~te])
                        iv = np.isin(dt[~te], inner[-max(1, len(inner)//5):])
                        tf = TRANSFORMS["zscore"](tr_X[~iv])
                        m, _ = train_mlp(tf(tr_X[~iv]), tr_y[~iv], tf(tr_X[iv]), tr_y[iv],
                                         d_in, seed=sd, epochs=60)
                    else:
                        inner = np.unique(ds)
                        iv = np.isin(ds, inner[-max(1, len(inner)//5):])
                        tf = TRANSFORMS["zscore"](Xs[~iv])
                        m, _ = train_mlp(tf(Xs[~iv]), ys[~iv], tf(Xs[iv]), ys[iv],
                                         d_in, seed=sd, epochs=60)
                    yp, _ = predict(m, tf(Xt))
                    yh = causal_median(yp, SEG[tgt])[te]      # smooth on full stream, score held-out
                    mm = metrics(yt[te], yh)
                    f1.append(mm["macroF1"]); mc.append(mm["mcc"])
            res[mode] = (np.mean(f1), np.std(f1), np.mean(mc))
            print(f"  {tgt:<8}{mode:<26}{np.mean(f1):>18.4f}{np.std(f1):>8.3f}"
                  f"{np.mean(mc):>9.4f}{ch:>9.4f}")
            rows.append(dict(target=tgt, mode=mode, f1=np.mean(f1), sd=np.std(f1),
                             mcc=np.mean(mc), chance=ch))
        o = res["oracle (target's own days)"][0]; t = res["transfer (source building)"][0]
        print(f"  {'':8}{'-> cross-building penalty':<26}{(o-t)*100:>17.1f} pts\n")
        rows.append(dict(target=tgt, mode="penalty", f1=o - t, sd=np.nan,
                         mcc=np.nan, chance=ch))

    pd.DataFrame(rows).to_csv(os.path.join(R, "exp21_domain_gap.csv"), index=False)
    print("-> results/exp21_domain_gap.csv")

# ---------------------------------------------------------------- matched-n control
def matched_run():
    """Same comparison with the two training sets forced to equal size.

    The uncontrolled version confounds domain shift with training-set size: S2 has 3412
    rows against S1's 5031, so S2's own-building model is trained on less data than the
    model transferred into it. Subsampling the larger training set to the smaller one
    isolates where the labels came from as the only difference.
    """
    X1, y1, d1, cols = load_inv("S1"); X2, y2, d2, _ = load_inv("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols])
    A = {"S1": (X1[:, mask], y1, d1), "S2": (X2[:, mask], y2, d2)}
    d_in = int(mask.sum()); rows = []
    print("\n" + "=" * 94)
    print("MATCHED-n CONTROL — identical training-set size, identical evaluation set")
    print("=" * 94)
    print(f"  {'target':<8}{'train on':<26}{'n_train':>9}{'held-out macroF1':>18}{'sd':>8}{'MCC':>9}")
    print("  " + "-" * 78)
    for tgt, src in (("S2", "S1"), ("S1", "S2")):
        Xt, yt, dt = A[tgt]; Xs, ys, ds = A[src]
        folds = np.array_split(np.unique(dt), 5)
        res = {}
        for mode in ("oracle (target's own days)", "transfer (source building)"):
            f1, mc, ns = [], [], []
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                for f in folds:
                    te = np.isin(dt, f)
                    n_cap = int((~te).sum())          # what the oracle has available
                    if mode.startswith("oracle"):
                        trX, trY = Xt[~te], yt[~te]
                    else:
                        idx = rng.permutation(len(Xs))[:n_cap]
                        trX, trY = Xs[idx], ys[idx]
                    cut = max(1, int(0.2 * len(trX)))
                    p = rng.permutation(len(trX)); vi, ti = p[:cut], p[cut:]
                    tf = TRANSFORMS["zscore"](trX[ti])
                    m, _ = train_mlp(tf(trX[ti]), trY[ti], tf(trX[vi]), trY[vi],
                                     d_in, seed=sd, epochs=60)
                    yp, _ = predict(m, tf(Xt))
                    yh = causal_median(yp, SEG[tgt])[te]
                    mm = metrics(yt[te], yh)
                    f1.append(mm["macroF1"]); mc.append(mm["mcc"]); ns.append(len(ti))
            res[mode] = np.mean(f1)
            print(f"  {tgt:<8}{mode:<26}{np.mean(ns):>9.0f}{np.mean(f1):>18.4f}"
                  f"{np.std(f1):>8.3f}{np.mean(mc):>9.4f}")
            rows.append(dict(target=tgt, mode="matched_" + mode, f1=np.mean(f1),
                             sd=np.std(f1), mcc=np.mean(mc), n=np.mean(ns)))
        pen = res["oracle (target's own days)"] - res["transfer (source building)"]
        print(f"  {'':8}{'-> cross-building penalty':<26}{'':>9}{pen*100:>17.1f} pts\n")
        rows.append(dict(target=tgt, mode="matched_penalty", f1=pen))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp21_matched.csv"), index=False)
    print("-> results/exp21_matched.csv")

matched_run()
