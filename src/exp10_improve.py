"""
EXPERIMENT 10 — getting more out of the source model, legally.

Experiment 5 left the deployable checkpoint at 0.4133 target macro-F1 with 9 features and
`cleaning` stuck at F1 0.009. Three things in that setup were never tuned, and all three
are source-side, so none of them touches a target label or a target statistic:

  1. FEATURES    the 9-d set keeps only 5 global + 4 temporal numbers and throws the whole
                 sensor array away. build_features_v2 adds layout-invariant descriptions of
                 the array -- a sorted activation profile, run-length structure, and the
                 global block at several window widths -- which are permutation-invariant
                 in sensor identity and so still cross buildings.
  2. CLASS       the source loss is unweighted while S1 is 1598:442 imbalanced, and macro-F1
     WEIGHTING   is the reported metric. Inverse-frequency weights cost nothing.
  3. SMOOTHING   activities persist for minutes while predictions are made per 30 s window.
                 A CAUSAL median filter over the last L predictions within the current
                 recording segment uses only past outputs, so it remains deployable online.

Also swept: network width, since 256-128-64 over 9 inputs is heavily overparameterized.

Selection is GroupKFold over year_day throughout. Target labels are used only to score.
"""
import os, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import K, CLASS_NAMES, train_mlp, predict, metrics, trivial_baselines, TRANSFORMS
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)


def load_inv(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def load_old(tag):
    z = np.load(os.path.join(D, f"feat_causal_recording_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def seg_ids(name):
    return segments(load_raw(name), "recording")


def causal_median(pred, seg, L):
    """Median of the last L predictions, never crossing a recording-segment boundary.
    Uses only samples at or before i, so it is valid for online deployment."""
    if L <= 1: return pred
    out = np.empty_like(pred)
    for i in range(len(pred)):
        lo = i
        while lo > 0 and i - lo < L - 1 and seg[lo - 1] == seg[i]:
            lo -= 1
        w = pred[lo:i + 1]
        c = np.bincount(w, minlength=K)
        out[i] = c.argmax()
    return out


GROUPS = {
    "g+t (9-d equivalent)":      lambda c: c.startswith(("g_", "t_")),
    "g+s+t":                     lambda c: c.startswith(("g_", "s_", "t_")),
    "g+s+r+t":                   lambda c: c.startswith(("g_", "s_", "r_", "t_")),
    "all invariant (60-d)":      lambda c: True,
}

if __name__ == "__main__":
    X1, y1, d1, cols = load_inv("S1")
    X2, y2, d2, _ = load_inv("S2")
    seg2 = seg_ids("S2")
    tb = trivial_baselines(y2)
    days = np.unique(d1); folds = np.array_split(days, 5)

    print("=" * 118)
    print("EXPERIMENT 10 — source-side improvements on the deployable checkpoint")
    print(f"  n_S1={len(y1)} n_S2={len(y2)}  GroupKFold over {len(days)} days, {len(SEEDS)} seeds")
    print(f"  chance: stratified macroF1={tb['strat_macroF1']:.4f} | prior best (9-d causal) = 0.4133")
    print("=" * 118)

    def evaluate(Xa, Xb, mask, cw, widths, L, tname="zscore"):
        f1s, mccs, accs, bals, per = [], [], [], [], []
        for sd in SEEDS:
            for f in folds:
                va = np.isin(d1, f); tr = ~va
                tf = TRANSFORMS[tname](Xa[tr][:, mask])
                w = None
                if cw:
                    cnt = np.bincount(y1[tr], minlength=K).astype(float)
                    w = (len(y1[tr]) / (K * np.maximum(cnt, 1)))
                m, _ = train_mlp(tf(Xa[tr][:, mask]), y1[tr], tf(Xa[va][:, mask]), y1[va],
                                 int(mask.sum()), seed=sd, epochs=60, class_weight=w, widths=widths)
                yp, _ = predict(m, tf(Xb[:, mask]))
                yp = causal_median(yp, seg2, L)
                mm = metrics(y2, yp)
                f1s.append(mm["macroF1"]); mccs.append(mm["mcc"]); accs.append(mm["acc"])
                bals.append(mm["bal"]); per.append(mm["f1"])
        return (np.mean(f1s), np.std(f1s), np.mean(mccs), np.mean(accs), np.mean(bals),
                np.mean(per, 0))

    rows = []
    hdr = (f"{'features':<24}{'d':>4}{'cls-w':>7}{'width':>12}{'smooth':>8} | "
           f"{'TGT macroF1':>13}{'MCC':>9}{'acc':>9}{'bal':>8}   {'vs 0.4133':>10}")

    # ---- stage 1: features and class weighting, no smoothing, default width
    print("\nSTAGE 1 — features x class weighting")
    print(hdr); print("-" * len(hdr))
    best = None
    for gname, pred_fn in GROUPS.items():
        mask = np.array([pred_fn(c) for c in cols])
        for cw in (False, True):
            f1, sd, mcc, acc, bal, per = evaluate(X1, X2, mask, cw, (256, 128, 64), 1)
            d = f1 - 0.4133
            print(f"{gname:<24}{int(mask.sum()):>4}{str(cw):>7}{'256-128-64':>12}{1:>8} | "
                  f"{f1:>8.4f}±{sd:<4.3f}{mcc:>9.4f}{acc:>9.4f}{bal:>8.4f}   {d*100:>+9.1f}pp")
            rows.append(dict(stage=1, features=gname, d=int(mask.sum()), class_weight=cw,
                             width="256-128-64", smooth=1, macroF1=f1, macroF1_sd=sd,
                             mcc=mcc, acc=acc, bal=bal,
                             **{f"f1_{CLASS_NAMES[k]}": per[k] for k in range(K)}))
            if best is None or f1 > best[0]: best = (f1, gname, mask, cw)

    # ---- stage 2: width and causal smoothing on the stage-1 winner
    _, gname, mask, cw = best
    print(f"\nSTAGE 2 — width x causal smoothing, on '{gname}' (class_weight={cw})")
    print(hdr); print("-" * len(hdr))
    for widths in [(256, 128, 64), (64, 32), (128, 64)]:
        for L in (1, 5, 11, 21, 41):
            f1, sd, mcc, acc, bal, per = evaluate(X1, X2, mask, cw, widths, L)
            d = f1 - 0.4133
            wn = "-".join(map(str, widths))
            print(f"{gname:<24}{int(mask.sum()):>4}{str(cw):>7}{wn:>12}{L:>8} | "
                  f"{f1:>8.4f}±{sd:<4.3f}{mcc:>9.4f}{acc:>9.4f}{bal:>8.4f}   {d*100:>+9.1f}pp")
            rows.append(dict(stage=2, features=gname, d=int(mask.sum()), class_weight=cw,
                             width=wn, smooth=L, macroF1=f1, macroF1_sd=sd, mcc=mcc,
                             acc=acc, bal=bal,
                             **{f"f1_{CLASS_NAMES[k]}": per[k] for k in range(K)}))

    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "exp10_improve.csv"), index=False)
    b = df.loc[df.macroF1.idxmax()]
    print(f"\nBEST: {b.features} d={b.d} class_weight={b.class_weight} width={b.width} "
          f"smooth={b.smooth}")
    print(f"      macroF1={b.macroF1:.4f}  MCC={b.mcc:.4f}  acc={b.acc:.4f}  bal={b.bal:.4f}")
    print("      per-class F1: " + "  ".join(f"{c}={b['f1_'+c]:.3f}" for c in CLASS_NAMES))
    print("-> results/exp10_improve.csv")
