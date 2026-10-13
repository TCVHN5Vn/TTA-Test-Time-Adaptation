"""
EXPERIMENT 11 — does the story hold in the reverse direction, S2 -> S1?

Everything so far rests on a single transfer pair. The two central claims are:

    (a) with causal layout-invariant features the source->target gap is ~9 points,
        not the ~83 the submitted paper reports;
    (b) per-sensor identity features are actively harmful across buildings,
        scoring below chance on their own.

Both are properties of a direction, not of the data, until the direction is swapped.
This runs the identical protocol with S2 as source and S1 as target. Nothing about the
feature definitions is source-specific in the invariant set, so the swap is clean; the
135-d set is included for reference, with the caveat that its 20 `pair_*` columns were
defined from S1's most co-active pairs and therefore mildly favour the forward direction.

Protocol, both directions: GroupKFold over the SOURCE's year_day, 3 seeds x 5 folds,
source-fit z-score, causal median smoothing (L=41) on the target stream.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import K, CLASS_NAMES, train_mlp, predict, metrics, trivial_baselines, TRANSFORMS, family
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)
L_SMOOTH = 41


def load(kind, tag):
    f = f"feat_inv_{tag}.npz" if kind == "inv" else f"feat_causal_recording_{tag}.npz"
    z = np.load(os.path.join(D, f), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


SEG = {t: segments(load_raw(t), "recording") for t in ("S1", "S2")}


def causal_median(pred, seg, L):
    if L <= 1: return pred
    out = np.empty_like(pred)
    for i in range(len(pred)):
        lo = i
        while lo > 0 and i - lo < L - 1 and seg[lo - 1] == seg[i]:
            lo -= 1
        out[i] = np.bincount(pred[lo:i + 1], minlength=K).argmax()
    return out


def run(kind, mask_fn, src, tgt):
    Xs, ys, ds, cols = load(kind, src)
    Xt, yt, dt, _ = load(kind, tgt)
    mask = mask_fn(cols)
    folds = np.array_split(np.unique(ds), 5)
    sv, tf1, tm, ta = [], [], [], []
    per = []
    for sd in SEEDS:
        for f in folds:
            va = np.isin(ds, f); tr = ~va
            tf = TRANSFORMS["zscore"](Xs[tr][:, mask])
            m, _ = train_mlp(tf(Xs[tr][:, mask]), ys[tr], tf(Xs[va][:, mask]), ys[va],
                             int(mask.sum()), seed=sd, epochs=60)
            yv, _ = predict(m, tf(Xs[va][:, mask]))
            sv.append(metrics(ys[va], yv)["macroF1"])
            yp, _ = predict(m, tf(Xt[:, mask]))
            mm = metrics(yt, causal_median(yp, SEG[tgt], L_SMOOTH))
            tf1.append(mm["macroF1"]); tm.append(mm["mcc"]); ta.append(mm["acc"]); per.append(mm["f1"])
    return dict(d=int(mask.sum()), src_val=np.mean(sv), src_val_sd=np.std(sv),
                tgt_f1=np.mean(tf1), tgt_f1_sd=np.std(tf1), tgt_mcc=np.mean(tm),
                tgt_acc=np.mean(ta), gap=np.mean(sv) - np.mean(tf1), per=np.mean(per, 0))


SETS = [
    ("13-d g+t (transferable)", "inv", lambda c: np.array([x.startswith(("g_", "t_")) for x in c])),
    ("all 135-d (paper)",       "raw", lambda c: np.ones(len(c), bool)),
    ("per-sensor ids only",     "raw", lambda c: np.array([family(x) == "per_sensor" for x in c])),
    ("no per-sensor (29-d)",    "raw", lambda c: np.array([family(x) != "per_sensor" for x in c])),
]

if __name__ == "__main__":
    print("=" * 112)
    print("EXPERIMENT 11 — forward vs reverse transfer, identical protocol")
    print("=" * 112)
    rows = []
    for src, tgt in (("S1", "S2"), ("S2", "S1")):
        _, yt, _, _ = load("inv", tgt)
        tb = trivial_baselines(yt)
        print(f"\n{src} -> {tgt}   (target n={len(yt)}, stratified-random macroF1={tb['strat_macroF1']:.4f})")
        hdr = (f"  {'feature set':<26}{'d':>5}{'src-val F1':>13}{'TGT F1':>11}{'TGT MCC':>10}"
               f"{'TGT acc':>10}{'gap':>9}{'vs chance':>11}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for name, kind, mf in SETS:
            r = run(kind, mf, src, tgt)
            g = r["tgt_f1"] - tb["strat_macroF1"]
            print(f"  {name:<26}{r['d']:>5}{r['src_val']:>8.4f}±{r['src_val_sd']:<4.3f}"
                  f"{r['tgt_f1']:>11.4f}{r['tgt_mcc']:>10.4f}{r['tgt_acc']:>10.4f}"
                  f"{r['gap']*100:>8.1f}p{('+' if g>0 else '')+f'{g*100:.1f}pp':>11}")
            rows.append(dict(direction=f"{src}->{tgt}", feature_set=name, chance=tb["strat_macroF1"],
                             **{k: v for k, v in r.items() if k != "per"},
                             **{f"f1_{CLASS_NAMES[k]}": r["per"][k] for k in range(K)}))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "exp11_reverse.csv"), index=False)

    print("\n" + "=" * 112)
    print("CLAIM CHECK")
    t = df[df.feature_set == "13-d g+t (transferable)"]
    for _, r in t.iterrows():
        print(f"  (a) gap, {r.direction}: source {r.src_val:.4f} -> target {r.tgt_f1:.4f}"
              f"  = {r.gap*100:.1f} points")
    p = df[df.feature_set == "per-sensor ids only"]
    for _, r in p.iterrows():
        v = r.tgt_f1 - r.chance
        print(f"  (b) per-sensor only, {r.direction}: {r.tgt_f1:.4f} vs chance {r.chance:.4f}"
              f"  = {v*100:+.1f} points  {'(below chance)' if v < 0 else '(ABOVE chance)'}")
    print("-> results/exp11_reverse.csv")
