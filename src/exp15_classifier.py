"""
EXPERIMENT 15 — is the MLP the bottleneck, and is W=20 the right window?

Target performance is capped by source performance: the layout-invariant checkpoint
validates at roughly 0.53 in-domain and transfers at 0.447, a gap of under nine points.
Raising the target therefore means raising the source, and two cheap levers have never
been tested on the causal features.

  (a) classifier family.  Every result so far uses a three-layer MLP. On ~4k rows and
      13 features, gradient boosting and random forests are often stronger, and a linear
      model establishes whether the problem is nonlinear at all.
  (b) window length.  W=20 was inherited from the original pipeline and never revisited
      under causal windowing, where the window extends backwards only.

Both are evaluated under the same protocol as everything else: GroupKFold over year_day
for source selection, both transfer directions, causal recording-segmented features,
causal median smoothing at L=41. Source-fit transforms only; no target information.
"""
import os, sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import K, metrics, trivial_baselines, TRANSFORMS, train_mlp, predict
from build_features import load_raw, segments, top_pairs
from build_features_v2 import extract as extract_inv, names as inv_names, G_NAMES

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)
L_SMOOTH = 41


def load_inv(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


SEG = {t: segments(load_raw(t), "recording") for t in ("S1", "S2")}


def causal_median(pred, seg, L=L_SMOOTH):
    if L <= 1: return pred
    out = np.empty_like(pred)
    for i in range(len(pred)):
        lo = i
        while lo > 0 and i - lo < L - 1 and seg[lo - 1] == seg[i]:
            lo -= 1
        out[i] = np.bincount(pred[lo:i + 1], minlength=K).argmax()
    return out


CLFS = {
    "MLP 256-128-64":   None,                      # handled separately
    "HistGradBoost":    lambda sd: HistGradientBoostingClassifier(
                            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                            l2_regularization=1.0, random_state=sd),
    "RandomForest":     lambda sd: RandomForestClassifier(
                            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                            random_state=sd),
    "ExtraTrees":       lambda sd: ExtraTreesClassifier(
                            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                            random_state=sd),
    "LogisticRegression": lambda sd: LogisticRegression(
                            max_iter=3000, C=1.0),
}


def evaluate(Xs, ys, ds, Xt, yt, tgt_tag, clf_name, d_in):
    folds = np.array_split(np.unique(ds), 5)
    sv, tf1, tm, ta = [], [], [], []
    for sd in SEEDS:
        for f in folds:
            va = np.isin(ds, f); tr = ~va
            tf = TRANSFORMS["zscore"](Xs[tr])
            Ztr, Zva, Zt = tf(Xs[tr]), tf(Xs[va]), tf(Xt)
            if clf_name.startswith("MLP"):
                m, bv = train_mlp(Ztr, ys[tr], Zva, ys[va], d_in, seed=sd, epochs=60)
                yv, _ = predict(m, Zva); yp, _ = predict(m, Zt)
            else:
                m = CLFS[clf_name](sd).fit(Ztr, ys[tr])
                yv, yp = m.predict(Zva), m.predict(Zt)
                bv = metrics(ys[va], yv)["macroF1"]
            sv.append(bv)
            mm = metrics(yt, causal_median(yp, SEG[tgt_tag]))
            tf1.append(mm["macroF1"]); tm.append(mm["mcc"]); ta.append(mm["acc"])
    return (np.mean(sv), np.mean(tf1), np.std(tf1), np.mean(tm), np.mean(ta))


def inv_features_at(W):
    """Rebuild the layout-invariant features at a different window length."""
    import build_features_v2 as B
    old = B.W; B.W = W
    out = {}
    s1, s2 = load_raw("S1"), load_raw("S2")
    sens = sorted(set(c for c in s1.columns if c.startswith("sensor_id_"))
                  & set(c for c in s2.columns if c.startswith("sensor_id_")),
                  key=lambda c: int(c.split("_")[-1]))
    for tag, df in (("S1", s1), ("S2", s2)):
        X, y, day = B.extract(df, sens)
        out[tag] = (X.astype(np.float64), y, day)
    B.W = old
    return out, inv_names()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="both")
    a = ap.parse_args()

    X1, y1, d1, cols = load_inv("S1"); X2, y2, d2, _ = load_inv("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols])
    A1, A2 = X1[:, mask], X2[:, mask]
    tb2, tb1 = trivial_baselines(y2), trivial_baselines(y1)
    rows = []

    if a.part in ("both", "clf"):
        print("=" * 104)
        print("PART A — classifier family, 13-d layout-invariant causal features, L=41")
        print(f"  chance macro-F1: S2 {tb2['strat_macroF1']:.4f} | S1 {tb1['strat_macroF1']:.4f}")
        print("=" * 104)
        print(f"  {'classifier':<22}{'dir':<10}{'src-val':>9}{'TGT macroF1':>13}{'sd':>7}"
              f"{'MCC':>9}{'acc':>8}")
        print("  " + "-" * 76)
        for name in CLFS:
            for dirn, (Xs, ys, ds, Xt, yt, tag) in {
                "S1->S2": (A1, y1, d1, A2, y2, "S2"),
                "S2->S1": (A2, y2, d2, A1, y1, "S1")}.items():
                sv, f1, sd_, mc, ac = evaluate(Xs, ys, ds, Xt, yt, tag, name, Xs.shape[1])
                print(f"  {name:<22}{dirn:<10}{sv:>9.4f}{f1:>13.4f}{sd_:>7.3f}{mc:>9.4f}{ac:>8.4f}")
                rows.append(dict(part="classifier", setting=name, direction=dirn,
                                 src_val=sv, tgt_f1=f1, tgt_f1_sd=sd_, tgt_mcc=mc, tgt_acc=ac))

    if a.part in ("both", "win"):
        print("\n" + "=" * 104)
        print("PART B — causal window length (MLP, 13-d, L=41)")
        print("=" * 104)
        print(f"  {'window W':<22}{'dir':<10}{'src-val':>9}{'TGT macroF1':>13}{'sd':>7}"
              f"{'MCC':>9}{'acc':>8}")
        print("  " + "-" * 76)
        for W in (10, 20, 40, 60):
            feats, nm = inv_features_at(W)
            mk = np.array([c.startswith(("g_", "t_")) for c in nm])
            B1, yb1, db1 = feats["S1"]; B2, yb2, db2 = feats["S2"]
            B1, B2 = B1[:, mk], B2[:, mk]
            for dirn, (Xs, ys, ds, Xt, yt, tag) in {
                "S1->S2": (B1, yb1, db1, B2, yb2, "S2"),
                "S2->S1": (B2, yb2, db2, B1, yb1, "S1")}.items():
                sv, f1, sd_, mc, ac = evaluate(Xs, ys, ds, Xt, yt, tag,
                                               "MLP 256-128-64", Xs.shape[1])
                print(f"  {'W = '+str(W):<22}{dirn:<10}{sv:>9.4f}{f1:>13.4f}{sd_:>7.3f}"
                      f"{mc:>9.4f}{ac:>8.4f}")
                rows.append(dict(part="window", setting=f"W={W}", direction=dirn,
                                 src_val=sv, tgt_f1=f1, tgt_f1_sd=sd_, tgt_mcc=mc, tgt_acc=ac))

    pd.DataFrame(rows).to_csv(os.path.join(R, "exp15_classifier.csv"), index=False)
    print("\n  reference: MLP / W=20 / L=41 gives 0.4466 (S1->S2) and 0.3068 (S2->S1)")
    print("-> results/exp15_classifier.csv")
