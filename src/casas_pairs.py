"""
EXPERIMENT 18 — the inversion across many buildings.

The inversion (in-domain fit anti-correlating with cross-building transfer) rests on two
apartments. This runs every ordered pair of CASAS homes under the same protocol, which
turns five observations into a scatter plot with a correlation coefficient.

For each ordered pair (A, B) and each feature set, the model is selected on A by
GroupKFold over recording day, then evaluated on B. The question is whether source
validation score predicts target score, and with what sign.

Prerequisite:  python src/casas_parse.py <folder of CASAS homes>
"""
import os, sys, glob, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import K, metrics, trivial_baselines, TRANSFORMS, train_mlp, predict
import build_features_v2 as B2

D = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(D, "raw")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1)
W = 20


def load_home(path):
    d = pd.read_csv(path).sort_values(["year_day", "time"]).reset_index(drop=True)
    sens = [c for c in d.columns if c.startswith("sensor_id_")]
    return d, sens


def segments_rec(df, step=30):
    new = (df.year_day != df.year_day.shift()) | (df.time.diff() != step)
    return new.cumsum().values


def features(df, sens):
    """Layout-invariant features: the g_ (global) and t_ (temporal) blocks of
    build_features_v2, which are defined without reference to sensor identity."""
    S = df[sens].values.astype(np.float32)
    tv, dv = df.time.values, df.year_day.values
    seg = segments_rec(df)
    starts = {s: int(np.searchsorted(seg, s, "left")) for s in np.unique(seg)}
    X = []
    for i in range(len(df)):
        w = S[max(i - W + 1, starts[seg[i]]): i + 1]
        g = B2._global_block(w)
        tn = (tv[i] % 86400) / 86400.0; dd = dv[i] % 7
        t = np.array([np.sin(2*np.pi*tn), np.cos(2*np.pi*tn),
                      np.sin(2*np.pi*dd/7.0), np.cos(2*np.pi*dd/7.0)])
        X.append(np.concatenate([g, t]))
    return np.array(X, np.float64), dv, seg


def per_sensor(df, sens):
    S = df[sens].values.astype(np.float32)
    seg = segments_rec(df)
    starts = {s: int(np.searchsorted(seg, s, "left")) for s in np.unique(seg)}
    out = []
    for i in range(len(df)):
        w = S[max(i - W + 1, starts[seg[i]]): i + 1]
        out.append(np.concatenate([w.mean(0), w.std(0)]))
    return np.array(out, np.float64)


def causal_median(pred, seg, L=41):
    out = np.empty_like(pred)
    for i in range(len(pred)):
        lo = i
        while lo > 0 and i - lo < L - 1 and seg[lo - 1] == seg[i]:
            lo -= 1
        out[i] = np.bincount(pred[lo:i + 1], minlength=max(pred.max() + 1, 2)).argmax()
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RAW, "casas_*.csv")))
    if len(files) < 2:
        print(f"need at least 2 homes in {RAW} (found {len(files)}).")
        print("run:  python src/casas_parse.py <folder of CASAS homes>")
        sys.exit(1)

    homes = {}
    print(f"{'home':<18}{'rows':>8}{'sensors':>9}{'classes':>9}")
    for f in files:
        name = os.path.basename(f)[6:-4]
        df, sens = load_home(f)
        acts = sorted(df.activity.unique())
        lab = {a: i for i, a in enumerate(acts)}
        y = df.activity.map(lab).values
        Xi, day, seg = features(df, sens)
        homes[name] = dict(df=df, sens=sens, y=y, inv=Xi, day=day, seg=seg,
                           ps=per_sensor(df, sens), classes=acts)
        print(f"{name:<18}{len(df):>8}{len(sens):>9}{len(acts):>9}")

    # only pairs sharing a comparable label space are meaningful
    print(f"\n{'pair':<22}{'shared':>8}{'feat':<18}{'src-val':>9}{'TGT F1':>9}{'chance':>9}")
    print("-" * 76)
    rows = []
    for a, b in itertools.permutations(homes, 2):
        A, B = homes[a], homes[b]
        shared = sorted(set(A["classes"]) & set(B["classes"]))
        if len(shared) < 3:
            continue
        ia = np.isin(A["df"].activity.values, shared)
        ib = np.isin(B["df"].activity.values, shared)
        lab = {c: i for i, c in enumerate(shared)}
        ya = A["df"].activity.values[ia]; ya = np.array([lab[v] for v in ya])
        yb = B["df"].activity.values[ib]; yb = np.array([lab[v] for v in yb])
        chance = trivial_baselines(yb, K=len(shared))["strat_macroF1"]
        for fname, ka, kb in [("layout-invariant", A["inv"][ia], B["inv"][ib]),
                              ("per-sensor", A["ps"][ia], B["ps"][ib])]:
            if ka.shape[1] != kb.shape[1]:
                continue                       # per-sensor only comparable if same width
            da = A["day"][ia]; folds = np.array_split(np.unique(da), 5)
            sv, tf = [], []
            for sd in SEEDS:
                for f in folds:
                    va = np.isin(da, f); tr = ~va
                    t = TRANSFORMS["zscore"](ka[tr])
                    m, bv = train_mlp(t(ka[tr]), ya[tr], t(ka[va]), ya[va],
                                      ka.shape[1], seed=sd, epochs=40, K_out=len(shared))
                    sv.append(bv)
                    yp, _ = predict(m, t(kb))
                    tf.append(metrics(yb, causal_median(yp, B["seg"][ib]),
                                      K=len(shared))["macroF1"])
            print(f"{a[:9]+'->'+b[:9]:<22}{len(shared):>8}{fname:<18}"
                  f"{np.mean(sv):>9.4f}{np.mean(tf):>9.4f}{chance:>9.4f}")
            rows.append(dict(source=a, target=b, n_shared=len(shared), features=fname,
                             src_val=np.mean(sv), tgt_f1=np.mean(tf), chance=chance))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "exp18_casas_pairs.csv"), index=False)
    if len(df):
        print("\n" + "=" * 76)
        for fname, g in df.groupby("features"):
            if len(g) > 2:
                r = np.corrcoef(g.src_val, g.tgt_f1)[0, 1]
                print(f"  {fname:<20}{len(g):>3} pairs   corr(src-val, target) = {r:+.3f}")
        li = df[df.features == "layout-invariant"]
        ps = df[df.features == "per-sensor"]
        if len(li): print(f"  layout-invariant above chance: "
                          f"{(li.tgt_f1 > li.chance).sum()}/{len(li)}")
        if len(ps): print(f"  per-sensor above chance:       "
                          f"{(ps.tgt_f1 > ps.chance).sum()}/{len(ps)}")
    print("-> results/exp18_casas_pairs.csv")
