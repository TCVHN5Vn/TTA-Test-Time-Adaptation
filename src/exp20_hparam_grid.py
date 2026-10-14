"""
EXPERIMENT 20 — a full hyperparameter search for all three methods on HAR.

Section~\ref{sec:res-occgrid} searches DIGA's published grid on occupancy and finds that
every configuration loses to the unadapted model. The same search has not been run on the
activity task, where only two points per method were compared. A reader is entitled to ask
whether the activity results are a statement about the methods or about one setting of them.

Three grids, taken from the hyperparameters each paper actually exposes:

  DIGA    lam_bn x P0 x rho_P x topK                                   (64 configurations)
  PETAL   lambda_bar x pi x delta                                      (24 configurations)
  PeTTA   alpha0 x lam0 x lr                                           (16 configurations)

Two selection rules are reported for each, because the difference between them is the
practical question:

  oracle    the configuration with the best TARGET macro-F1. Requires target labels, so it
            is not available in deployment; it bounds what tuning could ever achieve.
  entropy   the configuration with the lowest mean prediction entropy on the target stream.
            Label-free, therefore actually deployable, and the criterion TTA work uses when
            it reports unsupervised model selection.

If the two rules disagree, the reported headroom of a method is not reachable without the
labels the setting assumes are absent.

Protocol as elsewhere: 13-d layout-invariant causal features, GroupKFold over year_day for
the source split, 3 seeds, both transfer directions, source-fit z-score, batch size 64.
"""
import os, sys, itertools, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import K, MLP, train_mlp, predict, metrics, TRANSFORMS
from tta import SourceOnly, source_feature_stats
from tta_faithful import PETALFaithful, DIGAFaithful, PeTTAFaithful
from petal_fixed import SWAGDiag

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS, SEEDS = 64, (0, 1, 2)


def load(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def stream(mk, Z, y):
    """Run one adapter over the target stream; return predictions and mean entropy."""
    ad = mk(); pr = np.zeros(len(y), int); H = []
    for i in range(0, len(Z), BS):
        xb = torch.tensor(Z[i:i + BS], dtype=torch.float32)
        out = ad(xb)
        p, pb = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        pr[i:i + BS] = p
        if pb is not None:
            q = np.asarray(pb, dtype=np.float64)
            if q.ndim == 2:
                q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
                H.append(-(q * np.log(q + 1e-12)).sum(1))
    return pr, (float(np.concatenate(H).mean()) if H else float("nan"))


def fit_swag(d_in, src, Z, y, epochs=30):
    """SWAG posterior collected at a constant LR so it is not degenerate."""
    sw = MLP(d_in); sw.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Z, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    snaps = []
    for _ in range(epochs):
        sw.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            if len(j) < 2: continue
            opt.zero_grad(); lf(sw(Xt[j]), yt[j]).backward(); opt.step()
        snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
    return SWAGDiag().fit(snaps)


# ------------------------------------------------------------------------ grids
DIGA_GRID = [dict(lam_bn=a, P0=b, rho_P=c, topK=k)
             for a, b, c, k in itertools.product((0.8, 0.9, 0.95, 1.0),
                                                 (0.5, 0.6, 0.65, 0.7),
                                                 (0.02, 0.1), (16, 32))]
PETAL_GRID = [dict(lambda_bar=a, pi=b, delta=c)
              for a, b, c in itertools.product((0.0, 0.1, 1.0, 10.0),
                                               (0.99, 0.999), (0.01, 0.03, 0.1))]
PETTA_GRID = [dict(alpha0=a, lam0=b, lr=c)
              for a, b, c in itertools.product((1e-4, 1e-3, 1e-2, 5e-2),
                                               (1.0, 10.0), (1e-4, 1e-3))]


if __name__ == "__main__":
    X1, y1, d1, cols = load("S1"); X2, y2, d2, _ = load("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols])
    d_in = int(mask.sum())
    DIRS = {"S1->S2": (X1[:, mask], y1, d1, X2[:, mask], y2),
            "S2->S1": (X2[:, mask], y2, d2, X1[:, mask], y1)}

    rows = []
    for dirn, (Xs, ys, ds, Xt, yt) in DIRS.items():
        folds = np.array_split(np.unique(ds), 5)
        for sd in SEEDS:
            va = np.isin(ds, folds[-1]); tr = ~va
            tf = TRANSFORMS["zscore"](Xs[tr])
            Ztr, Zva, Zt = tf(Xs[tr]), tf(Xs[va]), tf(Xt)
            src, _ = train_mlp(Ztr, ys[tr], Zva, ys[va], d_in, seed=sd, epochs=60)
            p0, _ = stream(lambda: SourceOnly(src), Zt, yt)
            m0 = metrics(yt, p0)
            st = source_feature_stats(src, Ztr, ys[tr])
            swag = fit_swag(d_in, src, Ztr, ys[tr])

            jobs = ([("DIGA", c, (lambda c=c: DIGAFaithful(src, cfg=c))) for c in DIGA_GRID]
                    + [("PETAL", c, (lambda c=c: PETALFaithful(src, swag, cfg=c)))
                       for c in PETAL_GRID]
                    + [("PeTTA", c, (lambda c=c: PeTTAFaithful(src, src_stats=st, cfg=c)))
                       for c in PETTA_GRID])
            for meth, cfg, mk in jobs:
                pr, h = stream(mk, Zt, yt)
                mm = metrics(yt, pr)
                rows.append(dict(direction=dirn, seed=sd, method=meth,
                                 cfg="|".join(f"{k}={v}" for k, v in cfg.items()),
                                 entropy=h, changed=int((pr != p0).sum()),
                                 macroF1=mm["macroF1"], mcc=mm["mcc"], acc=mm["acc"],
                                 d_macroF1=mm["macroF1"] - m0["macroF1"],
                                 d_mcc=mm["mcc"] - m0["mcc"],
                                 src_macroF1=m0["macroF1"]))
            print(f"  {dirn} seed {sd}: done ({len(jobs)} configs), "
                  f"source macro-F1 {m0['macroF1']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, "exp20_hparam_grid.csv"), index=False)

    print("=" * 100)
    print("EXPERIMENT 20 — hyperparameter grids on HAR, 13-d layout-invariant features")
    print("=" * 100)
    for dirn in DIRS:
        sub = df[df.direction == dirn]
        print(f"\n{dirn}   source-only macro-F1 {sub.src_macroF1.mean():.4f}")
        print(f"  {'method':<8}{'cfgs':>6}{'hurt':>9}{'median':>10}{'oracle':>10}"
              f"{'entropy-sel':>13}{'changed':>10}")
        print("  " + "-" * 66)
        for meth in ("DIGA", "PETAL", "PeTTA"):
            g = sub[sub.method == meth].groupby("cfg").agg(
                d_macroF1=("d_macroF1", "mean"), entropy=("entropy", "mean"),
                changed=("changed", "mean"))
            orc = g.d_macroF1.max()
            ent = g.loc[g.entropy.idxmin(), "d_macroF1"] if g.entropy.notna().any() else np.nan
            print(f"  {meth:<8}{len(g):>6}{(g.d_macroF1<0).sum():>5}/{len(g):<3}"
                  f"{g.d_macroF1.median():>+10.4f}{orc:>+10.4f}{ent:>+13.4f}"
                  f"{g.changed.mean():>10.0f}")
    print("\n-> results/exp20_hparam_grid.csv")
