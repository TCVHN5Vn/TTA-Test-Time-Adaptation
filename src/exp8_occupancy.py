"""
EXPERIMENT 8 — occupancy estimation: reproduce the submitted numbers, then test whether
they survive uncertainty.

Section 6 reports "DIGA improving MCC by up to 13.8 percentage points" as the paper's
strongest adaptation result. That number is a point estimate on Target_test, which has
**150 rows, 34 of them minority class**. This script reproduces it and puts a confidence
interval on it.

Three protocol problems are also measured here:

  (A) Target_test overlaps Target_train          28/150 rows (18.7%) are exact duplicates,
                                                 27 of them class 0. Source_test overlaps
                                                 Source_train on 90/200 (45%).
  (B) the 9 features are PCA components          variance ladder 5.2e4 -> 1.1, and the
      apparently fitted on POOLED data           POOLED source+target set is MORE
                                                 decorrelated (mean |off-diag r| = 0.026)
                                                 than either split alone (0.043 / 0.063).
                                                 PCA zeroes correlation exactly on what it
                                                 was fit on, so the basis looks like it saw
                                                 the target -- including Target_test.
  (C) n_test = 150                               a 13.8-point MCC difference on 150 rows
                                                 needs an interval, not a point estimate.

(A) and (C) are measured below. (B) is reported as evidence, not proof -- confirming it
needs the OE preprocessing script, which is not on this machine.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, fit_zscore, TRANSFORMS
from tta import SourceOnly, AdaBN, TENT, DIGA, PeTTA, PETALAdapter, source_feature_stats
from petal_fixed import SWAGDiag, PETALFixed
import tta as _tta

OE = "/Users/mariemkbaier/Downloads/OPDA_FINAL/data"
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS = 64


def m2(yt, yp):
    """Binary metrics: accuracy, macro-F1, MCC."""
    yt, yp = np.asarray(yt).astype(int), np.asarray(yp).astype(int)
    M = np.zeros((2, 2), np.int64); np.add.at(M, (yt, yp), 1)
    N = M.sum(); tp = np.diag(M).astype(float)
    fp = M.sum(0) - tp; fn = M.sum(1) - tp
    pr = np.divide(tp, tp + fp, out=np.zeros(2), where=(tp + fp) > 0)
    rc = np.divide(tp, tp + fn, out=np.zeros(2), where=(tp + fn) > 0)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros(2), where=(pr + rc) > 0)
    c, s = np.trace(M), N
    p, t = M.sum(0).astype(float), M.sum(1).astype(float)
    den = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return dict(acc=np.trace(M) / N, macroF1=f1.mean(),
                mcc=(c * s - (p * t).sum()) / den if den > 0 else 0.0)


def train_bin(Xtr, ytr, Xva, yva, d, seed=0, epochs=60, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(d, K=2)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, st = -1, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            if len(j) < 2: continue
            opt.zero_grad(); lf(m(Xt[j]), yt[j]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = m(Xv).argmax(1).numpy()
        f = m2(yva, pv)["macroF1"]
        if f > best: best, st = f, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(st); m.eval(); return m, best


def run_stream(mk, Z, y, device="cpu"):
    ad = mk(); pred = np.zeros(len(y), dtype=int)
    for i in range(0, len(Z), BS):
        x = torch.tensor(Z[i:i + BS], dtype=torch.float32, device=device)
        p, _ = ad(x); pred[i:i + BS] = p
    return pred


def boot(y, pa, pb, key, n=5000, seed=0):
    rng = np.random.default_rng(seed); N = len(y); d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        if len(np.unique(y[i])) < 2: continue
        d.append(m2(y[i], pb[i])[key] - m2(y[i], pa[i])[key])
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


if __name__ == "__main__":
    sp = {n: pd.read_csv(f"{OE}/{n}.csv") for n in
          ["Source_train", "Source_test", "Target_train", "Target_test"]}
    Xs, ys = sp["Source_train"].iloc[:, :9].values, sp["Source_train"].labels.values.astype(int)
    Xsv, ysv = sp["Source_test"].iloc[:, :9].values, sp["Source_test"].labels.values.astype(int)
    Xtt, ytt = sp["Target_train"].iloc[:, :9].values, sp["Target_train"].labels.values.astype(int)
    Xte, yte = sp["Target_test"].iloc[:, :9].values, sp["Target_test"].labels.values.astype(int)
    _tta.K = 2                                      # DIGA/PeTTA prototype count

    # (A) which Target_test rows were already seen in Target_train
    A = {tuple(np.round(r, 9)) for r in Xtt}
    seen = np.array([tuple(np.round(r, 9)) in A for r in Xte])

    print("=" * 100)
    print("EXPERIMENT 8 — occupancy estimation under the paper's target-normalized protocol")
    print("=" * 100)
    print(f"  Source_train {len(ys):5d}  Target_train {len(ytt):5d}  Target_test {len(yte):5d}"
          f"  (minority n={int((yte==1).sum())})")
    print(f"  Target_test rows that are exact duplicates of Target_train rows: "
          f"{int(seen.sum())}/{len(yte)} ({100*seen.mean():.1f}%)")
    maj = m2(yte, np.zeros(len(yte), int))
    print(f"  TRIVIAL BASELINE (always 'not occupied'): acc={maj['acc']:.4f} "
          f"macroF1={maj['macroF1']:.4f} MCC={maj['mcc']:.4f}")
    print("=" * 100)

    rows, preds = [], {}
    SEEDS = (0, 1, 2, 3, 4)
    for seed in SEEDS:
        # paper's protocol, Sec 4.7: scaler fit on unlabeled Target_train
        tf = fit_zscore(Xtt)
        Zs, Zsv, Zte = tf(Xs), tf(Xsv), tf(Xte)
        src, _ = train_bin(Zs, ys, Zsv, ysv, 9, seed=seed)
        stt = source_feature_stats(src, Zs, ys)

        sw = MLP(9, K=2); sw.load_state_dict(src.state_dict())
        opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
        Xt_ = torch.tensor(Zs, dtype=torch.float32); yt_ = torch.tensor(ys, dtype=torch.long)
        snaps = []
        for ep in range(30):
            sw.train(); perm = torch.randperm(len(Xt_))
            for i in range(0, len(perm), 64):
                j = perm[i:i + 64]
                if len(j) < 2: continue
                opt.zero_grad(); lf(sw(Xt_[j]), yt_[j]).backward(); opt.step()
            snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
        swag = SWAGDiag().fit(snaps)

        for name, mk in [("Source-only", lambda: SourceOnly(src)),
                         ("AdaBN", lambda: AdaBN(src)),
                         ("TENT", lambda: TENT(src)),
                         ("DIGA", lambda: DIGA(src)),
                         ("PETAL-fixed", lambda: PETALAdapter(PETALFixed(src, swag, strict=False))),
                         ("PeTTA", lambda: PeTTA(src, src_stats=stt))]:
            pr = run_stream(mk, Zte, yte)
            mm = m2(yte, pr)
            mm_un = m2(yte[~seen], pr[~seen])        # unseen-only evaluation
            rows.append(dict(seed=seed, method=name, **mm,
                             acc_unseen=mm_un["acc"], macroF1_unseen=mm_un["macroF1"],
                             mcc_unseen=mm_un["mcc"]))
            if seed == 0: preds[name] = pr

    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "exp8_occupancy.csv"), index=False)
    g = df.groupby("method").agg(["mean", "std"])
    print(f"\n{'method':<14}{'acc':>18}{'macro-F1':>18}{'MCC':>18}   | unseen-only MCC")
    order = ["Source-only", "AdaBN", "TENT", "DIGA", "PETAL-fixed", "PeTTA"]
    for n in order:
        r = g.loc[n]
        print(f"{n:<14}{r[('acc','mean')]:>11.4f}+/-{r[('acc','std')]:<5.3f}"
              f"{r[('macroF1','mean')]:>11.4f}+/-{r[('macroF1','std')]:<5.3f}"
              f"{r[('mcc','mean')]:>11.4f}+/-{r[('mcc','std')]:<5.3f}   | "
              f"{r[('mcc_unseen','mean')]:.4f}+/-{r[('mcc_unseen','std')]:.3f}")

    print(f"\n95% paired-bootstrap CI vs source-only on Target_test (n={len(yte)}, 5000 resamples):")
    pa = preds["Source-only"]
    for n in order[1:]:
        lo1, hi1 = boot(yte, pa, preds[n], "macroF1")
        lo2, hi2 = boot(yte, pa, preds[n], "mcc")
        sig = "" if (lo2 < 0 < hi2) else "  *"
        print(f"  {n:<13} dMacroF1 [{lo1:+.4f}, {hi1:+.4f}]   dMCC [{lo2:+.4f}, {hi2:+.4f}]{sig}")
    print("\n-> results/exp8_occupancy.csv")
