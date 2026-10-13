"""
EXPERIMENT 12 — why the manuscript's PETAL was a no-op, proven from PETAL's own algorithm.

PETAL's Algorithm 2 (Brahma & Rai, CVPR 2023, lines 10-11):

    L = ( log q(theta) - (lambda_bar / |B|) * H )        theta <- theta + eta * grad L

lambda_bar weights the DATA term (the student-teacher cross-entropy). The prior weight is
fixed at 1. So lambda_bar = 0 removes the cross-entropy entirely and the update becomes
pure gradient ascent on the source posterior: the student is pulled toward the SWAG mean
and never sees the target. No adaptation is possible by construction.

Section 4.5 of the manuscript sets lambda_bar = 0.0 and describes it as "the regularization
weight ... set to 0, which means that FIM-based restoration provides the main anti-forgetting
mechanism instead of an explicit regularization term." That reads lambda_bar as weighting a
regularizer. It weights the likelihood.

The grid in Section 4.5 was lambda_bar in {0.0, 1.0}, so the working setting was available.
Section 4.6 selected "the configuration that achieved the best target-domain macro-F1 ...
on held-out target labels". This script tests the resulting hypothesis:

    on a source model that already beats what self-training produces, selecting on target
    labels PREFERS the degenerate lambda_bar = 0 -- and the no-op is then reported as
    "PETAL yields no measurable gain", a statement about the search, not about PETAL.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import K, MLP, train_mlp, predict, metrics, TRANSFORMS, family
from tta import SourceOnly, source_feature_stats
from tta_faithful import PETALFaithful, DIGAFaithful, PeTTAFaithful
from petal_fixed import SWAGDiag
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS = 64


def load(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def stream(mk, Z, y):
    ad = mk(); pr = np.zeros(len(y), int)
    for i in range(0, len(Z), BS):
        p, _ = ad(torch.tensor(Z[i:i+BS], dtype=torch.float32)); pr[i:i+BS] = p
    return pr, getattr(ad, "moved", float("nan"))


if __name__ == "__main__":
    X1, y1, d1, cols = load("S1"); X2, y2, _, _ = load("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols])
    days = np.unique(d1); va = np.isin(d1, np.array_split(days, 5)[-1]); tr = ~va
    tf = TRANSFORMS["zscore"](X1[tr][:, mask])
    Z1, Z1v, Z2 = tf(X1[tr][:, mask]), tf(X1[va][:, mask]), tf(X2[:, mask])
    src, _ = train_mlp(Z1, y1[tr], Z1v, y1[va], int(mask.sum()), seed=0, epochs=60)
    p0, _ = stream(lambda: SourceOnly(src), Z2, y2)
    m0 = metrics(y2, p0)

    # SWAG posterior collected at a constant LR so it is not degenerate
    sw = MLP(int(mask.sum())); sw.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Z1, dtype=torch.float32); yt = torch.tensor(y1[tr], dtype=torch.long)
    snaps = []
    for ep in range(30):
        sw.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            j = perm[i:i+128]
            if len(j) < 2: continue
            opt.zero_grad(); lf(sw(Xt[j]), yt[j]).backward(); opt.step()
        snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
    swag = SWAGDiag().fit(snaps)

    print("=" * 96)
    print("EXPERIMENT 12 — PETAL's lambda_bar, and what oracle selection does with it")
    print("=" * 96)
    print(f"SWAG posterior degeneracy: {swag.degeneracy():.4f}  (sigma median "
          f"{swag.sigma.median():.3e})")
    print(f"source-only: acc={m0['acc']:.4f} macroF1={m0['macroF1']:.4f} MCC={m0['mcc']:.4f}\n")
    print("PETAL, published parameterization  L = log q - lambda_bar * H   (Alg.2)")
    print(f"  {'lambda_bar':<12}{'drift':>10}{'changed':>12}{'acc':>9}{'macroF1':>10}{'MCC':>9}")
    rows = []
    for lb in (0.0, 0.1, 1.0, 10.0):
        pr, drift = stream(lambda: PETALFaithful(src, swag, cfg=dict(lambda_bar=lb)), Z2, y2)
        mm = metrics(y2, pr); ch = int((pr != p0).sum())
        print(f"  {lb:<12.1f}{drift:>10.4f}{ch:>8}/{len(y2)}{mm['acc']:>9.4f}"
              f"{mm['macroF1']:>10.4f}{mm['mcc']:>9.4f}")
        rows.append(dict(method="PETAL", lambda_bar=lb, drift=drift, changed=ch,
                         acc=mm["acc"], macroF1=mm["macroF1"], mcc=mm["mcc"]))
    df = pd.DataFrame(rows)
    grid = df[df.lambda_bar.isin([0.0, 1.0])]          # the manuscript's actual grid
    best = grid.loc[grid.macroF1.idxmax()]
    print(f"\n  Manuscript grid was lambda_bar in {{0.0, 1.0}}. Selecting on TARGET macro-F1")
    print(f"  picks lambda_bar = {best.lambda_bar:.1f}  (macroF1 {best.macroF1:.4f}), which "
          f"changes {int(best.changed)}/{len(y2)} predictions.")
    if best.lambda_bar == 0.0:
        print("  -> oracle selection chose the configuration that cannot adapt.")

    print("\nFaithful hyperparameters for all three methods (published values):")
    print(f"  {'method':<28}{'changed':>12}{'acc':>9}{'macroF1':>10}{'MCC':>9}")
    st = source_feature_stats(src, Z1, y1[tr])
    for nm, mk in [("source-only", lambda: SourceOnly(src)),
                   ("DIGA  (lam=.8, P0=.9)", lambda: DIGAFaithful(src)),
                   ("DIGA  (manuscript cfg)", lambda: DIGAFaithful(
                       src, cfg=dict(lam_bn=0.95, P0=0.50, lam_F=0.80, topK=32))),
                   ("PETAL (lambda_bar=1)", lambda: PETALFaithful(src, swag)),
                   ("PeTTA (alpha0=1e-3)", lambda: PeTTAFaithful(src, src_stats=st)),
                   ("PeTTA (manuscript 5e-2)", lambda: PeTTAFaithful(
                       src, src_stats=st, cfg=dict(alpha0=0.05)))]:
        pr, _ = stream(mk, Z2, y2); mm = metrics(y2, pr)
        print(f"  {nm:<28}{int((pr!=p0).sum()):>8}/{len(y2)}{mm['acc']:>9.4f}"
              f"{mm['macroF1']:>10.4f}{mm['mcc']:>9.4f}")
        rows.append(dict(method=nm, lambda_bar=np.nan, drift=np.nan,
                         changed=int((pr != p0).sum()), acc=mm["acc"],
                         macroF1=mm["macroF1"], mcc=mm["mcc"]))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp12_lambda_bar.csv"), index=False)
    print("\n-> results/exp12_lambda_bar.csv")
