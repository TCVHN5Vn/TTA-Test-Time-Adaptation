"""
EXPERIMENT 22 — is PETAL inert because of lambda_bar, or because of the step budget?

Section~\ref{sec:res-mech} shows that lambda_bar = 0 removes the data term, and that
selection on target labels prefers exactly that value. But the lambda_bar sweep also shows
the predictions barely moving at lambda_bar = 10, where the data term is weighted ten
times the prior. Something else is holding the method still.

The stream is 3412 samples, which at batch size 64 is 54 gradient steps. PETAL's published
learning rate is 1e-4 and its teacher is an EMA at pi = 0.999, so over the whole stream the
teacher that produces the predictions moves 1 - 0.999^54 = 5.3% of the way toward the
student. Those constants are calibrated for corruption benchmarks supplying tens of
thousands of batches.

This script removes the constraint, freeing the teacher (pi = 0.9) and sweeping the
learning rate. If the step budget is the binding constraint, PETAL should start to move.
What it does once it moves is the result worth reporting.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import K, MLP, train_mlp, metrics, TRANSFORMS
from tta import SourceOnly
from tta_faithful import PETALFaithful
from petal_fixed import SWAGDiag

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS = 64


def load(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


if __name__ == "__main__":
    X1, y1, d1, cols = load("S1"); X2, y2, _, _ = load("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols])
    days = np.unique(d1); va = np.isin(d1, np.array_split(days, 5)[-1]); tr = ~va
    tf = TRANSFORMS["zscore"](X1[tr][:, mask])
    Z1, Z1v, Z2 = tf(X1[tr][:, mask]), tf(X1[va][:, mask]), tf(X2[:, mask])
    src, _ = train_mlp(Z1, y1[tr], Z1v, y1[va], int(mask.sum()), seed=0, epochs=60)

    # SWAG posterior at constant LR so it is not degenerate
    torch.manual_seed(0); np.random.seed(0)
    sw = MLP(int(mask.sum())); sw.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Z1, dtype=torch.float32); yt = torch.tensor(y1[tr], dtype=torch.long)
    snaps = []
    for _ in range(30):
        sw.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            if len(j) < 2: continue
            opt.zero_grad(); lf(sw(Xt[j]), yt[j]).backward(); opt.step()
        snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
    swag = SWAGDiag().fit(snaps)

    p0 = np.concatenate([SourceOnly(src)(torch.tensor(Z2[i:i + BS], dtype=torch.float32))[0]
                         for i in range(0, len(Z2), BS)])
    nb = int(np.ceil(len(Z2) / BS))
    print("=" * 84)
    print("EXPERIMENT 22 — PETAL's step budget")
    print(f"  stream: {len(Z2)} samples = {nb} gradient steps at batch {BS}")
    print(f"  teacher EMA reach over the stream, 1 - pi^{nb}:")
    for pi in (0.999, 0.99, 0.95, 0.9):
        print(f"     pi={pi:<7}{100*(1-pi**nb):5.1f}%")
    print("=" * 84)

    rows = []
    print(f"\n  {'setting':<34}{'changed':>12}{'macroF1':>10}{'MCC':>9}")
    print("  " + "-" * 64)
    for tag, cfg in [("published (pi=.999, lr=1e-4)", dict(lambda_bar=1.0)),
                     ("teacher freed, lr=1e-4", dict(lambda_bar=1.0, pi=0.9, lr=1e-4)),
                     ("teacher freed, lr=1e-3", dict(lambda_bar=1.0, pi=0.9, lr=1e-3)),
                     ("teacher freed, lr=1e-2", dict(lambda_bar=1.0, pi=0.9, lr=1e-2)),
                     ("teacher freed, lr=5e-2", dict(lambda_bar=1.0, pi=0.9, lr=5e-2))]:
        ad = PETALFaithful(src, swag, cfg=cfg)
        pr = np.concatenate([ad(torch.tensor(Z2[i:i + BS], dtype=torch.float32))[0]
                             for i in range(0, len(Z2), BS)])
        m = metrics(y2, pr)
        print(f"  {tag:<34}{int((pr != p0).sum()):>8}/{len(y2)}{m['macroF1']:>10.4f}"
              f"{m['mcc']:>9.4f}")
        rows.append(dict(setting=tag, changed=int((pr != p0).sum()),
                         macroF1=m["macroF1"], mcc=m["mcc"]))
    m0 = metrics(y2, p0)
    print(f"\n  source-only{'':23}{0:>8}/{len(y2)}{m0['macroF1']:>10.4f}{m0['mcc']:>9.4f}")
    rows.append(dict(setting="source-only", changed=0, macroF1=m0["macroF1"], mcc=m0["mcc"]))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp22_petal_budget.csv"), index=False)
    print("\n-> results/exp22_petal_budget.csv")
