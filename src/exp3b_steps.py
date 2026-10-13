"""
EXPERIMENT 3b — Is the target stream simply too short for gradient-based TTA?

After the fix, PETAL adapts (nonzero drift) but changes only ~21/3412 predictions.
A single pass over 3412 samples at batch size 256 is only 13 optimizer steps.
The image benchmarks these methods were tuned on give 10k+ samples per domain and
often several passes. This script varies the number of update steps while holding
the data fixed, to separate "the method does not work here" from "the method was
never given enough steps to do anything".
"""
import sys, os, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from petal_fixed import SWAGDiag, PETALFixed
import torch.nn as nn

D = os.path.join(os.path.dirname(__file__), "..", "data")
s1 = pd.read_csv(os.path.join(D, "S1_features_engineered.csv")); s2 = pd.read_csv(os.path.join(D, "S2_features_engineered.csv"))
cols = [c for c in s1.columns if c != "label"]
mu, sc = np.load(os.path.join(D, "sc_mean.npy")), np.load(os.path.join(D, "sc_scale.npy"))
X1 = s1[cols].values * sc + mu; y1 = s1.label.values
X2 = s2[cols].values * sc + mu; y2 = s2.label.values
n = len(X1); cut = int(.8 * n); tr, va = np.arange(0, cut - 20), np.arange(cut, n)
tf = fit_zscore(X1[tr]); Z1tr, Z1va, Z2 = tf(X1[tr]), tf(X1[va]), tf(X2)

src, _ = train_mlp(Z1tr, y1[tr], Z1va, y1[va], len(cols), seed=0, epochs=60)
yp0, _ = predict(src, Z2); m0 = metrics(y2, yp0)
print("=" * 96); print("EXPERIMENT 3b — does PETAL move at all, given enough steps?"); print("=" * 96)
print(f"source-only: acc={m0['acc']:.4f} macroF1={m0['macroF1']:.4f} MCC={m0['mcc']:.4f}\n")

def snaps(lr, k=30):
    m = MLP(len(cols)); m.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Z1tr, dtype=torch.float32); yt = torch.tensor(y1[tr], dtype=torch.long)
    out = []
    for _ in range(k):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]
            if len(idx) < 2: continue
            opt.zero_grad(); lf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        out.append(torch.cat([p.data.view(-1) for p in m.parameters()]).detach().clone())
    return out
sw = SWAGDiag(eps=1e-12).fit(snaps(1e-3))

print(f"{'batch':>6}{'passes':>8}{'steps':>7}{'lr':>8} | {'drift':>8}{'changed':>9} | {'acc':>8}{'macroF1':>9}{'MCC':>8}")
print("-" * 76)
rows = []
for bs, passes, lr in [(256, 1, 1e-4), (128, 1, 1e-4), (64, 1, 1e-4), (64, 5, 1e-4),
                       (64, 5, 1e-3), (64, 20, 1e-3), (32, 20, 1e-3)]:
    ad = PETALFixed(src, sw, cfg=dict(lambda_bar=1e-2, lr=lr, K=8), device="cpu", strict=False)
    last = None; steps = 0
    for p in range(passes):
        P = []
        for i in range(0, len(Z2), bs):
            xb = torch.tensor(Z2[i:i+bs], dtype=torch.float32)
            if len(xb) < 2: break
            P.append(ad.adapt_and_predict(xb)["preds"].numpy()); steps += 1
        last = np.concatenate(P)
    yt = y2[:len(last)]; mm = metrics(yt, last)
    ch = int((last != yp0[:len(last)]).sum())
    print(f"{bs:>6}{passes:>8}{steps:>7}{lr:>8.0e} | {ad._moved:>8.3f}{ch:>9} | "
          f"{mm['acc']:>8.4f}{mm['macroF1']:>9.4f}{mm['mcc']:>8.4f}")
    rows.append(dict(bs=bs, passes=passes, steps=steps, lr=lr, drift=ad._moved, changed=ch,
                     **{k: mm[k] for k in ("acc", "macroF1", "mcc")}))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "exp3b_steps.csv"), index=False)
print("\n-> results/exp3b_steps.csv")
