"""
EXPERIMENT 3 — Prove the original PETAL is a no-op and that the corrected one adapts.

Uses the paper's exact HAR setup (135-d, source-fit z-score, HAR-MLP) so the comparison
is apples-to-apples with Table 4. Reports, for each variant:
  - whether predictions differ from source-only AT ALL (n_changed)
  - the L2 drift of the student from the source weights
  - the resulting target metrics
"""
import sys, os, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from petal_fixed import SWAGDiag, PETALFixed

D = os.path.join(os.path.dirname(__file__), "..", "data")
s1 = pd.read_csv(os.path.join(D, "S1_features_engineered.csv"))
s2 = pd.read_csv(os.path.join(D, "S2_features_engineered.csv"))
cols = [c for c in s1.columns if c != "label"]
mu, sc = np.load(os.path.join(D, "sc_mean.npy")), np.load(os.path.join(D, "sc_scale.npy"))
X1 = s1[cols].values * sc + mu; y1 = s1.label.values
X2 = s2[cols].values * sc + mu; y2 = s2.label.values
n = len(X1); cut = int(.8 * n)
tr, va = np.arange(0, cut - 20), np.arange(cut, n)
tf = fit_zscore(X1[tr]); Z1tr, Z1va, Z2 = tf(X1[tr]), tf(X1[va]), tf(X2)
dev = "cpu"

print("=" * 96); print("EXPERIMENT 3 — PETAL: no-op proof and fix"); print("=" * 96)
src, _ = train_mlp(Z1tr, y1[tr], Z1va, y1[va], len(cols), seed=0, epochs=60)
yp0, _ = predict(src, Z2); m0 = metrics(y2, yp0)
print(f"\nsource-only on target: acc={m0['acc']:.4f} macroF1={m0['macroF1']:.4f} MCC={m0['mcc']:.4f}")

# ---- SWAG posteriors: degenerate (as shipped) vs properly collected ------------
import torch.nn as nn
def make_snaps(lr_const, n_snap=30):
    m = MLP(len(cols)).to(dev); m.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr_const)
    Xt = torch.tensor(Z1tr, dtype=torch.float32); yt = torch.tensor(y1[tr], dtype=torch.long)
    lf = nn.CrossEntropyLoss(); snaps = []
    for ep in range(n_snap):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]
            if len(idx) < 2: continue
            opt.zero_grad(); lf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        snaps.append(torch.cat([p.data.view(-1) for p in m.parameters()]).detach().clone())
    return snaps

print("\n--- SWAG posterior quality ---")
sw_bad  = SWAGDiag(eps=1e-6).fit(make_snaps(1e-6))     # reproduces the shipped checkpoint
sw_good = SWAGDiag(eps=1e-12).fit(make_snaps(1e-3))    # SWA-style, constant moderate LR
print(f"  as-shipped recipe (lr=1e-6, eps=1e-6): degeneracy={sw_bad.degeneracy()*100:5.2f}% "
      f"sigma_med={sw_bad.sigma.median():.3e}")
print(f"  corrected recipe  (lr=1e-3, eps=1e-12): degeneracy={sw_good.degeneracy()*100:5.2f}% "
      f"sigma_med={sw_good.sigma.median():.3e}")

# ---- run a single-pass stream under each configuration -------------------------
def run(swag, lam, strict, label, bs=256):
    ad = PETALFixed(src, swag, cfg=dict(lambda_bar=lam), device=dev, strict=strict)
    P = []
    for i in range(0, len(Z2), bs):
        xb = torch.tensor(Z2[i:i+bs], dtype=torch.float32)
        if len(xb) < 2: break
        P.append(ad.adapt_and_predict(xb)["preds"].numpy())
    yp = np.concatenate(P); yt = y2[:len(yp)]
    mm = metrics(yt, yp); ch = int((yp != yp0[:len(yp)]).sum())
    print(f"  {label:<46} drift={ad._moved:7.4f}  n_changed={ch:5d}/{len(yp)}  "
          f"acc={mm['acc']:.4f} macroF1={mm['macroF1']:.4f} MCC={mm['mcc']:.4f}")
    return dict(variant=label, drift=ad._moved, n_changed=ch, n=len(yp), **{k: mm[k] for k in ("acc","macroF1","mcc")})

print("\n--- single-pass target stream ---")
rows = []
# Faithful reproduction of the published configuration: prior weight = 1/lambda_bar -> inf
class OriginalPETAL(PETALFixed):
    """lambda_bar weights the DATA term, as in the published code, with lambda_bar=0."""
    def adapt_and_predict(self, x):
        import torch.nn.functional as F
        x = x.to(self.device); c = self.cfg
        with torch.no_grad():
            y_prime = F.softmax(self.teacher(x), 1)
        preds = y_prime.argmax(1).cpu()
        self.opt.zero_grad(); self.student.train()
        H = -(y_prime * F.log_softmax(self.student(x), 1)).sum(1).mean()
        logq = self.swag.log_q_per_dim(self.student, self.device)
        (-(logq - 0.0 * H)).backward()           # <-- the published loss with lambda_bar = 0
        self.opt.step()
        with torch.no_grad():
            cur = torch.cat([p.data.view(-1) for p in self.student.parameters()])
            self._moved = float((cur - self._t0flat).norm())
        return dict(preds=preds, probs=y_prime.cpu(), drift=self._moved, loss=0.0)

ad = OriginalPETAL(src, sw_bad, cfg=dict(lambda_bar=0.0), device=dev, strict=False)
P = []
for i in range(0, len(Z2), 256):
    xb = torch.tensor(Z2[i:i+256], dtype=torch.float32)
    if len(xb) < 2: break
    P.append(ad.adapt_and_predict(xb)["preds"].numpy())
yp = np.concatenate(P); mm = metrics(y2[:len(yp)], yp)
print(f"  {'PUBLISHED: lam_bar=0 + degenerate SWAG':<46} drift={ad._moved:7.4f}  "
      f"n_changed={int((yp!=yp0[:len(yp)]).sum()):5d}/{len(yp)}  acc={mm['acc']:.4f} "
      f"macroF1={mm['macroF1']:.4f} MCC={mm['mcc']:.4f}")
rows.append(dict(variant="PUBLISHED (lam=0, degenerate SWAG)", drift=ad._moved,
                 n_changed=int((yp!=yp0[:len(yp)]).sum()), n=len(yp), **{k: mm[k] for k in ("acc","macroF1","mcc")}))

for lam, sw, lbl in [(0.0, sw_good, "FIXED: prior weight 0 (pure self-training)"),
                     (1e-2, sw_good, "FIXED: prior weight 1e-2"),
                     (1e-1, sw_good, "FIXED: prior weight 1e-1"),
                     (1.0,  sw_good, "FIXED: prior weight 1.0")]:
    rows.append(run(sw, lam, False, lbl))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "exp3_petal.csv"), index=False)
print("\n-> results/exp3_petal.csv")
