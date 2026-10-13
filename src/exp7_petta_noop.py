"""
EXPERIMENT 7 — why PeTTA does not move, and whether that is a bug or the method.

PeTTA gates BOTH the teacher EMA rate and the regularizer on one scalar (Eq. 11-13):

    gamma_t^y = 1 - exp( -(mu_t - mu_0)^T Sigma_0^-1 (mu_t - mu_0) )
    alpha_t   = (1 - gamma_bar) * alpha_0        <- teacher update rate
    lambda_t  =      gamma_bar  * lambda_0       <- pull back to source

The quadratic form is a SUM over embedding dimensions. With a 64-d embedding, even a
small per-dimension discrepancy makes the sum large, exp(-d) underflows, gamma_bar
saturates at 1.0, and then alpha_t = 0 exactly: the teacher is frozen and PeTTA returns
the source model's predictions forever. The failure is silent -- losses still look fine.

This script logs gamma_bar over the stream and compares three variants:
    as-published   d = sum_j (.)^2 / var_j
    per-dim        d = mean_j (.)^2 / var_j      (scale-free in embedding width)
    floored alpha  alpha_t = max(alpha_t, alpha_min)
to separate "PeTTA cannot help here" from "PeTTA was never allowed to run".
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from tta import PeTTA, SourceOnly, source_feature_stats
import torch.nn.functional as F

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS = 64


class PeTTAProbe(PeTTA):
    """PeTTA with the divergence reduction and an alpha floor made switchable."""

    def __init__(self, *a, reduce="sum", alpha_min=0.0, **kw):
        super().__init__(*a, **kw)
        self.reduce = reduce; self.alpha_min = alpha_min
        self.log = []

    def __call__(self, x):
        x = x.to(self.device); c = self.c
        with torch.no_grad():
            logit_t, z_t = self.teacher(x, return_feat=True)
            p_t = F.softmax(logit_t, 1)
        pred, prob = p_t.argmax(1).cpu().numpy(), p_t.cpu().numpy()
        if len(x) < 2: return pred, prob
        with torch.no_grad():
            cls = p_t.argmax(1); seen = []
            for k in torch.unique(cls):
                k = int(k)
                self.mu_t[k] = (1 - c["ema_feat"]) * self.mu_t[k] + c["ema_feat"] * z_t[cls == k].mean(0)
                seen.append(k)
            q = (self.mu_t[seen] - self.mu0[seen]) ** 2 / (self.var0[seen] + 1e-6)
            d = q.sum(1) if self.reduce == "sum" else q.mean(1)
            gamma = float((1 - torch.exp(-d)).mean())
            alpha_t = max((1 - gamma) * c["alpha0"], self.alpha_min)
            self.log.append((gamma, alpha_t, float(d.mean())))
        xin = x + torch.randn_like(x) * c["noise_std"]
        with torch.no_grad():
            pseudo = F.softmax(self.teacher(xin), 1)
            p0 = F.softmax(self.theta0(xin), 1)
        from tta import _sce
        logit_s = self.student(xin)
        loss = _sce(logit_s, pseudo) - (p0 * F.log_softmax(logit_s, 1)).sum(1).mean()
        reg = sum(((ps - pt) ** 2).sum() for ps, pt in
                  zip(self.student.parameters(), self.theta0.parameters()))
        loss = loss + gamma * c["lam0"] * 1e-4 * reg
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.data = (1 - alpha_t) * pt.data + alpha_t * ps.data
        return pred, prob


def load(var, nm):
    z = np.load(os.path.join(D, f"feat_{var}_{nm}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


if __name__ == "__main__":
    X1, y1, d1, cols = load("causal_recording", "S1")
    X2, y2, _, _ = load("causal_recording", "S2")
    fam = np.array([family(c) for c in cols]); mask = np.isin(fam, ["global", "temporal"])
    days = np.unique(d1); va = np.isin(d1, np.array_split(days, 5)[-1]); tr = ~va
    tf = TRANSFORMS["rank_gauss"](X1[tr][:, mask])
    Z1tr, Z1va, Z2 = tf(X1[tr][:, mask]), tf(X1[va][:, mask]), tf(X2[:, mask])
    src, _ = train_mlp(Z1tr, y1[tr], Z1va, y1[va], int(mask.sum()), seed=0, epochs=60)
    st = source_feature_stats(src, Z1tr, y1[tr])
    base = SourceOnly(src)
    p0 = np.concatenate([base(torch.tensor(Z2[i:i+BS], dtype=torch.float32))[0]
                         for i in range(0, len(Z2), BS)])
    m0 = metrics(y2, p0)
    print("=" * 96)
    print("EXPERIMENT 7 — PeTTA divergence saturation")
    print("=" * 96)
    print(f"source-only: acc={m0['acc']:.4f} macroF1={m0['macroF1']:.4f} MCC={m0['mcc']:.4f}\n")
    rows = []
    for tag, kw in [("as-published (sum)", dict(reduce="sum")),
                    ("per-dim mean", dict(reduce="mean")),
                    ("sum + alpha floor 1e-3", dict(reduce="sum", alpha_min=1e-3)),
                    ("per-dim + alpha floor 1e-3", dict(reduce="mean", alpha_min=1e-3))]:
        ad = PeTTAProbe(src, src_stats=st, **kw)
        pr = np.concatenate([ad(torch.tensor(Z2[i:i+BS], dtype=torch.float32))[0]
                             for i in range(0, len(Z2), BS)])
        mm = metrics(y2, pr)
        g = np.array([l[0] for l in ad.log]); al = np.array([l[1] for l in ad.log])
        dd = np.array([l[2] for l in ad.log])
        print(f"{tag:<28} gamma_bar med={np.median(g):.6f} (min {g.min():.4f})  "
              f"alpha_t med={np.median(al):.2e}  mahal_d med={np.median(dd):.2e}")
        print(f"{'':<28} changed={int((pr!=p0).sum()):4d}/{len(p0)}   "
              f"acc={mm['acc']:.4f} macroF1={mm['macroF1']:.4f} MCC={mm['mcc']:.4f}\n")
        rows.append(dict(variant=tag, gamma_med=np.median(g), alpha_med=np.median(al),
                         mahal_med=np.median(dd), changed=int((pr != p0).sum()),
                         acc=mm["acc"], macroF1=mm["macroF1"], mcc=mm["mcc"]))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp7_petta_noop.csv"), index=False)
    print("-> results/exp7_petta_noop.csv")


# ---------------------------------------------------------------------------------------
# FOLLOW-UP: does PeTTA's reported gain come from its adaptive mechanism, or just from BN?
#
# The submitted Table 4 gives PeTTA acc=0.392 / macroF1=0.244 / MCC=0.181, and the text
# gives AdaBN acc=0.376 / macroF1=0.265 / MCC=0.168 -- nearly the same model. PeTTA runs
# its student with BN momentum 0.01 in train mode, so the student's BN running statistics
# drift toward the target no matter what the teacher EMA does. If predictions are read off
# the STUDENT rather than the teacher, that drift alone moves them, and the "gain" is an
# AdaBN effect wearing PeTTA's name. Predicting from the teacher (standard mean-teacher,
# what tta.PeTTA does) isolates the adaptive mechanism.
# ---------------------------------------------------------------------------------------
def _followup():
    import torch.nn.functional as F
    from tta import AdaBN
    X1, y1, d1, cols = load("causal_recording", "S1")
    X2, y2, _, _ = load("causal_recording", "S2")
    fam = np.array([family(c) for c in cols]); mask = np.isin(fam, ["global", "temporal"])
    days = np.unique(d1); va = np.isin(d1, np.array_split(days, 5)[-1]); tr = ~va
    tf = TRANSFORMS["zscore"](X1[tr][:, mask])
    Z1tr, Z1va, Z2 = tf(X1[tr][:, mask]), tf(X1[va][:, mask]), tf(X2[:, mask])
    src, _ = train_mlp(Z1tr, y1[tr], Z1va, y1[va], int(mask.sum()), seed=0, epochs=60)
    st = source_feature_stats(src, Z1tr, y1[tr])
    base = SourceOnly(src)
    p0 = np.concatenate([base(torch.tensor(Z2[i:i+BS], dtype=torch.float32))[0]
                         for i in range(0, len(Z2), BS)])

    class StudentPred(PeTTAProbe):
        """Identical adaptation; predictions read from the student instead of the teacher."""
        def __call__(self, x):
            self.student.eval()
            with torch.no_grad():
                p = F.softmax(self.student(x.to(self.device)), 1)
            self.student.train()
            super().__call__(x)                       # run the normal update
            return p.argmax(1).cpu().numpy(), p.cpu().numpy()

    print("\n" + "=" * 96)
    print("FOLLOW-UP — PeTTA teacher-readout vs student-readout vs AdaBN   [9-d, zscore]")
    print("=" * 96)
    print(f"{'variant':<34}{'changed':>10}{'acc':>10}{'macroF1':>10}{'MCC':>10}")
    print(f"{'source-only':<34}{0:>10}{metrics(y2,p0)['acc']:>10.4f}"
          f"{metrics(y2,p0)['macroF1']:>10.4f}{metrics(y2,p0)['mcc']:>10.4f}")
    rows = []
    for tag, mk in [("PeTTA (teacher readout)", lambda: PeTTAProbe(src, src_stats=st, reduce="sum")),
                    ("PeTTA (student readout)", lambda: StudentPred(src, src_stats=st, reduce="sum")),
                    ("AdaBN", lambda: AdaBN(src))]:
        ad = mk()
        pr = np.concatenate([ad(torch.tensor(Z2[i:i+BS], dtype=torch.float32))[0]
                             for i in range(0, len(Z2), BS)])
        mm = metrics(y2, pr)
        print(f"{tag:<34}{int((pr!=p0).sum()):>10}{mm['acc']:>10.4f}"
              f"{mm['macroF1']:>10.4f}{mm['mcc']:>10.4f}")
        rows.append(dict(variant=tag, changed=int((pr != p0).sum()), acc=mm["acc"],
                         macroF1=mm["macroF1"], mcc=mm["mcc"]))
        np.save(os.path.join(R, f"pred_followup_{tag.split()[0]}_{tag[-9:-1]}.npy"), pr)
    # how similar are student-readout PeTTA and AdaBN to each other?
    pa = np.load(os.path.join(R, "pred_followup_PeTTA_t readou.npy")) if False else None
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp7b_followup.csv"), index=False)
    print("-> results/exp7b_followup.csv")


if __name__ == "__main__":
    _followup()
