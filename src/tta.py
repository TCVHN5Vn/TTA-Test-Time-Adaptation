"""
The five test-time adaptation methods, implemented against the 1-D HAR-MLP.

Each follows the description in the manuscript's Section 3/4 and uses the hyperparameters
the manuscript reports, so the new Table 4 is comparable to the submitted one. The source
model is never given target labels; labels are used only to score predictions afterwards.

Every method is ONLINE and SINGLE-PASS: a batch is predicted BEFORE the update that batch
induces, so no prediction is ever informed by its own adaptation step.

  AdaBN   normalization-only diagnostic: use target batch statistics in BN, no gradients
  TENT    entropy minimization on BN affine parameters only
  DIGA    backward-free: DAM (BN statistic mixing) + SAM (prototype classifier), Sec 3.2
  PETAL   probabilistic lifelong TTA, corrected objective (see petal_fixed.py), Sec 3.3
  PeTTA   mean teacher + Mahalanobis divergence sensing + anchor loss, Sec 3.4
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
from common import K


def _bn_layers(m):
    return [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d)]


# --------------------------------------------------------------------------- source-only
class SourceOnly:
    name = "Source-only"

    def __init__(self, model, device="cpu", **kw):
        self.m = deepcopy(model).to(device).eval(); self.device = device

    @torch.no_grad()
    def __call__(self, x):
        p = F.softmax(self.m(x.to(self.device)), 1)
        return p.argmax(1).cpu().numpy(), p.cpu().numpy()


# ------------------------------------------------------------------------------- AdaBN
class AdaBN:
    """Replace frozen source BN statistics with the current target batch's. No gradients."""
    name = "AdaBN"

    def __init__(self, model, device="cpu", **kw):
        self.m = deepcopy(model).to(device); self.device = device
        for bn in _bn_layers(self.m):
            bn.train()                      # use batch stats in forward
            bn.track_running_stats = False
            bn.running_mean = bn.running_var = None

    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device)
        if len(x) < 2:                      # BN needs >1 sample for a batch statistic
            self.m.eval(); p = F.softmax(self.m(x), 1); self.m.train()
        else:
            p = F.softmax(self.m(x), 1)
        return p.argmax(1).cpu().numpy(), p.cpu().numpy()


# -------------------------------------------------------------------------------- TENT
class TENT:
    """Minimize prediction entropy; only BN affine parameters are trainable."""
    name = "TENT"
    DEFAULTS = dict(lr=1e-4)

    def __init__(self, model, device="cpu", cfg=None, **kw):
        c = {**self.DEFAULTS, **(cfg or {})}
        self.m = deepcopy(model).to(device); self.device = device
        for p in self.m.parameters(): p.requires_grad_(False)
        ps = []
        for bn in _bn_layers(self.m):
            bn.train(); bn.track_running_stats = False
            bn.running_mean = bn.running_var = None
            bn.weight.requires_grad_(True); bn.bias.requires_grad_(True)
            ps += [bn.weight, bn.bias]
        self.opt = torch.optim.Adam(ps, lr=c["lr"])

    def __call__(self, x):
        x = x.to(self.device)
        if len(x) < 2:
            self.m.eval()
            with torch.no_grad(): p = F.softmax(self.m(x), 1)
            self.m.train(); return p.argmax(1).cpu().numpy(), p.cpu().numpy()
        out = self.m(x)
        p = F.softmax(out, 1)
        pred, prob = p.argmax(1).cpu().numpy(), p.detach().cpu().numpy()
        ent = -(p * torch.log(p + 1e-12)).sum(1).mean()
        self.opt.zero_grad(); ent.backward(); self.opt.step()
        return pred, prob


# -------------------------------------------------------------------------------- DIGA
class DIGA:
    """Sec 3.2. DAM mixes source/target BN statistics; SAM adds a prototype classifier.
    Backward-free: no parameter is ever updated by a gradient."""
    name = "DIGA"
    DEFAULTS = dict(lam_bn=0.95, P0=0.50, rho_P=0.10, topK=32, lam_F=0.80)

    def __init__(self, model, device="cpu", cfg=None, **kw):
        self.c = {**self.DEFAULTS, **(cfg or {})}
        self.m = deepcopy(model).to(device).eval(); self.device = device
        self.bns = _bn_layers(self.m)
        self.src = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in self.bns]
        self.proto = None                      # (K, d) EMA prototypes

    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device); c = self.c
        if len(x) >= 2:
            # --- DAM: mix frozen source statistics with this batch's, per BN layer.
            # Forward once in train mode to read off the batch statistics, then write the
            # mixture into running_mean/var and forward in eval mode.
            hooks, stats = [], {}
            def mk(i):
                def h(mod, inp, out):
                    z = inp[0].detach()
                    stats[i] = (z.mean(0), z.var(0, unbiased=False))
                return h
            for i, bn in enumerate(self.bns): hooks.append(bn.register_forward_hook(mk(i)))
            self.m(x)
            for h in hooks: h.remove()
            for i, bn in enumerate(self.bns):
                mu_s, var_s = self.src[i]; mu_t, var_t = stats[i]
                bn.running_mean = c["lam_bn"] * mu_s + (1 - c["lam_bn"]) * mu_t
                bn.running_var = c["lam_bn"] * var_s + (1 - c["lam_bn"]) * var_t
        logit, z = self.m(x, return_feat=True)
        p_hat = F.softmax(logit, 1)

        # --- SAM: per-class prototypes from the most confident samples, EMA-carried.
        conf, cls = p_hat.max(1)
        cur = torch.zeros(K, z.shape[1], device=self.device)
        hit = torch.zeros(K, dtype=torch.bool, device=self.device)
        for k in range(K):
            sel = (cls == k) & (conf >= c["P0"])
            if sel.sum() == 0: continue
            zi, ci = z[sel], conf[sel]
            if len(zi) > c["topK"]:
                zi = zi[ci.topk(c["topK"]).indices]
            cur[k] = zi.mean(0); hit[k] = True
        if self.proto is None:
            self.proto = cur.clone()
        else:
            self.proto[hit] = (1 - c["rho_P"]) * self.proto[hit] + c["rho_P"] * cur[hit]

        if self.proto.abs().sum() > 0:
            sim = F.normalize(z, dim=1) @ F.normalize(self.proto, dim=1).T
            p_tilde = F.softmax(sim / 0.1, 1)
            p = c["lam_F"] * p_tilde + (1 - c["lam_F"]) * p_hat
        else:
            p = p_hat
        return p.argmax(1).cpu().numpy(), p.cpu().numpy()


# ------------------------------------------------------------------------------- PeTTA
def _sce(logit, target_p, alpha=1.0, beta=1.0):
    """Symmetric cross-entropy, as Sec 4.6 specifies."""
    p = F.softmax(logit, 1)
    ce = -(target_p * torch.log(p + 1e-7)).sum(1).mean()
    rce = -(p * torch.log(target_p + 1e-7)).sum(1).mean()
    return alpha * ce + beta * rce


class PeTTA:
    """Sec 3.4 / 4.6. Mean teacher whose update rate and regularization are driven by a
    Mahalanobis divergence signal between per-class target feature means and source ones."""
    name = "PeTTA"
    DEFAULTS = dict(alpha0=0.05, lam0=10.0, lr=5e-5, bank=64, noise_std=0.05, ema_feat=0.05)

    def __init__(self, model, device="cpu", cfg=None, src_stats=None, **kw):
        self.c = {**self.DEFAULTS, **(cfg or {})}; self.device = device
        self.theta0 = deepcopy(model).to(device).eval()
        for p in self.theta0.parameters(): p.requires_grad_(False)
        self.student = deepcopy(model).to(device).train()
        self.teacher = deepcopy(model).to(device).eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.student.parameters(), lr=self.c["lr"])
        for bn in _bn_layers(self.student): bn.momentum = 0.01
        # source per-class feature mean / diagonal covariance, precomputed from S1
        self.mu0, self.var0 = src_stats
        self.mu0 = self.mu0.to(device); self.var0 = self.var0.to(device)
        self.mu_t = self.mu0.clone()
        self.bank_x, self.bank_y = [], []

    def __call__(self, x):
        x = x.to(self.device); c = self.c
        with torch.no_grad():
            logit_t, z_t = self.teacher(x, return_feat=True)
            p_t = F.softmax(logit_t, 1)
        pred, prob = p_t.argmax(1).cpu().numpy(), p_t.cpu().numpy()
        if len(x) < 2: return pred, prob

        with torch.no_grad():
            # --- divergence sensing, Eq. 11-13
            cls = p_t.argmax(1)
            seen = []
            for k in torch.unique(cls):
                k = int(k)
                self.mu_t[k] = (1 - c["ema_feat"]) * self.mu_t[k] + c["ema_feat"] * z_t[cls == k].mean(0)
                seen.append(k)
            d = ((self.mu_t[seen] - self.mu0[seen]) ** 2 / (self.var0[seen] + 1e-6)).sum(1)
            gamma = (1 - torch.exp(-d)).mean()
            lam_t = float(gamma) * c["lam0"]
            alpha_t = (1 - float(gamma)) * c["alpha0"]

            # --- category-balanced memory bank, to break temporal label correlation
            for xi, yi in zip(x, cls):
                self.bank_x.append(xi); self.bank_y.append(int(yi))
            if len(self.bank_x) > c["bank"]:
                by = np.array(self.bank_y); keep = []
                per = max(1, c["bank"] // K)
                for k in range(K):
                    idx = np.where(by == k)[0]
                    keep += list(idx[-per:])
                keep = sorted(keep)[-c["bank"]:]
                self.bank_x = [self.bank_x[i] for i in keep]
                self.bank_y = [self.bank_y[i] for i in keep]
            xb = torch.stack(self.bank_x) if len(self.bank_x) > 1 else x

        xin = xb + torch.randn_like(xb) * c["noise_std"]
        with torch.no_grad():
            pseudo = F.softmax(self.teacher(xin), 1)
            p0 = F.softmax(self.theta0(xin), 1)
        logit_s = self.student(xin)
        loss = _sce(logit_s, pseudo)
        loss = loss - (p0 * F.log_softmax(logit_s, 1)).sum(1).mean()          # anchor, Eq. 14
        reg = sum(((ps - pt) ** 2).sum() for ps, pt in
                  zip(self.student.parameters(), self.theta0.parameters()))
        loss = loss + lam_t * 1e-4 * reg                                      # adaptive R(theta)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.data = (1 - alpha_t) * pt.data + alpha_t * ps.data
        return pred, prob


@torch.no_grad()
def source_feature_stats(model, X, y, device="cpu"):
    """Per-class mean and diagonal variance of the 64-d source embedding, for PeTTA."""
    model.eval()
    _, z = model(torch.tensor(X, dtype=torch.float32, device=device), return_feat=True)
    mu = torch.zeros(K, z.shape[1]); var = torch.ones(K, z.shape[1])
    for k in range(K):
        s = y == k
        if s.sum() > 1:
            mu[k] = z[s].mean(0).cpu(); var[k] = z[s].var(0, unbiased=False).cpu()
    return mu, var


class PETALAdapter:
    """Uniform __call__ interface over PETALFixed, whose own API returns a dict."""
    name = "PETAL-fixed"

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, x):
        r = self.inner.adapt_and_predict(x)
        return r["preds"].numpy(), r["probs"].numpy()

    @property
    def drift(self): return self.inner._moved
