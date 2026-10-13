"""
Implementations that follow the three source papers EXACTLY, including their published
hyperparameters, so the comparison cannot be dismissed as a faulty port.

Differences from the manuscript's versions are each traceable to a specific equation:

DIGA  (Wang et al., CVPR 2023)
  Eq.7   q_bar^c_t = rho_P * q_bar^c_{t-1} + (1 - rho_P) * q^c_t
         Note the published rho_P = 0.1 weights the HISTORICAL term, so the "historical"
         prototype is 90% current instance. Implemented as written; flagged as an
         ambiguity in the source paper rather than silently corrected.
  Eq.8   p(c|x,q) = exp(-<z, q_c>) / sum_c' exp(-<z, q_c'>)      <- NEGATIVE inner product
  Eq.9   p_tilde = lambda_P * p(c|x,q_bar) + (1 - lambda_P) * p(c|x,q)
         TWO non-parametric classifiers (historical AND instance). The manuscript's
         Eq.5-6 keeps only one path and so has no lambda_P at all.
  Eq.10  p = lambda_F * p_tilde + (1 - lambda_F) * p_hat
  Sec4.1 rho_P = rho_BN = 0.1 ; lambda_BN = lambda_P = lambda_F = 0.8 ; P0 = 0.9

PETAL (Brahma & Rai, CVPR 2023)
  Alg.2  L = log q(theta) - (lambda_bar/|B|) * H   ;  theta <- theta + eta * grad L
         lambda_bar weights the DATA term; the prior weight is fixed at 1. Therefore
         lambda_bar = 0 deletes the cross-entropy entirely and the update becomes pure
         ascent on the source posterior -- no adaptation by construction.
  Sec5.2 delta = 0.03, K = 32

PeTTA (Hoang et al., NeurIPS 2024)
  Eq.6   gamma^y_t = 1 - exp(-(mu_hat - mu_0)^T Sigma_0^-1 (mu_hat - mu_0))
  Eq.7   lambda_t = gamma_bar * lambda_0 ; alpha_t = (1 - gamma_bar) * alpha_0
  Sec5.2 alpha_0 = 1e-3 (the manuscript used 0.05, fifty times larger),
         lambda_0 = 10, EMA rate for BN and feature statistics = 5e-2,
         cosine-similarity regularizer.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
import tta as _t
from tta import _bn_layers, _sce


class DIGAFaithful:
    """DIGA with its published hyperparameters and both prototype classifiers."""
    name = "DIGA"
    DEFAULTS = dict(lam_bn=0.8, P0=0.9, rho_P=0.1, lam_P=0.8, lam_F=0.8, topK=None,
                    sim="neg_l2")

    def __init__(self, model, device="cpu", cfg=None, K=None, **kw):
        self.c = {**self.DEFAULTS, **(cfg or {})}
        self.K = K if K is not None else _t.K
        self.m = deepcopy(model).to(device).eval(); self.device = device
        self.bns = _bn_layers(self.m)
        self.src = [(b.running_mean.clone(), b.running_var.clone()) for b in self.bns]
        self.hist = None

    @torch.no_grad()
    def __call__(self, x):
        x = x.to(self.device); c = self.c; K = self.K
        if len(x) >= 2:                                   # --- DAM, Eq.3
            hooks, st = [], {}
            def mk(i):
                def h(mod, inp, out):
                    z = inp[0].detach(); st[i] = (z.mean(0), z.var(0, unbiased=False))
                return h
            for i, b in enumerate(self.bns): hooks.append(b.register_forward_hook(mk(i)))
            self.m(x)
            for h in hooks: h.remove()
            for i, b in enumerate(self.bns):
                mu_s, var_s = self.src[i]; mu_t, var_t = st[i]
                b.running_mean = c["lam_bn"] * mu_s + (1 - c["lam_bn"]) * mu_t
                b.running_var = c["lam_bn"] * var_s + (1 - c["lam_bn"]) * var_t
        logit, z = self.m(x, return_feat=True)
        p_hat = F.softmax(logit, 1)

        # --- SAM: instance-aware prototypes, Eq.6
        conf, cls = p_hat.max(1)
        inst = torch.zeros(K, z.shape[1], device=self.device)
        hit = torch.zeros(K, dtype=torch.bool, device=self.device)
        for k in range(K):
            sel = (cls == k) & (conf >= c["P0"])
            if sel.sum() == 0: continue
            zi = z[sel]
            if c["topK"] and len(zi) > c["topK"]:
                zi = zi[conf[sel].topk(c["topK"]).indices]
            inst[k] = zi.mean(0); hit[k] = True
        if self.hist is None:
            self.hist = inst.clone()
        else:                                             # Eq.7, as published
            self.hist[hit] = c["rho_P"] * self.hist[hit] + (1 - c["rho_P"]) * inst[hit]

        def proto_pred(q):
            """Eq.8 reads p(c|x,q) ~ exp(-<z, q_c>). Taken literally as an inner product
            this ranks the LEAST similar prototype highest, i.e. an anti-classifier, so
            <.,.> must denote a dissimilarity. Both readings are provided."""
            if q.abs().sum() == 0: return p_hat
            if c["sim"] == "neg_inner":                   # Eq.8 read literally
                return F.softmax(-(z @ q.T), 1)
            if c["sim"] == "neg_l2":                      # <.,.> as squared distance
                return F.softmax(-torch.cdist(z, q) ** 2, 1)
            return F.softmax(F.normalize(z, dim=1) @ F.normalize(q, dim=1).T / 0.1, 1)
        p_tilde = c["lam_P"] * proto_pred(self.hist) + (1 - c["lam_P"]) * proto_pred(inst)
        p = c["lam_F"] * p_tilde + (1 - c["lam_F"]) * p_hat   # Eq.10
        return p.argmax(1).cpu().numpy(), p.cpu().numpy()


class PETALFaithful:
    """PETAL exactly as Algorithm 2: the coefficient weights the DATA term."""
    name = "PETAL"
    DEFAULTS = dict(pi=0.999, K=32, lambda_bar=1.0, delta=0.03, tau=0.5,
                    lr=1e-4, weight_decay=0.0, noise_std=0.05)

    def __init__(self, model, swag, device="cpu", cfg=None, **kw):
        self.c = {**self.DEFAULTS, **(cfg or {})}; self.device = device
        self.swag = swag
        self.theta0 = deepcopy(model).to(device).eval()
        for p in self.theta0.parameters(): p.requires_grad_(False)
        self.student = deepcopy(model).to(device).train()
        self.teacher = deepcopy(model).to(device).eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.student.parameters(), lr=self.c["lr"],
                                    weight_decay=self.c["weight_decay"])
        self._t0 = torch.cat([p.data.view(-1) for p in self.theta0.parameters()]).to(device)
        self.moved = 0.0

    def __call__(self, x):
        x = x.to(self.device); c = self.c
        with torch.no_grad():
            src_conf = F.softmax(self.theta0(x), 1).max(1).values
            y_hat = F.softmax(self.teacher(x), 1)
            acc = torch.zeros_like(y_hat)
            for _ in range(c["K"]):
                acc += F.softmax(self.teacher(x + torch.randn_like(x) * c["noise_std"]), 1)
            y_tilde = acc / c["K"]
            keep = (src_conf >= c["tau"]).unsqueeze(1).float()
            y_prime = keep * y_hat + (1 - keep) * y_tilde
        pred, prob = y_prime.argmax(1).cpu().numpy(), y_prime.cpu().numpy()

        self.opt.zero_grad(); self.student.train()
        H = -(y_prime * F.log_softmax(self.student(x), 1)).sum(1).mean()
        logq = self.swag.log_q_per_dim(self.student, self.device)
        # maximize  log q - lambda_bar * H   <=>   minimize  lambda_bar * H - log q
        loss = c["lambda_bar"] * H - logq
        loss.backward()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).view(-1)
                       for p in self.student.parameters()])
        fim = g ** 2
        self.opt.step()
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.data = c["pi"] * pt.data + (1 - c["pi"]) * ps.data
            gam = torch.quantile(fim, c["delta"]); mask = (fim < gam).float()
            off = 0
            for p in self.student.parameters():
                k = p.numel()
                p.data = (mask[off:off+k].view_as(p) * self._t0[off:off+k].view_as(p)
                          + (1 - mask[off:off+k].view_as(p)) * p.data)
                off += k
            cur = torch.cat([p.data.view(-1) for p in self.student.parameters()])
            self.moved = float((cur - self._t0).norm())
        return pred, prob


class PeTTAFaithful(_t.PeTTA):
    """PeTTA with the published alpha_0 = 1e-3 instead of the manuscript's 0.05."""
    name = "PeTTA"
    DEFAULTS = dict(alpha0=1e-3, lam0=10.0, lr=1e-3, bank=64, noise_std=0.0, ema_feat=5e-2)
