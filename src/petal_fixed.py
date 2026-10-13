"""
Corrected PETAL adapter.

TWO BUGS in the original implementation made PETAL a mathematical no-op, which is why
it reproduced the source-only predictions to 15 decimal places on both tasks and every
stream ordering.

BUG 1 - lambda_bar multiplied the DATA term, not the prior.
    original:  loss = -(log_q - lambda_bar * H)      with lambda_bar = 0.0
            =>  loss = -log_q                        (the pseudo-label term is deleted)
    PETAL's MAP objective is  argmax_theta [ log p(D|theta) + log p(theta) ], i.e.
            minimize  H_xe(y', y_student)  -  w * log q(theta).
    Dividing the original by lambda_bar shows its effective prior weight is 1/lambda_bar,
    so lambda_bar -> 0 means prior weight -> infinity: the student is frozen at source.
    FIX: reparameterize so the coefficient weights the PRIOR. Then lambda_bar = 0 means
    "no regularization" (plain self-training, which still adapts) instead of "no learning",
    which is also what the paper's prose claims it means.

BUG 2 - the SWAG posterior was degenerate.
    swag_har_engineered_fixed.pt had 99.46% of sigma entries pinned at sqrt(eps)=1e-3
    because all 20 snapshots were collected from an already-converged model at a low
    learning rate, so the weight-space variance was ~0. That makes log q a spring of
    stiffness 1/sigma^2 ~ 1e6 pinning theta to mu, which freezes the student even when
    the data term is restored.
    FIX: collect snapshots SWA-style under a constant moderate LR so the weights actually
    wander (see collect_swag below), normalize log_q by dimension so the coefficient has
    a scale-free meaning, and refuse to run on a degenerate posterior.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy


class SWAGDiag:
    """Diagonal SWAG posterior with an explicit degeneracy check."""

    def __init__(self, eps=1e-12):
        self.mu = self.sigma = None; self.eps = eps; self.n = 0

    def fit(self, snapshots):
        S = torch.stack([s.cpu() for s in snapshots])
        self.mu = S.mean(0)
        self.sigma = torch.sqrt(S.var(0, unbiased=True) + self.eps)
        self.n = len(S)
        return self

    def degeneracy(self):
        """Fraction of coordinates whose posterior std sits at the numerical floor."""
        floor = np.sqrt(self.eps) * 1.01
        return float((self.sigma <= floor).float().mean())

    def check(self, max_degenerate=0.25):
        f = self.degeneracy()
        msg = (f"SWAG posterior: n={self.n} snapshots, sigma median={self.sigma.median():.3e}, "
               f"{f*100:.2f}% of coords at the numerical floor")
        if f > max_degenerate:
            raise RuntimeError(
                msg + f"\n  -> DEGENERATE (>{max_degenerate*100:.0f}%). The snapshots are "
                "effectively identical, so log q becomes a ~1/eps spring pinning the student "
                "to the source weights and PETAL cannot adapt. Re-collect with collect_swag("
                "lr_const=...) at a higher constant LR, or over more epochs.")
        return msg

    def log_q_per_dim(self, model, device="cpu"):
        """log q(theta) / D -- normalized so the prior weight is scale-free."""
        mu, sg = self.mu.to(device), self.sigma.to(device)
        th = torch.cat([p.view(-1) for p in model.parameters()])
        return -0.5 * (((th - mu) / sg) ** 2).mean()


def collect_swag(model, train_fn, n_snapshots=30, every=1):
    """SWA-style collection: train_fn(epoch) must take ONE optimizer step pass at a
    constant, non-trivial LR so successive weights differ. Returns flat snapshots."""
    snaps = []
    for ep in range(n_snapshots * every):
        train_fn(ep)
        if ep % every == 0:
            snaps.append(torch.cat([p.data.view(-1) for p in model.parameters()]).detach().clone())
    return snaps


def augment_tabular(x, noise_std=0.1, drop_rate=0.1):
    """Stand-in for PETAL's image augmentations. Still a weak surrogate for sensor data --
    this is a real limitation and must be stated, not a bug."""
    xf = x.clone()
    if torch.rand(1).item() < 0.5:
        xf = xf + torch.randn_like(xf) * noise_std
    else:
        xf = xf * (torch.rand_like(xf) > drop_rate).float()
    return xf


class PETALFixed:
    DEFAULTS = dict(pi=0.95, K=32, lambda_bar=1e-2, delta=0.1, tau=0.5,
                    lr=1e-4, weight_decay=1e-4, noise_std=0.1, drop_rate=0.1)

    def __init__(self, source_model, swag: SWAGDiag, cfg=None, device="cpu", strict=True):
        self.cfg = {**self.DEFAULTS, **(cfg or {})}; self.device = device; self.swag = swag
        print("[PETAL-fixed] " + (swag.check() if strict else f"posterior degeneracy="
                                  f"{swag.degeneracy():.3f} (unchecked)"))
        self.theta_0 = deepcopy(source_model).to(device).eval()
        for p in self.theta_0.parameters(): p.requires_grad_(False)
        self.student = deepcopy(source_model).to(device).train()
        self.teacher = deepcopy(source_model).to(device).eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.student.parameters(), lr=self.cfg["lr"],
                                    weight_decay=self.cfg["weight_decay"])
        self._t0flat = torch.cat([p.data.view(-1) for p in self.theta_0.parameters()]).to(device)
        self._moved = 0.0   # cumulative L2 drift from source: a no-op detector

    def adapt_and_predict(self, x):
        x = x.to(self.device); c = self.cfg
        with torch.no_grad():
            src_conf = F.softmax(self.theta_0(x), 1).max(1).values
            y_hat = F.softmax(self.teacher(x), 1)
            acc = torch.zeros_like(y_hat)
            for _ in range(c["K"]):
                acc += F.softmax(self.teacher(augment_tabular(x, c["noise_std"], c["drop_rate"])), 1)
            y_tilde = acc / c["K"]
            use_plain = (src_conf >= c["tau"]).unsqueeze(1).float()
            y_prime = use_plain * y_hat + (1 - use_plain) * y_tilde
        preds, probs = y_prime.argmax(1).cpu(), y_prime.cpu()   # online prediction, pre-update

        # ---- CORRECTED OBJECTIVE: data fit + weighted prior -------------------
        self.opt.zero_grad(); self.student.train()
        logit_s = self.student(x)
        H = -(y_prime * F.log_softmax(logit_s, 1)).sum(1).mean()          # pseudo-label CE
        logq = self.swag.log_q_per_dim(self.student, self.device)          # per-dim prior
        loss = H - c["lambda_bar"] * logq
        loss.backward()
        grads = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).view(-1)
                           for p in self.student.parameters()])
        fim = grads ** 2
        self.opt.step()

        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.data = c["pi"] * pt.data + (1 - c["pi"]) * ps.data
            # FIM restoration: send the least-important coordinates back to source
            gamma = torch.quantile(fim, c["delta"]); mask = (fim < gamma).float()
            off = 0
            for p in self.student.parameters():
                k = p.numel()
                p.data = (mask[off:off+k].view_as(p) * self._t0flat[off:off+k].view_as(p)
                          + (1 - mask[off:off+k].view_as(p)) * p.data)
                off += k
            cur = torch.cat([p.data.view(-1) for p in self.student.parameters()])
            self._moved = float((cur - self._t0flat).norm())
        return dict(preds=preds, probs=probs, drift=self._moved, loss=float(loss))
