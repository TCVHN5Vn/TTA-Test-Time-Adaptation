"""Shared utilities: scaler recovery, feature families, transforms, metrics, MLP."""
import sys, pickle, re, numpy as np, torch, torch.nn as nn

# numpy>=2 pickles reference numpy._core; alias onto numpy 1.x
for sub in ["", "multiarray", "numeric", "umath", "_multiarray_umath"]:
    n = "numpy._core" + (("." + sub) if sub else ""); t = "numpy.core" + (("." + sub) if sub else "")
    try: sys.modules.setdefault(n, __import__(t, fromlist=["x"]))
    except Exception: pass

K = 5
# Activity codes 15/60/65/70/85 -> 0..4, names per feature_engineering_wsu.ipynb LABEL_NAMES.
# The submitted manuscript's Table 2 ("Other/Cook/Sleep/Bathe/Toilet") disagrees with its own
# Figure 2 legend and with the time-of-day evidence: class 2 peaks at 11:34, so it is not Sleep.
CLASS_NAMES = ["toileting", "meal_prep", "watching_tv", "music", "cleaning"]
PAPER_NAMES = ["Other", "Cook", "Sleep", "Bathe", "Toilet"]  # as submitted; incorrect


class _ScalerStub:
    """Stand-in so a StandardScaler pickle can be opened without sklearn installed."""
    def __setstate__(self, st): self.__dict__.update(st)


def load_scaler(path):
    """Recover mean_/scale_/var_ from the original sklearn StandardScaler pickle."""
    import types
    for mod, cls in [("sklearn.preprocessing._data", "StandardScaler"),
                     ("sklearn.preprocessing.data", "StandardScaler"),
                     ("sklearn.base", "BaseEstimator")]:
        m = sys.modules.setdefault(mod, types.ModuleType(mod))
        setattr(m, cls, _ScalerStub)
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    sys.modules.setdefault("sklearn.preprocessing", types.ModuleType("sklearn.preprocessing"))
    d = pickle.load(open(path, "rb")).__dict__
    return d["mean_"], d["scale_"], d["var_"]


def family(c):
    if re.match(r"(mean|std)_sensor_id_", c): return "per_sensor"
    if c.startswith("pair_"):                 return "pair"
    if c in ("time_sin", "time_cos", "day_sin", "day_cos"): return "temporal"
    return "global"


# ---------------------------------------------------------------- transforms
def fit_zscore(Xs):
    mu, sd = Xs.mean(0), Xs.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return lambda X: (X - mu) / sd


def fit_zscore_clip(Xs, c=5.0):
    f = fit_zscore(Xs)
    return lambda X: np.clip(f(X), -c, c)


def fit_rank_gauss(Xs, n_q=1000):
    """Source-fit quantile transform -> uniform[0,1] -> bounded. Strictly source-only.
    Target values outside the source support saturate at the boundary instead of
    reaching +100 sigma, which is what destroys a BN-MLP's first layer."""
    qs = np.linspace(0, 1, n_q)
    knots = np.quantile(Xs, qs, axis=0)                       # (n_q, D)
    def tf(X):
        out = np.empty_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            kn = knots[:, j]
            if kn[-1] - kn[0] < 1e-12:                        # constant in source
                out[:, j] = 0.0; continue
            out[:, j] = np.interp(X[:, j], kn, qs)            # clamps at [0,1] by construction
        return (out - 0.5) * 2.0                              # -> [-1, 1]
    return tf


TRANSFORMS = {"zscore": fit_zscore, "zscore_clip5": fit_zscore_clip, "rank_gauss": fit_rank_gauss}


# ------------------------------------------------------------------- metrics
def confusion(yt, yp, K=K):
    M = np.zeros((K, K), dtype=np.int64); np.add.at(M, (yt, yp), 1); return M


def metrics(yt, yp, K=K):
    M = confusion(yt, yp, K); N = M.sum()
    tp = np.diag(M).astype(float); fp = M.sum(0) - tp; fn = M.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(K), where=(tp + fp) > 0)
    rec  = np.divide(tp, tp + fn, out=np.zeros(K), where=(tp + fn) > 0)
    f1   = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    c, s = np.trace(M), N
    p, t = M.sum(0).astype(float), M.sum(1).astype(float)
    den = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return dict(acc=np.trace(M) / N, macroF1=f1.mean(), mcc=(c * s - (p * t).sum()) / den if den > 0 else 0.0,
                bal=rec.mean(), f1=f1, n_pred_classes=int((M.sum(0) > 0).sum()))


def trivial_baselines(yt, K=K, seed=0):
    pri = np.bincount(yt, minlength=K) / len(yt)
    maj = metrics(yt, np.full_like(yt, pri.argmax()), K)
    rng = np.random.default_rng(seed)
    f1s = [metrics(yt, rng.choice(K, len(yt), p=pri), K)["macroF1"] for _ in range(400)]
    accs = [(pri ** 2).sum()]
    return dict(majority_acc=maj["acc"], majority_macroF1=maj["macroF1"],
                strat_acc=accs[0], strat_macroF1=float(np.mean(f1s)))


# --------------------------------------------------------------------- model
class MLP(nn.Module):
    """Same topology as the paper's HAR-MLP: 3 BN blocks, 256/128/64, dropout."""
    def __init__(self, d_in, K=K, p=0.3, widths=(256, 128, 64)):
        super().__init__()
        L, d = [], d_in
        for w in widths:
            L += [nn.Linear(d, w), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(p)]; d = w
        self.feat = nn.Sequential(*L); self.head = nn.Linear(d, K)
    def forward(self, x, return_feat=False):
        z = self.feat(x); o = self.head(z)
        return (o, z) if return_feat else o


def train_mlp(Xtr, ytr, Xva, yva, d_in, seed=0, epochs=60, lr=1e-3, wd=1e-4,
              class_weight=None, device="cpu", bs=128, verbose=False,
              widths=(256, 128, 64), K_out=K):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(d_in, K=K_out, widths=widths).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); loss = lossf(m(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = m(Xva_t).argmax(1).cpu().numpy()
        f1 = metrics(yva, pv)["macroF1"]
        if f1 > best: best, best_state = f1, {k: v.detach().clone() for k, v in m.state_dict().items()}
        if verbose and ep % 10 == 0: print(f"   ep{ep:3d} val_macroF1={f1:.4f}")
    m.load_state_dict(best_state); m.eval()
    return m, best


@torch.no_grad()
def predict(m, X, device="cpu"):
    m.eval()
    o = m(torch.tensor(X, dtype=torch.float32, device=device))
    return o.argmax(1).cpu().numpy(), torch.softmax(o, 1).cpu().numpy()
