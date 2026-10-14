"""
EXPERIMENT 16 — does modelling the sequence beat smoothing it?

Each 30 s window is currently classified independently, while activities persist for
minutes. The causal median filter patches this after the fact and keeps improving out to
L=41, which says there is temporal structure the classifier never sees. Three ways of
using it properly, all strictly causal so they remain deployable online:

  stack     feed the last T feature vectors to the MLP as one flattened input. The
            cheapest way to give the model history.
  tcn       a small dilated 1-D convolution over the same stack, which shares weights
            across time instead of learning a separate weight per lag.
  hmm       leave the classifier alone and decode its output sequence with a first-order
            Markov chain whose transition matrix is estimated on the SOURCE only, using
            forward filtering (causal; no backward pass, no Viterbi over the future).

The HMM is the interesting one for this paper: transitions are a property of how
activities follow one another, which may transfer across buildings even when the sensors
do not. It also costs nothing at inference.

Protocol unchanged: 13-d layout-invariant causal features, GroupKFold over year_day,
3 seeds, both directions, source-fit z-score, no target information anywhere.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import K, metrics, trivial_baselines, TRANSFORMS, train_mlp, predict
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)
SEG = {t: segments(load_raw(t), "recording") for t in ("S1", "S2")}


def load_inv(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def causal_median(pred, seg, L=41):
    if L <= 1: return pred
    out = np.empty_like(pred)
    for i in range(len(pred)):
        lo = i
        while lo > 0 and i - lo < L - 1 and seg[lo - 1] == seg[i]:
            lo -= 1
        out[i] = np.bincount(pred[lo:i + 1], minlength=K).argmax()
    return out


def make_stack(X, seg, T):
    """[x_{t-T+1} .. x_t] flattened, repeating the segment's first frame at the start."""
    n, d = X.shape
    out = np.empty((n, T * d), X.dtype)
    starts = {}
    for i in range(n):
        s = seg[i]
        if s not in starts: starts[s] = i
        idx = [max(i - k, starts[s]) for k in range(T - 1, -1, -1)]
        out[i] = X[idx].ravel()
    return out


# ------------------------------------------------------------------ causal TCN
class TCN(nn.Module):
    def __init__(self, d_in, T, K=K, ch=64):
        super().__init__()
        self.T, self.d = T, d_in
        self.net = nn.Sequential(
            nn.Conv1d(d_in, ch, 3, padding=2, dilation=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=4, dilation=4),   nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Dropout(0.3))
        self.head = nn.Linear(ch, K)

    def forward(self, x):
        b = x.shape[0]
        z = x.view(b, self.T, self.d).transpose(1, 2)      # (B, d, T)
        h = self.net(z)[:, :, :self.T]                      # causal crop
        return self.head(h[:, :, -1])


def train_tcn(Xtr, ytr, Xva, yva, d_in, T, seed=0, epochs=60, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TCN(d_in, T)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, st = -1, None
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            if len(j) < 2: continue
            opt.zero_grad(); lf(m(Xt[j]), yt[j]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = m(Xv).argmax(1).numpy()
        f = metrics(yva, pv)["macroF1"]
        if f > best: best, st = f, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(st); m.eval(); return m, best


@torch.no_grad()
def tcn_predict(m, X):
    o = m(torch.tensor(X, dtype=torch.float32))
    return o.argmax(1).numpy(), torch.softmax(o, 1).numpy()


# -------------------------------------------------------- causal HMM filtering
def fit_transitions(y, seg, eps=1.0):
    """First-order transition matrix, estimated on the SOURCE only."""
    A = np.full((K, K), eps)
    for i in range(1, len(y)):
        if seg[i] == seg[i - 1]:
            A[y[i - 1], y[i]] += 1
    return A / A.sum(1, keepdims=True)


def forward_filter(prob, A, seg, prior=None):
    """Causal forward recursion: alpha_t ∝ p(x_t|y_t) * sum_y' A[y',y] alpha_{t-1}.
    Uses only the past, so it is valid online."""
    n = len(prob)
    prior = prior if prior is not None else np.full(K, 1.0 / K)
    out = np.empty(n, int)
    a = None
    for i in range(n):
        if i == 0 or seg[i] != seg[i - 1]:
            a = prior * np.maximum(prob[i], 1e-12)
        else:
            a = (a @ A) * np.maximum(prob[i], 1e-12)
        a = a / a.sum()
        out[i] = a.argmax()
    return out


if __name__ == "__main__":
    X1, y1, d1, cols = load_inv("S1"); X2, y2, d2, _ = load_inv("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols])
    A1, A2 = X1[:, mask], X2[:, mask]
    d = A1.shape[1]
    tb2 = trivial_baselines(y2)

    print("=" * 102)
    print("EXPERIMENT 16 — temporal modelling on the 13-d layout-invariant checkpoint")
    print(f"  chance macro-F1 on S2 {tb2['strat_macroF1']:.4f}; "
          f"reference MLP+median(41): 0.4466 / 0.3068")
    print("=" * 102)
    print(f"  {'model':<30}{'dir':<10}{'src-val':>9}{'TGT macroF1':>13}{'sd':>7}"
          f"{'MCC':>9}{'acc':>8}")
    print("  " + "-" * 76)

    DIRS = {"S1->S2": (A1, y1, d1, "S1", A2, y2, "S2"),
            "S2->S1": (A2, y2, d2, "S2", A1, y1, "S1")}
    rows = []

    def report(tag, dirn, sv, f1, sd_, mc, ac):
        print(f"  {tag:<30}{dirn:<10}{np.mean(sv):>9.4f}{np.mean(f1):>13.4f}"
              f"{np.std(f1):>7.3f}{np.mean(mc):>9.4f}{np.mean(ac):>8.4f}")
        rows.append(dict(model=tag, direction=dirn, src_val=np.mean(sv),
                         tgt_f1=np.mean(f1), tgt_f1_sd=np.std(f1),
                         tgt_mcc=np.mean(mc), tgt_acc=np.mean(ac)))

    for tag, kind, T in [("MLP + median(41)  [baseline]", "base", 1),
                         ("MLP + causal HMM", "hmm", 1),
                         ("MLP + HMM + median(41)", "hmm_sm", 1),
                         ("stacked MLP T=5", "stack", 5),
                         ("stacked MLP T=11", "stack", 11),
                         ("stacked MLP T=11 + median", "stack_sm", 11),
                         ("causal TCN T=21", "tcn", 21),
                         ("causal TCN T=21 + median", "tcn_sm", 21)]:
        for dirn, (Xs, ys, ds, stag, Xt, yt, ttag) in DIRS.items():
            folds = np.array_split(np.unique(ds), 5)
            sv, f1, mc, ac = [], [], [], []
            Ss, St = SEG[stag], SEG[ttag]
            if kind in ("stack", "stack_sm", "tcn", "tcn_sm"):
                Xs_, Xt_ = make_stack(Xs, Ss, T), make_stack(Xt, St, T)
            else:
                Xs_, Xt_ = Xs, Xt
            for sd in SEEDS:
                for f in folds:
                    va = np.isin(ds, f); tr = ~va
                    tf = TRANSFORMS["zscore"](Xs_[tr])
                    Ztr, Zva, Zt = tf(Xs_[tr]), tf(Xs_[va]), tf(Xt_)
                    if kind in ("tcn", "tcn_sm"):
                        m, bv = train_tcn(Ztr, ys[tr], Zva, ys[va], d, T, seed=sd)
                        yp, pp = tcn_predict(m, Zt)
                    else:
                        m, bv = train_mlp(Ztr, ys[tr], Zva, ys[va], Xs_.shape[1],
                                          seed=sd, epochs=60)
                        yp, pp = predict(m, Zt)
                    sv.append(bv)
                    if kind == "base":
                        yh = causal_median(yp, St)
                    elif kind in ("hmm", "hmm_sm"):
                        A = fit_transitions(ys[tr], Ss[tr])
                        yh = forward_filter(pp, A, St)
                        if kind == "hmm_sm": yh = causal_median(yh, St)
                    elif kind in ("stack_sm", "tcn_sm"):
                        yh = causal_median(yp, St)
                    else:
                        yh = yp
                    mm = metrics(yt, yh)
                    f1.append(mm["macroF1"]); mc.append(mm["mcc"]); ac.append(mm["acc"])
            report(tag, dirn, sv, f1, np.std(f1), mc, ac)

    pd.DataFrame(rows).to_csv(os.path.join(R, "exp16_temporal.csv"), index=False)
    print("-> results/exp16_temporal.csv")
