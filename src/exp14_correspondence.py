"""
EXPERIMENT 14 — can sensors be matched across buildings without labels?

Per-sensor features carry real signal: on S1 they reach 0.62 in-domain source validation
against 0.53 for the layout-invariant set. They are discarded because index 100 is a
different physical device in each apartment, so a rule learned about it is wrong rather
than merely uninformative in the target.

The question here is whether the index can be replaced by a *behavioural* signature that
is comparable across buildings. A kitchen sensor fires around meal times, in bursts,
alongside the other kitchen sensors, and does so in both apartments. If that signature is
stable, a correspondence can be estimated from unlabelled streams alone and the
per-sensor features become transferable.

Signature per sensor (all computed from unlabelled data only):
  activity   firing rate, burst length, inter-arrival statistics
  diurnal    24-bin normalised time-of-day profile
  topology   rank-correlation with the sensor's co-activation neighbours, summarised
             in a way that does not depend on sensor identity

Matching is a balanced assignment between the two sensor sets (Hungarian algorithm on a
cosine-distance matrix over signatures). The source model is then trained on S1 with
per-sensor features, and at test time S2's columns are permuted into S1's index space
through the learned correspondence.

Controls, which decide whether any gain is real:
  identity    no permutation (what the manuscript currently does)
  learned     the estimated correspondence
  random      a random permutation, averaged over draws -- the null
  oracle      best achievable permutation using target labels, as an upper bound
              (not deployable; reported only to bound the headroom)
"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
from common import K, CLASS_NAMES, train_mlp, predict, metrics, trivial_baselines, fit_zscore
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)


# --------------------------------------------------------------- signatures
def sensor_signatures(df, sensors):
    """One behavioural vector per sensor, from unlabelled data only."""
    S = df[sensors].values.astype(np.float32)
    t = df.time.values
    hod = ((t % 86400) / 3600.0).astype(int).clip(0, 23)
    n, m = S.shape
    sig = []
    for j in range(m):
        c = S[:, j]
        rate = c.mean()
        # diurnal profile, normalised so only the shape matters
        prof = np.array([c[hod == h].sum() for h in range(24)], dtype=np.float64)
        prof = prof / prof.sum() if prof.sum() > 0 else np.ones(24) / 24
        # burst structure
        d = np.diff(np.concatenate([[0], c, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        blen = (ends - starts) if len(starts) else np.array([0])
        gaps = np.diff(starts) if len(starts) > 1 else np.array([0])
        act = np.array([
            rate,
            np.log1p(blen.mean()), np.log1p(blen.std()), np.log1p(blen.max()),
            np.log1p(np.median(gaps)) if len(gaps) else 0.0,
            len(starts) / max(n, 1),
        ])
        sig.append((prof, act, c))
    # topology: distribution of co-activation strengths, sorted so it is index-free
    C = np.zeros((m, m))
    A = S.T
    for j in range(m):
        C[j] = (A * A[j]).sum(1) / max(A[j].sum(), 1)
    topo = np.sort(C, axis=1)[:, ::-1][:, :10]           # top-10 co-activation profile
    topo = topo / np.maximum(topo[:, :1], 1e-9)
    out = []
    for j in range(m):
        prof, act, _ = sig[j]
        out.append(np.concatenate([prof, act, topo[j]]))
    V = np.array(out)
    # standardise each signature dimension within the building so the two sets are
    # comparable without ever pooling them
    V = (V - V.mean(0)) / np.maximum(V.std(0), 1e-9)
    return V


def match(V1, V2):
    """Balanced assignment of S2 sensors onto S1 sensors by signature similarity."""
    A = V1 / np.maximum(np.linalg.norm(V1, axis=1, keepdims=True), 1e-9)
    B = V2 / np.maximum(np.linalg.norm(V2, axis=1, keepdims=True), 1e-9)
    cost = 1.0 - A @ B.T
    r, c = linear_sum_assignment(cost)
    perm = np.empty(len(V1), dtype=int)
    perm[r] = c
    return perm, cost


# --------------------------------------------------------- per-sensor features
def per_sensor_features(df, sensors, W=20):
    """Causal per-sensor window mean and std, recording-segmented."""
    S = df[sensors].values.astype(np.float32)
    seg = segments(df, "recording")
    starts = {s: int(np.searchsorted(seg, s, "left")) for s in np.unique(seg)}
    out = []
    for i in range(len(df)):
        w = S[max(i - W + 1, starts[seg[i]]): i + 1]
        out.append(np.concatenate([w.mean(0), w.std(0)]))
    y = df.activity.map({15.0: 0, 60.0: 1, 65.0: 2, 70.0: 3, 85.0: 4}).values
    return np.array(out, np.float32), y.astype(np.int64), df.year_day.values


def apply_perm(X, perm, m):
    """Reorder a [mean | std] block so target column j lands in source slot perm^-1."""
    inv = np.empty(m, int); inv[perm] = np.arange(m)
    return np.concatenate([X[:, :m][:, perm], X[:, m:][:, perm]], axis=1)


if __name__ == "__main__":
    s1, s2 = load_raw("S1"), load_raw("S2")
    sens = sorted(set(c for c in s1.columns if c.startswith("sensor_id_"))
                  & set(c for c in s2.columns if c.startswith("sensor_id_")),
                  key=lambda c: int(c.split("_")[-1]))
    m = len(sens)
    X1, y1, d1 = per_sensor_features(s1, sens)
    X2, y2, d2 = per_sensor_features(s2, sens)
    tb = trivial_baselines(y2)

    V1, V2 = sensor_signatures(s1, sens), sensor_signatures(s2, sens)
    perm, cost = match(V1, V2)
    fixed = int((perm == np.arange(m)).sum())

    print("=" * 96)
    print("EXPERIMENT 14 — unsupervised sensor correspondence")
    print(f"  {m} shared sensors   signature dim {V1.shape[1]}")
    print(f"  learned assignment leaves {fixed}/{m} sensors at their own index")
    print(f"  mean matched cost {cost[np.arange(m), perm].mean():.4f} "
          f"vs mean off-diagonal {cost.mean():.4f}")
    print(f"  TARGET chance: stratified macroF1={tb['strat_macroF1']:.4f}")
    print("=" * 96)

    days = np.unique(d1); folds = np.array_split(days, 5)

    def run(tag, permutation=None, n_rand=0, oracle=False):
        f1s, mccs, accs, svs = [], [], [], []
        for sd in SEEDS:
            for f in folds:
                va = np.isin(d1, f); tr = ~va
                tf = fit_zscore(X1[tr])
                mdl, bv = train_mlp(tf(X1[tr]), y1[tr], tf(X1[va]), y1[va],
                                    X1.shape[1], seed=sd, epochs=60)
                svs.append(bv)
                if n_rand:                       # null: average over random permutations
                    rng = np.random.default_rng(sd)
                    acc = []
                    for _ in range(n_rand):
                        p = rng.permutation(m)
                        yp, _ = predict(mdl, tf(apply_perm(X2, p, m)))
                        acc.append([metrics(y2, yp)[k] for k in ("macroF1", "mcc", "acc")])
                    v = np.mean(acc, 0)
                elif oracle:                     # upper bound, uses target labels
                    best = None
                    rng = np.random.default_rng(sd)
                    p = np.arange(m)
                    cur = metrics(y2, predict(mdl, tf(apply_perm(X2, p, m)))[0])["macroF1"]
                    for _ in range(400):         # greedy pairwise swaps
                        i, j = rng.integers(0, m, 2)
                        q = p.copy(); q[[i, j]] = q[[j, i]]
                        s = metrics(y2, predict(mdl, tf(apply_perm(X2, q, m)))[0])["macroF1"]
                        if s > cur: p, cur = q, s
                    yp, _ = predict(mdl, tf(apply_perm(X2, p, m)))
                    mm = metrics(y2, yp); v = [mm["macroF1"], mm["mcc"], mm["acc"]]
                else:
                    p = permutation if permutation is not None else np.arange(m)
                    yp, _ = predict(mdl, tf(apply_perm(X2, p, m)))
                    mm = metrics(y2, yp); v = [mm["macroF1"], mm["mcc"], mm["acc"]]
                f1s.append(v[0]); mccs.append(v[1]); accs.append(v[2])
        print(f"  {tag:<34}{np.mean(svs):>10.4f}{np.mean(f1s):>12.4f}"
              f"{np.std(f1s):>8.3f}{np.mean(mccs):>10.4f}{np.mean(accs):>9.4f}")
        return dict(variant=tag, src_val=np.mean(svs), tgt_f1=np.mean(f1s),
                    tgt_f1_sd=np.std(f1s), tgt_mcc=np.mean(mccs), tgt_acc=np.mean(accs))

    print(f"  {'variant':<34}{'src-val':>10}{'TGT macroF1':>12}{'sd':>8}{'MCC':>10}{'acc':>9}")
    print("  " + "-" * 81)
    rows = [run("identity (no matching)"),
            run("learned correspondence", permutation=perm),
            run("random permutation (null)", n_rand=5),
            run("oracle permutation (bound)", oracle=True)]
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp14_correspondence.csv"), index=False)
    print(f"\n  reference: layout-invariant 13-d reaches 0.4466 target macro-F1")
    print("-> results/exp14_correspondence.csv")
