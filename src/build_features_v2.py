"""
Extended CAUSAL, LAYOUT-INVARIANT feature set.

Experiment 5 showed that per-sensor features are actively harmful across apartments
(0.1484 macro-F1 alone, below chance) because sensor_id_100 is a different physical
device in S1 and S2, while the 9 global+temporal features transfer at 0.4133. But 9
features is a very thin description of a 10-minute window: the whole per-sensor signal
is discarded rather than made transferable.

This builds features that use the sensor array WITHOUT binding to sensor identity. The
key device is the SORTED activation profile: sort the per-sensor window means in
descending order and keep the first k. That is invariant to any permutation of sensor
ids, so it carries "how many sensors are busy and how concentrated the activity is"
across buildings with different layouts and different numbers of sensors.

Every feature is computed on the causal window [i-W+1, i], clipped at recording-segment
boundaries (day change or a time gap != 30 s). Nothing here reads a label, and nothing
reads the target.

Groups, by prefix, so a sweep can switch them on and off:
  g_   global         per-row-sum statistics over the window
  s_   shape          sorted activation profile + concentration
  r_   run            run-length / recency structure
  m_   multiscale     the global block recomputed at shorter and longer windows
  t_   temporal       cyclic time of day and day of week
"""
import os, numpy as np, pandas as pd
from scipy.stats import entropy as scipy_entropy
from build_features import load_raw, segments, LABEL_MAP

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
W = 20
SCALES = (5, 10, 40)      # extra window lengths for the multiscale block
TOPK = 10                 # length of the sorted activation profile


def _global_block(win):
    """Statistics of the per-timestep active-sensor count, plus activation entropy."""
    rs = win.sum(1)
    fm = win.mean(0)
    p = fm + 1e-10; p = p / p.sum()
    if len(win) > 1:
        d = np.abs(np.diff(win, axis=0))
        tc, tr = float(d.sum()), float(d.mean())
    else:
        tc = tr = 0.0
    return np.array([float(rs.mean()), float(rs.max()), float(scipy_entropy(p)), tc, tr,
                     float(rs.std()), float(rs.min()), float(np.median(rs)),
                     float((rs == 0).mean())])


G_NAMES = ["mean_active", "max_active", "act_entropy", "trans_count", "trans_rate",
           "std_active", "min_active", "med_active", "frac_idle"]


def _shape_block(win):
    """Permutation-invariant description of WHICH-sensor-how-much, without the which."""
    fm = win.mean(0)
    srt = np.sort(fm)[::-1][:TOPK]
    srt = np.pad(srt, (0, max(0, TOPK - len(srt))))
    ever = float((fm > 0).sum())
    always = float((fm >= 1.0).sum())
    tot = fm.sum()
    if tot > 1e-12:                                   # Gini concentration of activity
        v = np.sort(fm); n = len(v); idx = np.arange(1, n + 1)
        gini = float((2 * idx - n - 1).dot(v) / (n * tot))
    else:
        gini = 0.0
    sd = win.std(0)
    return np.concatenate([srt, [ever, always, gini, float(sd.mean()), float(sd.max())]])


S_NAMES = ([f"sorted{i}" for i in range(TOPK)]
           + ["n_ever_active", "n_always_active", "gini", "mean_sensor_sd", "max_sensor_sd"])


def _run_block(win):
    """Run-length and recency structure of 'any sensor active'."""
    act = (win.sum(1) > 0).astype(np.int8)
    n = len(act)
    best = cur = 0
    for v in act:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    tail = 0
    for v in act[::-1]:
        if v: tail += 1
        else: break
    rs = win.sum(1)
    chg = np.nonzero(np.abs(np.diff(rs)) > 0)[0] if n > 1 else np.array([])
    since = float(n - 1 - chg[-1]) if len(chg) else float(n)
    return np.array([float(best), float(tail), since, float(len(chg)), float(n)])


R_NAMES = ["max_active_run", "tail_active_run", "steps_since_change", "n_changes", "win_len"]


def extract(df, sensors):
    S = df[sensors].values.astype(np.float32)
    tv, dv = df.time.values, df.year_day.values
    seg = segments(df, "recording")
    n = len(df)
    starts = {s: int(np.searchsorted(seg, s, "left")) for s in np.unique(seg)}
    X = []
    for i in range(n):
        lo0 = starts[seg[i]]
        win = S[max(i - W + 1, lo0): i + 1]
        f = [_global_block(win), _shape_block(win), _run_block(win)]
        for w2 in SCALES:                              # m_: same global block, other widths
            f.append(_global_block(S[max(i - w2 + 1, lo0): i + 1]))
        tn = (tv[i] % 86400) / 86400.0; dd = dv[i] % 7
        f.append(np.array([np.sin(2 * np.pi * tn), np.cos(2 * np.pi * tn),
                           np.sin(2 * np.pi * dd / 7.0), np.cos(2 * np.pi * dd / 7.0)]))
        X.append(np.concatenate(f))
    y = df.activity.map(LABEL_MAP).values
    return np.array(X, np.float32), y.astype(np.int64), dv


def names():
    nm = [f"g_{x}" for x in G_NAMES] + [f"s_{x}" for x in S_NAMES] + [f"r_{x}" for x in R_NAMES]
    for w2 in SCALES:
        nm += [f"m{w2}_{x}" for x in G_NAMES]
    return nm + ["t_time_sin", "t_time_cos", "t_day_sin", "t_day_cos"]


if __name__ == "__main__":
    s1, s2 = load_raw("S1"), load_raw("S2")
    sens = sorted(set(c for c in s1.columns if c.startswith("sensor_id_"))
                  & set(c for c in s2.columns if c.startswith("sensor_id_")),
                  key=lambda c: int(c.split("_")[-1]))
    nm = names()
    print(f"{len(sens)} shared sensors -> {len(nm)} layout-invariant causal features")
    for tag, df in (("S1", s1), ("S2", s2)):
        X, y, day = extract(df, sens)
        assert X.shape[1] == len(nm), (X.shape, len(nm))
        np.savez_compressed(os.path.join(OUT, f"feat_inv_{tag}.npz"),
                            X=X, y=y, day=day, names=np.array(nm))
        print(f"  {tag}: n={len(y)} d={X.shape[1]} classes={np.bincount(y, minlength=5)}")
    print("-> data/feat_inv_*.npz")
