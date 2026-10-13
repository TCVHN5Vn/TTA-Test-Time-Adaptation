"""
Rebuild the WSU engineered features from the RAW csvs, with the windowing policy
as an explicit switch, so each defect can be turned off one at a time.

Two independent policy axes:

  causality  : "centered" -> window is [i-W/2, i+W/2]   (the paper; sees the future)
               "causal"   -> window is [i-W+1, i]       (deployable online)

  segmenting : "label"    -> segment = maximal run of a constant ACTIVITY LABEL
                            (the paper, via get_blocks(); uses ground truth at
                             feature-construction time -> oracle segmentation)
               "recording"-> segment = maximal run of contiguous samples
                            (same year_day, consecutive 30 s steps). Label-free,
                            which is all a deployed system can actually know.

Feature definitions are copied verbatim from feature_engineering_wsu.ipynb so that
the only thing that changes between variants is the window, never the arithmetic.

Windows are truncated at the segment start rather than dropped, which is what an
online system genuinely has after a recording gap. --require_full drops them instead.

Outputs data/feat_{causality}_{segmenting}.npz with X, y, day, row (original row id
in day/time-sorted order) and the feature-name list.
"""
import os, sys, argparse, pickle
import numpy as np, pandas as pd
from scipy.stats import entropy as scipy_entropy

RAW = "/Users/mariemkbaier/Desktop/dataset"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
LABEL_MAP = {15.0: 0, 60.0: 1, 65.0: 2, 70.0: 3, 85.0: 4}
W = 20


def load_raw(name):
    d = pd.read_csv(f"{RAW}/Copie de Final_Data_{name}_AR (1).csv")
    return d.sort_values(["year_day", "time"]).reset_index(drop=True)


def segments(df, mode):
    """Integer segment id per row."""
    if mode == "label":
        # verbatim get_blocks(): runs of constant label, NOT reset at recording gaps
        return (df.activity != df.activity.shift()).cumsum().values
    if mode == "recording":
        new = (df.year_day != df.year_day.shift()) | (df.time.diff() != 30)
        return new.cumsum().values
    raise ValueError(mode)


def top_pairs(df, sensors, k=20):
    """Top-k co-active sensor pairs, counted on the SOURCE only (as in the notebook)."""
    a = df[sensors].values.astype(np.float32)
    cnt = {}
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            c = int(((a[:, i] == 1) & (a[:, j] == 1)).sum())
            if c > 0:
                cnt[(i, j)] = c
    return sorted(cnt, key=cnt.get, reverse=True)[:k]


def extract(df, sensors, pairs, causality, segmenting, require_full=False):
    S = df[sensors].values.astype(np.float32)
    tv, dv = df.time.values, df.year_day.values
    seg = segments(df, segmenting)
    n = len(df)
    half = W // 2
    X, keep = [], []
    for i in range(n):
        s = seg[i]
        if causality == "centered":
            lo, hi = i - half, i + half + 1
        else:                                   # causal
            lo, hi = i - W + 1, i + 1
        # clip to the segment the row belongs to
        lo = max(lo, int(np.searchsorted(seg, s, "left")))
        hi = min(hi, int(np.searchsorted(seg, s, "right")))
        if require_full and (hi - lo) < W:
            continue
        win = S[lo:hi]

        fm, fs = win.mean(0), win.std(0)
        rs = win.sum(1)
        p = fm + 1e-10
        p = p / p.sum()
        g = np.array([float(rs.mean()), float(rs.max()), float(scipy_entropy(p))])
        if len(win) > 1:
            dfs = np.abs(np.diff(win, axis=0))
            tr = np.array([float(dfs.sum()), float(dfs.mean())])
        else:
            tr = np.zeros(2)
        pr = np.array([float(((win[:, a] == 1) & (win[:, b] == 1)).sum()) for a, b in pairs])
        tn = (tv[i] % 86400) / 86400.0
        dd = dv[i] % 7
        te = np.array([np.sin(2 * np.pi * tn), np.cos(2 * np.pi * tn),
                       np.sin(2 * np.pi * dd / 7.0), np.cos(2 * np.pi * dd / 7.0)])
        X.append(np.concatenate([fm, fs, g, tr, pr, te]))
        keep.append(i)
    keep = np.array(keep)
    y = df.activity.map(LABEL_MAP).values[keep]
    return np.array(X, dtype=np.float32), y.astype(np.int64), dv[keep], keep


def names(sensors, pairs):
    return ([f"mean_{s}" for s in sensors] + [f"std_{s}" for s in sensors]
            + ["global_mean_active", "global_max_active", "activation_entropy"]
            + ["transition_count", "transition_rate"]
            + [f"pair_{a}_{b}" for a, b in pairs]
            + ["time_sin", "time_cos", "day_sin", "day_cos"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--require_full", action="store_true")
    a = ap.parse_args()

    s1, s2 = load_raw("S1"), load_raw("S2")
    sens = sorted(set(c for c in s1.columns if c.startswith("sensor_id_"))
                  & set(c for c in s2.columns if c.startswith("sensor_id_")),
                  key=lambda c: int(c.split("_")[-1]))
    pairs = top_pairs(s1, sens)
    fn = names(sens, pairs)
    print(f"{len(sens)} shared sensors -> {len(fn)} features\n")
    os.makedirs(OUT, exist_ok=True)
    tag = "_full" if a.require_full else ""
    for cz in ("centered", "causal"):
        for sg in ("label", "recording"):
            for nm, df in (("S1", s1), ("S2", s2)):
                X, y, day, row = extract(df, sens, pairs, cz, sg, a.require_full)
                p = os.path.join(OUT, f"feat_{cz}_{sg}{tag}_{nm}.npz")
                np.savez_compressed(p, X=X, y=y, day=day, row=row, names=np.array(fn))
                print(f"{cz:<9}{sg:<10}{nm}  n={len(y):5d}  d={X.shape[1]}  "
                      f"classes={np.bincount(y, minlength=5)}")
    print("\n-> data/feat_*.npz")
