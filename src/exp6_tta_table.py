"""
EXPERIMENT 6 — the replacement for Table 4 and Table 5.

What is different from the submitted version:
  * source checkpoint   causal, recording-segmented features (deployable online), with the
                        feature set chosen by EXPERIMENT 5 -- not the 135-d vector
  * source selection    GroupKFold over year_day, so no window overlaps the split
  * methods             AdaBN and TENT are included as the normalization-only and
                        entropy-only baselines. Omitting them made DIGA/PeTTA look like
                        the only options; AdaBN in particular was competitive in the
                        submitted text but kept out of the table.
  * PETAL               corrected objective (petal_fixed.py). The submitted PETAL could
                        not change a prediction, so "no measurable gain" was not a result.
  * baselines           majority-class and stratified-random are reported IN the table.
                        Several submitted numbers sit below stratified-random macro-F1.
  * uncertainty         paired bootstrap over the target stream, 2000 resamples, giving a
                        95% CI on each method's DIFFERENCE from source-only. Point
                        estimates alone cannot support "DIGA improves macro-F1" when the
                        gap is 1.6 points.

Streams follow Sec 4.3: the target is cut into contiguous segments; non-recurring visits
each once, recurring makes three passes, and each recurring run is repeated over three
independently randomized segment orderings.
"""
import os, sys, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (K, CLASS_NAMES, MLP, train_mlp, predict, metrics, trivial_baselines,
                    TRANSFORMS, family)
from tta import (SourceOnly, AdaBN, TENT, DIGA, PeTTA, PETALAdapter,
                 source_feature_stats)
from petal_fixed import SWAGDiag, PETALFixed

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS = 64


def load(var, nm):
    z = np.load(os.path.join(D, f"feat_{var}_{nm}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def make_segments(n, n_seg=8):
    return np.array_split(np.arange(n), n_seg)


def stream_order(segs, mode, rng, passes=3):
    if mode == "non-recurring":
        return list(segs)
    out = []
    for _ in range(passes):
        o = rng.permutation(len(segs))
        out += [segs[i] for i in o]
    return out


def run_stream(make_adapter, Z2, y2, segs, mode, rng, device="cpu"):
    """Returns predictions aligned to the FINAL visit of each target sample."""
    ad = make_adapter()
    pred = np.full(len(y2), -1, dtype=np.int64)
    for seg in stream_order(segs, mode, rng):
        for i in range(0, len(seg), BS):
            idx = seg[i:i + BS]
            x = torch.tensor(Z2[idx], dtype=torch.float32, device=device)
            p, _ = ad(x)
            pred[idx] = p
    return pred


def paired_bootstrap(y, pa, pb, fn, n=2000, seed=0):
    """95% CI on fn(b) - fn(a), resampling target samples (paired)."""
    rng = np.random.default_rng(seed); N = len(y); d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(fn(y[i], pb[i]) - fn(y[i], pa[i]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="causal_recording")
    ap.add_argument("--featset", default="global+temporal",
                    help="global+temporal | all135 | no_per_sensor")
    ap.add_argument("--transform", default="rank_gauss")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    X1, y1, d1, cols = load(a.variant, "S1")
    X2, y2, d2, _ = load(a.variant, "S2")
    fam = np.array([family(c) for c in cols])
    mask = {"global+temporal": np.isin(fam, ["global", "temporal"]),
            "no_per_sensor": fam != "per_sensor",
            "all135": np.ones(len(cols), bool)}[a.featset]
    dsel = int(mask.sum())

    tb = trivial_baselines(y2)
    print("=" * 104)
    print(f"EXPERIMENT 6 — target-stream TTA comparison   [{a.variant}, {a.featset} ({dsel}-d), {a.transform}]")
    print(f"  n_S1={len(y1)}  n_S2={len(y2)}   source selection: GroupKFold over {len(np.unique(d1))} days")
    print(f"  TARGET BASELINES  majority: acc={tb['majority_acc']:.4f} macroF1={tb['majority_macroF1']:.4f}"
          f"   stratified-random: acc={tb['strat_acc']:.4f} macroF1={tb['strat_macroF1']:.4f}")
    print("=" * 104)

    rows = []
    for seed in range(a.seeds):
        # ---- honest source checkpoint: hold out the last fold of days for selection
        days = np.unique(d1); va_days = np.array_split(days, 5)[-1]
        va = np.isin(d1, va_days); tr = ~va
        tf = TRANSFORMS[a.transform](X1[tr][:, mask])
        Z1tr, Z1va, Z2 = tf(X1[tr][:, mask]), tf(X1[va][:, mask]), tf(X2[:, mask])
        src, svf1 = train_mlp(Z1tr, y1[tr], Z1va, y1[va], dsel, seed=seed, epochs=60)
        sstats = source_feature_stats(src, Z1tr, y1[tr])

        # SWAG posterior for PETAL, collected at a constant LR so it is not degenerate
        import torch.nn as nn
        sw = MLP(dsel); sw.load_state_dict(src.state_dict())
        opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Z1tr, dtype=torch.float32); yt = torch.tensor(y1[tr], dtype=torch.long)
        snaps = []
        for ep in range(30):
            sw.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 128):
                j = perm[i:i + 128]
                if len(j) < 2: continue
                opt.zero_grad(); lf(sw(Xt[j]), yt[j]).backward(); opt.step()
            snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
        swag = SWAGDiag().fit(snaps)

        segs = make_segments(len(y2))
        METHODS = [
            ("Source-only", "non-recurring", lambda: SourceOnly(src)),
            ("AdaBN",       "non-recurring", lambda: AdaBN(src)),
            ("TENT",        "non-recurring", lambda: TENT(src)),
            ("DIGA",        "non-recurring", lambda: DIGA(src)),
            ("PETAL-fixed", "non-recurring", lambda: PETALAdapter(PETALFixed(src, swag, strict=False))),
            ("PeTTA",       "recurring",     lambda: PeTTA(src, src_stats=sstats)),
            ("DIGA",        "recurring",     lambda: DIGA(src)),
            ("PETAL-fixed", "recurring",     lambda: PETALAdapter(PETALFixed(src, swag, strict=False))),
            ("Source-only", "recurring",     lambda: SourceOnly(src)),
        ]
        for name, mode, mk in METHODS:
            orders = range(3) if mode == "recurring" else range(1)
            for o in orders:
                rng = np.random.default_rng(1000 * seed + o)
                pred = run_stream(mk, Z2, y2, segs, mode, rng)
                mm = metrics(y2, pred)
                rows.append(dict(seed=seed, method=name, stream=mode, order=o,
                                 src_val_macroF1=svf1, acc=mm["acc"], macroF1=mm["macroF1"],
                                 mcc=mm["mcc"], bal=mm["bal"], n_pred_classes=mm["n_pred_classes"],
                                 **{f"f1_{CLASS_NAMES[k]}": mm["f1"][k] for k in range(K)}))
                np.save(os.path.join(R, f"pred_{name}_{mode}_{seed}_{o}.npy"), pred)
            print(f"  seed{seed} {name:<12}{mode:<14} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, f"exp6_tta_{a.featset}.csv"), index=False)

    print("\n" + "=" * 104)
    g = df.groupby(["method", "stream"]).agg(
        acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("macroF1", "mean"),
        f1_sd=("macroF1", "std"), mcc=("mcc", "mean"), mcc_sd=("mcc", "std")).reset_index()
    print(f"{'method':<14}{'stream':<15}{'acc':>16}{'macro-F1':>17}{'MCC':>17}")
    for _, r in g.iterrows():
        print(f"{r.method:<14}{r.stream:<15}{r.acc:>9.4f}+/-{r.acc_sd:<5.3f}"
              f"{r.f1:>10.4f}+/-{r.f1_sd:<5.3f}{r.mcc:>10.4f}+/-{r.mcc_sd:<5.3f}")

    # paired bootstrap vs source-only, on seed 0 / order 0
    print("\n95% paired-bootstrap CI on the DIFFERENCE from source-only (same stream, seed 0):")
    f1fn = lambda y, p: metrics(y, p)["macroF1"]
    mcfn = lambda y, p: metrics(y, p)["mcc"]
    for mode in ("non-recurring", "recurring"):
        base = os.path.join(R, f"pred_Source-only_{mode}_0_0.npy")
        if not os.path.exists(base): continue
        pa = np.load(base)
        for name in ("AdaBN", "TENT", "DIGA", "PETAL-fixed", "PeTTA"):
            f = os.path.join(R, f"pred_{name}_{mode}_0_0.npy")
            if not os.path.exists(f): continue
            pb = np.load(f)
            lo1, hi1 = paired_bootstrap(y2, pa, pb, f1fn)
            lo2, hi2 = paired_bootstrap(y2, pa, pb, mcfn)
            sig = "" if (lo1 < 0 < hi1) else "  *"
            print(f"  {mode:<14}{name:<13} dMacroF1 [{lo1:+.4f},{hi1:+.4f}]"
                  f"   dMCC [{lo2:+.4f},{hi2:+.4f}]{sig}")
    print(f"\n-> results/exp6_tta_{a.featset}.csv")
