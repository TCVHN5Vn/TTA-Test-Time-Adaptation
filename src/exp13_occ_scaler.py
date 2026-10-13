"""
EXPERIMENT 13 — the occupancy result, explained.

Section 4.7 of the manuscript states: "the StandardScaler is fit on unlabeled Target_train
feature statistics rather than Source_train." The occupancy notebooks do something else:

    DIGA_Occupancy.ipynb, DIGA_HAR_Occupancy.ipynb, DIGA_combined.ipynb, ... :
        scaler_tgt = StandardScaler().fit(pd.concat([tgt_train, tgt_test])[FEAT].values)

The normalization is fitted on Target_train **concatenated with Target_test** -- the set the
numbers are reported on. (One file, DIGA_Occupancy_fixed.ipynb, carries the comment
"Scaler fit on Source_train only - no target data leakage", so the issue was noticed at
some point and then not propagated.)

This script runs the identical model and method under three normalizations:

    A  source   fit on Source_train            no target information at all
    B  target   fit on Target_train            what Section 4.7 describes
    C  target+  fit on Target_train+Target_test  what the notebooks ran

and reports the gap, so the occupancy section can be rewritten with a protocol that is
stated accurately and a number that survives it.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, fit_zscore
from tta import SourceOnly, AdaBN, TENT, source_feature_stats
from tta_faithful import DIGAFaithful, PETALFaithful, PeTTAFaithful
from petal_fixed import SWAGDiag
import tta as _tta
from exp8_occupancy import m2, train_bin, OE, BS, boot

R = os.path.join(os.path.dirname(__file__), "..", "results")
_tta.K = 2


def stream(mk, Z, y):
    ad = mk(); pr = np.zeros(len(y), int)
    for i in range(0, len(Z), BS):
        p, _ = ad(torch.tensor(Z[i:i+BS], dtype=torch.float32)); pr[i:i+BS] = p
    return pr


if __name__ == "__main__":
    sp = {n: pd.read_csv(f"{OE}/{n}.csv") for n in
          ["Source_train", "Source_test", "Target_train", "Target_test"]}
    F9 = [c for c in sp["Source_train"].columns if c != "labels"]
    Xs, ys = sp["Source_train"][F9].values, sp["Source_train"].labels.values.astype(int)
    Xsv, ysv = sp["Source_test"][F9].values, sp["Source_test"].labels.values.astype(int)
    Xtt = sp["Target_train"][F9].values
    Xte, yte = sp["Target_test"][F9].values, sp["Target_test"].labels.values.astype(int)

    PROTO = {
        "A source   (Source_train)": Xs,
        "B target   (Target_train)": Xtt,
        "C target+  (Train+TEST)": np.vstack([Xtt, Xte]),
    }
    maj = m2(yte, np.zeros(len(yte), int))
    print("=" * 102)
    print("EXPERIMENT 13 — occupancy under three normalization protocols")
    print(f"  Target_test n={len(yte)} (minority {int((yte==1).sum())}). "
          f"Always-'not occupied': acc={maj['acc']:.4f} macroF1={maj['macroF1']:.4f}")
    print("=" * 102)

    rows = []
    for pname, Xfit in PROTO.items():
        tf = fit_zscore(Xfit)
        Zs, Zsv, Zte = tf(Xs), tf(Xsv), tf(Xte)
        accs = {}
        preds0 = None
        for seed in range(5):
            src, _ = train_bin(Zs, ys, Zsv, ysv, 9, seed=seed)
            st = source_feature_stats(src, Zs, ys)
            sw = MLP(9, K=2); sw.load_state_dict(src.state_dict())
            opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
            Xt_ = torch.tensor(Zs, dtype=torch.float32); yt_ = torch.tensor(ys, dtype=torch.long)
            snaps = []
            for ep in range(30):
                sw.train(); perm = torch.randperm(len(Xt_))
                for i in range(0, len(perm), 64):
                    j = perm[i:i+64]
                    if len(j) < 2: continue
                    opt.zero_grad(); lf(sw(Xt_[j]), yt_[j]).backward(); opt.step()
                snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
            swag = SWAGDiag().fit(snaps)
            for nm, mk in [("Source-only", lambda: SourceOnly(src)),
                           ("AdaBN", lambda: AdaBN(src)),
                           ("DIGA", lambda: DIGAFaithful(src, K=2)),
                           ("PETAL", lambda: PETALFaithful(src, swag)),
                           ("PeTTA", lambda: PeTTAFaithful(src, src_stats=st))]:
                pr = stream(mk, Zte, yte); mm = m2(yte, pr)
                accs.setdefault(nm, []).append([mm["acc"], mm["macroF1"], mm["mcc"]])
                if seed == 0 and nm == "Source-only": preds0 = pr
                if seed == 0: np.save(os.path.join(R, f"occ_{pname[0]}_{nm}.npy"), pr)
        print(f"\n{pname}")
        print(f"  {'method':<14}{'acc':>18}{'macro-F1':>18}{'MCC':>18}")
        for nm, v in accs.items():
            v = np.array(v)
            print(f"  {nm:<14}{v[:,0].mean():>11.4f}+/-{v[:,0].std():<5.3f}"
                  f"{v[:,1].mean():>11.4f}+/-{v[:,1].std():<5.3f}"
                  f"{v[:,2].mean():>11.4f}+/-{v[:,2].std():<5.3f}")
            rows.append(dict(protocol=pname, method=nm, acc=v[:,0].mean(),
                             macroF1=v[:,1].mean(), mcc=v[:,2].mean(),
                             mcc_sd=v[:,2].std()))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "exp13_occ_scaler.csv"), index=False)

    print("\n" + "=" * 102)
    print("What the normalization choice is worth (MCC):")
    piv = df.pivot_table(index="method", columns="protocol", values="mcc")
    print(piv.round(4).to_string())
    a, c = piv.columns[0], piv.columns[2]
    print(f"\n  leakage premium (C - A), MCC: "
          + "  ".join(f"{m}: {piv.loc[m, c]-piv.loc[m, a]:+.4f}" for m in piv.index))
    print("\n  manuscript Table 6 reported: Source-only MCC 0.587, DIGA 0.725 (+0.138)")
    print("-> results/exp13_occ_scaler.csv")
