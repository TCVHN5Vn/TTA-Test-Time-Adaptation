"""
EXPERIMENT 17 — occupancy in both transfer directions.

The HAR results are reported for S1->S2 and S2->S1, because a single direction cannot
distinguish a property of the transfer from a property of the weaker source. Occupancy was
reported in one direction only. This runs the reverse: train on the target office, test on
the source office, under the same three normalisation protocols.

Forward : train Source_train, select on Source_test, evaluate on Target_test.
Reverse : train Target_train, select on Target_test, evaluate on Source_test.

Both report the trivial baseline for their own evaluation set, since the two offices have
different class balance.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, fit_zscore
from tta import SourceOnly, AdaBN, source_feature_stats
from tta_faithful import DIGAFaithful, PETALFaithful, PeTTAFaithful
from petal_fixed import SWAGDiag
import tta as _tta
from exp8_occupancy import m2, train_bin, OE, BS
_tta.K = 2
R = os.path.join(os.path.dirname(__file__), "..", "results")


def stream(mk, Z, y):
    ad = mk(); pr = np.zeros(len(y), int)
    for i in range(0, len(Z), BS):
        p, _ = ad(torch.tensor(Z[i:i+BS], dtype=torch.float32)); pr[i:i+BS] = p
    return pr


if __name__ == "__main__":
    sp = {n: pd.read_csv(f"{OE}/{n}.csv") for n in
          ["Source_train", "Source_test", "Target_train", "Target_test"]}
    F = [c for c in sp["Source_train"].columns if c != "labels"]
    g = lambda n: (sp[n][F].values, sp[n].labels.values.astype(int))
    Xs, ys = g("Source_train"); Xsv, ysv = g("Source_test")
    Xt, yt = g("Target_train"); Xtv, ytv = g("Target_test")

    DIRS = {
        "source office -> target office": (Xs, ys, Xsv, ysv, Xt, Xtv, ytv),
        "target office -> source office": (Xt, yt, Xtv, ytv, Xs, Xsv, ysv),
    }
    print("=" * 100)
    print("EXPERIMENT 17 — occupancy, both transfer directions")
    print("=" * 100)
    rows = []
    for dirn, (Xtr, ytr, Xva, yva, Xadapt, Xev, yev) in DIRS.items():
        maj = m2(yev, np.zeros(len(yev), int))
        print(f"\n{dirn}   (eval n={len(yev)}, minority {int((yev==1).sum())}; "
              f"always-'not occupied' MCC {maj['mcc']:.3f}, acc {maj['acc']:.3f})")
        for pname, Xfit in [("A fit on source-of-this-direction", Xtr),
                            ("B fit on adaptation split", Xadapt),
                            ("C fit on adaptation + eval", np.vstack([Xadapt, Xev]))]:
            tf = fit_zscore(Xfit)
            Ztr, Zva, Zev = tf(Xtr), tf(Xva), tf(Xev)
            acc = {}
            for seed in range(5):
                src, _ = train_bin(Ztr, ytr, Zva, yva, 9, seed=seed)
                st = source_feature_stats(src, Ztr, ytr)
                sw = MLP(9, K=2); sw.load_state_dict(src.state_dict())
                opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
                Xq = torch.tensor(Ztr, dtype=torch.float32); yq = torch.tensor(ytr, dtype=torch.long)
                snaps = []
                for ep in range(30):
                    sw.train(); perm = torch.randperm(len(Xq))
                    for i in range(0, len(perm), 64):
                        j = perm[i:i+64]
                        if len(j) < 2: continue
                        opt.zero_grad(); lf(sw(Xq[j]), yq[j]).backward(); opt.step()
                    snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
                swag = SWAGDiag().fit(snaps)
                for nm, mk in [("Source-only", lambda: SourceOnly(src)),
                               ("AdaBN", lambda: AdaBN(src)),
                               ("DIGA", lambda: DIGAFaithful(src, K=2)),
                               ("PETAL", lambda: PETALFaithful(src, swag)),
                               ("PeTTA", lambda: PeTTAFaithful(src, src_stats=st))]:
                    mm = m2(yev, stream(mk, Zev, yev))
                    acc.setdefault(nm, []).append([mm["acc"], mm["macroF1"], mm["mcc"]])
            print(f"  {pname:<34}{'acc':>9}{'macroF1':>10}{'MCC':>10}")
            for nm, v in acc.items():
                v = np.array(v)
                print(f"    {nm:<32}{v[:,0].mean():>9.4f}{v[:,1].mean():>10.4f}"
                      f"{v[:,2].mean():>10.4f}")
                rows.append(dict(direction=dirn, protocol=pname, method=nm,
                                 acc=v[:,0].mean(), macroF1=v[:,1].mean(),
                                 mcc=v[:,2].mean(), mcc_sd=v[:,2].std()))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp17_occ_reverse.csv"), index=False)
    print("\n-> results/exp17_occ_reverse.csv")
