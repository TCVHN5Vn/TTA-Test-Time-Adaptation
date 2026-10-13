"""Consolidated summary of every measurement, built from the results CSVs."""
import pandas as pd, numpy as np, os
R = os.path.join(os.path.dirname(__file__), "..", "results")
g = lambda f: pd.read_csv(os.path.join(R, f))
L = []
P = lambda *a: L.append(" ".join(str(x) for x in a))

P("="*104); P("SUMMARY OF ALL RESULTS".center(104)); P("="*104)

P("\n1. THE SOURCE FIT  (all-135 features, 5 seeds)  -- which defect inflated 0.991?")
d = g("exp4_windowing.csv").pivot_table(index=["causality","segmenting"],columns="protocol",values="sf_mean")
P(f"   {'windowing':<26}{'random 80/20':>14}{'blocked':>10}{'grouped-by-day':>16}")
for (c,s),r in d.iterrows():
    tag=f"{c} + {s}"+("  <-PAPER" if (c,s)==("centered","label") else "")
    P(f"   {tag:<26}{r['random']:>14.4f}{r['blocked']:>10.4f}{r['grouped']:>16.4f}")
P("   -> the random split alone causes it. Honest source macro-F1 = 0.51-0.53")

P("\n2. TARGET TRANSFER BY WINDOW  (blocked-trained)")
d = g("exp4_windowing.csv").drop_duplicates(["causality","segmenting"])
P(f"   {'windowing':<26}{'TGT macroF1':>13}{'TGT MCC':>10}{'TGT acc':>10}")
for _,r in d.iterrows():
    tag=f"{r.causality} + {r.segmenting}"+("  <-PAPER" if (r.causality,r.segmenting)==("centered","label") else "")
    P(f"   {tag:<26}{r.tf_mean:>13.4f}{r.tm_mean:>10.4f}{r.ta_mean:>10.4f}")
P("   -> causal windows IMPROVE transfer by 7.3 points and are deployable")

P("\n3. FEATURE DESIGN  (causal, GroupKFold, both directions)   chance macroF1 ~0.20")
d = g("exp11_reverse.csv")
P(f"   {'direction':<10}{'feature set':<26}{'d':>5}{'src-val':>9}{'TGT F1':>9}{'TGT MCC':>9}{'vs chance':>11}")
for _,r in d.iterrows():
    v=(r.tgt_f1-r.chance)*100
    P(f"   {r.direction:<10}{r.feature_set:<26}{r.d:>5}{r.src_val:>9.4f}{r.tgt_f1:>9.4f}{r.tgt_mcc:>9.4f}{('+' if v>0 else '')+f'{v:.1f}pp':>11}")
P("   -> per-sensor features BELOW chance both directions; paper's 135-d at/below chance both directions")

P("\n4. HAR TTA COMPARISON  (9-d causal checkpoint, 3 seeds)")
d = g("exp6_tta_global+temporal.csv").groupby(["method","stream"]).agg(
    acc=("acc","mean"),f1=("macroF1","mean"),mcc=("mcc","mean")).reset_index()
P(f"   {'method':<14}{'stream':<15}{'acc':>9}{'macroF1':>10}{'MCC':>9}   verdict")
V={"DIGA":"only reliable gain (+1.7 F1)","PETAL-fixed":"null","PeTTA":"NO-OP (0/3412 changed)",
   "AdaBN":"HARMFUL (-17 F1)","TENT":"HARMFUL (-17 F1)","Source-only":"baseline"}
for _,r in d.iterrows():
    P(f"   {r.method:<14}{r.stream:<15}{r.acc:>9.4f}{r.f1:>10.4f}{r.mcc:>9.4f}   {V.get(r.method,'')}")

P("\n5. PeTTA SILENT FREEZE  (divergence signal saturates)")
d = g("exp7_petta_noop.csv")
P(f"   {'variant':<28}{'median gamma':>14}{'median alpha_t':>16}{'changed':>10}{'macroF1':>10}")
for _,r in d.iterrows():
    P(f"   {r.variant:<28}{r.gamma_med:>14.6f}{r.alpha_med:>16.2e}{int(r.changed):>7}/3412{r.macroF1:>10.4f}")

P("\n6. SOURCE-SIDE IMPROVEMENTS  (what helped, what did not)")
d = g("exp10_improve.csv")
b = d.loc[d.macroF1.idxmax()]
P(f"   baseline 9-d causal                      macroF1=0.4133")
P(f"   + 4 extra global stats (13-d)            macroF1={d[(d.stage==1)&(d.features.str.startswith('g+t'))&(~d.class_weight)].macroF1.iloc[0]:.4f}   WORKED (+2.2pp)")
P(f"   + causal smoothing L=41                  macroF1={b.macroF1:.4f}   WORKED (+1.2pp)")
for f_ in ["g+s+t","g+s+r+t","all invariant (60-d)"]:
    v=d[(d.stage==1)&(d.features==f_)&(~d.class_weight)].macroF1.iloc[0]
    P(f"   richer invariant features: {f_:<14} macroF1={v:.4f}   FAILED")
v=d[(d.stage==1)&(d.features.str.startswith('g+t'))&(d.class_weight)].macroF1.iloc[0]
P(f"   class weighting                          macroF1={v:.4f}   FAILED")
P(f"   -> BEST: {b.macroF1:.4f} +/- {b.macroF1_sd:.3f}  (note: fold SD exceeds the gain)")

P("\n7. OCCUPANCY  (paper's protocol, 5 seeds)")
d = g("exp8_occupancy.csv").groupby("method").agg(acc=("acc","mean"),f1=("macroF1","mean"),mcc=("mcc","mean"))
PAP={"Source-only":0.587,"DIGA":0.725,"PeTTA":0.632,"PETAL-fixed":0.587}
P(f"   {'method':<14}{'acc':>9}{'macroF1':>10}{'MCC':>9}{'paper MCC':>12}")
for n in ["Source-only","AdaBN","TENT","DIGA","PETAL-fixed","PeTTA"]:
    r=d.loc[n]; p=PAP.get(n); P(f"   {n:<14}{r.acc:>9.4f}{r.f1:>10.4f}{r.mcc:>9.4f}{(f'{p:.3f}' if p else '-'):>12}")
o=g("exp9_oracle_selection.csv")
P(f"   -> our source-only MCC 0.689 is 10 pts above the paper's source-only 0.587;")
P(f"      DIGA then adds nothing (-0.8). The paper's +13.8 gain was measured against a weak source model.")
P(f"   -> DIGA grid ({len(o)//3} configs): {100*(o.d_mcc<0).mean():.0f}% HURT, best gain {o.d_mcc.max()*100:+.1f}pp (not 13.8)")

P("\n8. HEADLINE  (same protocol both sides)")
P(f"   {'':<28}{'source':>10}{'target':>10}{'gap':>10}")
P(f"   {'paper as submitted':<28}{0.991:>10.3f}{0.162:>10.3f}{82.9:>9.1f}p")
P(f"   {'corrected, S1->S2':<28}{0.5341:>10.4f}{0.4466:>10.4f}{8.8:>9.1f}p")
P(f"   {'corrected, S2->S1':<28}{0.4538:>10.4f}{0.3068:>10.4f}{14.7:>9.1f}p")
P("   -> the cross-building gap is 9-15 points, not 83")
P("\n"+"="*104)
out="\n".join(L); print(out)
open(os.path.join(R,"SUMMARY.txt"),"w").write(out+"\n")
