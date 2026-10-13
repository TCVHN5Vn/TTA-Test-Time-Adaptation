"""
Generates the data-driven figures for the manuscript as vector PDFs.

Figure 2  PCA projection of the engineered feature space, S1 vs S2 (as in the previous
          submission, but on the causal recording-segmented features actually used).
Figure 3  The leakage decomposition: source validation macro-F1 by windowing policy and
          validation protocol. This is the paper's headline and deserves a figure.
Figure 7  Feature-set transfer in both directions against the chance baseline.
Figure 8  PeTTA divergence saturation: the distribution of the Mahalanobis statistic and
          where 1 - exp(-d) lands.

Style follows the previous submission: no chartjunk, greyscale-safe markers, serif labels
so the figures sit comfortably next to the body text.
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from common import CLASS_NAMES, TRANSFORMS, family

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
GREY = "#555555"
PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load(var, tag):
    z = np.load(os.path.join(D, f"feat_{var}_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], [str(s) for s in z["names"]]


# ---------------------------------------------------------------- Figure 2: PCA
def fig_pca():
    X1, y1, cols = load("causal_recording", "S1")
    X2, y2, _ = load("causal_recording", "S2")
    mu, sd = X1.mean(0), X1.std(0); sd[sd < 1e-12] = 1.0
    Z1, Z2 = (X1 - mu) / sd, (X2 - mu) / sd
    U, S, Vt = np.linalg.svd(Z1 - Z1.mean(0), full_matrices=False)
    P1, P2 = (Z1 - Z1.mean(0)) @ Vt[:2].T, (Z2 - Z1.mean(0)) @ Vt[:2].T
    ev = (S ** 2 / (S ** 2).sum())[:2] * 100

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True, sharey=True)
    for a, (P, y, t) in zip(ax, [(P1, y1, "S1 (source)"), (P2, y2, "S2 (target)")]):
        for k in range(5):
            m = y == k
            a.scatter(P[m, 0], P[m, 1], s=3, alpha=0.45, c=PAL[k],
                      label=CLASS_NAMES[k], linewidths=0, rasterized=True)
        a.set_title(t); a.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax[0].set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    h = [Line2D([], [], marker="o", ls="", ms=4, c=PAL[k], label=CLASS_NAMES[k])
         for k in range(5)]
    ax[1].legend(handles=h, loc="upper right", frameon=False, handletextpad=0.2,
                 borderpad=0.1, labelspacing=0.25)
    fig.savefig(os.path.join(FIG, "fig_pca.pdf")); plt.close(fig)
    print("  fig_pca.pdf")


# ------------------------------------------------- Figure 3: leakage decomposition
def fig_leakage():
    d = pd.read_csv(os.path.join(R, "exp4_windowing.csv"))
    lab = {("centered", "label"): "centred\n+ oracle seg.",
           ("centered", "recording"): "centred\n+ recording",
           ("causal", "label"): "causal\n+ oracle seg.",
           ("causal", "recording"): "causal\n+ recording"}
    order = [("centered", "label"), ("centered", "recording"),
             ("causal", "label"), ("causal", "recording")]
    protos = ["random", "blocked", "grouped"]
    pname = {"random": "random 80/20", "blocked": "blocked", "grouped": "grouped by day"}
    hatch = {"random": "//", "blocked": "", "grouped": ""}
    col = {"random": "#C44E52", "blocked": "#BBBBBB", "grouped": "#4C72B0"}

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w, x = 0.26, np.arange(len(order))
    for i, p in enumerate(protos):
        v = [d[(d.causality == c) & (d.segmenting == s) & (d.protocol == p)].sf_mean.iloc[0]
             for c, s in order]
        e = [d[(d.causality == c) & (d.segmenting == s) & (d.protocol == p)].sf_sd.iloc[0]
             for c, s in order]
        ax.bar(x + (i - 1) * w, v, w, yerr=e, capsize=2, label=pname[p],
               color=col[p], hatch=hatch[p], edgecolor="white", linewidth=0.6)
    ax.axhline(0.5287, ls=":", lw=0.9, c=GREY)
    ax.annotate("honest source fit (0.53)", xy=(3.42, 0.5287), xytext=(3.42, 0.72),
                fontsize=7.5, c=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels([lab[o] for o in order])
    ax.set_ylabel("source validation macro-F$_1$"); ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(os.path.join(FIG, "fig_leakage.pdf")); plt.close(fig)
    print("  fig_leakage.pdf")


# ------------------------------------------------- Figure 7: feature sets, both ways
def fig_features():
    d = pd.read_csv(os.path.join(R, "exp11_reverse.csv"))
    short = {"13-d g+t (transferable)": "layout-invariant (13)",
             "all 135-d (paper)": "conventional (135)",
             "per-sensor ids only": "per-sensor only (106)",
             "no per-sensor (29-d)": "no per-sensor (29)"}
    order = ["layout-invariant (13)", "no per-sensor (29)",
             "conventional (135)", "per-sensor only (106)"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for a, dirn in zip(ax, ["S1->S2", "S2->S1"]):
        g = d[d.direction == dirn].copy()
        g["s"] = g.feature_set.map(short)
        g = g.set_index("s").loc[order]
        y = np.arange(len(order))[::-1]
        c = ["#4C72B0" if v > ch else "#C44E52"
             for v, ch in zip(g.tgt_f1, g.chance)]
        a.barh(y, g.tgt_f1, 0.6, xerr=g.tgt_f1_sd, capsize=2, color=c,
               edgecolor="white", linewidth=0.6)
        a.axvline(g.chance.iloc[0], ls="--", lw=1.0, c=GREY)
        a.set_yticks(y); a.set_yticklabels(order)
        a.set_xlabel("target macro-F$_1$")
        a.set_title(dirn.replace("->", r" $\rightarrow$ "))
        a.set_xlim(0, 0.56)
    ax[1].tick_params(labelleft=False)          # sharey: blank only the right panel
    fig.savefig(os.path.join(FIG, "fig_features.pdf")); plt.close(fig)
    print("  fig_features.pdf")


# ---------------------------------------------- Figure 8: PeTTA saturation
def fig_saturation():
    d = np.logspace(-2, 3.2, 400)
    g = 1 - np.exp(-d)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.semilogx(d, g, lw=1.6, c="#4C72B0")
    # both markers terminate on their own vertical line, low on the axis, so neither
    # implies a value read off the curve
    ax.axvline(587, ls="--", lw=1.0, c="#C44E52")
    ax.annotate("as published\n(sum over 64 dims)\nmedian $d=587$",
                xy=(587, 0.30), xytext=(0.013, 0.60), fontsize=7, c="#C44E52",
                ha="left", arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8,
                connectionstyle="arc3,rad=-0.18"))
    ax.axvline(9.16, ls=":", lw=1.0, c=GREY)
    ax.annotate("per-dimension\nmedian $d=9.2$",
                xy=(9.16, 0.30), xytext=(0.013, 0.18), fontsize=7, c=GREY,
                ha="left", arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8,
                connectionstyle="arc3,rad=0.12"))
    ax.axhline(1.0, ls="-", lw=0.6, c="#999999", zorder=0)
    ax.set_xlabel("Mahalanobis statistic $d$")
    ax.set_ylabel(r"$\gamma = 1-e^{-d}$")
    ax.set_ylim(0, 1.06)
    fig.savefig(os.path.join(FIG, "fig_saturation.pdf")); plt.close(fig)
    print("  fig_saturation.pdf")


if __name__ == "__main__":
    print("writing figures to paper/figures/")
    fig_pca(); fig_leakage(); fig_features(); fig_saturation()
    print("done")
