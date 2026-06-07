# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  generate_metric_figures.py
#  Stage: 6 - Visualization / appendix
#
#  PURPOSE
#    Build two figures for the V13 model from its threshold-sweep table.
#    Figure 1 draws ROC and PR curves with the reported AUROC / AUPRC and an
#    operating point. Figure 2 draws Accuracy, F1+, F1-, and Macro F1 across
#    thresholds, marking the Macro F1 peak and the naive-accuracy ceiling.
#
#  INPUTS
#    /PATH/TO/PROJECT/technical/Results/67_table_V13_Threshold_Sweep.csv
#  OUTPUTS
#    /PATH/TO/PROJECT/fig1_roc_pr.png
#    /PATH/TO/PROJECT/fig1_roc_pr.pdf
#    /PATH/TO/PROJECT/fig2_threshold_metrics.png
#    /PATH/TO/PROJECT/fig2_threshold_metrics.pdf
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    SWEEP_CSV    -  path to the V13 threshold-sweep CSV
#    OUT_DIR      -  output directory for the four figure files
#    AUROC_V13    -  reported V13 AUROC (0.7710)
#    AUPRC_V13    -  reported V13 AUPRC (0.0758)
#    P_TOTAL      -  total positive (sepsis) count (12,778), drives prevalence
#
#  REQUIRES: numpy, pandas, matplotlib
# ============================================================================

"""
generate_metric_figures.py  (v2 — clean, fixed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# EDIT: path to the V13 threshold-sweep CSV
SWEEP_CSV = (
    "/PATH/TO/PROJECT/"
    "technical/Results/67_table_V13_Threshold_Sweep.csv"
)
# EDIT: output directory for the four figure files
OUT_DIR = "/PATH/TO/PROJECT/"

AUROC_V13  = 0.7710   # EDIT: reported V13 AUROC (hardcoded)
AUPRC_V13  = 0.0758   # EDIT: reported V13 AUPRC (hardcoded)
P_TOTAL    = 12_778   # EDIT: total positive (sepsis) count, drives prevalence

BLUE  = "#0072B2"
GREEN = "#009E73"
RED   = "#D55E00"
LBLUE = "#56B4E9"
GRAY  = "#757575"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   10.5,
    "axes.titleweight": "normal",
    "axes.labelsize":   10,
    "legend.fontsize":  8.5,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.22,
    "grid.linestyle":   ":",
    "grid.color":       "0.6",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "legend.framealpha":0.90,
    "legend.edgecolor": "0.82",
})

# ── load ──────────────────────────────────────────────────────────────────────
df_raw  = pd.read_csv(SWEEP_CSV)
N_TOTAL = round(df_raw["fp"].iloc[0] / df_raw["fpr"].iloc[0])
PREV    = P_TOTAL / (P_TOTAL + N_TOTAL)

df = df_raw[df_raw["threshold"] <= 0.63].copy().sort_values("threshold").reset_index(drop=True)

df["precision"] = df["tp"] / (df["tp"] + df["fp"])
df["recall"]    = df["sensitivity"]
df["tn"]        = N_TOTAL - df["fp"]
df["accuracy"]  = (df["tp"] + df["tn"]) / (P_TOTAL + N_TOTAL)
df["prec_neg"]  = df["tn"] / (df["tn"] + df["fn"])
df["spec"]      = 1.0 - df["fpr"]

dp = df["precision"] + df["recall"]
dn = df["prec_neg"]  + df["spec"]
df["f1_pos"]   = np.where(dp > 0, 2*df["precision"]*df["recall"]/dp, 0.0)
df["f1_neg"]   = np.where(dn > 0, 2*df["prec_neg"] *df["spec"]   /dn, 0.0)
df["macro_f1"] = (df["f1_pos"] + df["f1_neg"]) / 2.0

best  = df.loc[df["macro_f1"].idxmax()]
op    = df.sort_values("threshold").iloc[0]  # lowest threshold = most aggressive

# ── FIGURE 1 — ROC and PR ─────────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
fig1.subplots_adjust(wspace=0.34)

# ─ ROC ─
roc  = df.sort_values("fpr")
rfpr = np.concatenate([[0.0], roc["fpr"].values])
rrec = np.concatenate([[0.0], roc["recall"].values])

# Solid curve from sweep data; dashed extrapolation to (1,1)
ax1.plot(rfpr, rrec, color=BLUE, lw=1.9, zorder=3,
         label=f"V13 model  (AUROC = {AUROC_V13})")
ax1.plot([rfpr[-1], 1.0], [rrec[-1], 1.0],
         color=BLUE, lw=1.4, ls="--", zorder=2, alpha=0.45)
ax1.plot([0, 1], [0, 1], color=GRAY, lw=1.1, ls="--", zorder=1,
         label="Random baseline  (AUROC = 0.50)")

ax1.scatter(op["fpr"], op["recall"],
            color=BLUE, s=58, zorder=5, marker="o", clip_on=False)
ax1.annotate(
    f"t = {op['threshold']:.2f}\nrecall = {op['recall']:.2f},  FPR = {op['fpr']:.2f}",
    xy=(op["fpr"], op["recall"]),
    xytext=(op["fpr"] + 0.14, op["recall"] - 0.17),
    fontsize=7.5, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8, shrinkA=4, shrinkB=4),
)

ax1.set_xlim(-0.01, 1.0)
ax1.set_ylim(0.0, 1.02)
ax1.set_xlabel("False positive rate")
ax1.set_ylabel("True positive rate (recall)")
ax1.set_title(f"ROC curve — reported AUROC = {AUROC_V13}")
ax1.legend(loc="lower right")
ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax1.text(0.97, 0.04, "dashed = extrapolated\n(data limited to t ≥ 0.10)",
         ha="right", va="bottom", fontsize=7, color=GRAY,
         transform=ax1.transAxes)

# ─ PR ─
pr   = df.sort_values("recall")
prec = np.concatenate([[1.0], pr["precision"].values])
prec_recall = np.concatenate([[0.0], pr["recall"].values])

ax2.plot(prec_recall, prec, color=GREEN, lw=1.9, zorder=3,
         label=f"V13 model  (AUPRC = {AUPRC_V13})")
# Dashed anchor tail from (0,1) to first data point
ax2.plot([0.0, prec_recall[1]], [1.0, prec[1]],
         color=GREEN, lw=1.4, ls="--", zorder=2, alpha=0.45)
ax2.axhline(PREV, color=GRAY, lw=1.1, ls="--", zorder=1,
            label=f"Random baseline  (AUPRC = prevalence ≈ {PREV:.3f})")

ax2.scatter(op["recall"], op["precision"],
            color=GREEN, s=58, zorder=5, marker="o", clip_on=False)
ax2.annotate(
    f"t = {op['threshold']:.2f}\nrecall = {op['recall']:.2f},  prec = {op['precision']:.3f}",
    xy=(op["recall"], op["precision"]),
    xytext=(op["recall"] - 0.35, op["precision"] + 0.11),
    fontsize=7.5, color=GREEN,
    arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8, shrinkA=4, shrinkB=4),
)

ax2.set_xlim(-0.01, 1.0)
ax2.set_ylim(-0.01, 1.05)
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title(f"Precision-recall curve — reported AUPRC = {AUPRC_V13}")
ax2.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99))
ax2.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax2.text(0.97, 0.30,
         f"Ratio to random baseline:\n{AUPRC_V13/PREV:.1f}×",
         ha="right", va="top", fontsize=8, color=GREEN,
         transform=ax2.transAxes)

fig1.savefig(OUT_DIR + "fig1_roc_pr.png")
fig1.savefig(OUT_DIR + "fig1_roc_pr.pdf")
print("Saved fig1_roc_pr")

# ── FIGURE 2 — Metrics across thresholds ─────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7.8, 4.2))

t = df["threshold"].values

ax.plot(t, df["accuracy"],  color=GRAY,  lw=1.8, ls="-",
        label="Accuracy")
ax.plot(t, df["f1_neg"],    color=LBLUE, lw=1.4, ls="--",
        label="F1–  (negative class / non-sepsis)")
ax.plot(t, df["f1_pos"],    color=RED,   lw=1.4, ls="-.",
        label="F1+  (positive class / sepsis)")
ax.plot(t, df["macro_f1"],  color=BLUE,  lw=2.2, ls="-",
        label="Macro F1  (preferred)")

naive_acc = 1.0 - PREV

# Naive ceiling
ax.axhline(naive_acc, color=GRAY, lw=0.9, ls=":", alpha=0.65)
ax.text(df["threshold"].max() - 0.01,
        naive_acc + 0.008,
        f"naive-model ceiling  {naive_acc:.3f}",
        ha="right", va="bottom", fontsize=7.5, color=GRAY)

# Peak Macro F1
ax.axvline(best["threshold"], color=BLUE, lw=0.9, ls=":", alpha=0.65)
ax.scatter([best["threshold"]], [best["macro_f1"]],
           color=BLUE, s=55, zorder=5, clip_on=False)
ax.annotate(
    f"Macro F1 peak\nt = {best['threshold']:.2f},  {best['macro_f1']:.3f}",
    xy=(best["threshold"], best["macro_f1"]),
    xytext=(best["threshold"] + 0.07, best["macro_f1"] - 0.07),
    fontsize=7.5, color=BLUE,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8, shrinkA=4, shrinkB=4),
)

# Annotate accuracy at the peak-F1 threshold for contrast
acc_at_peak = df.loc[df["macro_f1"].idxmax(), "accuracy"]
ax.annotate(
    f"accuracy = {acc_at_peak:.3f}\nat same threshold",
    xy=(best["threshold"], acc_at_peak),
    xytext=(best["threshold"] + 0.07, acc_at_peak + 0.02),
    fontsize=7.5, color=GRAY,
    arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8, shrinkA=4, shrinkB=4),
)

ax.set_xlim(t.min() - 0.01, t.max() + 0.01)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel("Classification threshold")
ax.set_ylabel("Metric value")
ax.set_title("Metric behaviour across thresholds — V13 model, natural prevalence (1.73%)")
ax.legend(loc="center right", ncol=1)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

fig2.savefig(OUT_DIR + "fig2_threshold_metrics.png")
fig2.savefig(OUT_DIR + "fig2_threshold_metrics.pdf")
print("Saved fig2_threshold_metrics")

plt.close("all")
print("Done.")
