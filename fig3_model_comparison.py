# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  fig3_model_comparison.py
#  Stage: 6 - Visualization / appendix
#
#  PURPOSE
#    Two-panel figure comparing the raw-vitals XGBoost and Random Forest
#    operating points. Left panel is a schematic ROC (AUROC-parameterised
#    power law); right panel is a schematic PR (PCHIP through each operating
#    point) against the prevalence baseline. All metric values are hardcoded.
#
#  INPUTS
#    none / numbers hardcoded inline
#  OUTPUTS
#    /PATH/TO/PROJECT/fig3_model_comparison.png
#    /PATH/TO/PROJECT/fig3_model_comparison.pdf
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PREV          -  empirical prevalence = 7552 / 400815 = 0.01884
#    xgb dict      -  recall 0.0048, fpr 0.0011, precision 36/(36+457),
#                     auroc 0.6269, auprc 0.0307, macro_f1 0.4995, tp 36, fn 7516
#    rf dict       -  recall 0.4349, fpr 0.2630, precision 3284/(3284+103424),
#                     auroc 0.6250, auprc 0.0301, macro_f1 0.4504, tp 3284, fn 4268
#    OUT           -  output directory for the PNG and PDF
#
#  REQUIRES: numpy, matplotlib, scipy
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import PchipInterpolator

PREV = 7552 / 400815  # EDIT: empirical prevalence = 0.01884 (positives / total)

# EDIT: XGBoost raw-vitals operating point and discrimination metrics (hardcoded)
xgb = dict(recall=0.0048, fpr=0.0011, precision=36/(36+457),
           auroc=0.6269, auprc=0.0307, macro_f1=0.4995,
           tp=36,   fn=7516, color="#E69F00", marker="s")
# EDIT: Random Forest raw-vitals operating point and discrimination metrics (hardcoded)
rf  = dict(recall=0.4349, fpr=0.2630, precision=3284/(3284+103424),
           auroc=0.6250, auprc=0.0301, macro_f1=0.4504,
           tp=3284, fn=4268, color="#D55E00", marker="o")

# ── Schematic ROC (power-law, AUROC-parameterised) ────────────────────────────
auroc_avg = (xgb["auroc"] + rf["auroc"]) / 2
c = auroc_avg / (1.0 - auroc_avg)
x_roc = np.linspace(0, 1, 400)
y_roc = 1.0 - (1.0 - x_roc) ** c

# ── Schematic PR curves (PCHIP through 3 anchors each) ───────────────────────
# Anchors: (0, 1.0)  →  actual operating point  →  (1, prevalence)
# This gives a smooth monotone curve that passes through the real operating point.
r_curve = np.linspace(0, 1, 600)

def pr_curve(r_op, p_op):
    anchors_r = np.array([0.0,  r_op, 1.0 ])
    anchors_p = np.array([1.0,  p_op, PREV])
    interp = PchipInterpolator(anchors_r, anchors_p)
    return np.clip(interp(r_curve), PREV, 1.0)

xgb_pr = pr_curve(xgb["recall"], xgb["precision"])
rf_pr  = pr_curve(rf["recall"],  rf["precision"])

# ── Style ─────────────────────────────────────────────────────────────────────
GRAY = "#757575"
BLUE = "#0072B2"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 10, "axes.titleweight": "normal",
    "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.20, "grid.linestyle": ":",
    "grid.color": "0.6", "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "legend.framealpha": 0.92, "legend.edgecolor": "0.82",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.6))
fig.subplots_adjust(wspace=0.38)

# ── LEFT: ROC ─────────────────────────────────────────────────────────────────
ax1.plot([0,1],[0,1], color=GRAY, lw=1.0, ls="--", zorder=1,
         label="Random baseline  (AUROC = 0.50)")
ax1.plot(x_roc, y_roc, color=BLUE, lw=1.9, zorder=2,
         label=f"Shared ROC curve  (AUROC ≈ {auroc_avg:.3f})")

ax1.scatter(xgb["fpr"], xgb["recall"], color=xgb["color"], s=90,
            marker=xgb["marker"], zorder=5, clip_on=False,
            label="XGBoost  (raw vitals)  — operating point")
ax1.scatter(rf["fpr"],  rf["recall"],  color=rf["color"],  s=90,
            marker=rf["marker"],  zorder=5, clip_on=False,
            label="Random Forest  (raw vitals)  — operating point")

ax1.annotate(
    f"XGBoost\nrecall = {xgb['recall']:.3f}\n({xgb['tp']} TP of {xgb['tp']+xgb['fn']})",
    xy=(xgb["fpr"], xgb["recall"]),
    xytext=(0.22, 0.20),
    fontsize=7.5, color=xgb["color"],
    arrowprops=dict(arrowstyle="->", lw=0.8, color=xgb["color"],
                    connectionstyle="arc3,rad=-0.2", shrinkA=3, shrinkB=4),
)
ax1.annotate(
    f"RF\nrecall = {rf['recall']:.3f}\n({rf['tp']} TP of {rf['tp']+rf['fn']})",
    xy=(rf["fpr"], rf["recall"]),
    xytext=(0.56, 0.28),
    fontsize=7.5, color=rf["color"],
    arrowprops=dict(arrowstyle="->", lw=0.8, color=rf["color"],
                    shrinkA=3, shrinkB=4),
)

ax1.set_xlim(-0.01, 1.0)
ax1.set_ylim(0.0, 1.02)
ax1.set_xlabel("False positive rate")
ax1.set_ylabel("True positive rate (recall)")
ax1.set_title("ROC space — AUROC near-identical (0.625 vs 0.627)")
ax1.legend(loc="lower right", fontsize=7.5)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax1.text(0.98, 0.03, "ROC curve schematic (AUROC-parameterised)",
         ha="right", va="bottom", fontsize=6.5, color=GRAY,
         transform=ax1.transAxes)

# ── RIGHT: PR ─────────────────────────────────────────────────────────────────
Y_MAX = 0.20

# Draw schematic PR curves — clipped to visible frame
# Each curve enters from the top (precision > Y_MAX) and descends through its operating point
ax2.plot(r_curve, xgb_pr, color=xgb["color"], lw=1.6, zorder=2,
         label=f"XGBoost  (AUPRC = {xgb['auprc']:.3f})")
ax2.plot(r_curve, rf_pr,  color=rf["color"],  lw=1.6, zorder=2,
         label=f"RF  (AUPRC = {rf['auprc']:.3f})")

ax2.axhline(PREV, color=GRAY, lw=1.0, ls="--", zorder=1,
            label=f"Random baseline  (prevalence ≈ {PREV:.3f})")

ax2.scatter(xgb["recall"], xgb["precision"], color=xgb["color"], s=90,
            marker=xgb["marker"], zorder=5, edgecolors="white", linewidths=0.6)
ax2.scatter(rf["recall"],  rf["precision"],  color=rf["color"],  s=90,
            marker=rf["marker"],  zorder=5, edgecolors="white", linewidths=0.6)

# Annotations
ax2.annotate(
    f"XGBoost\nrecall = {xgb['recall']:.3f}\nprec = {xgb['precision']:.3f}\nMacro F1 = {xgb['macro_f1']:.3f}",
    xy=(xgb["recall"], xgb["precision"]),
    xytext=(0.10, 0.155),
    fontsize=7.5, color=xgb["color"],
    arrowprops=dict(arrowstyle="->", lw=0.8, color=xgb["color"],
                    connectionstyle="arc3,rad=-0.15", shrinkA=3, shrinkB=4),
)
ax2.annotate(
    f"RF\nrecall = {rf['recall']:.3f}\nprec = {rf['precision']:.3f}\nMacro F1 = {rf['macro_f1']:.3f}",
    xy=(rf["recall"], rf["precision"]),
    xytext=(rf["recall"] - 0.27, rf["precision"] + 0.046),
    fontsize=7.5, color=rf["color"],
    arrowprops=dict(arrowstyle="->", lw=0.8, color=rf["color"],
                    shrinkA=3, shrinkB=4),
)

# "Curves continue to prec = 1.0 at recall = 0" note at top of frame
ax2.text(0.01, 0.99, "↑ curves reach precision = 1.0 at recall → 0",
         ha="left", va="top", fontsize=6.5, color=GRAY,
         transform=ax2.transAxes, style="italic")

ax2.text(0.98, 0.97,
         f"AUPRC vs random\nXGBoost: {xgb['auprc']/PREV:.1f}×\nRF:            {rf['auprc']/PREV:.1f}×",
         ha="right", va="top", fontsize=8, transform=ax2.transAxes,
         bbox=dict(facecolor="white", edgecolor="0.80",
                   boxstyle="round,pad=0.35", alpha=0.93))

ax2.set_xlim(-0.01, 1.0)
ax2.set_ylim(-0.004, Y_MAX)
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("PR space — operating points diverge despite identical AUPRC")
ax2.legend(loc="upper right", fontsize=7.5)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.04))

ax2.text(0.98, 0.03, "PR curves schematic (PCHIP through operating point)",
         ha="right", va="bottom", fontsize=6.5, color=GRAY,
         transform=ax2.transAxes)

# EDIT: output directory for the PNG and PDF
OUT = "/PATH/TO/PROJECT/"
fig.savefig(OUT + "fig3_model_comparison.png")
fig.savefig(OUT + "fig3_model_comparison.pdf")
plt.close()
print("Done.")
