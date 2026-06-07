# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  build_correlation_matrix_appendix.py
#  Stage: 6 - Visualization / appendix
#
#  PURPOSE
#    Appendix figure showing the pairwise correlation profile of three novel
#    multiplicative-interaction features against the V13 engineered set. Three
#    horizontal heatmap rows plot |Pearson r| with every other engineered
#    feature, grouped by transformation family, plus a right-hand mean |r|
#    summary strip and an orthogonality footnote.
#
#  INPUTS
#    <script_dir>/technical/Results/table_V13_Full_Correlation_Matrix.csv
#  OUTPUTS
#    <script_dir>/fig_appendix_correlation_matrix.png
#    <script_dir>/fig_appendix_correlation_matrix.pdf
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    CSV            -  path to the full correlation matrix CSV
#    OUT            -  output directory for the PNG and PDF
#    SET_MEDIAN     -  engineered-set median of mean |r| yardstick (0.115)
#    SET_MEAN       -  engineered-set mean of mean |r| yardstick (0.107)
#    vmax           -  colour-scale upper bound (1.0, full |r| range)
#    png / pdf      -  output filenames
#
#  REQUIRES: numpy, pandas, matplotlib
# ============================================================================

"""
Appendix figure: pairwise correlation profile of the three novel features
against the V13 engineered set. Three horizontal heatmap rows (one per
feature) plot |Pearson r| with every other engineered feature, grouped by
transformation family. A right-hand summary strip shows each feature's mean
|r| with the rest of the set, and a footnote anchors the median.

Outputs:
  /PATH/TO/PROJECT/fig_appendix_correlation_matrix.png
  /PATH/TO/PROJECT/fig_appendix_correlation_matrix.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
# EDIT: path to the full correlation matrix CSV
CSV  = HERE / "technical" / "Results" / "table_V13_Full_Correlation_Matrix.csv"
# EDIT: output directory for the PNG and PDF (defaults to the script directory)
OUT  = HERE

M = pd.read_csv(CSV, index_col=0)

NOVEL = [
    ("Fever-Driven Tachycardia\nHR · max(Temp − 37, 0)",     "fever_tachycardia"),
    ("Cardiorespiratory Distress Index\nRR · (100 − SpO₂)",  "resp_distress"),
    ("Generalised Perfusion Adequacy\nHR / (SBP · SpO₂/100)", "perfusion_adequacy"),
]

RAW = ["heart_rate", "resprate", "sbp", "dbp", "temp_c", "spo2"]
CTX = ["age", "weight_kg", "time_since_ICU_admit_hours"]

# Group order (engineered features only, excluding the three NOVEL we're plotting)
GROUPS_DEF = [
    ("Raw vitals",        RAW),
    ("Expanding min",     [f"exp_min_{c}" for c in RAW]),
    ("Expanding max",     [f"exp_max_{c}" for c in RAW]),
    ("Expanding mean",    [f"exp_mean_{c}" for c in RAW]),
    ("Expanding SD",      [f"exp_std_{c}" for c in RAW]),
    ("Rolling SD (4h)",   [f"{c}_sd_4h" for c in RAW]),
    ("EWMA (3h)",         [f"{c}_ewma_3h" for c in RAW]),
    ("Slope (4h)",        [f"slope_4h_{c}" for c in RAW]),
    ("Lag diff (1h)",     [f"lag_diff_1h_{c}" for c in RAW]),
    ("Lag ratio (1h)",    [f"lag_ratio_1h_{c}" for c in RAW]),
    ("Acceleration",      [f"accel_{c}" for c in RAW]),
    ("CUSUM positive",    [f"cusum_pos_{c}" for c in RAW]),
    ("CUSUM negative",    [f"cusum_neg_{c}" for c in RAW]),
    ("Clinical composites", ["shock_index", "msi", "map", "pulse_pressure", "rpp",
                             "news_score", "news_delta", "partial_qsofa", "qsofa_delta"]),
    ("Other interactions",  ["tachycardia_excess", "ventilation_perfusion_proxy",
                             "cardiorespiratory_coupling_4h"]),
    ("Contextual",          CTX),
]

NOVEL_KEYS = [k for _, k in NOVEL]

order, group_breaks, group_centers, group_names = [], [], [], []
running = 0
for name, items in GROUPS_DEF:
    present = [f for f in items if f in M.columns and f not in NOVEL_KEYS]
    if not present:
        continue
    if running > 0:
        group_breaks.append(running)
    centre = running + len(present) / 2
    order.extend(present)
    running += len(present)
    group_centers.append(centre)
    group_names.append(name)

print(f"Plotting {len(order)} engineered features against {len(NOVEL)} novel features")

# Build 3 x N matrix of |r|
data = np.zeros((len(NOVEL), len(order)))
for i, (_, key) in enumerate(NOVEL):
    if key in M.index:
        data[i] = M.loc[key, order].abs().values
    else:
        print(f"WARN: {key} not found in matrix")

# Yardsticks
SET_MEDIAN = 0.115  # EDIT: engineered-set median of mean |r| yardstick
SET_MEAN   = 0.107  # EDIT: engineered-set mean of mean |r| yardstick

# Per-feature mean |r| with the rest of the engineered set
all_eng_present = [c for c in M.columns
                   if c not in RAW and c not in CTX]
mean_abs_r = {}
for label, key in NOVEL:
    others = [c for c in all_eng_present if c != key]
    mean_abs_r[key] = M.loc[key, others].abs().mean()
    print(f"  {key}: mean |r| = {mean_abs_r[key]:.4f}")

# Plot ------------------------------------------------------------------------
N = len(order)
fig = plt.figure(figsize=(20, 10), dpi=160)

axL = fig.add_axes([0.13, 0.34, 0.70, 0.46])   # main heatmap
axS = fig.add_axes([0.845, 0.34, 0.040, 0.46]) # summary strip
cax = fig.add_axes([0.910, 0.40, 0.014, 0.34]) # colorbar

cmap = LinearSegmentedColormap.from_list(
    "v13_corr", ["#f6f1e7", "#cfdcd7", "#8db5b1", "#3f7d7c", "#1a3a3a"])

vmax = 1.0  # EDIT: colour-scale upper bound; full 0..1 so off-diagonal r weakness reads
im = axL.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto",
                interpolation="nearest")

# Cell annotations removed for legibility — colour alone carries the signal.

# Row labels
axL.set_yticks(range(len(NOVEL)))
axL.set_yticklabels([lab for lab, _ in NOVEL], fontsize=10)

# Column labels
axL.set_xticks(range(N))
axL.set_xticklabels(order, rotation=90, fontsize=7)

# Group dividers + bracket labels above the matrix
for b in group_breaks:
    axL.axvline(b - 0.5, color="white", linewidth=1.6)
    axL.axvline(b - 0.5, color="#222", linewidth=0.6, linestyle=":")
for centre, name in zip(group_centers, group_names):
    axL.text(centre, -0.85, name, ha="left", va="bottom",
             fontsize=8.5, color="#222", rotation=40,
             rotation_mode="anchor", fontweight="bold")

# Frame ticks for every cell
axL.set_xticks(np.arange(-0.5, N, 1), minor=True)
axL.set_yticks(np.arange(-0.5, len(NOVEL), 1), minor=True)
axL.grid(which="minor", color="white", linewidth=0.7)
axL.tick_params(which="minor", length=0)

# Right summary strip: mean |r| per feature against the set median
mean_vals = np.array([[mean_abs_r[key]] for _, key in NOVEL])
axS.imshow(mean_vals, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
for i, (_, key) in enumerate(NOVEL):
    v = mean_abs_r[key]
    flag = "▼ below" if v < SET_MEDIAN else "▲ above"
    col = "white" if v > 0.55 else "#1a1a1a"
    axS.text(0, i, f"{v:.3f}\n{flag}", ha="center", va="center",
             color=col, fontsize=8)
axS.set_xticks([0]); axS.set_xticklabels(["mean |r|\nvs rest"], fontsize=9)
axS.set_yticks([])
axS.set_xticks([-0.5, 0.5], minor=True)
axS.set_yticks(np.arange(-0.5, len(NOVEL), 1), minor=True)
axS.grid(which="minor", color="white", linewidth=1.0)
axS.tick_params(which="minor", length=0)

# Colorbar
cb = fig.colorbar(im, cax=cax)
cb.set_label("|Pearson r|", fontsize=9.5)
cb.ax.tick_params(labelsize=8.5)

# Title and footer
fig.suptitle("Appendix: pairwise correlation profile of the three novel features against the V13 engineered set",
             fontsize=12, y=0.96, x=0.5)

footer = (f"Each row shows |Pearson r| between the named feature and every other "
          f"feature in the V13 engineered set, grouped by transformation family.  "
          f"Engineered-set yardstick, mean of mean |r| = {SET_MEAN:.3f}, median = "
          f"{SET_MEDIAN:.3f}.  "
          f"Fever-Driven Tachycardia (mean |r| = {mean_abs_r['fever_tachycardia']:.3f}) "
          f"sits below the median and is therefore orthogonal to the rest of the pipeline; "
          f"Cardiorespiratory Distress Index ({mean_abs_r['resp_distress']:.3f}) sits at "
          f"the median; Generalised Perfusion Adequacy ({mean_abs_r['perfusion_adequacy']:.3f}) "
          f"sits in the upper tail, sharing variance with HR-, SBP- and SpO₂-derived features.")

fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=8.8, color="#444",
         wrap=True)

# EDIT: output filenames for the appendix figure
png = OUT / "fig_appendix_correlation_matrix.png"
pdf = OUT / "fig_appendix_correlation_matrix.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print(f"\nSaved {png.name} and {pdf.name}")
