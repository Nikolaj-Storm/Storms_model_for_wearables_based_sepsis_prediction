# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  build_feature_engineering_heatmap.py
#  Stage: 6 - Visualization / appendix
#
#  PURPOSE
#    Heatmap of per-feature univariate AUPRC lift over the 1-hour prevalence
#    baseline for the V13 pipeline. Panel A covers kinematic / temporal-dynamic
#    features (transformation x vital-sign channel); Panel B is a tall strip of
#    clinical composites, biological interactions, and contextual features.
#
#  INPUTS
#    <script_dir>/technical/Results/feature_ranking_results copy.csv
#  OUTPUTS
#    <script_dir>/fig_feature_engineering_heatmap.png
#    <script_dir>/fig_feature_engineering_heatmap.pdf
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    CSV          -  path to the per-feature ranking results CSV
#    OUT_DIR      -  output directory for the PNG and PDF
#    PREVALENCE   -  1-hour resolution prevalence baseline (0.0183), lift divisor
#    png / pdf    -  output filenames
#
#  REQUIRES: numpy, pandas, matplotlib
# ============================================================================

"""
Figure 4 style heatmap of per-feature univariate AUPRC lift for the V13 pipeline.

Panel A: kinematic / temporal-dynamic features, transformation x vital-sign channel.
Panel B: clinical composites, biological interactions and contextual features
         as a single tall strip.

Cell colour = AUPRC lift over the 1-hour prevalence baseline (0.0183).
Lift = 1.0 means the feature is at chance.

Saves PNG and PDF to /PATH/TO/PROJECT/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_HERE = Path(__file__).resolve().parent
# EDIT: path to the per-feature ranking results CSV
CSV = _HERE / "technical" / "Results" / "feature_ranking_results copy.csv"
# EDIT: output directory for the PNG and PDF (defaults to the script directory)
OUT_DIR = _HERE
PREVALENCE = 0.0183  # EDIT: 1-hour resolution prevalence (dummy AUPRC), lift divisor

df = pd.read_csv(CSV)
df["AUPRC_lift"] = df["AUPRC_Effect"] / PREVALENCE


def lift(name: str) -> float:
    row = df[df["Feature"] == name]
    return float(row["AUPRC_lift"].iloc[0]) if not row.empty else np.nan


CHANNELS = ["heart_rate", "resprate", "sbp", "dbp", "temp_c", "spo2"]
CHANNEL_LABELS = ["HR", "RR", "SBP", "DBP", "Temp", "SpO2"]

TRANSFORMS = [
    ("Raw value",            lambda c: c),
    ("Expanding min",        lambda c: f"exp_min_{c}"),
    ("Expanding max",        lambda c: f"exp_max_{c}"),
    ("Expanding mean",       lambda c: f"exp_mean_{c}"),
    ("Expanding SD",         lambda c: f"exp_std_{c}"),
    ("Rolling SD (4h)",      lambda c: f"{c}_sd_4h"),
    ("EWMA (3h)",            lambda c: f"{c}_ewma_3h"),
    ("Slope (4h)",           lambda c: f"slope_4h_{c}"),
    ("Lag diff (1h)",        lambda c: f"lag_diff_1h_{c}"),
    ("Lag ratio (1h)",       lambda c: f"lag_ratio_1h_{c}"),
    ("Acceleration",         lambda c: f"accel_{c}"),
    ("CUSUM positive drift", lambda c: f"cusum_pos_{c}"),
    ("CUSUM negative drift", lambda c: f"cusum_neg_{c}"),
]

# Panel A matrix
mat = np.full((len(TRANSFORMS), len(CHANNELS)), np.nan)
for i, (_, fn) in enumerate(TRANSFORMS):
    for j, ch in enumerate(CHANNELS):
        mat[i, j] = lift(fn(ch))

# Panel B: composites + interactions + context, ordered by family
PANEL_B = [
    ("Modified Shock Index",         "msi"),
    ("Shock Index",                  "shock_index"),
    ("Mean Arterial Pressure",       "map"),
    ("Pulse Pressure",               "pulse_pressure"),
    ("Rate-Pressure Product",        "rpp"),
    ("NEWS Score",                   "news_score"),
    ("NEWS delta",                   "news_delta"),
    ("Partial qSOFA",                "partial_qsofa"),
    ("qSOFA delta",                  "qsofa_delta"),
    ("Fever-Driven Tachycardia",     "fever_tachycardia"),
    ("Tachycardia Excess",           "tachycardia_excess"),
    ("Perfusion Adequacy",           "perfusion_adequacy"),
    ("Resp Distress (CRDI)",         "resp_distress"),
    ("Ventilation/Perfusion proxy",  "ventilation_perfusion_proxy"),
    ("Cardiorespiratory Coupling",   "cardiorespiratory_coupling_4h"),
    ("Time since ICU admit",         "time_since_ICU_admit_hours"),
    ("Age",                          "age"),
    ("Weight",                       "weight_kg"),
]
panelB = np.array([lift(n) for _, n in PANEL_B]).reshape(-1, 1)
panelB_labels = [lab for lab, _ in PANEL_B]

# Family separators (after these indices we draw a divider line)
PANEL_B_GROUP_BREAKS = [9, 14]  # composites | interactions | context

# Colour scale
all_vals = np.concatenate([mat[~np.isnan(mat)], panelB[~np.isnan(panelB)]])
vmin = float(np.floor(all_vals.min() * 10) / 10)
vmax = float(np.ceil(all_vals.max() * 10) / 10)

cmap = LinearSegmentedColormap.from_list(
    "v13_teal", ["#f6f1e7", "#cfdcd7", "#8db5b1", "#3f7d7c", "#1a3a3a"],
)


def annotate(ax, M, fontsize=8.5):
    """Write each cell value as text, choosing colour by background luminance."""
    cut = vmin + 0.55 * (vmax - vmin)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]
            if np.isnan(v):
                ax.text(c, r, "n/a", ha="center", va="center",
                        color="#999", fontsize=fontsize - 1)
            else:
                col = "white" if v > cut else "#1a1a1a"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        color=col, fontsize=fontsize)


# Figure -----------------------------------------------------------------------
fig = plt.figure(figsize=(15.5, 11.0), dpi=160)

# Three axes via add_axes for precise control:
#   axA: kinematic heatmap   (left half)
#   axB: composites strip    (right half, taller because more rows)
#   cax: colorbar            (far right)
axA = fig.add_axes([0.075, 0.10, 0.40, 0.80])
axB = fig.add_axes([0.61, 0.10, 0.07, 0.80])
cax = fig.add_axes([0.93, 0.18, 0.018, 0.64])

imA = axA.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
axA.set_xticks(range(len(CHANNELS)))
axA.set_xticklabels(CHANNEL_LABELS, fontsize=11)
axA.set_yticks(range(len(TRANSFORMS)))
axA.set_yticklabels([t[0] for t in TRANSFORMS], fontsize=10)
axA.set_title("(a) Kinematic and temporal-dynamic features\n"
              "rows: transformation   columns: vital-sign channel",
              fontsize=11.5, loc="left", pad=12)
annotate(axA, mat)
axA.set_xticks(np.arange(-0.5, len(CHANNELS), 1), minor=True)
axA.set_yticks(np.arange(-0.5, len(TRANSFORMS), 1), minor=True)
axA.grid(which="minor", color="white", linewidth=1.2)
axA.tick_params(which="minor", length=0)

imB = axB.imshow(panelB, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
axB.yaxis.tick_right()
axB.set_yticks(range(len(panelB_labels)))
axB.set_yticklabels(panelB_labels, fontsize=10)
axB.set_xticks([0]); axB.set_xticklabels(["Lift"], fontsize=11)
axB.set_title("(b) Clinical composites,\nbiological interactions\nand contextual features",
              fontsize=11.5, loc="left", pad=12)
annotate(axB, panelB)
axB.set_xticks([-0.5, 0.5], minor=True)
axB.set_yticks(np.arange(-0.5, len(panelB_labels), 1), minor=True)
axB.grid(which="minor", color="white", linewidth=1.2)
axB.tick_params(which="minor", length=0)
# Family group dividers
for idx in PANEL_B_GROUP_BREAKS:
    axB.axhline(idx + 0.5, color="#222", linewidth=1.6)
# Family bracket labels on the LEFT (so they don't fight the long names on the right)
bracket_props = dict(fontsize=10, color="#444", rotation=90, ha="right",
                     va="center", fontstyle="italic")
axB.text(-0.85, (0 + 8) / 2, "clinical composites", **bracket_props)
axB.text(-0.85, (10 + 14) / 2, "biological interactions", **bracket_props)
axB.text(-0.85, (15 + 17) / 2, "contextual", **bracket_props)

cb = fig.colorbar(imA, cax=cax)
cb.set_label("AUPRC lift  (univariate AUPRC / prevalence)", fontsize=10)
cb.ax.tick_params(labelsize=9)
# Reference lines
cb.ax.axhline(1.0, color="#cc4040", linewidth=1.5)
cb.ax.text(2.3, 1.0, "chance",
           transform=cb.ax.get_yaxis_transform(),
           fontsize=8.5, color="#cc4040", va="center")

# Footer only (headline removed at user request) ------------------------------
footer = ("Lift = univariate AUPRC / prevalence baseline (0.0183).  "
          "1.00 marks chance; the strongest engineered features sit between "
          "1.45 and 1.55, while the raw vital-sign values cluster around "
          "1.07 – 1.28.  Read horizontally to compare a transformation "
          "across channels; read vertically to compare transformations "
          "within one channel.")
fig.text(0.5, 0.04, footer, ha="center", va="center",
         fontsize=9.2, color="#444", wrap=True)

# Save
# EDIT: output filenames for the feature-engineering heatmap
png = OUT_DIR / "fig_feature_engineering_heatmap.png"
pdf = OUT_DIR / "fig_feature_engineering_heatmap.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")

print(f"Saved {png.name} and {pdf.name}")
print(f"Lift range across all features: {vmin:.2f} – {vmax:.2f}")
print("\nTop 8 features by AUPRC lift:")
print(df.nlargest(8, "AUPRC_lift")[["Feature", "AUPRC_lift", "AUROC_Effect"]]
      .to_string(index=False))
