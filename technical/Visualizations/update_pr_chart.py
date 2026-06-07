# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  update_pr_chart.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Plots a recall-vs-precision scatter for the 1-hour models, overlaid
#    with iso-F1 reference curves and the dataset prevalence line. Every
#    model's (recall, precision) point is hard-coded in this script.
#
#  INPUTS
#    none (all metric points are hard-coded in this script)
#  OUTPUTS
#    /PATH/TO/PROJECT/technical/Visualizations/updated_pr_chart.png
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    figure DPI    -  figure.dpi=150 (in rcParams)
#    f1_levels     -  iso-F1 reference contour levels [0.05, 0.10, 0.15]
#    prevalence    -  0.018 (hard-coded dataset positive prevalence line)
#    points        -  hard-coded (recall, precision, label, color, marker)
#                     per model
#    output path   -  updated_pr_chart.png save location
#
#  REQUIRES: matplotlib, numpy, pandas
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.figsize': (10, 8),
    'figure.dpi': 150  # EDIT: figure DPI
})

# 1. Setup curves for F1 scores
def get_p_for_f1(r, f1):
    return (f1 * r) / (2 * r - f1)

r_range = np.linspace(0.01, 1.0, 100)
f1_levels = [0.05, 0.10, 0.15]  # EDIT: iso-F1 reference contour levels

fig, ax = plt.subplots()

for f1 in f1_levels:
    p_vals = get_p_for_f1(r_range, f1)
    # Mask values outside [0, 1]
    p_vals = np.where((p_vals > 0) & (p_vals <= 1), p_vals, np.nan)
    ax.plot(r_range, p_vals, color='steelblue', linestyle='--', alpha=0.4, zorder=1)
    # Label the curves
    idx = np.nanargmin(np.abs(p_vals - 0.12)) if f1 == 0.15 else np.nanargmin(np.abs(p_vals - 0.08))
    if not np.isnan(idx):
        ax.text(r_range[idx]+0.02, p_vals[idx], f'F1 = {f1}', color='steelblue', alpha=0.7, fontsize=9)

# 2. Prevalence line
prevalence = 0.018  # EDIT: dataset positive prevalence (1.8%)
ax.axhline(y=prevalence, color='gray', linestyle=':', alpha=0.6, zorder=0)
ax.text(0.85, prevalence + 0.002, 'Prevalence (~1.8%)', color='gray', fontsize=9)

# 3. Data points (1-hour resolution)
# Format: (Recall, Precision, Label, Color, Marker)
# EDIT: hard-coded per-model (recall, precision) metric points
points = [
    # Baseline (from standard_algos_performance_all_engineered.csv)
    (0.069, 0.1236, 'XGBoost', 'cyan', 'o'),
    (0.6128, 0.0393, 'Random Forest', 'forestgreen', 'o'),
    (0.6548, 0.0348, 'Logistic Regression', 'orange', 'o'),
    (0.6414, 0.0332, 'Decision Tree', 'firebrick', 'o'),

    # Meta Learners (from previous calculations)
    (0.174, 0.086, 'V13b NOSE XGBoost Meta', 'midnightblue', 'o'),
    (0.698, 0.0378, 'V13b NOSE LogReg Meta', 'steelblue', 'o'),

    # Dummy
    (0.0, 0.018, 'Dummy Classifier', 'gray', 'o'),

    # NEW OPTIMISED RUNS
    (0.7769, 0.0533, 'Optimised XGBoost', 'red', '*'),
    (0.9508, 0.0335, 'Optimised XGBoost (F2-opt)', 'crimson', '*'),
    (0.6616, 0.0523, 'Optimised Random Forest', 'darkgreen', '*'),
]

for r, p, label, color, marker in points:
    size = 150 if marker == '*' else 80
    ax.scatter(r, p, color=color, marker=marker, s=size, edgecolors='black', linewidth=0.5, label=label, zorder=5)

    # Adjust label positions to avoid overlaps
    ha = 'left'
    va = 'bottom'
    off_x, off_y = 0.02, 0.002

    if 'F2-opt' in label:
        ha = 'right'
        off_x = -0.02
    if 'Random Forest' == label:
        va = 'top'
        off_y = -0.005
    if 'Decision Tree' in label:
        va = 'top'
        off_y = -0.005
    if 'Optimised Random Forest' == label:
        va = 'bottom'
        off_y = 0.005

    ax.text(r + off_x, p + off_y, label, fontsize=9, ha=ha, va=va, fontweight='bold' if marker == '*' else 'normal')

# Final formatting
ax.set_xlabel('Recall (sensitivity)')
ax.set_ylabel('Precision (PPV)')
ax.set_title('1-hour models: Recall vs Precision (Updated with Optimised Runs)')
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 0.16)
ax.grid(True, which='both', linestyle='-', alpha=0.2)

# Remove legend to keep it clean (labels are on plot)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('/PATH/TO/PROJECT/technical/Visualizations/updated_pr_chart.png')  # EDIT: output figure path
print("✅ Saved: updated_pr_chart.png")
