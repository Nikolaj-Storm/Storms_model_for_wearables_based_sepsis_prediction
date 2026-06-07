# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  viz_v13_full_correlation_matrix.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Loads the full V13/V14 engineered feature matrix, computes a Pearson
#    correlation matrix, hierarchically clusters it for readability, and
#    renders an annotated heatmap. Also writes the ordered correlation CSV
#    and prints the top-20 features by absolute correlation with the target.
#
#  INPUTS
#    /PATH/TO/PROJECT/v13_full_features.parquet
#  OUTPUTS
#    viz_V13_Full_Correlation_Matrix.png
#    table_V13_Full_Correlation_Matrix.csv
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    input parquet   -  v13_full_features.parquet (feature matrix)
#    target_col      -  'Target (Sepsis 6h)' (column used for ranking)
#    output PNG      -  viz_V13_Full_Correlation_Matrix.png, dpi=250
#    output CSV      -  table_V13_Full_Correlation_Matrix.csv
#
#  REQUIRES: pandas, numpy, matplotlib, scipy
# ============================================================================
"""
Full Correlation Matrix: All 97 V13/V14 Features + Sepsis Target
Hierarchically clustered for readability.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# ── Readable labels ──────────────────────────────────────────────────────
def readable_name(col):
    r = {
        'heart_rate': 'HR', 'resprate': 'RR', 'spo2': 'SpO2',
        'temp_c': 'Temp', 'sbp': 'SBP', 'dbp': 'DBP', 'map': 'MAP',
        'ewma_3h': 'EWMA3h', 'sd_4h': 'SD4h',
        'exp_min_': 'ExpMin.', 'exp_max_': 'ExpMax.',
        'exp_mean_': 'ExpMean.', 'exp_std_': 'ExpStd.',
        'slope_4h_': 'Slope4h.', 'lag_diff_1h_': 'LagDiff.',
        'lag_ratio_1h_': 'LagRatio.',
        'shock_index': 'Shock Index', 'partial_qsofa': 'qSOFA(partial)',
        'news_score': 'NEWS', 'qsofa_delta': 'dqSOFA', 'news_delta': 'dNEWS',
        'ventilation_perfusion_proxy': 'V/Q Proxy',
        'tachycardia_excess': 'Tachy Excess',
        'perfusion_adequacy': 'Perfusion Adeq.',
        'cardiorespiratory_coupling_4h': 'CR Coupling',
        'time_since_ICU_admit_hours': 'ICU Hours',
        'weight_kg': 'Weight', 'age': 'Age',
        'accel_': 'Accel.', 'cusum_pos_': 'CUSUM+.',
        'cusum_neg_': 'CUSUM-.', 'pulse_pressure': 'Pulse Press.',
        'rpp': 'Rate-Press Prod', 'msi': 'Mod Shock Idx',
        'resp_distress': 'Resp Distress',
        'fever_tachycardia': 'Fever-Tachy',
        'Target (Sepsis 6h)': 'TARGET (Sepsis)',
    }
    label = col
    for old, new in r.items():
        label = label.replace(old, new)
    return label.replace('_', ' ').strip()

# ── Load ─────────────────────────────────────────────────────────────────
print("Loading feature matrix...")
feat_df = pd.read_parquet('/PATH/TO/PROJECT/v13_full_features.parquet')  # EDIT: input feature-matrix parquet path
n_feat = len(feat_df.columns)
print(f"Features: {n_feat}")

# ── Correlation ──────────────────────────────────────────────────────────
print("Computing correlation matrix...")
corr = feat_df.corr()

# ── Hierarchical clustering ──────────────────────────────────────────────
print("Clustering...")
dist = 1 - corr.abs().values
np.fill_diagonal(dist, 0)
dist = np.clip(dist, 0, None)
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method='ward')
order = leaves_list(Z)
ordered_cols = corr.columns[order]
corr_ordered = corr.loc[ordered_cols, ordered_cols]

# ── Plot ─────────────────────────────────────────────────────────────────
n = len(ordered_cols)
nice_labels = [readable_name(c) for c in ordered_cols]

cell = 0.48
figw = max(28, n * cell + 4)
figh = max(24, n * cell + 4)
fig, ax = plt.subplots(figsize=(figw, figh))

cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
data = corr_ordered.values

im = ax.imshow(data, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')

# Annotate
fontsize = max(3.5, min(5.5, 280 / n))
for i in range(n):
    for j in range(n):
        val = data[i, j]
        color = 'white' if abs(val) > 0.55 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=fontsize, color=color)

ax.set_xticks(range(n))
ax.set_xticklabels(nice_labels, rotation=90, fontsize=6, fontfamily='monospace')
ax.set_yticks(range(n))
ax.set_yticklabels(nice_labels, fontsize=6, fontfamily='monospace')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
cbar.set_label('Pearson Correlation', fontsize=12)

ax.set_title(
    f'Complete V13/V14 Feature Correlation Matrix: {n} Features (Hierarchically Clustered)',
    fontsize=16, fontweight='bold', pad=18
)

fig.tight_layout()
out = 'viz_V13_Full_Correlation_Matrix.png'  # EDIT: output figure filename
fig.savefig(out, dpi=250, bbox_inches='tight', facecolor='white')  # EDIT: figure DPI
plt.close(fig)
print(f"Saved: {out}")

# ── Also save CSV ────────────────────────────────────────────────────────
corr_ordered.to_csv('table_V13_Full_Correlation_Matrix.csv')  # EDIT: output correlation CSV filename
print("Saved: table_V13_Full_Correlation_Matrix.csv")

# ── Target correlation ranking ───────────────────────────────────────────
target_col = 'Target (Sepsis 6h)'  # EDIT: target column used for the correlation ranking
target_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
print(f"\nTop 20 features by |r| with {target_col}:")
for i, (feat, val) in enumerate(target_corr.head(20).items(), 1):
    sign = corr[target_col][feat]
    print(f"  {i:>2}. {readable_name(feat):<35s}  r = {sign:>+.4f}")
