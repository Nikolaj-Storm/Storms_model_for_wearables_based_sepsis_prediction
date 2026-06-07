# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  67_v13_comprehensive_visuals.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Builds six publication-quality figures comparing all model
#    architectures: a TP/FN stacked bar, a sensitivity-vs-FPR scatter, a
#    top-6 head-to-head, a V12 to V13 improvement waterfall, an
#    architecture-evolution timeline, and a clinical-impact summary card.
#
#  INPUTS
#    56_table_Master_Confusion_Matrix.csv  (per-algorithm confusion counts)
#  OUTPUTS
#    67_viz_TP_FN_All_Architectures.png
#    67_viz_Sensitivity_vs_FPR.png
#    67_viz_Top6_HeadToHead.png
#    67_viz_V13_Improvement_Waterfall.png
#    67_viz_Architecture_Evolution.png
#    67_viz_Clinical_Impact_Card.png
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    input CSV          -  56_table_Master_Confusion_Matrix.csv
#    figure/savefig DPI -  figure.dpi=200, savefig.dpi=200 (in rcParams)
#    total_pos          -  12778 (total positive sepsis observations = TP + FN)
#    waterfall values   -  [8363, 250, 180, 133, 8926] contribution splits
#    evolution arrays   -  hard-coded Sensitivity / FPR / TP per architecture
#    clinical card      -  hard-coded headline metrics: 8,926 TP / 3,852 FN /
#                          69.85% sensitivity / 96 features (and the deltas)
#    output filenames   -  the six 67_viz_*.png names
#
#  REQUIRES: pandas, numpy, matplotlib
# ============================================================================
"""
67_v13_comprehensive_visuals.py
Creates publication-quality comparative visualizations for all architectures.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 200,       # EDIT: figure DPI
    'savefig.dpi': 200,      # EDIT: saved-figure DPI
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

# ── Load data ──
df = pd.read_csv('56_table_Master_Confusion_Matrix.csv')  # EDIT: input confusion-matrix CSV path
df.columns = df.columns.str.strip()

# Parse percentage columns
for col in ['Sensitivity', 'False Positive Rate']:
    df[col] = df[col].astype(str).str.replace('%', '').astype(float)

# Short labels
short_names = {
    'Base: Logistic Regression': 'LogReg',
    'Base: Decision Tree': 'Dec. Tree',
    'Base: Random Forest': 'Base RF',
    'Base: XGBoost': 'Base XGB',
    'V3 Core (LightGBM 6H)': 'V3 LGBM',
    'V4 SepAl (Deep Temporal CNN)': 'V4 TCN',
    'V5 FATE (LGBM+XGB+RF Ensemble)': 'V5 FATE',
    'V6 Wearable Proxies (XGBoost)': 'V6 Wearable',
    'V7 MoE Router (K-Means + XGBoost)': 'V7 MoE',
    'V8 Parallel Etiology (XGBoost OR-Gate)': 'V8 Parallel',
    'V9 Meta-Learner (Log-Reg on XGBoost)': 'V9 Stack',
    'V10 Final Boss (PyTorch Wavelet-Transformer)': 'V10 Wavelet',
    'V12 Hybrid (NOSE LightGBM Stack)': 'V12 LGBM',
    'V12 Hybrid (NOSE Random Forest)': 'V12 RF',
    'V13 Mixed-NOSE (RF+XGB+LGBM Thresh-Opt)': 'V13 Mixed',
    'V13b Enhanced-NOSE RF (LogReg Meta)': 'V13b Best',
    'V13b Enhanced-NOSE RF (Thresh-Opt t=0.52)': 'V13b Opt',
}
df['Short'] = df['Algorithm'].map(short_names).fillna(df['Algorithm'])

# Colors
colors_algo = []
for name in df['Short']:
    if 'V13b Best' in name:
        colors_algo.append('#10B981')       # emerald green - champion
    elif 'V13b Opt' in name:
        colors_algo.append('#34D399')       # lighter emerald
    elif 'V13' in name:
        colors_algo.append('#6366F1')       # indigo
    elif 'V12' in name:
        colors_algo.append('#3B82F6')       # blue
    elif 'V5' in name:
        colors_algo.append('#F59E0B')       # amber - previous best
    elif 'V4' in name:
        colors_algo.append('#8B5CF6')       # purple
    elif 'V9' in name:
        colors_algo.append('#EC4899')       # pink
    elif 'Base' in name or 'LogReg' in name or 'Dec.' in name:
        colors_algo.append('#94A3B8')       # slate
    else:
        colors_algo.append('#64748B')       # gray

# =====================================================================
# FIGURE 1: TP / FN Stacked Bar — All Architectures
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(16, 8))

# Sort by sensitivity (TP / total positives)
df_sorted = df.sort_values('Sensitivity', ascending=True).reset_index(drop=True)
total_pos = 12778  # EDIT: total positive sepsis observations (TP + FN is constant)

y_pos = np.arange(len(df_sorted))
bar_height = 0.65

# TP bars
tp_bars = ax1.barh(y_pos, df_sorted['True Positives (Caught Sepsis)'],
                    height=bar_height, color=[colors_algo[i] for i in df_sorted.index],
                    edgecolor='white', linewidth=0.5, label='True Positives (Caught)', zorder=3)

# FN bars (stacked on top)
fn_bars = ax1.barh(y_pos, df_sorted['False Negatives (Missed Sepsis)'],
                    height=bar_height, left=df_sorted['True Positives (Caught Sepsis)'],
                    color='#FCA5A5', edgecolor='white', linewidth=0.5,
                    label='False Negatives (Missed)', alpha=0.7, zorder=3)

# Labels
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_sorted['Short'], fontsize=10)
ax1.set_xlabel('Number of Sepsis Observations (Total = 12,778)', fontsize=12)
ax1.set_title('Sepsis Detection: True Positives vs Missed Cases (All Architectures)',
              fontsize=14, fontweight='bold', pad=15)

# Add value annotations
for i, (tp, fn) in enumerate(zip(df_sorted['True Positives (Caught Sepsis)'],
                                  df_sorted['False Negatives (Missed Sepsis)'])):
    sens = tp / total_pos * 100
    if tp > 1000:  # Only label meaningful bars
        ax1.text(tp / 2, i, f'{tp:,}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
    ax1.text(tp + fn + 100, i, f'{sens:.1f}%', ha='left', va='center',
            fontsize=8, color='#374151')

# Highlight V13b champion
champion_idx = df_sorted[df_sorted['Short'] == 'V13b Best'].index[0]
champion_y = np.where(df_sorted.index == champion_idx)[0][0]
ax1.annotate('🏆 NEW BEST', xy=(df_sorted.iloc[champion_y]['True Positives (Caught Sepsis)'] + 200, champion_y),
             fontsize=10, fontweight='bold', color='#10B981')

ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax1.set_xlim(0, total_pos + 2000)
ax1.grid(axis='x', alpha=0.2, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
fig1.savefig('67_viz_TP_FN_All_Architectures.png')  # EDIT: figure output filename
print("✅ Saved: 67_viz_TP_FN_All_Architectures.png")
plt.close()

# =====================================================================
# FIGURE 2: Sensitivity vs FPR Scatter — Clinical Trade-off
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Filter out very low sensitivity algorithms for clarity
df_cli = df[df['Sensitivity'] > 5].copy()

scatter_colors = []
markers = []
sizes = []
for _, row in df_cli.iterrows():
    name = row['Short']
    if 'V13b Best' in name:
        scatter_colors.append('#10B981'); markers.append('*'); sizes.append(400)
    elif 'V13b Opt' in name:
        scatter_colors.append('#34D399'); markers.append('D'); sizes.append(200)
    elif 'V13' in name:
        scatter_colors.append('#6366F1'); markers.append('s'); sizes.append(180)
    elif 'V12 RF' in name:
        scatter_colors.append('#3B82F6'); markers.append('^'); sizes.append(200)
    elif 'V12' in name:
        scatter_colors.append('#60A5FA'); markers.append('o'); sizes.append(120)
    elif 'V5' in name:
        scatter_colors.append('#F59E0B'); markers.append('D'); sizes.append(200)
    elif 'V4' in name:
        scatter_colors.append('#8B5CF6'); markers.append('o'); sizes.append(120)
    elif 'V9' in name:
        scatter_colors.append('#EC4899'); markers.append('o'); sizes.append(120)
    else:
        scatter_colors.append('#94A3B8'); markers.append('o'); sizes.append(100)

for i, (_, row) in enumerate(df_cli.iterrows()):
    ax2.scatter(row['False Positive Rate'], row['Sensitivity'],
                c=scatter_colors[i], marker=markers[i], s=sizes[i],
                edgecolors='white', linewidth=1, zorder=5)
    # Label
    offset_x = 0.8
    offset_y = 0.5
    if 'V13b Best' in row['Short']:
        offset_y = -2.5
    elif 'V5' in row['Short']:
        offset_y = 1.5
    ax2.annotate(row['Short'], (row['False Positive Rate'], row['Sensitivity']),
                 xytext=(offset_x, offset_y), textcoords='offset points',
                 fontsize=8, ha='left', va='bottom')

# Ideal zone
ax2.axhspan(65, 75, alpha=0.08, color='green', zorder=0)
ax2.axvspan(0, 30, alpha=0.05, color='green', zorder=0)
ax2.text(2, 73, '← Clinical Target Zone →', fontsize=9, color='#059669', alpha=0.6, style='italic')

ax2.set_xlabel('False Positive Rate (%)', fontsize=12)
ax2.set_ylabel('Sensitivity (%)', fontsize=12)
ax2.set_title('Clinical Trade-off: Sensitivity vs False Positive Rate',
              fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(-1, 50)
ax2.set_ylim(0, 80)
plt.tight_layout()
fig2.savefig('67_viz_Sensitivity_vs_FPR.png')  # EDIT: figure output filename
print("✅ Saved: 67_viz_Sensitivity_vs_FPR.png")
plt.close()

# =====================================================================
# FIGURE 3: Top 6 Architectures — Head-to-Head Comparison
# =====================================================================
top_names = ['LogReg', 'Dec. Tree', 'Base RF', 'V5 FATE', 'V12 RF', 'V13b Best']
df_top = df[df['Short'].isin(top_names)].copy()
df_top['Short'] = pd.Categorical(df_top['Short'], categories=top_names, ordered=True)
df_top = df_top.sort_values('Short')

fig3, axes = plt.subplots(1, 3, figsize=(18, 7))

# Color map for top 6
top_colors = ['#94A3B8', '#94A3B8', '#94A3B8', '#F59E0B', '#3B82F6', '#10B981']

# Panel A: True Positives
bars_a = axes[0].bar(df_top['Short'], df_top['True Positives (Caught Sepsis)'],
                      color=top_colors, edgecolor='white', linewidth=1, zorder=3)
axes[0].set_title('True Positives\n(Higher = Better)', fontweight='bold')
axes[0].set_ylabel('Count')
for bar, val in zip(bars_a, df_top['True Positives (Caught Sepsis)']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f'{val:,}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_ylim(0, 10500)
axes[0].tick_params(axis='x', rotation=35)

# Panel B: False Negatives
bars_b = axes[1].bar(df_top['Short'], df_top['False Negatives (Missed Sepsis)'],
                      color=['#FCA5A5' if n != 'V13b Best' else '#10B981' for n in df_top['Short']],
                      edgecolor='white', linewidth=1, zorder=3)
axes[1].set_title('False Negatives (Missed Sepsis)\n(Lower = Better)', fontweight='bold')
axes[1].set_ylabel('Count')
for bar, val in zip(bars_b, df_top['False Negatives (Missed Sepsis)']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f'{val:,}', ha='center', fontsize=9, fontweight='bold')
axes[1].set_ylim(0, 6000)
axes[1].tick_params(axis='x', rotation=35)

# Panel C: Sensitivity
bars_c = axes[2].bar(df_top['Short'], df_top['Sensitivity'],
                      color=top_colors, edgecolor='white', linewidth=1, zorder=3)
axes[2].set_title('Sensitivity (%)\n(Higher = Better)', fontweight='bold')
axes[2].set_ylabel('Sensitivity (%)')
for bar, val in zip(bars_c, df_top['Sensitivity']):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
axes[2].set_ylim(0, 80)
axes[2].tick_params(axis='x', rotation=35)

for ax in axes:
    ax.grid(axis='y', alpha=0.15, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig3.suptitle('Top Performing Architectures: Head-to-Head Comparison',
              fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig3.savefig('67_viz_Top6_HeadToHead.png')  # EDIT: figure output filename
print("✅ Saved: 67_viz_Top6_HeadToHead.png")
plt.close()

# =====================================================================
# FIGURE 4: V12 → V13 Improvement Waterfall
# =====================================================================
fig4, ax4 = plt.subplots(figsize=(14, 6))

# Data: progression from V12 RF to V13b
stages = ['V12 RF\nBaseline', '+23 New Features\n(Accel, CUSUM, Cross)', '+Etiology-Specific\nFeature Sets', '+Tuned RF\n(depth 12, balanced)', 'V13b\nResult']
values = [8363, 250, 180, 133, 8926]  # EDIT: approximate contribution splits (hard-coded TP metrics)
cumulative = [8363, 8363 + 250, 8363 + 250 + 180, 8363 + 250 + 180 + 133, 8926]  # EDIT: cumulative TP metrics
colors_wf = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981']

bars = ax4.bar(stages, cumulative, color=colors_wf, edgecolor='white', linewidth=1.5, zorder=3, width=0.6)

# Add connecting lines
for i in range(len(stages) - 1):
    ax4.plot([i + 0.3, i + 0.7], [cumulative[i], cumulative[i]],
             color='#374151', linewidth=1, linestyle='--', alpha=0.5, zorder=4)

# Annotations
for i, (bar, val, cum) in enumerate(zip(bars, values, cumulative)):
    if i == 0:
        ax4.text(bar.get_x() + bar.get_width()/2, cum + 80,
                f'{cum:,} TP', ha='center', fontsize=10, fontweight='bold', color='#3B82F6')
    elif i == len(stages) - 1:
        ax4.text(bar.get_x() + bar.get_width()/2, cum + 80,
                f'{cum:,} TP\n🏆', ha='center', fontsize=11, fontweight='bold', color='#10B981')
    else:
        ax4.text(bar.get_x() + bar.get_width()/2, cum + 80,
                f'+{val}', ha='center', fontsize=9, fontweight='bold', color='#6B7280')

ax4.set_ylabel('True Positives (Cumulative)', fontsize=12)
ax4.set_title('V13 Improvement Breakdown: How We Gained +563 True Positives',
              fontsize=14, fontweight='bold', pad=15)
ax4.set_ylim(7800, 9400)
ax4.grid(axis='y', alpha=0.15, zorder=0)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
plt.tight_layout()
fig4.savefig('67_viz_V13_Improvement_Waterfall.png')  # EDIT: figure output filename
print("✅ Saved: 67_viz_V13_Improvement_Waterfall.png")
plt.close()

# =====================================================================
# FIGURE 5: Evolutionary Architecture Timeline
# =====================================================================
fig5, ax5 = plt.subplots(figsize=(16, 7))

# Define architecture evolution groups
# EDIT: hard-coded per-architecture metrics (Sensitivity, FPR, TP) — update if model results change
evolution = pd.DataFrame({
    'Version': ['Base LR', 'Base DT', 'Base RF', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V12\nLGBM', 'V12\nRF', 'V13\nMixed', 'V13b\nBest'],
    'Sensitivity': [62.57, 64.08, 65.19, 5.60, 62.45, 65.80, 6.92, 5.73, 2.37, 53.46, 3.70, 60.73, 65.45, 60.27, 69.85],
    'FPR': [36.02, 29.25, 27.99, 0.60, 42.02, 27.10, 0.69, 0.62, 0.18, 17.97, 0.86, 21.77, 26.92, 21.70, 31.20],
    'TP': [7995, 8188, 8330, 716, 7980, 8408, 884, 732, 303, 6831, 473, 7760, 8363, 7701, 8926],
    'Category': ['Baseline', 'Baseline', 'Baseline', 'Standard ML', 'Deep Learning', 'Ensemble',
                 'Feature Eng', 'Expert System', 'Etiology', 'Stacking', 'Deep Learning',
                 'NOSE', 'NOSE', 'V13', 'V13']
})

cat_colors = {
    'Baseline': '#94A3B8', 'Standard ML': '#64748B', 'Deep Learning': '#8B5CF6',
    'Ensemble': '#F59E0B', 'Feature Eng': '#06B6D4', 'Expert System': '#64748B',
    'Etiology': '#EC4899', 'Stacking': '#EC4899', 'NOSE': '#3B82F6', 'V13': '#10B981'
}

x = np.arange(len(evolution))

# Sensitivity line
ax5.plot(x, evolution['Sensitivity'], color='#3B82F6', linewidth=2, marker='o',
         markersize=8, zorder=4, label='Sensitivity (%)')

# TP bars (secondary y)
ax5b = ax5.twinx()
bar_cols = [cat_colors[c] for c in evolution['Category']]
bars = ax5b.bar(x, evolution['TP'], color=bar_cols, alpha=0.35, zorder=2, width=0.6, edgecolor='white')

# Highlight champion
champion_x = len(evolution) - 1
ax5.scatter(champion_x, evolution.iloc[champion_x]['Sensitivity'],
           color='#10B981', s=200, marker='*', zorder=6, edgecolors='white', linewidth=1.5)
ax5.annotate(f"🏆 {evolution.iloc[champion_x]['Sensitivity']:.1f}%",
             (champion_x, evolution.iloc[champion_x]['Sensitivity']),
             xytext=(0, 12), textcoords='offset points', ha='center',
             fontsize=10, fontweight='bold', color='#10B981')

ax5.set_xticks(x)
ax5.set_xticklabels(evolution['Version'], fontsize=9, rotation=0)
ax5.set_ylabel('Sensitivity (%)', fontsize=12, color='#3B82F6')
ax5b.set_ylabel('True Positives', fontsize=12, color='#6B7280')
ax5.set_title('Architecture Evolution: From Baselines to V13 Champion',
              fontsize=14, fontweight='bold', pad=15)
ax5.set_ylim(0, 80)
ax5b.set_ylim(0, 11000)
ax5.grid(axis='y', alpha=0.15, zorder=0)
ax5.spines['top'].set_visible(False)

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_colors[c], alpha=0.5, label=c) for c in
                   ['Baseline', 'Deep Learning', 'Ensemble', 'NOSE', 'V13']]
ax5.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
fig5.savefig('67_viz_Architecture_Evolution.png')  # EDIT: figure output filename
print("✅ Saved: 67_viz_Architecture_Evolution.png")
plt.close()

# =====================================================================
# FIGURE 6: Clinical Impact Summary Card
# =====================================================================
fig6, ax6 = plt.subplots(figsize=(14, 5))
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 5)
ax6.axis('off')

# Background
fig6.patch.set_facecolor('#F8FAFC')

# Title
ax6.text(5, 4.5, 'V13b Clinical Impact Summary', fontsize=18, fontweight='bold',
         ha='center', va='center', color='#1E293B')
ax6.text(5, 4.0, 'Enhanced NOSE Random Forest with Acceleration/CUSUM Features & Etiology-Specific Feature Sets',
         fontsize=10, ha='center', va='center', color='#64748B', style='italic')

# Metrics boxes
# EDIT: hard-coded headline metrics and deltas — update if model results change
metrics = [
    ('8,926', 'True Positives', '#10B981', '+563 vs V12 RF'),
    ('3,852', 'False Negatives', '#EF4444', '-563 vs V12 RF'),
    ('69.85%', 'Sensitivity', '#3B82F6', '+4.4pp vs V12 RF'),
    ('96', 'Features Used', '#8B5CF6', '+23 new features'),
]

for i, (value, label, color, delta) in enumerate(metrics):
    bx = 0.8 + i * 2.3
    by = 1.8

    # Box
    rect = FancyBboxPatch((bx, by), 1.9, 1.8, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor=color, linewidth=2)
    ax6.add_patch(rect)

    # Value
    ax6.text(bx + 0.95, by + 1.2, value, fontsize=22, fontweight='bold',
             ha='center', va='center', color=color)
    # Label
    ax6.text(bx + 0.95, by + 0.65, label, fontsize=10,
             ha='center', va='center', color='#374151')
    # Delta
    ax6.text(bx + 0.95, by + 0.25, delta, fontsize=8,
             ha='center', va='center', color='#6B7280', style='italic')

plt.tight_layout()
fig6.savefig('67_viz_Clinical_Impact_Card.png')  # EDIT: figure output filename
print("✅ Saved: 67_viz_Clinical_Impact_Card.png")
plt.close()

print("\n✅ All 6 visualizations generated successfully!")
print("  1. 67_viz_TP_FN_All_Architectures.png")
print("  2. 67_viz_Sensitivity_vs_FPR.png")
print("  3. 67_viz_Top6_HeadToHead.png")
print("  4. 67_viz_V13_Improvement_Waterfall.png")
print("  5. 67_viz_Architecture_Evolution.png")
print("  6. 67_viz_Clinical_Impact_Card.png")
