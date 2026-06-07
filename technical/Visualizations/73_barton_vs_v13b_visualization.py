# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  73_barton_vs_v13b_visualization.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Renders a head-to-head comparison of the Barton (2019) replication
#    against the V13b pipeline at the 6-hour-prior horizon (AUROC and
#    AUPRC bars), plus a line chart of the Barton replication's scores
#    across the 0h / 6h / 12h horizons. All metric values are hard-coded.
#
#  INPUTS
#    none (metric values are hard-coded in this script)
#  OUTPUTS
#    73_viz_Barton_vs_V13b_6h.png
#    73_viz_Barton_Horizons.png
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    data list        -  hard-coded AUROC / AUPRC per model and horizon
#                        (Barton 0h/6h/12h and V13b 6h)
#    figure outputs   -  73_viz_Barton_vs_V13b_6h.png (dpi=300),
#                        73_viz_Barton_Horizons.png (dpi=300)
#
#  REQUIRES: pandas, numpy, matplotlib, seaborn
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_comparison():
    print("Generating Comparative Bar Charts: Barton (2019) vs Pipeline V13b")

    # EDIT: hard-coded metric values (AUROC / AUPRC per model and horizon)
    # Static Data from test extractions
    data = [
        {"Model": "Barton (2019) Replication", "Horizon": "0h (Onset)", "AUROC": 0.7168, "AUPRC": 0.0208},
        {"Model": "Barton (2019) Replication", "Horizon": "6h Prior", "AUROC": 0.6719, "AUPRC": 0.0382},
        {"Model": "Barton (2019) Replication", "Horizon": "12h Prior", "AUROC": 0.6736, "AUPRC": 0.0614},

        # V13b values from previous logs
        {"Model": "V13b (NOSE+RF+Etiology Stack)", "Horizon": "6h Prior", "AUROC": 0.7606, "AUPRC": 0.0609},
        # For V13b 12h we usually had similar AUROC and slightly higher AUPRC due to larger pos bucket. Let's approximate based on prior scripts or leave out. Let's just compare 6h directly.
    ]

    # We'll plot a simple head-to-head for the 6h horizon
    h6_data = [d for d in data if d['Horizon'] == '6h Prior']
    df_h6 = pd.DataFrame(h6_data)

    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Head-to-Head: Barton (2019) Simple Features vs V13b Complex Representation\n(6-Hour Prior Horizon)', fontsize=16, fontweight='bold', y=1.05)

    # AUROC PIE/BAR
    sns.barplot(data=df_h6, x='Model', y='AUROC', ax=axes[0], palette=['#E74C3C', '#2ECC71'])
    axes[0].set_title('AUROC (Area Under ROC Curve)')
    axes[0].set_ylim(0.5, 0.8)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # AUPRC
    sns.barplot(data=df_h6, x='Model', y='AUPRC', ax=axes[1], palette=['#E74C3C', '#2ECC71'])
    axes[1].set_title('AUPRC (Precision-Recall Baseline ~0.017)')
    axes[1].set_ylim(0, 0.1)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('73_viz_Barton_vs_V13b_6h.png', dpi=300, bbox_inches='tight')  # EDIT: figure output filename and DPI
    plt.close()

    # Horizon comparison for Barton
    bart_data = [d for d in data if d['Model'] == 'Barton (2019) Replication']
    df_bart = pd.DataFrame(bart_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_bart, x='Horizon', y='AUROC', marker='o', linewidth=3, markersize=10, color='#3498DB', label='AUROC')
    sns.lineplot(data=df_bart, x='Horizon', y='AUPRC', marker='s', linewidth=3, markersize=10, color='#E67E22', label='AUPRC')
    plt.title('Barton (2019) Replication Across Horizons', fontsize=16, fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('73_viz_Barton_Horizons.png', dpi=300)  # EDIT: figure output filename and DPI
    plt.close()

    print("✅ Visualizations saved: 73_viz_Barton_vs_V13b_6h.png and 73_viz_Barton_Horizons.png")

if __name__ == "__main__":
    visualize_comparison()
