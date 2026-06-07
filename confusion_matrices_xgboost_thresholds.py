# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  confusion_matrices_xgboost_thresholds.py
#  Stage: 6 - Visualization / appendix
#
#  PURPOSE
#    Render a single publication-quality figure with three side-by-side
#    confusion matrices for the V13b stacked-ensemble XGBoost meta-learner at
#    1-hour resolution, one per operating point (low / mid / high threshold).
#    Each panel is annotated with counts and derived sensitivity, specificity,
#    PPV, and FPR. Counts are hardcoded from the benchmark results table.
#
#  INPUTS
#    none / numbers hardcoded inline (source: 67_v13b_benchmark_results.txt)
#  OUTPUTS
#    <script_dir>/v13b_xgboost_threshold_confusion_matrices.png
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    MODELS thresholds   -  0.3705, 0.5686, 0.8006 panel titles
#    MODELS tp/fn/fp/tn  -  confusion-matrix counts per operating point
#                           low:  tp 11170, fn 2091,  fp 362546, tn 364127
#                           mid:  tp 7510,  fn 5751,  fp 145271, tn 581402
#                           high: tp 2257,  fn 11004, fp 21185,  tn 705488
#    SHARED_METRICS      -  ROC AUC 0.7576, AUPRC 0.0677 (shared discrimination)
#    subtitle counts     -  n positives 13,261; n negatives 726,673
#    output PNG filename -  where the figure is written
#
#  REQUIRES: matplotlib, numpy
# ============================================================================

"""
Confusion matrices for the top three V13b stacked-ensemble XGBoost meta-learner
operating points at 1-hour resolution.

Generates a single publication-quality figure with three side-by-side confusion
matrices, one per threshold strategy:

    1. F1-maximising threshold (primary)
    2. Max sensitivity, FPR < 50%
    3. Max TP, FPR < 20%

Each panel is annotated with absolute counts and the derived sensitivity,
specificity, and PPV so the operating-point trade-offs are visible at a glance.

Numbers are pulled directly from the V13b benchmark results table; do NOT
recompute or substitute -- update only if the underlying benchmark file
changes.

Author: Author
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Confusion-matrix counts (XGBoost stacked-ensemble meta-learner, 1h resolution)
# Source: 67_v13b_benchmark_results.txt
# EDIT: thresholds and tp/fn/fp/tn counts below are hardcoded benchmark values;
#       update them if the underlying benchmark file changes.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "title": "Threshold = 0.3705\n(low-threshold operating point)",  # EDIT: threshold
        "tp": 11_170,   # EDIT: true positives
        "fn": 2_091,    # EDIT: false negatives
        "fp": 362_546,  # EDIT: false positives
        "tn": 364_127,  # EDIT: true negatives
    },
    {
        "title": "Threshold = 0.5686\n(mid-threshold operating point)",  # EDIT: threshold
        "tp": 7_510,    # EDIT: true positives
        "fn": 5_751,    # EDIT: false negatives
        "fp": 145_271,  # EDIT: false positives
        "tn": 581_402,  # EDIT: true negatives
    },
    {
        "title": "Threshold = 0.8006\n(high-threshold operating point)",  # EDIT: threshold
        "tp": 2_257,    # EDIT: true positives
        "fn": 11_004,   # EDIT: false negatives
        "fp": 21_185,   # EDIT: false positives
        "tn": 705_488,  # EDIT: true negatives
    },
]

# Shared properties pulled from the V13b benchmark table
# EDIT: shared discrimination metrics (ROC AUC, AUPRC) hardcoded from benchmark
SHARED_METRICS = {"ROC AUC": 0.7576, "AUPRC": 0.0677}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.dpi": 150,
    }
)

# Two-tone blue ramp; muted enough for academic print
CMAP = LinearSegmentedColormap.from_list(
    "academic_blue", ["#f5f9ff", "#1f4e79"], N=256
)


def derive_metrics(tp: int, fn: int, fp: int, tn: int) -> dict[str, float]:
    """Return sensitivity, specificity, PPV, and FPR from raw counts."""
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "FPR": fpr,
    }


def plot_confusion_panel(ax: plt.Axes, model: dict) -> None:
    """Draw a single 2x2 confusion matrix panel onto the supplied axes."""
    matrix = np.array(
        [
            [model["tn"], model["fp"]],
            [model["fn"], model["tp"]],
        ]
    )
    # Normalise per-row so colour intensity is comparable across thresholds
    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / row_totals

    ax.imshow(matrix_norm, cmap=CMAP, vmin=0, vmax=1, aspect="equal")

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            colour = "white" if matrix_norm[i, j] > 0.55 else "#222222"
            ax.text(
                j,
                i - 0.12,
                labels[i][j],
                ha="center",
                va="center",
                color=colour,
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                j,
                i + 0.18,
                f"{value:,}",
                ha="center",
                va="center",
                color=colour,
                fontsize=10,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNon-sepsis", "Predicted\nSepsis"])
    ax.set_yticklabels(["Actual\nNon-sepsis", "Actual\nSepsis"])
    ax.set_title(model["title"], pad=10)

    # Metric strip beneath the matrix
    metrics = derive_metrics(model["tp"], model["fn"], model["fp"], model["tn"])
    metric_text = (
        f"Sensitivity {metrics['Sensitivity']:.1%}   "
        f"Specificity {metrics['Specificity']:.1%}   "
        f"PPV {metrics['PPV']:.1%}   "
        f"FPR {metrics['FPR']:.1%}"
    )
    ax.text(
        0.5,
        -0.32,
        metric_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def build_figure(output_path: Path) -> Path:
    """Render the three-panel figure and write it to disk."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))

    for ax, model in zip(axes, MODELS):
        plot_confusion_panel(ax, model)

    suptitle = (
        "Confusion matrices for the V13b stacked-ensemble XGBoost meta-learner "
        "across operating points (1-hour resolution)"
    )
    # EDIT: n positives = 13,261 and n negatives = 726,673 are hardcoded class counts
    subtitle = (
        f"Shared discrimination: ROC AUC = {SHARED_METRICS['ROC AUC']:.4f}   "
        f"AUPRC = {SHARED_METRICS['AUPRC']:.4f}   "
        f"(n positives = 13,261; n negatives = 726,673)"
    )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, 0.96, subtitle, ha="center", fontsize=9.5, color="#444444")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    # EDIT: output PNG filename for the confusion-matrix figure
    png_path = build_figure(out_dir / "v13b_xgboost_threshold_confusion_matrices.png")
    print(f"Saved figure to: {png_path}")
