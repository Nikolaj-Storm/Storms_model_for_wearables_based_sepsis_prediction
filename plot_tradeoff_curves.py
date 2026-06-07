# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  plot_tradeoff_curves.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Render a two-panel ROC and Precision-Recall trade-off figure comparing the
#    V13b stacked ensemble against the Danish TOKS 2.1 proxy at 4-hour
#    resolution and the 6-hour Sepsis-3 horizon. Operating points are read from
#    the Excel workbook, full sweeps from a CSV and a parquet, plus a bi-normal
#    ROC fit and other estimators as default-threshold markers.
#
#  INPUTS
#    /PATH/TO/PROJECT/all results updated.xlsx  (operating points)
#    /PATH/TO/PROJECT/technical/Results/V13b_final/4_hour/
#      threshold_sweep_4_hour_is_sepsis_6h_xgb.csv  (V13b empirical sweep)
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_TOKS_danish.parquet
#      (TOKS per-row scores)
#  OUTPUTS
#    /PATH/TO/PROJECT/tradeoff_v13b_vs_TOKS_4h_6h.png
#    Also prints empirical and fitted metrics to the console.
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    EXCEL_PATH        -  results workbook to read operating points from
#    OUTPUT_PATH       -  PNG figure path to write
#    SHEET             -  worksheet name (6h Sepsis Prediction)
#    V13B_SWEEP_CSV    -  V13b threshold-sweep CSV
#    TOKS_PARQUET      -  TOKS per-row score parquet
#    TOKS_SCORE_COL    -  TOKS score column name
#    TARGET_COL        -  label column (is_sepsis_6h or is_sepsis_12h)
#    V13B_TARGETS / TOKS_TARGETS / OTHER_ESTIMATOR_TARGETS  -  operating-point
#                         (Version, Model, Variant) match tuples
#    F1_ISOLEVELS      -  F1 iso-curve levels overlaid on the PR panel
#    RESOLUTION_HEADER -  Excel block header to read (4-HOUR RESOLUTION)
#    savefig dpi       -  figure DPI passed to plt.savefig
#
#  REQUIRES: numpy, openpyxl, matplotlib, scipy, pandas
# ============================================================================
"""
TOKS vs V13b trade-off curve plotter.

Reads operating points from `all results updated.xlsx`, plots TP-vs-FP and
TPR-vs-FPR trade-off curves for V13b Stacked Ensemble (XGBoost meta) and the
Danish TOKS 2.1 proxy at 4-hour resolution, 6-hour Sepsis-3 horizon.

Each system gets a piecewise-linear curve through the operating points
extracted from the Excel, plus anchor points at (0, 0) and (total negatives,
total positives) for the no-flag and flag-everything corners.

Author: Author. Run `python plot_tradeoff_curves.py` from /PATH/TO/PROJECT.
"""

from pathlib import Path
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# EDIT: results workbook and output figure paths
EXCEL_PATH = Path("/PATH/TO/PROJECT/all results updated.xlsx")
OUTPUT_PATH = Path("/PATH/TO/PROJECT/tradeoff_v13b_vs_TOKS_4h_6h.png")
SHEET = "6h Sepsis Prediction"  # EDIT: worksheet name

# Empirical sweep sources, used to draw the full-resolution curves.
# EDIT: V13b sweep CSV and TOKS score parquet paths
V13B_SWEEP_CSV = Path("/PATH/TO/PROJECT/technical/Results/V13b_final/4_hour/threshold_sweep_4_hour_is_sepsis_6h_xgb.csv")
TOKS_PARQUET = Path("/PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_TOKS_danish.parquet")
TOKS_SCORE_COL = "TOKS_total"  # EDIT: TOKS score column (Region Nordjylland 2025)
TARGET_COL = "is_sepsis_6h"  # EDIT: change to "is_sepsis_12h" for the 12-hour figure

# Operating points to extract per system. The (Version, Model, Variant) tuple
# uniquely identifies each row in the 4-hour block of the sheet.
# EDIT: V13b operating-point match tuples
V13B_TARGETS = [
    ("Stacked Ensemble", "XGBoost (meta)", "F1-maximising threshold (primary)"),
    ("Stacked Ensemble", "XGBoost (meta)", "Max sensitivity, FPR < 50%"),
    ("Stacked Ensemble", "XGBoost (meta)", "Max TP, FPR < 20%"),
]

# EDIT: TOKS operating-point match tuples
TOKS_TARGETS = [
    ("Rule-Based Baseline", "Danish TOKS 2.1 proxy (Nemati 2018 replication", "Threshold >= 1 (Grøn)"),
    ("Rule-Based Baseline", "Danish TOKS 2.1 proxy (Nemati 2018 replication", "Threshold >= 6 (Gul)"),
    ("Rule-Based Baseline", "Danish TOKS 2.1 proxy (Nemati 2018 replication", "Threshold >= 8 (Orange)"),
    ("Rule-Based Baseline", "Danish TOKS 2.1 proxy (Nemati 2018 replication", "Threshold >= 10 (Rød)"),
]

# Other estimators to drop in as single (FPR, TPR) and (Recall, Precision)
# points at their default 0.5-threshold runs. Each tuple is (Version, Model,
# Variant prefix, short label for the marker).
# EDIT: other-estimator operating-point match tuples
OTHER_ESTIMATOR_TARGETS = [
    ("Engineered Features (Single Model)", "XGBoost", "Default threshold (0.50)", "Engineered XGBoost"),
    ("Engineered Features (Single Model)", "Random Forest", "Default threshold (0.50)", "Engineered Random Forest"),
    ("Engineered Features (Single Model)", "Logistic Regression", "Default threshold (0.50)", "Engineered Logistic Reg."),
    ("Raw Vitals (No Feat. Eng.)", "XGBoost", "Default threshold (0.50)", "Raw XGBoost"),
    ("Raw Vitals (No Feat. Eng.)", "Random Forest", "Default threshold (0.50)", "Raw Random Forest"),
    ("Raw Vitals (No Feat. Eng.)", "Logistic Regression", "Default threshold (0.50)", "Raw Logistic Reg."),
]

# Per-estimator visual styles (color, marker shape, scatter size).
# Colour identifies the underlying algorithm (XGBoost, Random Forest, Logistic
# Regression); marker shape identifies the feature set (filled plus =
# engineered features, triangle = raw vitals only).
ESTIMATOR_STYLES = {
    "Engineered XGBoost":         ("#2ca02c", "P", 150),   # green plus
    "Engineered Random Forest":   ("#17becf", "P", 150),   # cyan plus
    "Engineered Logistic Reg.":   ("#8c564b", "P", 150),   # brown plus
    "Raw XGBoost":                ("#2ca02c", "^", 150),   # green triangle
    "Raw Random Forest":          ("#17becf", "^", 150),   # cyan triangle
    "Raw Logistic Reg.":          ("#8c564b", "^", 150),   # brown triangle
}

# EDIT: F1 iso-curve levels overlaid on the PR panel
F1_ISOLEVELS = [0.05, 0.10, 0.15]

# Resolution block to read from in the Excel. Switching to 1-HOUR for the
# updated TOKS 2.1 strict-protocol comparison.
RESOLUTION_HEADER = "4-HOUR RESOLUTION"  # EDIT: Excel resolution block header


# ---------------------------------------------------------------------------
# Excel parsing
# ---------------------------------------------------------------------------

def load_4h_rows(path: Path, sheet: str) -> list[dict]:
    """Return a list of dicts representing rows in the 4-hour resolution block."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))

    # Find the row index where the 4-hour block starts and where its column
    # header row sits.
    block_start = None
    for i, row in enumerate(rows):
        first_cell = (row[0] or "").strip() if isinstance(row[0], str) else ""
        if RESOLUTION_HEADER in first_cell:
            block_start = i
            break
    if block_start is None:
        raise RuntimeError(f"Could not find '{RESOLUTION_HEADER}' in sheet '{sheet}'.")

    header = rows[block_start + 1]
    column_index = {col: idx for idx, col in enumerate(header) if col is not None}

    # Required columns
    required = ["Version", "Model", "Variant / Operating Point",
                "TP", "FN", "FP", "TN", "AUROC", "Recall", "FPR", "Threshold"]
    for col in required:
        if col not in column_index:
            raise RuntimeError(f"Missing column '{col}' in 4-hour block.")

    # Extract rows after the header until the sheet ends
    parsed = []
    for row in rows[block_start + 2:]:
        if row[column_index["Version"]] is None:
            continue
        record = {col: row[idx] for col, idx in column_index.items()}
        parsed.append(record)
    return parsed


def find_operating_points(rows: list[dict], targets: list[tuple]) -> list[dict]:
    """Return the rows matching the (Version, Model, Variant) target tuples.

    Matching uses substring containment for Model (so "TOKS Proxy" matches
    "TOKS Proxy (full 7-variable, 4h cadence)") and prefix match for Variant.
    """
    out = []
    for version, model, variant in targets:
        for row in rows:
            row_version = str(row.get("Version", "")).strip()
            row_model = str(row.get("Model", "")).strip()
            row_variant = str(row.get("Variant / Operating Point", "")).strip()
            if (row_version == version
                    and model in row_model
                    and row_variant.startswith(variant)):
                out.append(row)
                break
        else:
            print(f"WARNING: did not find row for {version} | {model} | {variant}")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def binormal_roc(fpr, a, b):
    """Classical bi-normal ROC model, TPR = Phi(a + b * Phi^-1(FPR)).

    The bi-normal model parameterises a smooth ROC curve with two parameters,
    where `a` controls the curve's elevation (signal-to-noise) and `b` controls
    its symmetry. A perfect classifier has a -> infinity, a random classifier
    has a = 0 and b = 1. The resulting curve passes through (0, 0) and (1, 1)
    by construction, so it can be fitted on operating points alone without
    needing to include the anchors as data.
    """
    fpr_clipped = np.clip(np.asarray(fpr, dtype=float), 1e-9, 1.0 - 1e-9)
    return norm.cdf(a + b * norm.ppf(fpr_clipped))


def load_v13b_empirical(csv_path: Path, max_points: int | None = 50):
    """Load V13b's full threshold sweep, return arrays sorted by FPR ascending.

    If `max_points` is set, evenly downsample across the sorted array so the
    rendered curve stays clean without losing the curve's shape. Set to None
    to keep the full sweep.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Drop degenerate rows where TP+FN or FP+TN is zero (ambiguous metrics).
    df = df[(df["tp"] + df["fn"] > 0) & (df["fp"] + df["tn"] > 0)].copy()
    # Recompute totals to anchor the (TP, FP) → (TPR, FPR) → precision identity.
    total_pos = float((df["tp"] + df["fn"]).max())
    total_neg = float((df["fp"] + df["tn"]).max())
    fpr = df["fpr"].to_numpy()
    tpr = df["sensitivity"].to_numpy()
    precision = df["precision"].to_numpy()
    # Sort by FPR ascending for monotonic plotting.
    order = np.argsort(fpr)
    fpr, tpr, precision = fpr[order], tpr[order], precision[order]

    if max_points is not None and len(fpr) > max_points:
        # Evenly distributed indices across the sorted array.
        idx = np.linspace(0, len(fpr) - 1, max_points).round().astype(int)
        idx = np.unique(idx)  # dedupe in case max_points >= len(fpr)
        fpr, tpr, precision = fpr[idx], tpr[idx], precision[idx]

    return {
        "fpr": fpr, "tpr": tpr, "precision": precision,
        "total_pos": total_pos, "total_neg": total_neg,
    }


def load_toks_empirical(parquet_path: Path, target_col: str = "is_sepsis_6h"):
    """Compute TOKS sweep from the saved per-row scores. Returns full curve."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    df = df[[TOKS_SCORE_COL, target_col]].dropna()
    scores = df[TOKS_SCORE_COL].to_numpy()
    labels = df[target_col].to_numpy()

    total_pos = float(labels.sum())
    total_neg = float((labels == 0).sum())

    # Iterate every integer cutoff. Cutoff = k means flag if score >= k.
    # Range covers [min_score, max_score+1] so we capture the all-flag and
    # no-flag corners.
    cutoffs = np.arange(int(scores.min()), int(scores.max()) + 2)
    rows = []
    for k in cutoffs:
        flagged = scores >= k
        tp = float(((flagged) & (labels == 1)).sum())
        fp = float(((flagged) & (labels == 0)).sum())
        fn = total_pos - tp
        tn = total_neg - fp
        sensitivity = tp / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rows.append((k, tp, fp, fn, tn, sensitivity, fpr, precision))
    arr = np.array(rows, dtype=float)
    order = np.argsort(arr[:, 6])  # sort by FPR
    return {
        "cutoff": arr[order, 0].astype(int),
        "tpr": arr[order, 5],
        "fpr": arr[order, 6],
        "precision": arr[order, 7],
        "total_pos": total_pos, "total_neg": total_neg,
    }


def fit_binormal(fpr_data, tpr_data):
    """Fit the bi-normal ROC model to operating-point data via least squares.

    Returns (a, b, auc_from_fit). The closed-form AUC of a bi-normal curve is
    Phi(a / sqrt(1 + b^2)).
    """
    popt, _ = curve_fit(binormal_roc, fpr_data, tpr_data, p0=[1.0, 1.0],
                        maxfev=20000)
    a, b = popt
    auc_fit = norm.cdf(a / np.sqrt(1.0 + b ** 2))
    return float(a), float(b), float(auc_fit)


def build_curve(points: list[dict], empirical: dict):
    """Combine empirical sweep with highlighted operating points and a bi-normal fit.

    The empirical sweep is the primary curve, namely the full threshold-by-threshold
    measurement plotted at native resolution. The bi-normal fit is computed across
    the empirical points so the parametric formula is anchored by the full sweep
    rather than just three operating points. AUROC and AUPRC come from the empirical
    curve via numerical integration.
    """
    fp = np.array([p["FP"] for p in points], dtype=float)
    tp = np.array([p["TP"] for p in points], dtype=float)
    fn = np.array([p["FN"] for p in points], dtype=float)
    tn = np.array([p["TN"] for p in points], dtype=float)
    precision_obs = np.array([p["Precision (PPV)"] for p in points], dtype=float)

    total_pos = empirical["total_pos"]
    total_neg = empirical["total_neg"]
    prevalence = total_pos / (total_pos + total_neg)

    # Highlighted operating points, in normalised space.
    fpr_obs = fp / total_neg
    tpr_obs = tp / total_pos

    # Empirical curve, sorted by FPR ascending.
    fpr_emp = empirical["fpr"]
    tpr_emp = empirical["tpr"]
    precision_emp = empirical["precision"]

    # Bi-normal fit on the empirical curve. Drop the degenerate corners where
    # FPR or TPR are exactly 0 or 1, since Phi^-1 is undefined there.
    mask = (fpr_emp > 1e-6) & (fpr_emp < 1 - 1e-6) & (tpr_emp > 1e-6) & (tpr_emp < 1 - 1e-6)
    a, b, auc_fit = fit_binormal(fpr_emp[mask], tpr_emp[mask])

    # Smooth bi-normal projection over the full FPR range.
    fpr_smooth = np.linspace(1e-4, 1.0 - 1e-4, 500)
    tpr_smooth = binormal_roc(fpr_smooth, a, b)
    precision_smooth = (total_pos * tpr_smooth) / (
        total_pos * tpr_smooth + total_neg * fpr_smooth)

    # AUROC and AUPRC from the empirical curve directly via trapezoid integration.
    auroc_emp = float(np.trapz(tpr_emp, fpr_emp))
    # AUPRC is integrated as precision over recall, so sort by recall ascending.
    pr_order = np.argsort(tpr_emp)
    auprc_emp = float(np.trapz(precision_emp[pr_order], tpr_emp[pr_order]))

    auroc_reported = points[0].get("AUROC")
    auprc_reported = points[0].get("AUPRC")

    return {
        "fpr_obs": fpr_obs, "tpr_obs": tpr_obs, "precision_obs": precision_obs,
        "fpr_emp": fpr_emp, "tpr_emp": tpr_emp, "precision_emp": precision_emp,
        "fpr_smooth": fpr_smooth, "tpr_smooth": tpr_smooth,
        "precision_smooth": precision_smooth,
        "a": a, "b": b,
        "auc_fit": auc_fit, "auroc_emp": auroc_emp,
        "auprc_emp": auprc_emp,
        "auroc_reported": auroc_reported, "auprc_reported": auprc_reported,
        "total_pos": total_pos, "total_neg": total_neg,
        "prevalence": prevalence,
    }


def combined_label(name: str, c: dict) -> str:
    """Legend label combining AUROC and AUPRC for the shared bottom legend."""
    return f"{name}, AUROC {c['auroc_emp']:.4f}, AUPRC {c['auprc_emp']:.4f}"


def split_by_observed_range(x_smooth, y_smooth, x_obs):
    """Split a smooth curve into in-range (anchored) and out-of-range (extrapolated) segments.

    The "in-range" segment lies between the smallest and largest observed
    x-coordinates, where the bi-normal fit is anchored by data. Beyond that
    range, the curve is purely model extrapolation.
    """
    x_min = float(np.min(x_obs))
    x_max = float(np.max(x_obs))
    in_range = (x_smooth >= x_min) & (x_smooth <= x_max)
    return in_range, (x_min, x_max)


def plot_curve_with_extrap(ax, x_smooth, y_smooth, x_obs, color, label):
    """Plot a smooth curve with solid in-range and dashed extrapolation segments."""
    in_range, _ = split_by_observed_range(x_smooth, y_smooth, x_obs)
    # Solid in-range line carries the legend entry.
    ax.plot(x_smooth[in_range], y_smooth[in_range], linewidth=2.2,
            color=color, label=label)
    # Dashed extrapolation segments (no legend entry to avoid clutter).
    below = x_smooth < float(np.min(x_obs))
    above = x_smooth > float(np.max(x_obs))
    if below.any():
        ax.plot(x_smooth[below], y_smooth[below], linewidth=1.6,
                color=color, linestyle="--", alpha=0.55)
    if above.any():
        ax.plot(x_smooth[above], y_smooth[above], linewidth=1.6,
                color=color, linestyle="--", alpha=0.55)


def f1_isocurve(f1: float, n_points: int = 200):
    """Return (recall, precision) arrays tracing a constant-F1 contour.

    Solves F1 = 2*P*R / (P + R) for P given fixed F1, P = F1*R / (2R - F1),
    valid for R > F1/2. Below that bound the curve is undefined.
    """
    r = np.linspace(f1 / 2 + 1e-4, 1.0, n_points)
    p = (f1 * r) / (2 * r - f1)
    return r, p


def plot_tradeoffs(v, t, v13b_points, toks_points, other_points, out_path: Path):
    # Global text-size bumps so the figure reads well at thesis-print scale.
    title_fs = 14
    axis_fs = 12
    tick_fs = 11
    legend_fs = 10
    annot_fs = 10

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ---- Panel 1, ROC ----
    ax = axes[0]
    # Empirical primary curves (full-resolution sweeps).
    ax.plot(v["fpr_emp"], v["tpr_emp"], linewidth=2.4, color="#1f77b4")
    ax.plot(t["fpr_emp"], t["tpr_emp"], linewidth=2.4, color="#ff7f0e")
    # Highlighted operating points.
    ax.scatter(v["fpr_obs"], v["tpr_obs"], color="#1f77b4", s=60, zorder=3, edgecolor="white")
    ax.scatter(t["fpr_obs"], t["tpr_obs"], color="#ff7f0e", s=60, marker="s",
               zorder=3, edgecolor="white")
    # Random classifier reference.
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, linewidth=1)

    for p in v13b_points:
        ax.annotate(f"thr {p['Threshold']:.3f}",
                    (p["FP"] / v["total_neg"], p["TP"] / v["total_pos"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=annot_fs)
    for p in toks_points:
        ax.annotate(f"≥{int(p['Threshold'])}",
                    (p["FP"] / t["total_neg"], p["TP"] / t["total_pos"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=annot_fs)

    ax.set_xlabel("False Positive Rate, FP / (FP + TN)", fontsize=axis_fs)
    ax.set_ylabel("True Positive Rate, TP / (TP + FN)", fontsize=axis_fs)
    ax.set_title("ROC trade-off\n4-hour resolution, 6-hour Sepsis-3 horizon", fontsize=title_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Other estimators at default 0.5 threshold, ROC space, per-estimator
    # colour and marker shape.
    for op in other_points:
        color, shape, marker_size = ESTIMATOR_STYLES.get(op["_label"], ("grey", "x", 130))
        fpr_op = op["FP"] / v["total_neg"]
        tpr_op = op["TP"] / v["total_pos"]
        ax.scatter(fpr_op, tpr_op, color=color, s=marker_size, marker=shape, zorder=4,
                   edgecolor="white", linewidths=1.2)

    # ---- Panel 2, Precision-Recall ----
    ax = axes[1]

    # F1 iso-curves underneath everything else, drawn first so they sit behind
    # the empirical curves and points.
    for f1 in F1_ISOLEVELS:
        r_iso, p_iso = f1_isocurve(f1)
        # Restrict to plot range.
        mask = (p_iso >= 0) & (p_iso <= 1)
        ax.plot(r_iso[mask], p_iso[mask], linestyle=":", color="#1f77b4",
                linewidth=1.2, alpha=0.5)
        # Place a small label near the right edge of each iso-curve.
        r_label = 0.95
        if 2 * r_label - f1 > 0:
            p_label = (f1 * r_label) / (2 * r_label - f1)
            if 0 < p_label < 1:
                ax.text(r_label, p_label, f"F1 = {f1:.2f}", fontsize=annot_fs,
                        color="#1f77b4", alpha=0.85,
                        verticalalignment="bottom", horizontalalignment="right")

    ax.plot(v["tpr_emp"], v["precision_emp"], linewidth=2.4, color="#1f77b4")
    ax.plot(t["tpr_emp"], t["precision_emp"], linewidth=2.4, color="#ff7f0e")
    # Highlighted operating points.
    ax.scatter(v["tpr_obs"], v["precision_obs"], color="#1f77b4", s=60,
               zorder=3, edgecolor="white")
    ax.scatter(t["tpr_obs"], t["precision_obs"], color="#ff7f0e", s=60, marker="s",
               zorder=3, edgecolor="white")
    # Dummy classifier line, grey dotted at prevalence.
    ax.axhline(0.025, color="grey", linestyle=":", alpha=0.7, linewidth=1.2)  # EDIT: prevalence reference line

    for p in v13b_points:
        ax.annotate(f"thr {p['Threshold']:.3f}",
                    (p["Recall"], p["Precision (PPV)"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=annot_fs)
    for p in toks_points:
        ax.annotate(f"≥{int(p['Threshold'])}",
                    (p["Recall"], p["Precision (PPV)"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=annot_fs)

    # Other estimators at default 0.5 threshold, PR space, per-estimator
    # colour and marker shape.
    for op in other_points:
        color, shape, marker_size = ESTIMATOR_STYLES.get(op["_label"], ("grey", "x", 130))
        recall_op = op["Recall"]
        precision_op = op["Precision (PPV)"]
        ax.scatter(recall_op, precision_op, color=color, s=marker_size,
                   marker=shape, zorder=4, edgecolor="white", linewidths=1.2)

    ax.set_xlabel("Recall, TP / (TP + FN)", fontsize=axis_fs)
    ax.set_ylabel("Precision, TP / (TP + FP)", fontsize=axis_fs)
    ax.set_title("Precision-Recall trade-off\n4-hour resolution, 6-hour Sepsis-3 horizon", fontsize=title_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    # Clip y-axis so the empirical curves stay readable. F1 iso-curves spike
    # vertically near recall = F1/2 and would otherwise dominate the panel.
    ax.set_ylim(-0.005, 0.22)

    # ---- Single shared legend below both panels ----
    from matplotlib.lines import Line2D

    bottom_fs = 12  # bumped above the previous in-chart legend size

    def legend_marker_size(scatter_size: int) -> int:
        # Convert scatter `s` (point area) to legend `markersize`.
        return max(9, int(np.sqrt(scatter_size) * 1.0))

    # Build all handles. matplotlib fills the legend column-major, so order
    # the list to land each item in the right column-row position.
    empty_handle = Line2D([0], [0], color="none", label="")

    line_se = Line2D([0], [0], color="#1f77b4", linewidth=2.6,
                     label=combined_label("Stacked Ensemble (XGB meta)", v))
    line_toks = Line2D([0], [0], color="#ff7f0e", linewidth=2.6,
                       label=combined_label("Danish TOKS 2.1 proxy", t))
    line_random = Line2D([0], [0], color="black", linestyle=":", linewidth=1.4,
                         label="random classifier (TPR = FPR)")
    line_dummy = Line2D([0], [0], color="grey", linestyle=":", linewidth=1.4,
                        label="Dummy Classifier (precision ≈ 0.025)")

    estimator_handles = {
        name: Line2D([0], [0], marker=shape, color="w", markerfacecolor=color,
                     markeredgecolor="white",
                     markersize=legend_marker_size(size), label=name)
        for name, (color, shape, size) in ESTIMATOR_STYLES.items()
    }

    # Three columns, three rows. Column-major fill order.
    # Col 1, the two curves with metrics. Col 2, the two reference lines.
    # Col 3, the engineered-feature estimators. Col 4, the raw-vitals estimators.
    handles = [
        # Col 1 (3 cells)
        line_se, line_toks, empty_handle,
        # Col 2 (3 cells)
        line_random, line_dummy, empty_handle,
        # Col 3 (3 cells)
        estimator_handles["Engineered XGBoost"],
        estimator_handles["Engineered Random Forest"],
        estimator_handles["Engineered Logistic Reg."],
        # Col 4 (3 cells)
        estimator_handles["Raw XGBoost"],
        estimator_handles["Raw Random Forest"],
        estimator_handles["Raw Logistic Reg."],
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        fontsize=bottom_fs,
        frameon=True,
        handlelength=2.5,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    # Reserve room at the bottom for the legend, then save.
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")  # EDIT: figure DPI
    print(f"Saved figure to {out_path}")
    print(f"\nEmpirical and fitted metrics")
    print(f"  V13b: empirical AUROC {v['auroc_emp']:.4f}, AUPRC {v['auprc_emp']:.4f}")
    print(f"        bi-normal a {v['a']:.4f}, b {v['b']:.4f}, AUC {v['auc_fit']:.4f}")
    print(f"  TOKS: empirical AUROC {t['auroc_emp']:.4f}, AUPRC {t['auprc_emp']:.4f}")
    print(f"        bi-normal a {t['a']:.4f}, b {t['b']:.4f}, AUC {t['auc_fit']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = load_4h_rows(EXCEL_PATH, SHEET)
    v13b_points = find_operating_points(rows, V13B_TARGETS)
    toks_points = find_operating_points(rows, TOKS_TARGETS)

    if len(v13b_points) != len(V13B_TARGETS):
        raise RuntimeError(f"Expected {len(V13B_TARGETS)} V13b points, got {len(v13b_points)}.")
    if len(toks_points) != len(TOKS_TARGETS):
        raise RuntimeError(f"Expected {len(TOKS_TARGETS)} TOKS points, got {len(toks_points)}.")

    print("V13b highlighted operating points (from Excel):")
    for p in v13b_points:
        print(f"  thr {p['Threshold']:.4f}, TP={p['TP']}, FP={p['FP']}, "
              f"recall={p['Recall']:.4f}, FPR={p['FPR']:.4f}")

    print("\nTOKS highlighted operating points (from Excel):")
    for p in toks_points:
        print(f"  thr ≥{int(p['Threshold'])}, TP={p['TP']}, FP={p['FP']}, "
              f"recall={p['Recall']:.4f}, FPR={p['FPR']:.4f}")

    print("\nLoading empirical sweeps...")
    v_emp = load_v13b_empirical(V13B_SWEEP_CSV)
    t_emp = load_toks_empirical(TOKS_PARQUET, target_col=TARGET_COL)
    print(f"  V13b sweep: {len(v_emp['fpr'])} thresholds")
    print(f"  TOKS sweep: {len(t_emp['fpr'])} cutoffs")

    v = build_curve(v13b_points, v_emp)
    t = build_curve(toks_points, t_emp)

    # Pull the additional 0.5-threshold rows. Each carries its short label.
    other_points = []
    for version, model, variant, label in OTHER_ESTIMATOR_TARGETS:
        for row in rows:
            if (str(row.get("Version", "")).strip() == version
                    and model in str(row.get("Model", ""))
                    and str(row.get("Variant / Operating Point", "")).strip().startswith(variant)):
                row = dict(row)
                row["_label"] = label
                other_points.append(row)
                break
        else:
            print(f"WARNING: did not find {version} | {model} | {variant}")

    print(f"\nOther estimator points: {len(other_points)}")
    for op in other_points:
        print(f"  {op['_label']:>12s}, recall {op['Recall']:.4f}, "
              f"precision {op['Precision (PPV)']:.4f}, FPR {op['FPR']:.4f}")

    plot_tradeoffs(v, t, v13b_points, toks_points, other_points, OUTPUT_PATH)


if __name__ == "__main__":
    main()
