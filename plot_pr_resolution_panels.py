# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  plot_pr_resolution_panels.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Render Precision-Recall panels for the 4-hour, 1-hour, and 15-minute
#    sampling cadences. Each panel overlays the V13b stacked-ensemble sweep,
#    engineered and raw XGB/RF/LR curves from cached predictions, F1 iso-curves
#    and a prevalence line. The Danish TOKS 2.1 proxy step curve appears only on
#    the 4-hour panel. Also writes a combined vertically stacked figure.
#
#  INPUTS
#    /PATH/TO/PROJECT/all results updated.xlsx  (operating points)
#    /PATH/TO/PROJECT/technical/Results/V13b_final/<res>/
#      threshold_sweep_<res>_is_sepsis_6h_xgb.csv  (V13b sweeps)
#    /PATH/TO/PROJECT/pr_curve_predictions/preds_<res>_<featset>_<algo>.parquet
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_TOKS_danish.parquet
#  OUTPUTS
#    /PATH/TO/PROJECT/pr_panel_4_hour_6h.png
#    /PATH/TO/PROJECT/pr_panel_1_hour_6h.png
#    /PATH/TO/PROJECT/pr_panel_15_min_6h.png
#    /PATH/TO/PROJECT/pr_panel_stacked_6h.png
#    Also prints per-panel metrics to the console.
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    _CANDIDATE_BASES  -  project base-path candidate list, first existing wins
#    EXCEL_PATH        -  results workbook to read operating points from
#    SHEET             -  worksheet name (6h Sepsis Prediction)
#    TARGET_COL        -  label column (is_sepsis_6h)
#    V13B_SWEEP        -  per-resolution V13b threshold-sweep CSV paths
#    PRED_DIR          -  cached prediction parquet directory
#    TOKS_PARQUET      -  TOKS per-row score parquet
#    TOKS_SCORE_COL    -  TOKS score column name
#    RESOLUTION_HEADERS / RESOLUTION_TITLES  -  Excel block headers and titles
#    OUTPUT_PATHS / COMBINED_OUTPUT  -  per-panel and combined PNG paths
#    V13B_TARGETS / TOKS_TARGETS  -  operating-point match tuples
#    F1_ISOLEVELS      -  F1 iso-curve levels overlaid on each panel
#    savefig dpi       -  figure DPI passed to plt.savefig
#
#  REQUIRES: numpy, pandas, openpyxl, matplotlib, scikit-learn
# ============================================================================
"""
Three-panel resolution sweep, Precision-Recall trade-off only.

For each of 4-hour, 1-hour, 15-minute sampling cadences, this script renders a
single Precision-Recall panel showing:
  * V13b stacked ensemble (XGBoost meta) empirical PR sweep,
  * Engineered XGBoost / Random Forest / Logistic Regression as solid colour-
    coded curves built from cached test-set predicted probabilities,
  * Raw-vitals XGBoost / Random Forest / Logistic Regression as dotted curves
    in the same colour-by-algorithm scheme so the engineered-vs-raw shift is
    visible at a glance,
  * F1 = 0.05 / 0.10 / 0.15 iso-curves and prevalence reference line.

The Danish TOKS 2.1 proxy step curve is included only on the 4-hour panel,
because the proxy is defined on the 4-hour cadence the actual ward charts use.

Run:
    python plot_pr_resolution_panels.py
"""

from pathlib import Path

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# EDIT: project base-path candidate list, first existing path is used
_CANDIDATE_BASES = [
    Path("/PATH/TO/PROJECT"),
]
BASE = next((p for p in _CANDIDATE_BASES if p.exists()), _CANDIDATE_BASES[0])

EXCEL_PATH = BASE / "all results updated.xlsx"  # EDIT: results workbook path
SHEET = "6h Sepsis Prediction"  # EDIT: worksheet name
TARGET_COL = "is_sepsis_6h"  # EDIT: label column

# EDIT: per-resolution V13b threshold-sweep CSV paths
V13B_SWEEP = {
    "4_hour": BASE / "technical/Results/V13b_final/4_hour/threshold_sweep_4_hour_is_sepsis_6h_xgb.csv",
    "1_hour": BASE / "technical/Results/V13b_final/1_hour/threshold_sweep_1_hour_is_sepsis_6h_xgb.csv",
    "15_min": BASE / "technical/Results/V13b_final/15_min/threshold_sweep_15_min_is_sepsis_6h_xgb.csv",
}

PRED_DIR = BASE / "pr_curve_predictions"  # EDIT: cached prediction parquet directory

# EDIT: TOKS per-row score parquet and score column
TOKS_PARQUET = BASE / "technical/Data/v3_dataset_4h_test_TOKS_danish.parquet"
TOKS_SCORE_COL = "TOKS_total"

# EDIT: Excel resolution block headers
RESOLUTION_HEADERS = {
    "4_hour": "4-HOUR RESOLUTION",
    "1_hour": "1-HOUR RESOLUTION",
    "15_min": "15-MINUTE RESOLUTION",
}

RESOLUTION_TITLES = {
    "4_hour": "4-hour resolution, 6-hour Sepsis-3 horizon",
    "1_hour": "1-hour resolution, 6-hour Sepsis-3 horizon",
    "15_min": "15-minute resolution, 6-hour Sepsis-3 horizon",
}

# EDIT: per-panel and combined output PNG paths
OUTPUT_PATHS = {
    "4_hour": BASE / "pr_panel_4_hour_6h.png",
    "1_hour": BASE / "pr_panel_1_hour_6h.png",
    "15_min": BASE / "pr_panel_15_min_6h.png",
}
COMBINED_OUTPUT = BASE / "pr_panel_stacked_6h.png"

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

# Colour identifies the algorithm; alpha distinguishes feature set
# (engineered = full opacity, raw = half-transparent solid line).
ENG_ALPHA = 1.0
RAW_ALPHA = 0.22

# Engineered curves use a darker nuance of the same hue as the raw curves so
# the family relationship stays visible while the engineered tier dominates.
ESTIMATOR_STYLES = {
    "engineered_xgb": {"color": "#125c12", "alpha": ENG_ALPHA, "label": "Eng. XGB"},
    "engineered_rf":  {"color": "#17becf", "alpha": ENG_ALPHA, "label": "Eng. RF"},
    "engineered_lr":  {"color": "#4a2820", "alpha": ENG_ALPHA, "label": "Eng. LR"},
    "raw_xgb":        {"color": "#2ca02c", "alpha": RAW_ALPHA, "label": "Raw XGB"},
    "raw_rf":         {"color": "#17becf", "alpha": RAW_ALPHA, "label": "Raw RF"},
    "raw_lr":         {"color": "#8c564b", "alpha": RAW_ALPHA, "label": "Raw LR"},
}

ESTIMATOR_KEYS = list(ESTIMATOR_STYLES.keys())

# EDIT: F1 iso-curve levels overlaid on each panel
F1_ISOLEVELS = [0.05, 0.10, 0.15]


# ---------------------------------------------------------------------------
# Excel parsing (kept only for V13b and TOKS operating-point markers)
# ---------------------------------------------------------------------------

def load_block_rows(path: Path, sheet: str, resolution_header: str) -> list[dict]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))

    block_start = None
    for i, row in enumerate(rows):
        first_cell = (row[0] or "").strip() if isinstance(row[0], str) else ""
        if resolution_header in first_cell:
            block_start = i
            break
    if block_start is None:
        raise RuntimeError(f"Could not find '{resolution_header}' in sheet '{sheet}'.")

    other_headers = [h for h in RESOLUTION_HEADERS.values() if h != resolution_header]
    block_end = len(rows)
    for j in range(block_start + 1, len(rows)):
        first_cell = (rows[j][0] or "").strip() if isinstance(rows[j][0], str) else ""
        if any(h in first_cell for h in other_headers):
            block_end = j
            break

    header = rows[block_start + 1]
    column_index = {col: idx for idx, col in enumerate(header) if col is not None}
    required = ["Version", "Model", "Variant / Operating Point",
                "TP", "FN", "FP", "TN", "AUROC", "AUPRC", "Recall", "FPR", "Threshold",
                "Precision (PPV)"]
    for col in required:
        if col not in column_index:
            raise RuntimeError(f"Missing column '{col}' in '{resolution_header}' block.")

    parsed = []
    for row in rows[block_start + 2: block_end]:
        if row[column_index["Version"]] is None:
            continue
        record = {col: row[idx] for col, idx in column_index.items()}
        parsed.append(record)
    return parsed


def find_operating_points(rows: list[dict], targets: list[tuple]) -> list[dict]:
    out = []
    for entry in targets:
        version, model, variant = entry[:3]
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
# Curves and metrics
# ---------------------------------------------------------------------------

def load_v13b_empirical(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[(df["tp"] + df["fn"] > 0) & (df["fp"] + df["tn"] > 0)].copy()
    total_pos = float((df["tp"] + df["fn"]).max())
    total_neg = float((df["fp"] + df["tn"]).max())
    fpr = df["fpr"].to_numpy()
    tpr = df["sensitivity"].to_numpy()
    precision = df["precision"].to_numpy()
    order = np.argsort(fpr)
    fpr, tpr, precision = fpr[order], tpr[order], precision[order]
    return {
        "fpr": fpr, "tpr": tpr, "precision": precision,
        "total_pos": total_pos, "total_neg": total_neg,
    }


def load_toks_empirical(parquet_path: Path, target_col: str = "is_sepsis_6h"):
    df = pd.read_parquet(parquet_path)
    df = df[[TOKS_SCORE_COL, target_col]].dropna()
    scores = df[TOKS_SCORE_COL].to_numpy()
    labels = df[target_col].to_numpy()
    total_pos = float(labels.sum())
    total_neg = float((labels == 0).sum())
    cutoffs = np.arange(int(scores.min()), int(scores.max()) + 2)
    rows = []
    for k in cutoffs:
        flagged = scores >= k
        tp = float(((flagged) & (labels == 1)).sum())
        fp = float(((flagged) & (labels == 0)).sum())
        sensitivity = tp / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rows.append((k, sensitivity, fpr, precision))
    arr = np.array(rows, dtype=float)
    order = np.argsort(arr[:, 2])
    return {
        "cutoff": arr[order, 0].astype(int),
        "tpr": arr[order, 1],
        "fpr": arr[order, 2],
        "precision": arr[order, 3],
        "total_pos": total_pos, "total_neg": total_neg,
    }


def auroc_auprc_from_empirical(empirical: dict) -> tuple[float, float]:
    fpr = empirical["fpr"]
    tpr = empirical["tpr"]
    precision = empirical["precision"]
    auroc = float(np.trapezoid(tpr, fpr))
    pr_order = np.argsort(tpr)
    auprc = float(np.trapezoid(precision[pr_order], tpr[pr_order]))
    return auroc, auprc


def f1_isocurve(f1: float, n_points: int = 200):
    r = np.linspace(f1 / 2 + 1e-4, 1.0, n_points)
    p = (f1 * r) / (2 * r - f1)
    return r, p


def estimator_pred_path(resolution: str, key: str) -> Path:
    return PRED_DIR / f"preds_{resolution}_{key.replace('_', '_', 1)}.parquet"


def load_estimator_curve(resolution: str, key: str) -> dict | None:
    """Load cached predictions, build a PR curve, and return summary metrics."""
    feature_set, algorithm = key.split("_", 1)
    path = PRED_DIR / f"preds_{resolution}_{feature_set}_{algorithm}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    y = df["y_true"].to_numpy()
    p = df["y_prob"].to_numpy()
    precision, recall, _ = precision_recall_curve(y, p)
    auroc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    return {"recall": recall, "precision": precision,
            "auroc": auroc, "auprc": auprc, "n": len(y), "pos": int(y.sum())}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def gather_panel_data(resolution: str, include_toks: bool):
    rows = load_block_rows(EXCEL_PATH, SHEET, RESOLUTION_HEADERS[resolution])
    v13b_points = find_operating_points(rows, V13B_TARGETS)
    if len(v13b_points) != len(V13B_TARGETS):
        raise RuntimeError(
            f"[{resolution}] expected {len(V13B_TARGETS)} V13b rows, "
            f"got {len(v13b_points)}.")

    v_emp = load_v13b_empirical(V13B_SWEEP[resolution])
    v_auroc, v_auprc = auroc_auprc_from_empirical(v_emp)
    print(f"  V13b empirical: AUROC {v_auroc:.4f}, AUPRC {v_auprc:.4f}")

    estimator_curves = {}
    for key in ESTIMATOR_KEYS:
        c = load_estimator_curve(resolution, key)
        if c is None:
            print(f"  WARNING: missing predictions for {resolution} {key}")
            continue
        estimator_curves[key] = c
        print(f"  {key}: AUROC {c['auroc']:.4f}, AUPRC {c['auprc']:.4f}, n={c['n']:,}, pos={c['pos']:,}")

    t_emp = None
    t_auroc = t_auprc = None
    if include_toks:
        t_emp = load_toks_empirical(TOKS_PARQUET, target_col=TARGET_COL)
        t_auroc, t_auprc = auroc_auprc_from_empirical(t_emp)
        print(f"  TOKS empirical: AUROC {t_auroc:.4f}, AUPRC {t_auprc:.4f}")

    return {
        "v_emp": v_emp, "v_auroc": v_auroc, "v_auprc": v_auprc,
        "estimator_curves": estimator_curves,
        "t_emp": t_emp, "t_auroc": t_auroc, "t_auprc": t_auprc,
        "include_toks": include_toks,
    }


def render_panel(fig, ax, legend_ax, data: dict, resolution: str):
    title = RESOLUTION_TITLES[resolution]
    title_fs = 14
    axis_fs = 12
    tick_fs = 11
    annot_fs = 10
    legend_fs = 13

    v_emp = data["v_emp"]
    v_auprc = data["v_auprc"]
    estimator_curves = data["estimator_curves"]
    t_emp = data["t_emp"]
    t_auprc = data["t_auprc"]
    include_toks = data["include_toks"]

    # F1 iso-curves first so they sit behind everything.
    for f1 in F1_ISOLEVELS:
        r_iso, p_iso = f1_isocurve(f1)
        mask = (p_iso >= 0) & (p_iso <= 1)
        ax.plot(r_iso[mask], p_iso[mask], linestyle=":", color="#1f77b4",
                linewidth=1.0, alpha=0.4)
        r_label = 0.95
        if 2 * r_label - f1 > 0:
            p_label = (f1 * r_label) / (2 * r_label - f1)
            if 0 < p_label < 1:
                ax.text(r_label, p_label, f"F1 = {f1:.2f}", fontsize=annot_fs,
                        color="#1f77b4", alpha=0.7,
                        verticalalignment="bottom", horizontalalignment="right")

    # V13b empirical curve, drawn thick so it dominates the panel.
    ax.plot(v_emp["tpr"], v_emp["precision"], linewidth=2.6, color="#1f77b4",
            zorder=4)

    # TOKS empirical step curve.
    if include_toks and t_emp is not None:
        ax.plot(t_emp["tpr"], t_emp["precision"], linewidth=2.4, color="#ff7f0e",
                zorder=4)

    # Estimator full PR curves.
    for key in ESTIMATOR_KEYS:
        if key not in estimator_curves:
            continue
        c = estimator_curves[key]
        style = ESTIMATOR_STYLES[key]
        ax.plot(c["recall"], c["precision"],
                color=style["color"], linestyle="-",
                linewidth=1.8, alpha=style["alpha"], zorder=3)

    # Operating-point markers and threshold annotations have been removed for
    # the stacked ensemble and the TOKS proxy; the curves now read as smooth
    # trade-off frontiers without callouts.

    # Prevalence reference.
    prev = v_emp["total_pos"] / (v_emp["total_pos"] + v_emp["total_neg"])
    ax.axhline(prev, color="grey", linestyle=":", alpha=0.7, linewidth=1.2)

    ax.set_xlabel("Recall, TP / (TP + FN)", fontsize=axis_fs)
    ax.set_ylabel("Precision, TP / (TP + FP)", fontsize=axis_fs)
    # Title is placed in figure coordinates so it spans the PR axes plus the
    # legend column rather than centering only over the main axes.
    ax_pos = ax.get_position()
    legend_pos = legend_ax.get_position()
    title_x = (ax_pos.x0 + legend_pos.x1) / 2
    title_y = max(ax_pos.y1, legend_pos.y1) + 0.025
    fig.text(title_x, title_y,
             f"Precision-Recall trade-off, {title}",
             fontsize=title_fs, ha="center", va="bottom", weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.005, 0.22)

    # Custom three-column legend on the right (line sample, name, AUPRC).
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_edgecolor("#bbbbbb")
        spine.set_linewidth(0.8)

    # Build the row entries in display order.
    # Show AUPRC lift (AUPRC / prevalence) so resolutions are comparable;
    # the dummy classifier's lift is exactly 1.0 by construction.
    legend_rows = [
        {"name": "Stacked Ens.", "lift": v_auprc / prev,
         "color": "#1f77b4", "alpha": 1.0, "linestyle": "-", "lw": 2.6},
    ]
    if include_toks:
        legend_rows.append({"name": "TOKS 2.1", "lift": t_auprc / prev,
                            "color": "#ff7f0e", "alpha": 1.0, "linestyle": "-", "lw": 2.4})
    legend_rows.append({"name": "Dummy", "lift": 1.0,
                        "color": "grey", "alpha": 0.7, "linestyle": ":", "lw": 1.4})
    for key in ESTIMATOR_KEYS:
        if key not in estimator_curves:
            continue
        style = ESTIMATOR_STYLES[key]
        c = estimator_curves[key]
        legend_rows.append({"name": style["label"], "lift": c["auprc"] / prev,
                            "color": style["color"], "alpha": style["alpha"],
                            "linestyle": "-", "lw": 2.0})

    # Header band at the top of the legend. "AUPRC lift" is split onto two
    # stacked lines so the value column header fits inside its narrow column
    # without crossing the vertical divider.
    header_top_y = 0.965
    header_bot_y = 0.915
    underline_y = 0.88
    divider_x = 0.62
    legend_ax.text(0.04, (header_top_y + header_bot_y) / 2,
                   "Estimator", fontsize=legend_fs,
                   weight="bold", va="center", ha="left")
    legend_ax.text(0.97, header_top_y, "AUPRC",
                   fontsize=legend_fs, weight="bold", va="center", ha="right")
    legend_ax.text(0.97, header_bot_y, "lift",
                   fontsize=legend_fs, weight="bold", va="center", ha="right")
    legend_ax.plot([0.03, 0.97], [underline_y, underline_y],
                   color="#888888", linewidth=0.9)
    legend_ax.plot([divider_x, divider_x], [0.02, header_top_y + 0.025],
                   color="#888888", linewidth=0.9)

    # Distribute rows evenly between the underline and the bottom of the panel.
    n = len(legend_rows)
    top = underline_y - 0.04
    bottom = 0.05
    if n == 1:
        ys = [(top + bottom) / 2]
    else:
        ys = [top - i * (top - bottom) / (n - 1) for i in range(n)]

    line_x0, line_x1 = 0.03, 0.18
    name_x = 0.22
    value_x = 0.97
    for row, y in zip(legend_rows, ys):
        legend_ax.plot([line_x0, line_x1], [y, y],
                       color=row["color"], linestyle=row["linestyle"],
                       linewidth=row["lw"], alpha=row["alpha"],
                       solid_capstyle="round")
        legend_ax.text(name_x, y, row["name"], fontsize=legend_fs, va="center")
        legend_ax.text(value_x, y, f"{row['lift']:.2f}×",
                       fontsize=legend_fs, va="center", ha="right",
                       family="monospace")


def plot_pr_panel(resolution: str, include_toks: bool, output_path: Path):
    print(f"\n=== Building PR panel for {resolution}, include_toks={include_toks} ===")
    data = gather_panel_data(resolution, include_toks)

    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_axes([0.07, 0.10, 0.67, 0.82])
    legend_ax = fig.add_axes([0.77, 0.10, 0.21, 0.82], frameon=True)
    render_panel(fig, ax, legend_ax, data, resolution)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")  # EDIT: figure DPI
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_combined(output_path: Path):
    """Stack all three resolutions vertically into a single figure, 4-hour on top,
    1-hour middle, 15-minute bottom. Each row keeps its own legend on the right."""
    print(f"\n=== Building combined stacked figure ===")
    panels = [
        ("4_hour", True),
        ("1_hour", False),
        ("15_min", False),
    ]
    datas = []
    for res, inc in panels:
        print(f"  loading data for {res}")
        datas.append((res, gather_panel_data(res, inc)))

    n_rows = len(panels)
    fig = plt.figure(figsize=(13, 7.5 * n_rows))

    # Vertical layout — leave a small top/bottom margin and equal gaps between rows.
    margin_top = 0.025
    margin_bottom = 0.025
    inter_gap = 0.045
    total_h = 1.0 - margin_top - margin_bottom - inter_gap * (n_rows - 1)
    row_h = total_h / n_rows

    for i, (res, data) in enumerate(datas):
        # Row 0 sits at the top, so y_top decreases as i grows.
        y_top = 1.0 - margin_top - i * (row_h + inter_gap)
        y_bottom = y_top - row_h
        ax = fig.add_axes([0.07, y_bottom + 0.04 * row_h,
                           0.67, row_h * 0.92])
        legend_ax = fig.add_axes([0.77, y_bottom + 0.04 * row_h,
                                  0.21, row_h * 0.92], frameon=True)
        render_panel(fig, ax, legend_ax, data, res)

    plt.savefig(output_path, dpi=180, bbox_inches="tight")  # EDIT: figure DPI
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    plot_pr_panel("4_hour", include_toks=True, output_path=OUTPUT_PATHS["4_hour"])
    plot_pr_panel("1_hour", include_toks=False, output_path=OUTPUT_PATHS["1_hour"])
    plot_pr_panel("15_min", include_toks=False, output_path=OUTPUT_PATHS["15_min"])
    plot_combined(COMBINED_OUTPUT)


if __name__ == "__main__":
    main()
