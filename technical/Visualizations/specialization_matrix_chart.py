# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  specialization_matrix_chart.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Renders the 4x4 ensemble-by-target lift-over-random matrix as a single
#    grouped bar chart. Defaults to the approved hard-coded lift values; an
#    optional --from-csv path recomputes specialist rows from a results CSV
#    by dividing AUPRC by hard-coded cohort prevalences.
#
#  INPUTS
#    /PATH/TO/PROJECT/technical/Results/V13b_final/4_hour/specialisation_matrix_4_hour.csv
#      (only read when --from-csv is passed)
#  OUTPUTS
#    /PATH/TO/PROJECT/Drafts/figures/specialization_matrix_v3.png
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    savefig DPI     -  figure.dpi=200, savefig.dpi=200 (in rcParams)
#    SPEC_CSV        -  recompute-path input CSV (relative to repo root)
#    OUT_PNG         -  output figure path (relative to repo root)
#    PREVALENCE      -  hard-coded 4-hour cohort positive-share per target
#    LIFT_FALLBACK   -  hard-coded approved 4x4 lift matrix values
#
#  REQUIRES: matplotlib, numpy, pandas
# ============================================================================
"""
specialization_matrix_chart.py

Renders the 4x4 ensemble x target lift-over-random matrix as a single
grouped bar chart (formerly "panel (b)").

Three changes vs. the Excel original:
  1. The all-cause / general label is renamed "Global" everywhere
     (matching the ensemble of the same name); previously it appeared
     as "General" in the legend and "Global" on the x-axis.
  2. The "(b)" sub-panel prefix is removed from the title.
  3. The y-axis carries a vertical label naming the measurement.

Data sources
------------
The default lift matrix matches the approved figure
(Drafts/figures/specialization_payoff_4h_v2.xlsx) and the numbers
cited in the prose of Drafts/specialisation_analysis_section_draft.md
(e.g. respiratory 3.12, urinary 2.67, other 2.60).

A CSV-driven recompute path is available behind --from-csv: it reads
technical/Results/V13b_final/4_hour/specialisation_matrix_4_hour.csv
and divides AUPRC by the cohort prevalences hard-coded below. The
two paths produce slightly different specialist lifts because the
approved figure was computed against a different prevalence basis;
keep the default unless the prose is updated in lockstep.

The global ensemble's row is not in the CSV in either case and is
taken from the approved figure.

Usage
-----
    python specialization_matrix_chart.py             # approved values
    python specialization_matrix_chart.py --from-csv  # recompute path
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 200,        # EDIT: figure DPI
    "savefig.dpi": 200,       # EDIT: saved-figure DPI
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SPEC_CSV = (  # EDIT: recompute-path input CSV (relative to repo root)
    ROOT / "technical" / "Results" / "V13b_final" / "4_hour"
    / "specialisation_matrix_4_hour.csv"
)
OUT_PNG = ROOT / "Drafts" / "figures" / "specialization_matrix_v3.png"  # EDIT: output figure path
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ── Inputs ─────────────────────────────────────────────────────────────
# 4-hour resolution, V13b. "Global" replaces the prior "General" label.
PREVALENCE = {           # EDIT: share of positive rows for each target (hard-coded)
    "Global": 0.0245,
    "Respiratory": 0.0084,
    "Urinary": 0.0036,
    "Other": 0.0124,
}

ENSEMBLES = ["Global", "Respiratory", "Urinary", "Other"]
TARGETS = ["Global", "Respiratory", "Urinary", "Other"]

# Global ensemble row taken from the approved figure (xlsx).
# Specialist rows are pulled from the CSV below; this matrix is only
# used for the global row and as a fallback if the CSV is unavailable.
LIFT_FALLBACK = pd.DataFrame(  # EDIT: hard-coded approved 4x4 lift-over-random matrix
    [
        # Global, Resp, Uri, Other
        [2.49, 2.70, 2.58, 2.45],   # Global ensemble
        [2.35, 3.12, 2.05, 2.10],   # Respiratory ensemble
        [2.29, 2.27, 2.67, 2.27],   # Urinary ensemble
        [2.25, 1.85, 2.27, 2.60],   # Other ensemble
    ],
    index=ENSEMBLES,
    columns=TARGETS,
)


def load_lift_matrix(from_csv: bool = False) -> pd.DataFrame:
    """Return the 4x4 lift matrix.

    If from_csv is False (default) the matrix is the approved figure's
    values. If True, specialist rows are recomputed from the CSV using
    the cohort prevalences declared above; the global row is always
    taken from the approved figure.
    """
    if not from_csv or not SPEC_CSV.exists():
        return LIFT_FALLBACK.copy()

    raw = pd.read_csv(SPEC_CSV)
    label_to_target = {
        "is_sepsis_6h": "Global",
        "target_resp_6h": "Respiratory",
        "target_uri_6h": "Urinary",
        "target_other_6h": "Other",
    }
    ensemble_to_row = {
        "target_resp_6h": "Respiratory",
        "target_uri_6h": "Urinary",
        "target_other_6h": "Other",
    }

    matrix = LIFT_FALLBACK.copy()
    for _, row in raw.iterrows():
        ens = ensemble_to_row.get(row["ensemble"])
        tgt = label_to_target.get(row["label"])
        if ens is None or tgt is None:
            continue
        matrix.loc[ens, tgt] = row["AUPRC"] / PREVALENCE[tgt]
    return matrix


# ── Render ─────────────────────────────────────────────────────────────
def render(lift: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "Global": "#E07B7B",        # coral
        "Respiratory": "#7BC9A0",   # mint
        "Urinary": "#A47BC9",       # violet
        "Other": "#C9C97B",         # khaki
    }

    n_groups = len(ENSEMBLES)
    n_bars = len(TARGETS)
    bar_width = 0.18
    group_centres = np.arange(n_groups)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    for j, target in enumerate(TARGETS):
        heights = [lift.loc[ens, target] for ens in ENSEMBLES]
        bars = ax.bar(
            group_centres + offsets[j],
            heights,
            width=bar_width,
            color=colors[target],
            edgecolor="white",
            linewidth=0.6,
            label=target,
        )
        # Outline the diagonal (own-target) bar in each group
        for i, bar in enumerate(bars):
            if ENSEMBLES[i] == target:
                bar.set_edgecolor("#222222")
                bar.set_linewidth(1.6)

    ax.set_xticks(group_centres)
    ax.set_xticklabels(ENSEMBLES)
    ax.set_ylim(0, max(3.5, lift.values.max() * 1.08))
    ax.set_ylabel("Lift over random (AUPRC ÷ prevalence)")
    ax.set_title(
        "Diagonal bar (own-target) leads within each ensemble's group",
        pad=28,
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(
        ncol=n_bars,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        handlelength=1.1,
    )

    fig.subplots_adjust(top=0.82)

    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Recompute specialist rows from the CSV (different prevalence basis).",
    )
    args = parser.parse_args()

    matrix = load_lift_matrix(from_csv=args.from_csv)
    print("Lift matrix used (rows = ensemble, columns = target):")
    print(matrix.round(2).to_string())
    render(matrix, OUT_PNG)
    print(f"\nSaved -> {OUT_PNG}")
