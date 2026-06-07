# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  check_orthogonality.py
#  Stage: 6 - Visualization / appendix
#
#  PURPOSE
#    Verify the orthogonality claim about the three novel multiplicative-
#    interaction features. For each engineered feature it computes the mean
#    absolute Pearson correlation with all other engineered features, reports
#    the set-wide yardstick (mean and median), and ranks the named features.
#    Prints results to stdout only; writes no files.
#
#  INPUTS
#    <script_dir>/technical/Results/table_V13_Full_Correlation_Matrix.csv
#  OUTPUTS
#    none (prints a console report)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    CSV  -  path to the full correlation matrix CSV
#    No hardcoded metric constants; the yardstick is computed from the matrix.
#
#  REQUIRES: pandas, numpy
# ============================================================================

"""
Verify the orthogonality claim about the three novel multiplicative-interaction
features. For each feature in the V13 engineered set, compute the mean absolute
Pearson correlation with all OTHER features. Report:
  1. The set-wide mean of those mean-|r| values (the "average pairwise
     correlation" yardstick).
  2. Where Fever-Driven Tachycardia (fever_tachycardia),
     Cardiorespiratory Distress Index (resp_distress), and
     Generalised Perfusion Adequacy (perfusion_adequacy) sit relative to it.
  3. Their rank within the engineered set, low to high (1 = least correlated
     with the rest).
"""

from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
# EDIT: path to the full correlation matrix CSV
CSV  = HERE / "technical" / "Results" / "table_V13_Full_Correlation_Matrix.csv"

# Load full correlation matrix (square)
M = pd.read_csv(CSV, index_col=0)

# Sanity check: square and symmetric
assert M.shape[0] == M.shape[1], f"Not square: {M.shape}"
print(f"Loaded correlation matrix with {M.shape[0]} features\n")

# Drop columns/rows that should not be counted as "engineered" yardsticks.
# We exclude pure context features (age, weight, time_since_ICU_admit_hours)
# and the raw vital signs themselves, since the claim is specifically about
# "the engineered set". Keep all derived features.
RAW_VITALS = {"heart_rate", "resprate", "sbp", "dbp", "temp_c", "spo2"}
CONTEXT    = {"age", "weight_kg", "time_since_ICU_admit_hours"}

eng = [c for c in M.columns if c not in RAW_VITALS and c not in CONTEXT]
print(f"Engineered features in matrix: {len(eng)}")

E = M.loc[eng, eng].copy()

# Mean absolute correlation with all OTHER engineered features
np.fill_diagonal(E.values, np.nan)
mean_abs_r = E.abs().mean(axis=1).sort_values()

# Set-wide statistics
yardstick = mean_abs_r.mean()
median    = mean_abs_r.median()
print(f"\nSet-wide mean of (mean |r| with rest) : {yardstick:.4f}")
print(f"Set-wide median                       : {median:.4f}\n")

# Specific features of interest
TARGETS = {
    "fever_tachycardia":   "Fever-Driven Tachycardia",
    "resp_distress":       "Cardiorespiratory Distress Index (resp_distress)",
    "ventilation_perfusion_proxy": "Ventilation/Perfusion proxy",
    "perfusion_adequacy":  "Generalised Perfusion Adequacy",
    # For comparison, include some established composites:
    "msi":                 "Modified Shock Index",
    "shock_index":         "Shock Index",
    "news_score":          "NEWS Score",
    "partial_qsofa":       "Partial qSOFA",
    "map":                 "Mean Arterial Pressure",
}

print(f"{'Feature':50s} {'mean|r|':>9s} {'vs avg':>9s}  rank")
print("-" * 84)
for col in TARGETS:
    if col in mean_abs_r.index:
        v = mean_abs_r[col]
        rank = (mean_abs_r.index.get_loc(col) + 1)
        delta = v - yardstick
        flag = "BELOW" if v < yardstick else "above"
        print(f"{TARGETS[col]:50s} {v:9.4f} {delta:+9.4f}  {rank:>3d}/{len(mean_abs_r)} ({flag})")
    else:
        print(f"{TARGETS[col]:50s}  not in matrix")

print(f"\nLowest 10 mean|r| (most orthogonal):")
print(mean_abs_r.head(10).to_string())
print(f"\nHighest 10 mean|r| (most redundant):")
print(mean_abs_r.tail(10).to_string())
