#!/usr/bin/env bash
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  run_overnight.sh
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Overnight driver script. Deletes the previous V13b_final results, runs the
#    leakage-free optimised baselines, then runs V13b end-to-end at all three
#    resolutions with the tuned hyperparameters. Logs everything to
#    overnight_run.log and resumes V13b from checkpoints if interrupted.
#
#  INPUTS
#    optimise_features_v2.py, 67_v13b_final.py (in this directory)
#  OUTPUTS
#    overnight_run.log
#    ../../Results/optimised_performance_v2.csv
#    ../../Results/V13b_final/  (per-resolution outputs + Excel report)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    Assumed working directory  -  the script cd's to its own directory; run it
#      from anywhere but keep it inside technical/Models/V13b.
#    LOG path                   -  overnight_run.log location.
#    Step 0 abort delay         -  sleep 10 seconds before the destructive rm.
#    Step 0 rm targets          -  the 4_hour / 1_hour / 15_min result dirs and
#                                  the v13b_final_report.xlsx that get deleted.
#    Step 2 resolution flags    -  --resolution all --no-resume.
#    (Tuned V13b hyperparameters live in 67_v13b_final.py, not here.)
#
#  REQUIRES: bash, python3 (with the 67_v13b_final.py / optimise_features_v2.py
#            dependencies installed)
# ============================================================================
# ──────────────────────────────────────────────────────────────────────────
# Overnight pipeline:
#   Step 0: delete previous V13b_final results (sub-par regressed run)
#   Step 1: leakage-free optimised baselines (RF / LR / XGB at 3 resolutions)
#   Step 2: V13b end-to-end at 4h, 1h, 15m with the tuned hyperparameters
#
# Tuned hyperparameters baked into 67_v13b_final.py:
#   max_depth = 8
#   min_samples_leaf = 35
#   max_features = 'sqrt'
#   n_estimators = 150
#   class_weight = None  (NOSE already balances at stay level)
#   bootstrap = True
#   num_NOSE_subsets = 5
#   time_since_ICU_admit_hours INCLUDED as a feature
#
# Logs to overnight_run.log in this directory.
# Resumes from checkpoints if interrupted (V13b only; baselines are quick).
# ──────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# EDIT: LOG path (overnight run log location)
LOG="$SCRIPT_DIR/overnight_run.log"
echo "" > "$LOG"

banner() {
  echo "" | tee -a "$LOG"
  echo "==============================================================" | tee -a "$LOG"
  echo "$1  ($(date '+%Y-%m-%d %H:%M:%S'))" | tee -a "$LOG"
  echo "==============================================================" | tee -a "$LOG"
}

banner "STEP 0: Delete previous V13b_final results"
# EDIT: Step 0 abort delay (seconds before the destructive rm)
echo "About to delete the regressed 4h/1h/15m output. Press Ctrl+C in the next 10 seconds to abort." | tee -a "$LOG"
sleep 10
# EDIT: Step 0 rm targets (result dirs + Excel report deleted before rerun)
rm -rf ../../Results/V13b_final/4_hour 2>>"$LOG" || true
rm -rf ../../Results/V13b_final/1_hour 2>>"$LOG" || true
rm -rf ../../Results/V13b_final/15_min 2>>"$LOG" || true
rm -f  ../../Results/V13b_final/v13b_final_report.xlsx 2>>"$LOG" || true
echo "Cleanup done." | tee -a "$LOG"

banner "STEP 1: Leakage-free optimised baselines"
python3 -u optimise_features_v2.py 2>&1 | tee -a "$LOG"

banner "STEP 2: V13b at all three resolutions (4h -> 1h -> 15m)"
# EDIT: Step 2 resolution flags (--resolution all --no-resume)
python3 -u 67_v13b_final.py --resolution all --no-resume 2>&1 | tee -a "$LOG"

banner "ALL DONE"
echo "Optimised baselines: ../../Results/optimised_performance_v2.csv" | tee -a "$LOG"
echo "V13b results:        ../../Results/V13b_final/"                  | tee -a "$LOG"
echo "Excel report:        ../../Results/V13b_final/v13b_final_report.xlsx" | tee -a "$LOG"
echo "Run log:             $LOG" | tee -a "$LOG"
