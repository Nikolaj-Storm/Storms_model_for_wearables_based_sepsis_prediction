#!/bin/bash
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/
# 4-hour (ward-check) grid: same models, same patient split as the 1h run, then the paired 1h-vs-4h comparison.
# Waits for the 1h pipeline (run_all.sh) to finish first, so the two never compete for memory.
# Checkpointed + resumable exactly like run_all.sh: if it is cut off, start it again.
set -o pipefail
B2="/PATH/TO/PROJECT"   # EDIT: project root (folder that contains technical/)
PY="python3"   # EDIT: Python with xgboost 2.1, scikit-learn 1.6.1, pandas, pyarrow
MAIN="$B2/technical/Models/Ultimate/run_ultimate_full.py"
BP="$B2/technical/Models/Ultimate/barton_protocol_full.py"
CAD="$B2/technical/Models/Ultimate/cadence_compare.py"
OUT="$B2/technical/Results/ultimate_full"
export PYTHONWARNINGS="ignore::RuntimeWarning,ignore::FutureWarning"
mkdir -p "$OUT"; cd "$B2" || exit 1

while [ ! -f "$OUT/DONE_barton_kmeans_1h" ]; do echo ">>> waiting for the 1h run to finish ($(date '+%T'))"; sleep 120; done

stage () {
  local marker="$OUT/$1"; local logf="$OUT/$2"; shift 2
  if [ -f "$marker" ]; then echo ">>> skip (already done): $(basename "$marker")"; return 0; fi
  echo ">>> start $(basename "$marker")  $(date '+%F %T')" | tee -a "$logf"
  "$@" 2>&1 | tee -a "$logf"
  local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then echo "!!! stage failed (exit $rc). Inspect $logf, then re-run this script - it resumes."; exit $rc; fi
}

stage DONE_main_etiology_4h   log_etiology_4h.txt        "$PY" "$MAIN" --grid 4h --partition etiology
stage DONE_main_kmeans_4h     log_kmeans_4h.txt          "$PY" "$MAIN" --grid 4h --partition kmeans
stage DONE_compare_4h         log_compare_4h.txt         "$PY" "$MAIN" --grid 4h --compare
stage DONE_cadence            log_cadence.txt            "$PY" "$CAD"
stage DONE_barton_etiology_4h log_barton_etiology_4h.txt "$PY" "$BP" --grid 4h --partition etiology
stage DONE_barton_kmeans_4h   log_barton_kmeans_4h.txt   "$PY" "$BP" --grid 4h --partition kmeans
echo "ALL 4h DONE $(date '+%F %T'). Results in $OUT"
