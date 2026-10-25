#!/bin/bash
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/
# Full-scale ultimate-ensemble runs on this Mac. Safe to re-run at any time:
#   * every model fit is checkpointed (atomic writes) under technical/Results/ultimate_full/ckpt_*
#   * every finished stage leaves a DONE_* marker and is skipped on the next run
# So if the run is cut off (crash, reboot, Ctrl-C, battery), just start this script again.
set -o pipefail
B2="/PATH/TO/PROJECT"   # EDIT: project root (folder that contains technical/)
PY="python3"   # EDIT: Python with xgboost 2.1, scikit-learn 1.6.1, pandas, pyarrow
MAIN="$B2/technical/Models/Ultimate/run_ultimate_full.py"
BP="$B2/technical/Models/Ultimate/barton_protocol_full.py"
OUT="$B2/technical/Results/ultimate_full"
mkdir -p "$OUT"; cd "$B2" || exit 1

stage () {   # stage <marker> <logfile> <command...>
  local marker="$OUT/$1"; local logf="$OUT/$2"; shift 2
  if [ -f "$marker" ]; then echo ">>> skip (already done): $(basename "$marker")"; return 0; fi
  echo ">>> start $(basename "$marker")  $(date '+%F %T')" | tee -a "$logf"
  "$@" 2>&1 | tee -a "$logf"
  local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then echo "!!! stage failed (exit $rc). Fix/inspect $logf, then re-run this script - it resumes." ; exit $rc; fi
}

if [ ! -f "$OUT/DONE_smoke" ]; then
  echo ">>> smoke test (2% of patients, ~2 min)"
  "$PY" "$MAIN" --partition etiology --sample 0.02 --folds 3 --nose 2 --rf-trees 30 --xgb-trees 100 --B 50 --out /tmp/ult_smoke 2>&1 | tail -6
  [ ${PIPESTATUS[0]} -eq 0 ] && date > "$OUT/DONE_smoke" || { echo "!!! smoke test failed"; exit 1; }
fi
stage DONE_main_etiology_1h log_etiology.txt         "$PY" "$MAIN" --partition etiology
stage DONE_main_kmeans_1h   log_kmeans.txt           "$PY" "$MAIN" --partition kmeans
stage DONE_compare_1h       log_compare.txt          "$PY" "$MAIN" --compare
stage DONE_barton_etiology_1h log_barton_etiology.txt "$PY" "$BP" --partition etiology
stage DONE_barton_kmeans_1h   log_barton_kmeans.txt   "$PY" "$BP" --partition kmeans
echo "ALL DONE $(date '+%F %T'). Results in $OUT"
