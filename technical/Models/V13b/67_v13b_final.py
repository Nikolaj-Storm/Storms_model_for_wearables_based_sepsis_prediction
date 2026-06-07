# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  67_v13b_final.py
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Final V13b run. Refined NOSE base learners plus an etiology-stacked
#    meta-learner, with diagnostic logging, multi-threshold reporting,
#    checkpointing, and Excel aggregation. Runs at 15-min, 1-hour and 4-hour
#    sampling resolutions and emits per-resolution metrics plus a combined
#    multi-sheet Excel report.
#
#  INPUTS
#    ../../Data/All engineered features/Dataset_all_engineered_{15min,1h,4h}_{train,test}.parquet
#    Etiology label parquets (paths via --etiology-train / --etiology-test,
#      defaulting to the white-labeled ETIOLOGY_*_DEFAULT constants below)
#    EXTRA_TEST_PATHS fallback test parquet for the 1_hour resolution
#  OUTPUTS
#    ../../Results/V13b_final/<resolution>/  (checkpoints, sweeps, PR PNGs,
#      metrics_rows.parquet, metrics_rows.csv, specialisation_matrix CSV,
#      v13b_results_<resolution>.txt)
#    ../../Results/V13b_final/v13b_final_report.xlsx  (aggregated Excel)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    Assumed working directory  -  run from technical/Models/V13b so the
#      ../../Data and ../../Results relative paths resolve.
#    DATA_DIR                   -  engineered-feature parquet folder.
#    RESULTS_DIR                -  output root.
#    ETIOLOGY_TRAIN_DEFAULT     -  default v8 etiology train parquet path.
#    ETIOLOGY_TEST_DEFAULT      -  default v8 etiology test parquet path.
#    EXTRA_TEST_PATHS           -  per-resolution fallback test parquet path.
#    DATASETS                   -  resolution -> (train_fn, test_fn) map.
#    DROP_COLS                  -  columns excluded from the feature set.
#    RF_N_ESTIMATORS=150, RF_MAX_DEPTH=8, RF_MIN_SAMPLES_LEAF=35,
#      RF_MAX_FEATURES='sqrt', RF_CLASS_WEIGHT=None, RF_BOOTSTRAP=True
#                               -  NOSE Random Forest base-learner hyperparams.
#    NUM_NOSE_SUBSETS=5         -  NOSE negative-undersampling subsets.
#    NUM_FOLDS=5                -  StratifiedGroupKFold splits for OOF.
#    RANDOM_STATE=42            -  global random seed.
#    RF_N_JOBS=4                -  parallel jobs (CLI --max-jobs; -1 for all).
#    DOWNCAST=True              -  float64->float32 (CLI --no-downcast).
#    XGB meta-learner           -  n_estimators=200, max_depth=4,
#                                  learning_rate=0.05, scale_pos_weight=neg/pos,
#                                  subsample=0.8, colsample_bytree=0.8,
#                                  min_child_weight=5.
#    LR meta-learner            -  class_weight='balanced', max_iter=1000.
#    Threshold sweep            -  np.linspace(0.005, 0.95, 950).
#    Operating points           -  F1-max; max-sens FPR<0.50; max-TP FPR<0.20.
#    CLI args                   -  --resolution, --etiology-train,
#                                  --etiology-test, --no-resume, --build-excel,
#                                  --max-jobs, --no-downcast.
#
#  REQUIRES: pandas, numpy, scikit-learn, xgboost, joblib, matplotlib,
#            pyarrow, xlsxwriter or openpyxl, psutil (optional)
# ============================================================================
"""
V13b Final Run
==============
Refined NOSE + Etiology-Stacked Meta-Learner with diagnostic logging,
multi-threshold reporting, checkpointing, and Excel aggregation.

Pipeline (high level)
---------------------
  1-2. Load Dataset_all_engineered_{res}_{train,test}.parquet (V12+V13 features
       are pre-engineered upstream, so apply_v13_features() is dropped).
   3.  Merge stay-level etiology labels (broadcast from
       v8_dataset_1h_etiology_*.parquet at the etiology default paths).
       Logs per-bucket stay counts (resp / uri / other / non-septic).
   4.  Build feature list: ALL features used for every stream (no keyword filter).
   5.  Median imputation, fit on train.
   6.  OOF meta-features via 5-fold StratifiedGroupKFold and 5-subset NOSE RF.
       Logs per-NOSE-subset positive/negative stay counts (first fold).
       Checkpointed per fold and again at completion.
   7.  Train two meta-learners per horizon (LR balanced + XGB with dynamic
       scale_pos_weight = neg/pos).
   8.  Train 8 final NOSE ensembles on 100% of train. Checkpointed per stream.
   9.  Specialisation matrix: each etiology ensemble vs every label.
  10.  Test predictions, three threshold operating points per (horizon, meta):
         (a) F1-max          balanced precision/recall
         (b) Max-sens FPR<0.50  aggressive (fewer FN)
         (c) Max-TP   FPR<0.20  precision-focused (fair FPR)
       Emits PR-frontier PNG + threshold-sweep CSV per (horizon, meta).
       Writes per-resolution metrics_rows.parquet for the aggregator.
  11.  Aggregator (run with --build-excel) merges all resolutions into
       a single Excel file with one sheet per horizon, three tables per sheet.

Usage
-----
  cd technical/Models/V13b
  python 67_v13b_final.py --resolution 4_hour
  python 67_v13b_final.py --resolution 1_hour
  python 67_v13b_final.py --resolution 15_min
  python 67_v13b_final.py --resolution all
  python 67_v13b_final.py --build-excel        # only assemble Excel from existing results
  python 67_v13b_final.py --resolution 1_hour --no-resume   # ignore checkpoints
"""

import argparse
import gc
import os
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# EDIT: DATA_DIR - engineered-feature parquet folder (relative to working dir)
DATA_DIR = Path("../../Data/All engineered features")
# EDIT: RESULTS_DIR - output root (relative to working dir)
RESULTS_DIR = Path("../../Results/V13b_final")
# EDIT: ETIOLOGY_TRAIN_DEFAULT - default v8 etiology train parquet path
ETIOLOGY_TRAIN_DEFAULT = Path(
    "/PATH/TO/PROJECT/technical/v8_dataset_1h_etiology_train.parquet"
)
# EDIT: ETIOLOGY_TEST_DEFAULT - default v8 etiology test parquet path
ETIOLOGY_TEST_DEFAULT = Path(
    "/PATH/TO/PROJECT/technical/v8_dataset_1h_etiology_test.parquet"
)
# EDIT: EXTRA_TEST_PATHS - per-resolution fallback test parquet path(s)
EXTRA_TEST_PATHS = {
    "1_hour": Path(
        "/PATH/TO/PROJECT/technical/Dataset_all_engineered_1h_test.parquet"
    ),
}

# EDIT: DATASETS - resolution -> (train_fn, test_fn) map
DATASETS = {
    "15_min": ("Dataset_all_engineered_15min_train.parquet", "Dataset_all_engineered_15min_test.parquet"),
    "1_hour": ("Dataset_all_engineered_1h_train.parquet", "Dataset_all_engineered_1h_test.parquet"),
    "4_hour": ("Dataset_all_engineered_4h_train.parquet", "Dataset_all_engineered_4h_test.parquet"),
}

# EDIT: DROP_COLS - columns excluded from the feature set
DROP_COLS = [
    "is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
    "stay_id", "charttime", "intime", "sepsis3_time",
    # NOTE: time_since_ICU_admit_hours is INTENTIONALLY KEPT as a feature.
    # Excluding it (as the optimise_features.py inheritance did) cost ~0.013
    # AUROC at 1h. It is a top-importance feature in this cohort.
]

HORIZONS = ["is_sepsis_6h", "is_sepsis_12h"]
META_NAMES = ["lr", "xgb"]
OP_LABELS = {
    "f1_max": "F1-max (balanced)",
    "max_sens_fpr50": "Max sens, FPR<0.50",
    "max_tp_fpr20": "Max TP, FPR<0.20",
}

# Hyperparameters tuned via diagnostic sweep on 4h fold 0.
# See methodology section for the depth, min_samples_leaf, max_features,
# n_estimators, class_weight, bootstrap, and num_NOSE_subsets diagnostics.
# EDIT: RF base-learner hyperparameters (next seven constants)
RF_N_ESTIMATORS = 150       # was 200; plateau at 150
RF_MAX_DEPTH = 8            # was 12; sweet spot between underfit and overfit
RF_MIN_SAMPLES_LEAF = 35    # was 5; AUPRC peaks at 35
RF_MAX_FEATURES = "sqrt"    # unchanged; default is robust for ensembles
RF_CLASS_WEIGHT = None      # was 'balanced'; NOSE already balances at stay level
RF_BOOTSTRAP = True         # unchanged; default
NUM_NOSE_SUBSETS = 5        # EDIT: NOSE negative-undersampling subsets (saturation point)
NUM_FOLDS = 5               # EDIT: StratifiedGroupKFold splits for OOF
RANDOM_STATE = 42           # EDIT: global random seed

# Memory guardrails. Set via --max-jobs and --downcast/--no-downcast at the CLI.
# n_jobs = 4 keeps RF parallel-tree memory bounded on a 16 GB machine;
# set to -1 on machines with abundant RAM. Downcast trims float64 to float32
# (no clinical or statistical impact at this scale of vital-sign features).
RF_N_JOBS = 4               # EDIT: default parallel jobs (CLI --max-jobs)
DOWNCAST = True             # EDIT: float64->float32 downcast (CLI --no-downcast)

STREAMS = [
    "is_sepsis_6h", "is_sepsis_12h",
    "target_resp_6h", "target_resp_12h",
    "target_uri_6h", "target_uri_12h",
    "target_other_6h", "target_other_12h",
]

# ---------------------------------------------------------------------------
# CHECKPOINT HELPERS
# ---------------------------------------------------------------------------
def ckpt_dir_for(resolution):
    p = RESULTS_DIR / resolution / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_joblib(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    return joblib.load(path)


def save_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def touch_flag(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


# ---------------------------------------------------------------------------
# MEMORY HELPERS
# ---------------------------------------------------------------------------
def mem_snapshot(label=""):
    """Print current process RSS and a short label. No-op if psutil missing."""
    if not _HAVE_PSUTIL:
        return
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    avail = psutil.virtual_memory().available / 1024**3
    print(f"  [mem] {label}  rss={rss:.2f} GB  avail={avail:.2f} GB", flush=True)


def downcast_dataframe(df, label=""):
    """Cast float64 → float32 and shrink ints. Vital-sign features and
    derivatives are well within float32 precision, so this is safe and
    saves ~50% of the dataframe's footprint."""
    before = df.memory_usage(deep=False).sum() / 1024**3
    for c in df.columns:
        dt = df[c].dtype
        if dt == "float64":
            df[c] = df[c].astype("float32")
        elif dt == "int64":
            mn, mx = df[c].min(), df[c].max()
            if mn >= -32768 and mx <= 32767:
                df[c] = df[c].astype("int16")
            elif mn >= -2147483648 and mx <= 2147483647:
                df[c] = df[c].astype("int32")
    after = df.memory_usage(deep=False).sum() / 1024**3
    print(f"  [downcast] {label}  before={before:.2f} GB  after={after:.2f} GB  "
          f"saved={before - after:.2f} GB", flush=True)
    return df


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_engineered(resolution):
    train_fn, test_fn = DATASETS[resolution]
    train_path = DATA_DIR / train_fn
    test_path = DATA_DIR / test_fn

    if not train_path.exists():
        raise FileNotFoundError(f"Train file missing: {train_path}")
    if not test_path.exists():
        alt = EXTRA_TEST_PATHS.get(resolution)
        if alt and alt.exists():
            print(f"  [info] Using fallback test path: {alt}")
            test_path = alt
        else:
            raise FileNotFoundError(
                f"Test file missing: {test_path}. "
                f"For 15_min, generate it from v3_dataset_15m_test.parquet "
                f"by running V12 + V13 feature engineering."
            )

    print(f"  Loading train: {train_path.name}")
    df_train = pd.read_parquet(train_path)
    print(f"  Loading test:  {test_path.name}")
    df_test = pd.read_parquet(test_path)

    for df, name in [(df_train, "train"), (df_test, "test")]:
        if "stay_id" not in df.columns and "level_0" in df.columns:
            df.rename(columns={"level_0": "stay_id", "level_1": "charttime"}, inplace=True)
        for required in ["stay_id", "is_sepsis_6h", "is_sepsis_12h", "is_sepsis_stay"]:
            if required not in df.columns:
                raise ValueError(f"{name} dataset missing required column: {required}")

    return df_train, df_test


def load_etiology_table(train_path, test_path):
    cols = ["stay_id", "target_resp", "target_uri", "target_other"]
    df_tr = pd.read_parquet(train_path, columns=cols).groupby("stay_id").max().reset_index()
    df_te = pd.read_parquet(test_path, columns=cols).groupby("stay_id").max().reset_index()
    return df_tr, df_te


# ---------------------------------------------------------------------------
# STEP 3: MERGE ETIOLOGY + LOG BUCKET COUNTS
# ---------------------------------------------------------------------------
def merge_etiology_and_log(df_train, df_test, etio_tr, etio_te, log_lines):
    df_train = df_train.merge(etio_tr, on="stay_id", how="left").fillna({"target_resp": 0, "target_uri": 0, "target_other": 0})
    df_test = df_test.merge(etio_te, on="stay_id", how="left").fillna({"target_resp": 0, "target_uri": 0, "target_other": 0})

    for col in ["target_resp", "target_uri", "target_other"]:
        for h in ["6h", "12h"]:
            sepsis_h = f"is_sepsis_{h}"
            df_train[f"{col}_{h}"] = ((df_train[sepsis_h] == 1) & (df_train[col] == 1)).astype(int)
            df_test[f"{col}_{h}"] = ((df_test[sepsis_h] == 1) & (df_test[col] == 1)).astype(int)

    def bucket_counts(df, label):
        n_stays = df["stay_id"].nunique()
        septic_stays = df[df["is_sepsis_stay"] == 1]["stay_id"].unique()
        n_septic = len(septic_stays)
        n_nonseptic = n_stays - n_septic
        df_septic = df[df["stay_id"].isin(septic_stays)]
        n_resp = df_septic[df_septic["target_resp"] == 1]["stay_id"].nunique()
        n_uri = df_septic[df_septic["target_uri"] == 1]["stay_id"].nunique()
        n_other = df_septic[df_septic["target_other"] == 1]["stay_id"].nunique()
        return (
            f"  [{label}] stays total: {n_stays:,}  |  septic: {n_septic:,}  |  non-septic: {n_nonseptic:,}\n"
            f"            -> respiratory sepsis: {n_resp:,}\n"
            f"            -> urinary sepsis:     {n_uri:,}\n"
            f"            -> other sepsis:       {n_other:,}\n"
        )

    log_lines.append("Step 3: Etiology bucket counts\n")
    log_lines.append(bucket_counts(df_train, "train"))
    log_lines.append(bucket_counts(df_test, "test"))
    print("".join(log_lines[-3:]))
    return df_train, df_test


# ---------------------------------------------------------------------------
# STEP 4: FEATURE LIST (ALL FEATURES, NO KEYWORD FILTER)
# ---------------------------------------------------------------------------
def get_all_features(df_train, exclude_extra=None):
    extras = exclude_extra or []
    exclude = set(DROP_COLS) | {"target_resp", "target_uri", "target_other"} | {
        f"{c}_{h}" for c in ["target_resp", "target_uri", "target_other"] for h in ["6h", "12h"]
    } | set(extras)
    return [c for c in df_train.columns if c not in exclude]


# ---------------------------------------------------------------------------
# NOSE BASE-LEARNER ENSEMBLE
# ---------------------------------------------------------------------------
def train_nose_ensemble(df, features, target_col, num_subsets=NUM_NOSE_SUBSETS,
                        log_prefix=None, log_lines=None):
    pos_stays = df[df[target_col] == 1]["stay_id"].unique()
    all_stays = df["stay_id"].unique()
    neg_stays = np.setdiff1d(all_stays, pos_stays)

    rng = np.random.RandomState(RANDOM_STATE)
    neg_shuffled = rng.permutation(neg_stays)

    chunk_size = max(len(pos_stays), 1)
    models = []
    subset_logs = []

    for i in range(num_subsets):
        start = i * chunk_size
        end = min(start + chunk_size, len(neg_shuffled))
        subset_neg = neg_shuffled[start:end]
        if len(subset_neg) == 0:
            break

        subset_stays = np.concatenate([pos_stays, subset_neg])
        df_subset = df[df["stay_id"].isin(subset_stays)]
        X_sub = df_subset[features].values
        y_sub = df_subset[target_col].values

        n_pos_rows = int((y_sub == 1).sum())
        n_neg_rows = int((y_sub == 0).sum())
        subset_logs.append(
            f"      subset {i+1}/{num_subsets}: pos_stays={len(pos_stays):,}, "
            f"neg_stays={len(subset_neg):,} | rows pos={n_pos_rows:,}, neg={n_neg_rows:,}"
        )

        # EDIT: NOSE RF base-learner hyperparameters (driven by the RF_* constants)
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            class_weight=RF_CLASS_WEIGHT,
            bootstrap=RF_BOOTSTRAP,
            random_state=RANDOM_STATE + i,
            n_jobs=RF_N_JOBS,
        )
        model.fit(X_sub, y_sub)
        models.append(model)
        # Free the per-subset frame and arrays before the next iteration so
        # peak memory does not stack across NOSE subsets.
        del df_subset, X_sub, y_sub
        gc.collect()

    if log_prefix and log_lines is not None:
        log_lines.append(f"    {log_prefix}\n")
        for line in subset_logs:
            log_lines.append(line + "\n")

    return models


def predict_nose(models, df_eval, features):
    X = df_eval[features].values
    preds = np.zeros(len(df_eval))
    for m in models:
        preds += m.predict_proba(X)[:, 1]
    return preds / max(len(models), 1)


# ---------------------------------------------------------------------------
# THRESHOLD SWEEP + THREE OPERATING POINTS
# ---------------------------------------------------------------------------
def threshold_sweep(y_true, y_prob):
    # EDIT: threshold sweep grid (0.005 to 0.95, 950 points)
    thresholds = np.linspace(0.005, 0.95, 950)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        sens = tp / max((tp + fn), 1)
        fpr = fp / max((fp + tn), 1)
        prec = tp / max((tp + fp), 1)
        f1 = f1_score(y_true, yp, zero_division=0)
        f05 = fbeta_score(y_true, yp, beta=0.5, zero_division=0)
        f2 = fbeta_score(y_true, yp, beta=2, zero_division=0)
        rows.append({
            "threshold": t, "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
            "sensitivity": sens, "fpr": fpr,
            "precision": prec, "f0_5": f05, "f1": f1, "f2": f2,
        })
    return pd.DataFrame(rows)


def pick_three_operating_points(sweep):
    f1_idx = sweep["f1"].idxmax()
    f1_row = sweep.loc[f1_idx]

    # EDIT: max-sensitivity FPR ceiling (0.50)
    sub_a = sweep[sweep["fpr"] < 0.50]
    a_row = sub_a.loc[sub_a["sensitivity"].idxmax()] if len(sub_a) else f1_row

    # EDIT: max-TP FPR ceiling (0.20)
    sub_b = sweep[sweep["fpr"] < 0.20]
    b_row = sub_b.loc[sub_b["tp"].idxmax()] if len(sub_b) else f1_row

    return {"f1_max": f1_row, "max_sens_fpr50": a_row, "max_tp_fpr20": b_row}


def plot_pr_frontier(sweep, op_points, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"f1_max": "#c0392b", "max_sens_fpr50": "#16a085", "max_tp_fpr20": "#8e44ad"}

    ax = axes[0]
    ax.plot(sweep["sensitivity"], sweep["precision"], color="#3a6ea5", lw=2, label="PR frontier")
    for k, row in op_points.items():
        ax.scatter([row["sensitivity"]], [row["precision"]],
                   color=colors[k], s=80, zorder=5, edgecolor="black",
                   label=f"{OP_LABELS[k]} (t={row['threshold']:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ymax = max(0.05, sweep["precision"].max() * 1.1)
    ax.set_ylim(0, ymax)

    ax = axes[1]
    ax.plot(sweep["threshold"], sweep["sensitivity"], label="Sensitivity (Recall)", color="#16a085")
    ax.plot(sweep["threshold"], sweep["fpr"], label="FPR", color="#c0392b")
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision", color="#8e44ad")
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", color="#2c3e50", linestyle="--")
    ax.plot(sweep["threshold"], sweep["f2"], label="F2", color="#7f8c8d", linestyle="--")
    ax.plot(sweep["threshold"], sweep["f0_5"], label="F0.5", color="#bdc3c7", linestyle="--")
    for k, row in op_points.items():
        ax.axvline(row["threshold"], color=colors[k], alpha=0.4, linestyle=":")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Metrics vs threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# COMPREHENSIVE METRIC COMPUTATION
# ---------------------------------------------------------------------------
def compute_full_metrics(resolution, horizon, meta_name, op_label, threshold,
                         y_train_true, y_train_prob,
                         y_test_true, y_test_prob):
    yt_pred = (y_train_prob >= threshold).astype(int)
    ye_pred = (y_test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_true, ye_pred, labels=[0, 1]).ravel()
    total = len(y_test_true)

    return {
        "Resolution": resolution,
        "Horizon": horizon,
        "Meta-learner": meta_name.upper(),
        "Threshold Op": op_label,
        "Threshold": round(float(threshold), 4),
        "Train Accuracy (%)": round(accuracy_score(y_train_true, yt_pred) * 100, 2),
        "Test Accuracy (%)":  round(accuracy_score(y_test_true, ye_pred) * 100, 2),
        "Precision": round(precision_score(y_test_true, ye_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test_true, ye_pred, zero_division=0), 4),
        "F0.5 Score": round(fbeta_score(y_test_true, ye_pred, beta=0.5, zero_division=0), 4),
        "F1 Score":   round(f1_score(y_test_true, ye_pred, zero_division=0), 4),
        "F2 Score":   round(fbeta_score(y_test_true, ye_pred, beta=2, zero_division=0), 4),
        "F1 Macro":   round(f1_score(y_test_true, ye_pred, average="macro", zero_division=0), 4),
        "ROC AUC":    round(roc_auc_score(y_test_true, y_test_prob), 4),
        "AUPRC":      round(average_precision_score(y_test_true, y_test_prob), 4),
        "TP (Count)": int(tp), "FN (Count)": int(fn),
        "FP (Count)": int(fp), "TN (Count)": int(tn),
        "TP (%)": round(tp / max(total, 1) * 100, 2),
        "FN (%)": round(fn / max(total, 1) * 100, 2),
        "FP (%)": round(fp / max(total, 1) * 100, 2),
        "TN (%)": round(tn / max(total, 1) * 100, 2),
    }


METRIC_COLUMNS = [
    "Resolution", "Horizon", "Meta-learner", "Threshold Op", "Threshold",
    "Train Accuracy (%)", "Test Accuracy (%)",
    "Precision", "Recall",
    "F0.5 Score", "F1 Score", "F2 Score", "F1 Macro",
    "ROC AUC", "AUPRC",
    "TP (Count)", "FN (Count)", "FP (Count)", "TN (Count)",
    "TP (%)", "FN (%)", "FP (%)", "TN (%)",
]


# ---------------------------------------------------------------------------
# MAIN PIPELINE (per resolution)
# ---------------------------------------------------------------------------
def run_resolution(resolution, args):
    out_dir = RESULTS_DIR / resolution
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir_for(resolution)
    log_lines = []
    t0 = time.time()

    print("=" * 75)
    print(f"V13b FINAL RUN  |  resolution={resolution}  |  resume={'no' if args.no_resume else 'yes'}")
    print("=" * 75)

    # ---- Step 1-2: Load ----
    print("\n[Step 1-2] Loading engineered datasets")
    df_train, df_test = load_engineered(resolution)
    print(f"  train shape: {df_train.shape}  test shape: {df_test.shape}")
    mem_snapshot("after load")

    if DOWNCAST:
        print("\n[Step 1b] Downcasting numeric columns (float64 -> float32)")
        df_train = downcast_dataframe(df_train, label="train")
        df_test = downcast_dataframe(df_test, label="test")
        gc.collect()
        mem_snapshot("after downcast")

    # ---- Step 3: Etiology merge ----
    print("\n[Step 3] Merging etiology labels and counting buckets")
    etio_tr, etio_te = load_etiology_table(args.etiology_train, args.etiology_test)
    df_train, df_test = merge_etiology_and_log(df_train, df_test, etio_tr, etio_te, log_lines)
    del etio_tr, etio_te
    gc.collect()

    # ---- Step 4: Features ----
    print("\n[Step 4] Building feature list (all features, no keyword filter)")
    all_features = get_all_features(df_train)
    print(f"  Features used: {len(all_features)}")
    log_lines.append(f"\nStep 4: Total features = {len(all_features)}\n")

    # ---- Step 5: Imputation ----
    print("\n[Step 5] Median imputation (fit on train)")
    imputer = SimpleImputer(strategy="median")
    df_train[all_features] = imputer.fit_transform(df_train[all_features]).astype("float32")
    df_test[all_features] = imputer.transform(df_test[all_features]).astype("float32")
    del imputer
    gc.collect()
    mem_snapshot("after imputation")

    # ---- Step 6: OOF meta-features (with checkpointing) ----
    oof_full_path = ckpt / "oof_meta_full.parquet"
    if (not args.no_resume) and oof_full_path.exists():
        print(f"\n[Step 6] Loading OOF checkpoint -> {oof_full_path.name}")
        df_train_meta = pd.read_parquet(oof_full_path)
    else:
        print(f"\n[Step 6] Building OOF meta-features ({NUM_FOLDS}-fold, {NUM_NOSE_SUBSETS} NOSE subsets)")
        df_train_meta = df_train[["stay_id", "is_sepsis_6h", "is_sepsis_12h", "is_sepsis_stay"]].copy()
        for s in STREAMS:
            df_train_meta[f"meta_{s}"] = 0.0

        sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        log_lines.append("\nStep 7: NOSE subset positive/negative breakdown (first fold, all streams)\n")

        fold_splits = list(sgkf.split(df_train, df_train["is_sepsis_stay"], groups=df_train["stay_id"]))
        for fold, (tr_idx, va_idx) in enumerate(fold_splits):
            fold_path = ckpt / f"oof_fold_{fold:02d}.parquet"
            if (not args.no_resume) and fold_path.exists():
                print(f"  Fold {fold + 1}/{NUM_FOLDS}: loading checkpoint")
                fold_df = pd.read_parquet(fold_path)
                idx = df_train_meta.index[va_idx]
                for s in STREAMS:
                    df_train_meta.loc[idx, f"meta_{s}"] = fold_df[f"meta_{s}"].values
                continue

            print(f"  Fold {fold + 1}/{NUM_FOLDS}: training")
            X_tr = df_train.iloc[tr_idx]
            X_va = df_train.iloc[va_idx]
            fold_preds = {f"meta_{s}": np.zeros(len(va_idx)) for s in STREAMS}
            for stream in STREAMS:
                log_prefix = f"Stream {stream}, fold {fold + 1}" if fold == 0 else None
                ll = log_lines if fold == 0 else None
                nose = train_nose_ensemble(X_tr, all_features, stream,
                                           num_subsets=NUM_NOSE_SUBSETS,
                                           log_prefix=log_prefix, log_lines=ll)
                fold_preds[f"meta_{stream}"] = predict_nose(nose, X_va, all_features)
                # Free the per-stream ensemble before training the next stream.
                del nose
                gc.collect()

            idx = df_train_meta.index[va_idx]
            for s in STREAMS:
                df_train_meta.loc[idx, f"meta_{s}"] = fold_preds[f"meta_{s}"]

            fold_save = pd.DataFrame({
                "row_idx": va_idx,
                **{f"meta_{s}": fold_preds[f"meta_{s}"] for s in STREAMS},
            })
            save_parquet(fold_save, fold_path)
            print(f"    -> checkpoint saved: {fold_path.name}")
            del X_tr, X_va, fold_preds
            gc.collect()
            mem_snapshot(f"after fold {fold + 1}")

        df_train_meta["var_6h"] = df_train_meta[
            ["meta_is_sepsis_6h", "meta_target_resp_6h", "meta_target_uri_6h", "meta_target_other_6h"]
        ].var(axis=1)
        df_train_meta["var_12h"] = df_train_meta[
            ["meta_is_sepsis_12h", "meta_target_resp_12h", "meta_target_uri_12h", "meta_target_other_12h"]
        ].var(axis=1)
        save_parquet(df_train_meta, oof_full_path)
        print(f"  OOF complete -> {oof_full_path.name}")

    if "var_6h" not in df_train_meta.columns:
        df_train_meta["var_6h"] = df_train_meta[
            ["meta_is_sepsis_6h", "meta_target_resp_6h", "meta_target_uri_6h", "meta_target_other_6h"]
        ].var(axis=1)
        df_train_meta["var_12h"] = df_train_meta[
            ["meta_is_sepsis_12h", "meta_target_resp_12h", "meta_target_uri_12h", "meta_target_other_12h"]
        ].var(axis=1)

    meta_features = [f"meta_{s}" for s in STREAMS] + ["var_6h", "var_12h"]

    # ---- Step 7: Meta-learners ----
    print("\n[Step 7] Training meta-learners per horizon")
    meta_models = {}
    for horizon in HORIZONS:
        y_meta = df_train_meta[horizon]
        n_pos = int(y_meta.sum())
        n_neg = int(len(y_meta) - n_pos)
        spw = n_neg / max(n_pos, 1)
        print(f"  Horizon={horizon}  pos={n_pos:,}  neg={n_neg:,}  scale_pos_weight={spw:.1f}")
        log_lines.append(f"\nStep 8: Horizon={horizon} class ratio neg/pos = {spw:.1f}\n")

        # EDIT: LR meta-learner hyperparameters
        lr = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
        lr.fit(df_train_meta[meta_features], y_meta)

        # EDIT: XGBoost meta-learner hyperparameters (scale_pos_weight dynamic = neg/pos)
        xgb_meta = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=RF_N_JOBS, verbosity=0,
        )
        xgb_meta.fit(df_train_meta[meta_features], y_meta)
        meta_models[horizon] = {"lr": lr, "xgb": xgb_meta, "spw": spw}

    # ---- Step 8: Final ensembles (per-stream checkpointing) ----
    print(f"\n[Step 8] Training {len(STREAMS)} final NOSE ensembles on full train")
    final_ensembles = {}
    for stream in STREAMS:
        ens_path = ckpt / f"final_ensemble_{stream}.joblib"
        if (not args.no_resume) and ens_path.exists():
            print(f"  Stream {stream}: loading checkpoint")
            final_ensembles[stream] = load_joblib(ens_path)
        else:
            print(f"  Stream {stream}: training")
            final_ensembles[stream] = train_nose_ensemble(df_train, all_features, stream,
                                                          num_subsets=NUM_NOSE_SUBSETS)
            save_joblib(final_ensembles[stream], ens_path)
            print(f"    -> checkpoint saved: {ens_path.name}")
        gc.collect()
        mem_snapshot(f"after stream {stream}")

    # ---- Step 9: Specialisation matrix ----
    print("\n[Step 9] Etiology specialisation matrix (each etiology ensemble vs every label)")
    spec_streams = ["target_resp_6h", "target_uri_6h", "target_other_6h"]
    eval_labels = ["is_sepsis_6h", "target_resp_6h", "target_uri_6h", "target_other_6h"]
    spec_rows = []
    for stream in spec_streams:
        preds = predict_nose(final_ensembles[stream], df_test, all_features)
        for label in eval_labels:
            y = df_test[label]
            if y.sum() == 0:
                auroc = float("nan"); auprc = float("nan")
            else:
                auroc = roc_auc_score(y, preds)
                auprc = average_precision_score(y, preds)
            spec_rows.append({"ensemble": stream, "label": label,
                              "AUROC": round(auroc, 4) if auroc == auroc else auroc,
                              "AUPRC": round(auprc, 4) if auprc == auprc else auprc})
    spec_df = pd.DataFrame(spec_rows)
    print(spec_df.to_string(index=False))
    spec_df.to_csv(out_dir / f"specialisation_matrix_{resolution}.csv", index=False)
    log_lines.append("\nStep 9: Etiology specialisation matrix\n")
    log_lines.append(spec_df.to_string(index=False) + "\n")

    # ---- Step 10: Test predictions, three thresholds, full metric set ----
    print("\n[Step 10] Test predictions, three operating points, PR curves, full metrics")
    test_meta_path = ckpt / "test_meta_predictions.parquet"
    if (not args.no_resume) and test_meta_path.exists():
        print(f"  Loading test predictions checkpoint")
        df_test_meta = pd.read_parquet(test_meta_path)
    else:
        df_test_meta = df_test[["stay_id", "is_sepsis_6h", "is_sepsis_12h", "is_sepsis_stay"]].copy()
        for stream in STREAMS:
            df_test_meta[f"meta_{stream}"] = predict_nose(final_ensembles[stream], df_test, all_features)
        df_test_meta["var_6h"] = df_test_meta[
            ["meta_is_sepsis_6h", "meta_target_resp_6h", "meta_target_uri_6h", "meta_target_other_6h"]
        ].var(axis=1)
        df_test_meta["var_12h"] = df_test_meta[
            ["meta_is_sepsis_12h", "meta_target_resp_12h", "meta_target_uri_12h", "meta_target_other_12h"]
        ].var(axis=1)
        save_parquet(df_test_meta, test_meta_path)

    rows = []
    report_lines = []
    for horizon in HORIZONS:
        y_test = df_test_meta[horizon].astype(int)
        y_train = df_train_meta[horizon].astype(int)
        for meta_name in META_NAMES:
            model = meta_models[horizon][meta_name]
            p_test = model.predict_proba(df_test_meta[meta_features])[:, 1]
            p_train = model.predict_proba(df_train_meta[meta_features])[:, 1]

            sweep = threshold_sweep(y_test, p_test)
            ops = pick_three_operating_points(sweep)

            sweep_csv = out_dir / f"threshold_sweep_{resolution}_{horizon}_{meta_name}.csv"
            sweep.to_csv(sweep_csv, index=False)
            png_path = out_dir / f"pr_frontier_{resolution}_{horizon}_{meta_name}.png"
            plot_pr_frontier(sweep, ops,
                             title=f"V13b {resolution}  |  horizon={horizon}  |  meta={meta_name}",
                             out_path=png_path)

            block = (
                f"\n========== horizon={horizon}  meta={meta_name} ==========\n"
                f"  AUROC={roc_auc_score(y_test, p_test):.4f}  "
                f"AUPRC={average_precision_score(y_test, p_test):.4f}\n"
            )
            for op_key, op_row in ops.items():
                metrics = compute_full_metrics(
                    resolution=resolution, horizon=horizon,
                    meta_name=meta_name, op_label=OP_LABELS[op_key],
                    threshold=op_row["threshold"],
                    y_train_true=y_train, y_train_prob=p_train,
                    y_test_true=y_test, y_test_prob=p_test,
                )
                rows.append(metrics)
                block += (
                    f"  [{OP_LABELS[op_key]}] t={metrics['Threshold']:.3f}\n"
                    f"    Train Acc: {metrics['Train Accuracy (%)']}%  |  "
                    f"Test Acc: {metrics['Test Accuracy (%)']}%\n"
                    f"    Precision: {metrics['Precision']}  |  Recall: {metrics['Recall']}\n"
                    f"    F0.5: {metrics['F0.5 Score']}  |  F1: {metrics['F1 Score']}  |  "
                    f"F2: {metrics['F2 Score']}  |  F1 macro: {metrics['F1 Macro']}\n"
                    f"    AUROC: {metrics['ROC AUC']}  |  AUPRC: {metrics['AUPRC']}\n"
                    f"    TP: {metrics['TP (Count)']} ({metrics['TP (%)']}%) | "
                    f"FN: {metrics['FN (Count)']} ({metrics['FN (%)']}%) | "
                    f"FP: {metrics['FP (Count)']} ({metrics['FP (%)']}%) | "
                    f"TN: {metrics['TN (Count)']} ({metrics['TN (%)']}%)\n"
                )
            print(block)
            report_lines.append(block)

    metrics_df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    metrics_df.to_parquet(out_dir / f"metrics_rows.parquet", index=False)
    metrics_df.to_csv(out_dir / f"metrics_rows.csv", index=False)

    # ---- Persist text log ----
    txt_path = out_dir / f"v13b_results_{resolution}.txt"
    with open(txt_path, "w") as f:
        f.write(f"V13b Final Run  |  resolution={resolution}\n")
        f.write("=" * 75 + "\n")
        f.write(f"n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, "
                f"min_samples_leaf={RF_MIN_SAMPLES_LEAF}, "
                f"NOSE subsets={NUM_NOSE_SUBSETS}, folds={NUM_FOLDS}, seed={RANDOM_STATE}\n")
        f.write("=" * 75 + "\n")
        f.writelines(log_lines)
        f.writelines(report_lines)

    print(f"\n✓ Done in {(time.time() - t0) / 60:.1f} min  |  outputs: {out_dir}")
    return metrics_df


# ---------------------------------------------------------------------------
# EXCEL AGGREGATOR
# ---------------------------------------------------------------------------
def build_excel(args):
    """Combine metrics_rows.parquet from each resolution into a single
    multi-sheet Excel file. One sheet per horizon, three tables per sheet."""
    print("\n========== Building Excel report ==========")
    frames = []
    for res in DATASETS.keys():
        f = RESULTS_DIR / res / "metrics_rows.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            print(f"  loaded {f}")
        else:
            print(f"  [missing] {f}  (skipping)")
    if not frames:
        print("  no metrics rows found; nothing to write.")
        return

    df = pd.concat(frames, ignore_index=True)

    out_xlsx = RESULTS_DIR / "v13b_final_report.xlsx"
    res_order = ["15_min", "1_hour", "4_hour"]
    res_titles = {"15_min": "15-minute resolution",
                  "1_hour": "1-hour resolution",
                  "4_hour": "4-hour resolution"}

    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"
    print(f"  Excel engine: {engine}")

    with pd.ExcelWriter(out_xlsx, engine=engine) as writer:
        for horizon in HORIZONS:
            sheet_name = "6h prediction" if horizon == "is_sepsis_6h" else "12h prediction"
            row = 0
            sub = df[df["Horizon"] == horizon]
            if sub.empty:
                pd.DataFrame({"Note": ["No data"]}).to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            for res in res_order:
                res_sub = sub[sub["Resolution"] == res]
                if res_sub.empty:
                    title_df = pd.DataFrame({res_titles[res]: ["(no results — run --resolution " + res + ")"]})
                    title_df.to_excel(writer, sheet_name=sheet_name,
                                      startrow=row, startcol=0, index=False)
                    row += 3
                    continue

                title_df = pd.DataFrame({res_titles[res]: []})
                title_df.to_excel(writer, sheet_name=sheet_name,
                                  startrow=row, startcol=0, index=False)
                row += 1
                cols_for_table = [c for c in METRIC_COLUMNS if c not in ("Resolution", "Horizon")]
                res_sub[cols_for_table].to_excel(writer, sheet_name=sheet_name,
                                                 startrow=row, startcol=0, index=False)
                row += len(res_sub) + 3  # spacer

    print(f"✓ Excel written: {out_xlsx}")
    return out_xlsx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # EDIT: --resolution default ("all") and choices
    ap.add_argument("--resolution", default="all",
                    choices=list(DATASETS.keys()) + ["all"],
                    help="Which resolution to run.")
    # EDIT: --etiology-train default (ETIOLOGY_TRAIN_DEFAULT above)
    ap.add_argument("--etiology-train", default=str(ETIOLOGY_TRAIN_DEFAULT),
                    help="Path to v8_dataset_1h_etiology_train.parquet")
    # EDIT: --etiology-test default (ETIOLOGY_TEST_DEFAULT above)
    ap.add_argument("--etiology-test", default=str(ETIOLOGY_TEST_DEFAULT),
                    help="Path to v8_dataset_1h_etiology_test.parquet")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing checkpoints and rerun from scratch.")
    ap.add_argument("--build-excel", action="store_true",
                    help="Skip computation, only assemble Excel from existing per-resolution outputs.")
    # EDIT: --max-jobs default (RF_N_JOBS = 4)
    ap.add_argument("--max-jobs", type=int, default=RF_N_JOBS,
                    help="Cap on parallel jobs for RF/XGB. Default 4 (safe on 16 GB machine). "
                         "Use -1 for all cores on a machine with abundant RAM.")
    ap.add_argument("--no-downcast", action="store_true",
                    help="Skip float64->float32 downcast (use full precision; needs more RAM).")
    return ap.parse_args()


def main():
    global RF_N_JOBS, DOWNCAST
    args = parse_args()
    args.etiology_train = Path(args.etiology_train)
    args.etiology_test = Path(args.etiology_test)
    RF_N_JOBS = args.max_jobs
    DOWNCAST = not args.no_downcast
    print(f"[config] RF_N_JOBS={RF_N_JOBS}  DOWNCAST={DOWNCAST}  "
          f"psutil_available={_HAVE_PSUTIL}", flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.build_excel:
        build_excel(args)
        return

    if args.resolution == "all":
        # EDIT: resolution run order when --resolution all
        order = ["4_hour", "1_hour", "15_min"]
    else:
        order = [args.resolution]

    for res in order:
        try:
            run_resolution(res, args)
        except FileNotFoundError as e:
            print(f"\n[skip] {res}: {e}\n")

    build_excel(args)


if __name__ == "__main__":
    main()
