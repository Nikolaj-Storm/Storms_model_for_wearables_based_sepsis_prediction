# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  benchmark_basic_features.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Trains Random Forest, Logistic Regression and XGBoost on raw vitals only
#    (8 features, no time-since) against the 6-hour sepsis target across the
#    15-min, 1-hour and 4-hour v3 datasets. Supports resume from a prior CSV.
#    Output feeds compile_excel.py as a "Raw Vitals Only" sheet.
#
#  INPUTS
#    ../Data/v3_dataset_{15m,1h,4h}_{train,test}.parquet
#  OUTPUTS
#    ../Results/raw_vitals_performance.csv  (written incrementally; resumable)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    DATA_DIR / RESULTS_DIR / OUT_FILE  -  relative paths; assumes you run from
#                          the original technical/Models/ directory
#    DATASETS           -  resolution -> (train, test) parquet filenames
#    TARGET             -  label column, is_sepsis_6h
#    DROP_COLS          -  metadata/label columns excluded from features
#    BASIC_FEATURES     -  8 raw vital-sign features used
#    15-min downsample  -  train/test sample(frac=0.5, random_state=42)
#    Random Forest      -  n_estimators=100, max_depth=10,
#                          class_weight='balanced', random_state=42
#    Logistic Regression-  max_iter=500, solver='saga',
#                          class_weight='balanced', random_state=42
#    XGBoost            -  n_estimators=100, max_depth=6, learning_rate=0.1,
#                          scale_pos_weight=10, random_state=42
#
#  REQUIRES: scikit-learn, xgboost, pandas, numpy
# ============================================================================

"""
benchmark_basic_features.py
────────────────────────────────────────────────────────────────────────────
Train Random Forest, Logistic Regression, and XGBoost on raw vitals only
(no time-since feature):
    age, weight_kg, heart_rate, resprate, spo2, temp_c, sbp, dbp

Uses the v3 datasets (which already contain exactly these columns + targets).
Results are saved to ../Results/raw_vitals_performance.csv so that
compile_excel.py can pick them up and add a "Raw Vitals Only" sheet.
"""

import pandas as pd
import numpy as np
import os
import time
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = '../Data'  # EDIT: input parquet folder (assumes original technical/Models/ cwd)
RESULTS_DIR = '../Results'  # EDIT: output folder
OUT_FILE    = os.path.join(RESULTS_DIR, 'raw_vitals_performance.csv')  # EDIT: output CSV path

DATASETS = {  # EDIT: resolution -> (train, test) parquet filenames
    '15_min': ('v3_dataset_15m_train.parquet', 'v3_dataset_15m_test.parquet'),
    '1_hour': ('v3_dataset_1h_train.parquet',  'v3_dataset_1h_test.parquet'),
    '4_hour': ('v3_dataset_4h_train.parquet',  'v3_dataset_4h_test.parquet'),
}

TARGET    = 'is_sepsis_6h'  # EDIT: label column
DROP_COLS = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h',
             'stay_id', 'charttime', 'intime', 'sepsis3_time']  # EDIT: columns excluded from features

# 8 raw vital-sign features — time_since excluded to isolate pure signal.
BASIC_FEATURES = [  # EDIT: raw vital-sign feature list
    'age', 'weight_kg',
    'heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp',
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(train_file: str, test_file: str, res_name: str):
    print(f"  Loading {train_file} …")
    df_train = pd.read_parquet(train_file)
    df_test  = pd.read_parquet(test_file)

    # 15-min set is large – downsample by 50 % (same as benchmark_models.py)
    if res_name == '15_min':
        print("  → Downsampling 15_min by 50 % to save memory …")
        df_train = df_train.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed
        df_test  = df_test.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed

    # Drop rows where target is NaN
    df_train = df_train.dropna(subset=[TARGET])
    df_test  = df_test.dropna(subset=[TARGET])

    y_train = df_train[TARGET].astype(int)
    y_test  = df_test[TARGET].astype(int)

    # Keep only the basic features that actually exist in this file
    keep = [c for c in BASIC_FEATURES if c in df_train.columns]
    X_train = df_train[keep].fillna(0).astype('float32')
    X_test  = df_test[keep].fillna(0).astype('float32')

    print(f"  Features used ({len(keep)}): {keep}")
    return X_train, y_train, X_test, y_test


def get_metrics(name: str, res_name: str,
                y_true, y_pred, y_prob,
                y_train_true, y_train_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    return {
        'Resolution':         f"{res_name} ({name})",
        'Train Accuracy (%)': f"{accuracy_score(y_train_true, y_train_pred)*100:.2f}%",
        'Test Accuracy (%)':  f"{accuracy_score(y_true, y_pred)*100:.2f}%",
        'Precision':          f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        'Recall':             f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        'F1 Score':           f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        'ROC AUC':            f"{roc_auc_score(y_true, y_prob):.4f}",
        'AUPRC':              f"{average_precision_score(y_true, y_prob):.4f}",
        'TP (Count)': tp, 'FN (Count)': fn, 'FP (Count)': fp, 'TN (Count)': tn,
        'TP (%)': f"{tp/total*100:.2f}%", 'FN (%)': f"{fn/total*100:.2f}%",
        'FP (%)': f"{fp/total*100:.2f}%", 'TN (%)': f"{tn/total*100:.2f}%",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume support
    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE).to_dict('records')
    else:
        existing = []
    done_keys = {r['Resolution'] for r in existing}

    all_results = list(existing)

    models_cfg = [
        # EDIT: Random Forest hyperparameters
        ('Random Forest',
         RandomForestClassifier(n_estimators=100, max_depth=10,
                                random_state=42, class_weight='balanced',
                                n_jobs=-1)),
        # EDIT: Logistic Regression hyperparameters
        ('Logistic Regression',
         LogisticRegression(max_iter=500, solver='saga',
                            random_state=42, class_weight='balanced',
                            n_jobs=-1)),
        # EDIT: XGBoost hyperparameters
        ('XGBoost',
         XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                       random_state=42, scale_pos_weight=10, n_jobs=-1)),
    ]

    for res_name, (train_fn, test_fn) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Resolution: {res_name}")
        print(f"{'='*60}")

        train_path = os.path.join(DATA_DIR, train_fn)
        test_path  = os.path.join(DATA_DIR, test_fn)

        to_run = [n for n, _ in models_cfg
                  if f"{res_name} ({n})" not in done_keys]
        if not to_run:
            print("All models already done – skipping.")
            continue

        X_train, y_train, X_test, y_test = load_data(train_path, test_path, res_name)

        # Keep a clean float copy before potential in-place scaling
        X_train_orig = X_train.copy()
        X_test_orig  = X_test.copy()

        for name, model in models_cfg:
            if name not in to_run:
                continue

            print(f"\n  → Training {name} …")
            t0 = time.time()

            if name == 'Logistic Regression':
                scaler   = StandardScaler()
                X_tr_fit = scaler.fit_transform(X_train_orig)
                X_te_fit = scaler.transform(X_test_orig)
                X_tr_fit = np.clip(X_tr_fit, -1e6, 1e6)
                X_te_fit = np.clip(X_te_fit, -1e6, 1e6)
                X_tr_fit = np.nan_to_num(X_tr_fit, nan=0.0)
                X_te_fit = np.nan_to_num(X_te_fit, nan=0.0)
                model.fit(X_tr_fit, y_train)
                y_train_pred = model.predict(X_tr_fit)
                y_test_pred  = model.predict(X_te_fit)
                y_test_prob  = model.predict_proba(X_te_fit)[:, 1]
            else:
                model.fit(X_train_orig, y_train)
                y_train_pred = model.predict(X_train_orig)
                y_test_pred  = model.predict(X_test_orig)
                y_test_prob  = model.predict_proba(X_test_orig)[:, 1]

            metrics = get_metrics(name, res_name,
                                  y_test, y_test_pred, y_test_prob,
                                  y_train, y_train_pred)
            all_results.append(metrics)
            pd.DataFrame(all_results).to_csv(OUT_FILE, index=False)

            elapsed = time.time() - t0
            print(f"     Done in {elapsed:.1f}s | ROC AUC: {metrics['ROC AUC']} | AUPRC: {metrics['AUPRC']}")

    print(f"\n✓ All results saved → {OUT_FILE}")
    print(pd.DataFrame(all_results).to_string(index=False))


if __name__ == '__main__':
    run()
