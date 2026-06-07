# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  benchmark_models.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Benchmarks five standard classifiers (Dummy, Decision Tree, Random Forest,
#    XGBoost, Logistic Regression) on the full engineered-feature set against
#    the 6-hour sepsis target across the 15-min, 1-hour and 4-hour resolutions.
#    Writes results incrementally and supports resume.
#
#  INPUTS
#    ../Data/All engineered features/Dataset_all_engineered_{15min,1h,4h}_{train,test}.parquet
#  OUTPUTS
#    ../Results/standard_algos_performance_all_engineered.csv  (resumable)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    DATA_DIR / RESULTS_DIR  -  relative paths; assumes you run from the
#                          original technical/Models/ directory
#    DATASETS           -  resolution -> (train, test) parquet filenames
#    TARGET             -  label column, is_sepsis_6h
#    DROP_COLS          -  metadata/label columns excluded from features
#    15-min downsample  -  train/test sample(frac=0.5, random_state=42)
#    Dummy Classifier   -  strategy='prior'
#    Decision Tree      -  max_depth=10, class_weight='balanced', random_state=42
#    Random Forest      -  n_estimators=100, max_depth=10,
#                          class_weight='balanced', random_state=42
#    XGBoost            -  n_estimators=100, max_depth=6, learning_rate=0.1,
#                          scale_pos_weight=10, random_state=42
#    Logistic Regression-  max_iter=100, solver='saga',
#                          class_weight='balanced', random_state=42
#
#  REQUIRES: scikit-learn, xgboost, pandas, numpy
# ============================================================================

import pandas as pd
import numpy as np
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = '../Data/All engineered features'  # EDIT: input parquet folder (assumes original technical/Models/ cwd)
RESULTS_DIR = '../Results'  # EDIT: output folder

DATASETS = {  # EDIT: resolution -> (train, test) parquet filenames
    '15_min': ('Dataset_all_engineered_15min_train.parquet', 'Dataset_all_engineered_15min_test.parquet'),
    '1_hour': ('Dataset_all_engineered_1h_train.parquet', 'Dataset_all_engineered_1h_test.parquet'),
    '4_hour': ('Dataset_all_engineered_4h_train.parquet', 'Dataset_all_engineered_4h_test.parquet')
}

TARGET = 'is_sepsis_6h'  # EDIT: label column
DROP_COLS = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h', 'stay_id', 'charttime', 'intime', 'sepsis3_time', 'time_since_ICU_admit_hours']  # EDIT: columns excluded from features

def load_and_preprocess(train_file, test_file, res_name):
    print(f"Loading data from {train_file}...")
    df_train = pd.read_parquet(train_file)
    if res_name == '15_min':
        print("  -> Downsampling 15_min train set by 50% to save memory...")
        df_train = df_train.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed

    df_test = pd.read_parquet(test_file)
    if res_name == '15_min':
        print("  -> Downsampling 15_min test set by 50% to save memory...")
        df_test = df_test.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed

    # Separate features and target
    y_train = df_train[TARGET]
    X_train = df_train.drop(columns=[c for c in DROP_COLS if c in df_train.columns])

    y_test = df_test[TARGET]
    X_test = df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns])

    # Impute missing values with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Downcast to float32 to save memory
    for col in X_train.select_dtypes(include=['float64']).columns:
        X_train[col] = X_train[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')

    return X_train, y_train, X_test, y_test

def get_metrics(name, res_name, y_true, y_pred, y_prob, y_train_true, y_train_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)

    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc = accuracy_score(y_true, y_pred)

    return {
        'Resolution': f"{res_name} ({name})",
        'Train Accuracy (%)': f"{train_acc*100:.2f}%",
        'Test Accuracy (%)': f"{test_acc*100:.2f}%",
        'Precision': f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        'Recall': f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        'F1 Score': f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        'ROC AUC': f"{roc_auc_score(y_true, y_prob):.4f}",
        'AUPRC': f"{average_precision_score(y_true, y_prob):.4f}",
        'TP (Count)': tp,
        'FN (Count)': fn,
        'FP (Count)': fp,
        'TN (Count)': tn,
        'TP (%)': f"{(tp/total)*100:.2f}%",
        'FN (%)': f"{(fn/total)*100:.2f}%",
        'FP (%)': f"{(fp/total)*100:.2f}%",
        'TN (%)': f"{(tn/total)*100:.2f}%"
    }

def run_benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, 'standard_algos_performance_all_engineered.csv')

    # Load existing results to resume
    if os.path.exists(out_file):
        results_df = pd.read_csv(out_file)
        all_results = results_df.to_dict('records')
    else:
        all_results = []

    for res_name, (train_filename, test_filename) in DATASETS.items():
        print(f"\n=======================================================")
        print(f"Processing Resolution: {res_name}")
        print(f"=======================================================")
        train_path = os.path.join(DATA_DIR, train_filename)
        test_path = os.path.join(DATA_DIR, test_filename)

        models = [
            ('Dummy Classifier', DummyClassifier(strategy='prior')),  # EDIT: dummy strategy
            ('Decision Tree', DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')),  # EDIT: Decision Tree hyperparameters
            ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),  # EDIT: Random Forest hyperparameters
            ('XGBoost', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, scale_pos_weight=10, n_jobs=-1)),  # EDIT: XGBoost hyperparameters
            ('Logistic Regression', LogisticRegression(max_iter=100, solver='saga', random_state=42, class_weight='balanced', n_jobs=-1))  # EDIT: Logistic Regression hyperparameters
        ]

        models_to_run = [name for name, _ in models if not any(r['Resolution'] == f"{res_name} ({name})" for r in all_results)]

        if not models_to_run:
            print(f"All models for {res_name} already completed. Skipping.")
            continue

        X_train, y_train, X_test, y_test = load_and_preprocess(train_path, test_path, res_name)

        for name, model in models:
            if name not in models_to_run:
                continue

            print(f"  -> Training {name}...")
            t0 = time.time()

            if name == 'Logistic Regression':
                print("     Scaling data in-place for Logistic Regression to save memory...")
                scaler = StandardScaler()
                X_train[:] = scaler.fit_transform(X_train)
                X_test[:] = scaler.transform(X_test)

                # Clip inf and extremely large values that cause warnings in LR
                X_train = np.clip(X_train, -1e6, 1e6)
                X_test = np.clip(X_test, -1e6, 1e6)
                X_train[np.isnan(X_train)] = 0
                X_test[np.isnan(X_test)] = 0

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_test_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_test_prob = model.decision_function(X_test)

            metrics = get_metrics(name, res_name, y_test, y_test_pred, y_test_prob, y_train, y_train_pred)
            all_results.append(metrics)

            # Save incrementally
            pd.DataFrame(all_results).to_csv(out_file, index=False)

            print(f"  -> Completed {name} in {time.time()-t0:.1f}s. ROC AUC: {metrics['ROC AUC']}")

    print(f"\nAll Results saved to {out_file}")
    print(pd.DataFrame(all_results).to_string(index=False))

if __name__ == '__main__':
    run_benchmark()
