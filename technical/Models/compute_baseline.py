# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  compute_baseline.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Computes the random stratified-guessing baseline for the 6-hour sepsis
#    target across the 15-min, 1-hour and 4-hour v3 datasets. Loads only the
#    target column for speed, fits a stratified DummyClassifier, and writes a
#    CSV plus a markdown overview of the "just guessing" performance floor.
#
#  INPUTS
#    ../Data/v3_dataset_{15m,1h,4h}_{train,test}.parquet
#  OUTPUTS
#    baseline_performance/guessing_baselines.csv
#    baseline_performance/Guessing_Baselines_Overview.md
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    resolutions        -  resolution -> (train, test) parquet paths; relative
#                          ../Data/...; assumes you run from the original
#                          technical/Models/ directory
#    target             -  label column, is_sepsis_6h
#    Dummy Classifier   -  strategy='stratified', random_state=42
#    Output dir / files -  baseline_performance/ folder and the two output names
#
#  REQUIRES: scikit-learn, pandas, numpy
# ============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.dummy import DummyClassifier
import warnings

warnings.filterwarnings('ignore')

os.makedirs('baseline_performance', exist_ok=True)  # EDIT: output directory

resolutions = {  # EDIT: resolution -> (train, test) parquet paths (assumes original technical/Models/ cwd)
    '15_min': ('../Data/v3_dataset_15m_train.parquet', '../Data/v3_dataset_15m_test.parquet'),
    '1_hour': ('../Data/v3_dataset_1h_train.parquet', '../Data/v3_dataset_1h_test.parquet'),
    '4_hour': ('../Data/v3_dataset_4h_train.parquet', '../Data/v3_dataset_4h_test.parquet')
}

target = 'is_sepsis_6h'  # EDIT: label column
results = []

for res_name, (train_path, test_path) in resolutions.items():
    print(f"Processing {res_name}...")

    # Only load the target column to save memory and time
    df_train = pd.read_parquet(train_path, columns=[target]).dropna(subset=[target])
    df_test = pd.read_parquet(test_path, columns=[target]).dropna(subset=[target])

    y_train = df_train[target].values
    y_test = df_test[target].values

    # We use Stratified Random Guessing which guesses proportionately to the class distribution
    dummy = DummyClassifier(strategy='stratified', random_state=42)  # EDIT: dummy strategy / seed
    # create dummy features
    X_train_dummy = np.zeros((len(y_train), 1))
    X_test_dummy = np.zeros((len(y_test), 1))

    dummy.fit(X_train_dummy, y_train)

    # Train metrics
    y_train_pred = dummy.predict(X_train_dummy)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Test predictions
    y_test_pred = dummy.predict(X_test_dummy)
    y_test_prob = dummy.predict_proba(X_test_dummy)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    auroc = roc_auc_score(y_test, y_test_prob)
    auprc = average_precision_score(y_test, y_test_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    total = len(y_test)

    tn_pct = tn / total * 100
    fp_pct = fp / total * 100
    fn_pct = fn / total * 100
    tp_pct = tp / total * 100

    results.append({
        'Resolution': res_name,
        'Train Accuracy (%)': f"{train_acc*100:.2f}%",
        'Test Accuracy (%)': f"{test_acc*100:.2f}%",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1 Score': f"{f1:.4f}",
        'ROC AUC': f"{auroc:.4f}",
        'AUPRC': f"{auprc:.4f}",
        'TP (Count)': tp,
        'FN (Count)': fn,
        'FP (Count)': fp,
        'TN (Count)': tn,
        'TP (%)': f"{tp_pct:.2f}%",
        'FN (%)': f"{fn_pct:.2f}%",
        'FP (%)': f"{fp_pct:.2f}%",
        'TN (%)': f"{tn_pct:.2f}%"
    })

results_df = pd.DataFrame(results)
csv_out = 'baseline_performance/guessing_baselines.csv'  # EDIT: output CSV path
md_out = 'baseline_performance/Guessing_Baselines_Overview.md'  # EDIT: output markdown path

results_df.to_csv(csv_out, index=False)

with open(md_out, 'w') as f:
    f.write('# Baseline Performance (Random Stratified Guessing)\n\n')
    f.write('This overview shows the baseline "just guessing" performance for our three temporal resolutions. The guesses are generated using a stratified strategy proportional to the overall target class prevalence.\n\n')
    # Custom markdown generation
    cols = list(results_df.columns)
    f.write('| ' + ' | '.join(cols) + ' |\n')
    f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
    for _, row in results_df.iterrows():
        f.write('| ' + ' | '.join(str(row[c]) for c in cols) + ' |\n')


print("Completed. Results saved in baseline_performance/")
