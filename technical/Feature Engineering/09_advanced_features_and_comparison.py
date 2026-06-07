# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  09_advanced_features_and_comparison.py
#  Stage: 3 - Feature Engineering
#
#  PURPOSE
#    Engineers advanced clinical and temporal features (Shock Index, MAP,
#    EWMA, rolling std, diffs) at three sampling resolutions and trains a
#    LightGBM classifier on each to compare how monitoring frequency
#    (4h vs 1h vs 15m) affects sepsis prediction performance.
#
#  INPUTS
#    dataset_a_4h_train.parquet / dataset_a_4h_test.parquet
#    dataset_b_1h_train.parquet / dataset_b_1h_test.parquet
#    dataset_c_15m_train.parquet / dataset_c_15m_test.parquet
#  OUTPUTS
#    11_viz_Comparative_PR_Curves.png   (precision-recall comparison figure)
#    11_table_comparative_results.csv   (AUPRC / ROC-AUC per resolution)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    dataset parquet paths   -  the three train/test pairs to load
#    interval_hours          -  sampling RESOLUTION per dataset (4.0 / 1.0 / 0.25)
#    target column           -  'is_sepsis'
#    LightGBM hyperparams    -  n_estimators=200, learning_rate=0.05,
#                               scale_pos_weight=auto, max_depth=7, num_leaves=63,
#                               random_state=42
#    figure output           -  11_viz_Comparative_PR_Curves.png, dpi=300
#    table output            -  11_table_comparative_results.csv
#
#  REQUIRES: pandas, numpy, matplotlib, seaborn, lightgbm, scikit-learn
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import os

print("-------------------------------------------------------------------------")
print("Phase 6: V2 Feature Engineering & Comparative Interval Analysis")
print("-------------------------------------------------------------------------\n")

def v2_engineer_features(data, interval_hours):
    """
    Applies Advanced Clinical Feature Engineering (SI, MAP) and
    Temporal Dynamics (EWMA, 8H Lookbacks) adapted dynamically to the row interval.
    """
    # Defensive copy
    df = data.copy()

    # 1. Clinical Composite Scores
    # Shock Index = HR / SBP. (Add 1 to SBP to prevent zero division theoretically)
    df['shock_index'] = df['heart_rate'] / (df['sbp'] + 1)

    # Mean Arterial Pressure (MAP) = (SBP + 2*DBP) / 3
    df['map'] = (df['sbp'] + 2 * df['dbp']) / 3

    # Base Vitals + New Composites
    analyze_cols = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp', 'shock_index', 'map']

    # Calculate how many rows equal 4 hours and 8 hours based on the dataset's resolution
    rows_per_4h = max(1, int(4 / interval_hours))
    rows_per_8h = max(1, int(8 / interval_hours))

    grouped = df.groupby('stay_id')

    for v in analyze_cols:
        # A. Exponentially Weighted Moving Average (EWMA) to prioritize recent changes
        # span=rows_per_4h means the "half-life" of the weight aligns with a 4 hour window
        df[f'{v}_ewma_4h'] = grouped[v].transform(lambda x: x.ewm(span=rows_per_4h, min_periods=1).mean())

        # B. Rate of Change (Slope / Diff) over 4 hours
        df[f'{v}_diff_4h'] = grouped[v].diff(periods=rows_per_4h)

        # C. Extended Historical Volatility (8-Hour Standard Deviation)
        df[f'{v}_std_8h'] = grouped[v].transform(lambda x: x.rolling(window=rows_per_8h, min_periods=1).std())

    # Forward fill local NaNs created by early diffs/rolling, then fill absolute NaNs with 0
    df = df.fillna(method='ffill').fillna(0)

    # Separate features and target
    y = df['is_sepsis'].values  # EDIT: target/label column name

    # Drop identifying/target columns
    drop_cols = ['is_sepsis', 'stay_id', 'charttime', 'index', 'level_0']
    if df.index.name == 'charttime': df = df.reset_index()
    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return X, y

def load_and_prep(train_file, test_file, interval_hours):
    print(f"Loading {train_file}...")
    df_train = pd.read_parquet(train_file)
    df_test = pd.read_parquet(test_file)

    X_train, y_train = v2_engineer_features(df_train, interval_hours)
    X_test, y_test = v2_engineer_features(df_test, interval_hours)

    return X_train, y_train, X_test, y_test

# ---------------------------------------------------------------------------
# 1. Load and Engineer the 3 Datasets
# ---------------------------------------------------------------------------
print("\n[1/3] V2 Engineering: Applying Shock Index, MAP, and EWMA to all Datasets...")
# EDIT: dataset parquet paths and the sampling RESOLUTION (interval_hours) for each
datasets = {
    "4-Hour (Standard Care)": load_and_prep('dataset_a_4h_train.parquet', 'dataset_a_4h_test.parquet', interval_hours=4.0),
    "1-Hour (Wearable Proxy)": load_and_prep('dataset_b_1h_train.parquet', 'dataset_b_1h_test.parquet', interval_hours=1.0),
    "15-Minute (High-Frequency)": load_and_prep('dataset_c_15m_train.parquet', 'dataset_c_15m_test.parquet', interval_hours=0.25)
}

# ---------------------------------------------------------------------------
# 2. Train and Evaluate the Champion LightGBM Model
# ---------------------------------------------------------------------------
print("\n[2/3] Training Champion LightGBM Models across explicit Temporal Resolutions...")
results = {}

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))

colors = {"4-Hour (Standard Care)": "#e74c3c", "1-Hour (Wearable Proxy)": "#3498db", "15-Minute (High-Frequency)": "#2ecc71"}

for name, (X_train, y_train, X_test, y_test) in datasets.items():
    print(f" -> Training LightGBM on {name}...")

    # Calculate native class imbalance weight
    scale_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))

    # Initialize LightGBM (Gradient Boosting optimized for tabular data)
    model = lgb.LGBMClassifier(
        n_estimators=200,                 # EDIT: number of boosting trees
        learning_rate=0.05,               # EDIT: learning rate
        scale_pos_weight=scale_weight,    # EDIT: class-imbalance weight (auto from data)
        max_depth=7,                      # EDIT: max tree depth
        num_leaves=63,                    # EDIT: max leaves per tree
        random_state=42,                  # EDIT: random seed
        n_jobs=-1,
        verbose=-1 # Suppress warnings
    )

    # Fit and Predict
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    auprc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)

    results[name] = {'AUPRC': auprc, 'ROC-AUC': roc_auc}
    print(f"    [+] {name} | V2 AUPRC: {auprc:.4f} | V2 ROC-AUC: {roc_auc:.4f}")

    # Plot PR Curve for Comparison
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision, lw=3, color=colors[name], label=f'{name} (AUPRC = {auprc:.3f})')

# ---------------------------------------------------------------------------
# 3. Export Final Visualizations & Data
# ---------------------------------------------------------------------------
print("\n[3/3] Generating Final Comparative PR-Curve...")
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
plt.title('Core Thesis Analysis: Effect of Monitoring Frequency on AI Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="upper right", fontsize=11)
plt.tight_layout()
plt.savefig('11_viz_Comparative_PR_Curves.png', dpi=300)  # EDIT: figure output filename and DPI
plt.close()

# Save numerical results to CSV for thesis ingestion
res_df = pd.DataFrame(results).T
res_df.to_csv('11_table_comparative_results.csv')  # EDIT: results table output filename

print("\n✅ V2 Advanced Engineering and 3-Tier Comparative Analysis Complete!")
print("The final theoretical benchmarking visual (11_viz_Comparative_PR_Curves.png) is ready.")
