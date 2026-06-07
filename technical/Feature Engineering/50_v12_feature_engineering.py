# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  50_v12_feature_engineering.py
#  Stage: 3 - Feature Engineering
#
#  PURPOSE
#    Builds the V12 hybrid feature set on the 1-hour cohort, including
#    NEWS / partial-qSOFA clinical composites, expanding-window statistics,
#    rolling SDs and slopes, lag kinematics, and physiological interaction
#    terms. Saves the engineered parquet files and measures the feature
#    lift with a LightGBM baseline.
#
#  INPUTS
#    v3_dataset_1h_train.parquet
#    v3_dataset_1h_test.parquet
#  OUTPUTS
#    v12_dataset_1h_train.parquet
#    v12_dataset_1h_test.parquet
#    50_v12_feature_benchmark_results.txt
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    train_file / test_file  -  1-hour cohort parquet paths
#    target_col              -  'is_sepsis_6h' (prediction HORIZON = 6h)
#    drop_cols               -  identifier/leakage columns stripped before training
#    LightGBM params         -  learning_rate=0.05, num_leaves=31, max_depth=6,
#                               scale_pos_weight=10, random_state=42
#    num_boost_round         -  150
#    early stopping rounds   -  20
#    decision threshold      -  0.5 (probability cut for the confusion matrix)
#    parquet outputs         -  v12_dataset_1h_train.parquet / _test.parquet
#    results output          -  50_v12_feature_benchmark_results.txt
#
#  REQUIRES: pandas, numpy, lightgbm, scikit-learn
# ============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
import time

warnings.filterwarnings('ignore')

def compute_news_score(df):
    """Approximates NEWS score from vitals."""
    hr = df['heart_rate']
    rr = df['resprate']
    spo2 = df['spo2']
    temp = df['temp_c']
    sbp = df['sbp']

    news_hr = np.select([hr <= 40, (hr >= 41) & (hr <= 50), (hr >= 51) & (hr <= 90), (hr >= 91) & (hr <= 110), (hr >= 111) & (hr <= 130), hr >= 131], [3, 1, 0, 1, 2, 3], default=0)
    news_rr = np.select([rr <= 8, (rr >= 9) & (rr <= 11), (rr >= 12) & (rr <= 20), (rr >= 21) & (rr <= 24), rr >= 25], [3, 1, 0, 2, 3], default=0)
    news_spo2 = np.select([spo2 <= 91, (spo2 >= 92) & (spo2 <= 93), (spo2 >= 94) & (spo2 <= 95), spo2 >= 96], [3, 2, 1, 0], default=0)
    news_temp = np.select([temp <= 35.0, (temp >= 35.1) & (temp <= 36.0), (temp >= 36.1) & (temp <= 38.0), (temp >= 38.1) & (temp <= 39.0), temp >= 39.1], [3, 1, 0, 1, 2], default=0)
    news_sbp = np.select([sbp <= 90, (sbp >= 91) & (sbp <= 100), (sbp >= 101) & (sbp <= 110), (sbp >= 111) & (sbp <= 219), sbp >= 220], [3, 2, 1, 0, 3], default=0)

    return news_hr + news_rr + news_spo2 + news_temp + news_sbp

def apply_v12_features(df):
    print("  -> Initializing Layer 1 Feature Generation...")

    # Base Vitals & Forward Fills
    vitals = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']
    df[vitals] = df.groupby('stay_id')[vitals].ffill().bfill()

    # 1. Base Engineered Composites (V2/V3)
    print("  -> Computing Base Composites...")
    df['shock_index'] = df['heart_rate'] / df['sbp']
    df['map'] = (df['sbp'] + (2 * df['dbp'])) / 3
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. Clinical Composite Scores
    print("  -> Computing Clinical Knowledge Composites (qSOFA/NEWS)...")
    df['partial_qsofa'] = (df['sbp'] < 100).astype(int) + (df['resprate'] >= 20).astype(int)
    df['news_score'] = compute_news_score(df)

    grouped = df.groupby('stay_id')

    # Deltas
    df['qsofa_delta'] = df['partial_qsofa'] - grouped['partial_qsofa'].shift(1)
    df['news_delta'] = df['news_score'] - grouped['news_score'].shift(1)

    # 3. Expanding Window Statistics for Vitals
    print("  -> Computing Expanding Windows...")
    for v in vitals:
        # Expanding aggregates per patient
        df[f'exp_min_{v}'] = grouped[v].expanding().min().reset_index(0, drop=True)
        df[f'exp_max_{v}'] = grouped[v].expanding().max().reset_index(0, drop=True)
        df[f'exp_mean_{v}'] = grouped[v].expanding().mean().reset_index(0, drop=True)
        # For standard deviation, minimum 2 periods required. Fill NaNs with 0
        df[f'exp_std_{v}'] = grouped[v].expanding().std().reset_index(0, drop=True).fillna(0)

    # 4. Rolling base stats and Slopes (V3 preservation)
    print("  -> Computing Rolling Standard Deviations and Slopes...")
    for v in vitals:
        df[f'{v}_sd_4h'] = grouped[v].rolling(window=4, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        df[f'{v}_ewma_3h'] = grouped[v].transform(lambda x: x.ewm(span=3, adjust=False).mean())

        # Simple slope proxy: value(t) - value(t-4)
        df[f'slope_4h_{v}'] = df[v] - grouped[v].shift(4)

    # 5. Lagged Differences and Ratios
    print("  -> Computing Lagged Kinematics...")
    for v in vitals:
        lag_val = grouped[v].shift(1)
        df[f'lag_diff_1h_{v}'] = df[v] - lag_val
        # Add epsilon to prevent div by zero
        df[f'lag_ratio_1h_{v}'] = df[v] / (lag_val + 1e-6)

    # 6. Physiological Interaction Composites
    print("  -> Computing Biological Interactions...")
    df['ventilation_perfusion_proxy'] = df['resprate'] * (100 - df['spo2'])
    df['tachycardia_excess'] = df['heart_rate'] - (10 * (df['temp_c'] - 37))
    df['perfusion_adequacy'] = df['heart_rate'] / ((df['sbp'] + 1e-6) * (df['spo2'] / 100 + 1e-6))

    # Rolling correlation (requires rolling with 2 columns, done via pandas apply or rolling trick)
    # df.rolling().corr is efficient
    hr_rr_corr = grouped.apply(lambda g: g['heart_rate'].rolling(4, min_periods=2).corr(g['resprate'])).reset_index(0, drop=True)
    df['cardiorespiratory_coupling_4h'] = hr_rr_corr.fillna(0)

    # Cleanup any lingering numerical instabilities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0) # For differences/slopes at start of stay where shift() creates NaN

    # Restore any categorical non-numeric targets that might have been zeroed out inappropriately if they were NaN
    # We will assume targets were clean before passing.

    return df

def run_layer1_extraction():
    print("=====================================================================")
    print("LAYER 1: V12 Inspiration Hybrid Feature Engineering")
    print("=====================================================================")

    t0 = time.time()
    train_file = 'v3_dataset_1h_train.parquet'  # EDIT: 1-hour train cohort parquet path
    test_file = 'v3_dataset_1h_test.parquet'    # EDIT: 1-hour test cohort parquet path

    print(f"Loading Base Arrays...")
    df_train = pd.read_parquet(train_file).sort_values(['stay_id', 'charttime'])
    df_test = pd.read_parquet(test_file).sort_values(['stay_id', 'charttime'])

    print(f"\\n⚙️ Engineering Train Cohort ({len(df_train)} rows)...")
    df_train_v12 = apply_v12_features(df_train)

    print(f"\\n⚙️ Engineering Test Cohort ({len(df_test)} rows)...")
    df_test_v12 = apply_v12_features(df_test)

    t1 = time.time()
    print(f"\\n✅ Engineered {len(df_train_v12.columns)} features. Time: {(t1-t0)/60:.1f} mins.")

    # Save the huge V12 datasets so we never have to compute them again!
    print("Saving V12 Parquet files...")
    df_train_v12.to_parquet('v12_dataset_1h_train.parquet', engine='pyarrow')  # EDIT: engineered train parquet output
    df_test_v12.to_parquet('v12_dataset_1h_test.parquet', engine='pyarrow')    # EDIT: engineered test parquet output

    # -------------------------------------------------------------
    # LIGHTGBM BASELINE EVALUATION (Measure Lift)
    # -------------------------------------------------------------
    print("\\n🚀 Pipeline 1: Computing V12 Feature Baseline Lift (LightGBM)")

    # EDIT: identifier/leakage columns stripped before training
    drop_cols = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h', 'stay_id', 'charttime', 'intime', 'sepsis3_time']
    target_col = 'is_sepsis_6h'  # EDIT: target/label column (prediction HORIZON = 6h)

    X_train = df_train_v12.drop(columns=[c for c in drop_cols if c in df_train_v12.columns])
    y_train = df_train_v12[target_col]
    X_test = df_test_v12.drop(columns=[c for c in drop_cols if c in df_test_v12.columns])
    y_test = df_test_v12[target_col]

    # LightGBM handles NaN natively, so no strict imputation needed!
    # But we filled mostly with 0 above. We can just use it directly.
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # EDIT: LightGBM hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,       # EDIT: learning rate
        'num_leaves': 31,            # EDIT: max leaves per tree
        'max_depth': 6,              # EDIT: max tree depth
        'scale_pos_weight': 10,      # EDIT: class-imbalance weight (baseline)
        'n_jobs': -1,
        'random_state': 42,          # EDIT: random seed
        'verbose': -1
    }

    print("Training LightGBM on V12 Features...")
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=150,  # EDIT: number of boosting rounds
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]  # EDIT: early-stopping patience
    )

    p_test = lgb_model.predict(X_test)
    y_pred = (p_test >= 0.5).astype(int)  # EDIT: decision threshold

    auroc = roc_auc_score(y_test, p_test)
    auprc = average_precision_score(y_test, p_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / max((tp + fn), 1)
    fpr = fp / max((fp + tn), 1)

    # Extract Feature Importances to inform Pruning (if desired later)
    importance = lgb_model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': X_train.columns, 'gain': importance}).sort_values(by='gain', ascending=False)

    print("\\n=====================================================================")
    print("V12 Layer 1 Baseline Results (Natural Prevalence Evaluation)")
    print("=====================================================================")
    print(f"  ROC-AUC: {auroc:.4f}  (Was ~0.76 V3 Baseline)")
    print(f"  AUPRC:   {auprc:.4f}")
    print(f"  Sens:    {sens:.4f} | FPR: {fpr:.4f}")

    print("\\nTop 15 Features Driving Lift:")
    print(feat_imp.head(15).to_string(index=False))

    out_text = f"V12 Inspiration Feature Set Benchmark\\n"
    out_text += f"ROC-AUC: {auroc:.4f}\\n"
    out_text += f"AUPRC:   {auprc:.4f}\\n"
    out_text += f"Sensitivity: {sens:.4f}\\n"
    out_text += f"FPR: {fpr:.4f}\\n"

    with open('50_v12_feature_benchmark_results.txt', 'w') as f:  # EDIT: results output filename
        f.write(out_text)

if __name__ == '__main__':
    run_layer1_extraction()
