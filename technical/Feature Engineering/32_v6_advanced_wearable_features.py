# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  32_v6_advanced_wearable_features.py
#  Stage: 3 - Feature Engineering
#
#  PURPOSE
#    Benchmarks a V3 standard feature set against a V6 advanced
#    wearable-derivable feature set (Pulse Pressure, RPP, MSI, HRV/RRV
#    proxies, temperature trajectories). Trains two XGBoost classifiers on
#    the 1-hour cohort and reports ROC-AUC, AUPRC, sensitivity, and FPR.
#
#  INPUTS
#    v3_dataset_1h_train.parquet
#    v3_dataset_1h_test.parquet
#  OUTPUTS
#    32_v6_xgb_advanced_features_results.txt
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    train_file / test_file  -  1-hour cohort parquet paths
#    target_col              -  'is_sepsis_6h' (prediction HORIZON = 6h)
#    drop_cols               -  identifier/leakage columns stripped before training
#    SimpleImputer strategy  -  'median'
#    XGBoost hyperparams     -  scale_pos_weight=10, max_depth=6,
#                               learning_rate=0.05, n_estimators=100,
#                               random_state=42
#    results output          -  32_v6_xgb_advanced_features_results.txt
#
#  REQUIRES: pandas, numpy, xgboost, scikit-learn
# ============================================================================
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

def apply_base_v3_features(df):
    """
    Applies the simple V3 features for the Baseline evaluation.
    """
    df['shock_index'] = df['heart_rate'] / df['sbp']
    df['map'] = (df['sbp'] + (2 * df['dbp'])) / 3
    df['shock_index'] = df['shock_index'].replace([np.inf, -np.inf], np.nan)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    vitals_cols = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp', 'shock_index', 'map']
    grouped = df.groupby('stay_id')
    for vital in vitals_cols:
        df[f'{vital}_ewma_3h'] = grouped[vital].transform(lambda x: x.ewm(span=3, adjust=False).mean())
        df[f'{vital}_sd_8h'] = grouped[vital].rolling(window=8, min_periods=1).std().reset_index(0, drop=True)
        df[f'{vital}_sd_8h'] = df[f'{vital}_sd_8h'].fillna(0)
    return df

def apply_advanced_v6_features(df):
    """
    Engineers the complex biological interaction and trajectory features
    derivable explicitly from wearable proxies.
    """
    print("  -> Computing Biological Interactions (PP, RPP, MSI)...")
    df['map'] = (df['sbp'] + (2 * df['dbp'])) / 3

    # 1. Pulse Pressure (Arterial stiffness)
    df['pulse_pressure'] = df['sbp'] - df['dbp']

    # 2. Rate Pressure Product (Myocardial Oxygen Demand)
    df['rpp'] = df['heart_rate'] * df['sbp']

    # 3. Modified Shock Index (Systemic Hypoperfusion)
    df['msi'] = df['heart_rate'] / df['map']
    df['msi'] = df['msi'].replace([np.inf, -np.inf], np.nan)

    print("  -> Computing HRV/RRV Proxies and Temperature Trajectories...")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    grouped = df.groupby('stay_id')

    # 4. HRV/RRV High-Res Proxies (Rolling Variance across tight windows)
    for hours in [1, 2, 4]:
        df[f'hr_var_{hours}h'] = grouped['heart_rate'].rolling(window=hours, min_periods=1).var().reset_index(0, drop=True)
        df[f'rr_var_{hours}h'] = grouped['resprate'].rolling(window=hours, min_periods=1).var().reset_index(0, drop=True)
        # 5. Delta/Acceleration Features (How fast is it changing?)
        df[f'map_diff_{hours}h'] = df['map'] - grouped['map'].shift(hours)

    # Temperature Trajectory (Slope proxy: Current Temp - Average Past Temp)
    df['temp_traj_2h'] = df['temp_c'] - grouped['temp_c'].transform(lambda x: x.rolling(window=2, min_periods=1).mean())
    df['temp_traj_4h'] = df['temp_c'] - grouped['temp_c'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['spo2_traj_2h'] = df['spo2'] - grouped['spo2'].transform(lambda x: x.rolling(window=2, min_periods=1).mean())

    # Traditional EWMA to carry forward existing predictive power
    vitals_all = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp', 'map', 'pulse_pressure', 'rpp', 'msi']
    for vital in vitals_all:
        df[f'{vital}_ewma_3h'] = grouped[vital].transform(lambda x: x.ewm(span=3, adjust=False).mean())

    return df

def run_v6_advanced_features():
    print("---------------------------------------------------------------------")
    print("PHASE 13: V6 Advanced Wearable-Derivable Features Benchmark")
    print("---------------------------------------------------------------------")

    train_file = 'v3_dataset_1h_train.parquet'  # EDIT: 1-hour train cohort parquet path
    test_file = 'v3_dataset_1h_test.parquet'    # EDIT: 1-hour test cohort parquet path

    print(f"\\n⚙️ Loading 1-Hour Proxy Arrays...")
    df_train_raw = pd.read_parquet(train_file)
    df_test_raw = pd.read_parquet(test_file)

    # EDIT: identifier/leakage columns stripped before training
    drop_cols = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h', 'stay_id', 'charttime', 'intime', 'sepsis3_time']
    target_col = 'is_sepsis_6h'  # EDIT: target/label column (prediction HORIZON = 6h)

    print("\\n🚀 Pipeline 1: V3 Standard Features (XGBoost Baseline)")
    df_train_v3 = apply_base_v3_features(df_train_raw.copy())
    df_test_v3 = apply_base_v3_features(df_test_raw.copy())

    X_train_v3 = df_train_v3.drop(columns=[c for c in drop_cols if c in df_train_v3.columns])
    y_train_v3 = df_train_v3[target_col]
    X_test_v3 = df_test_v3.drop(columns=[c for c in drop_cols if c in df_test_v3.columns])
    y_test_v3 = df_test_v3[target_col]

    imputer_v3 = SimpleImputer(strategy='median')  # EDIT: imputation strategy
    X_train_v3_imputed = imputer_v3.fit_transform(X_train_v3)
    X_test_v3_imputed = imputer_v3.transform(X_test_v3)

    # EDIT: XGBoost hyperparameters (scale_pos_weight, max_depth, learning_rate, n_estimators, random_state)
    xgb_base = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05, n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_base.fit(X_train_v3_imputed, y_train_v3)

    p_base = xgb_base.predict_proba(X_test_v3_imputed)[:, 1]
    y_pred_base = xgb_base.predict(X_test_v3_imputed)
    auroc_base = roc_auc_score(y_test_v3, p_base)
    auprc_base = average_precision_score(y_test_v3, p_base)

    tn, fp, fn, tp = confusion_matrix(y_test_v3, y_pred_base).ravel()
    sens_base = tp / max((tp + fn), 1)
    fpr_base = fp / max((fp + tn), 1)

    print("\\n🚀 Pipeline 2: V6 Advanced Wearable Features (XGBoost Champion)")
    df_train_v6 = apply_advanced_v6_features(df_train_raw.copy())
    df_test_v6 = apply_advanced_v6_features(df_test_raw.copy())

    X_train_v6 = df_train_v6.drop(columns=[c for c in drop_cols if c in df_train_v6.columns])
    y_train_v6 = df_train_v6[target_col]
    X_test_v6 = df_test_v6.drop(columns=[c for c in drop_cols if c in df_test_v6.columns])
    y_test_v6 = df_test_v6[target_col]

    imputer_v6 = SimpleImputer(strategy='median')  # EDIT: imputation strategy
    X_train_v6_imputed = imputer_v6.fit_transform(X_train_v6)
    X_test_v6_imputed = imputer_v6.transform(X_test_v6)

    # EDIT: XGBoost hyperparameters (scale_pos_weight, max_depth, learning_rate, n_estimators, random_state)
    xgb_champ = XGBClassifier(scale_pos_weight=10, max_depth=6, learning_rate=0.05, n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_champ.fit(X_train_v6_imputed, y_train_v6)

    p_champ = xgb_champ.predict_proba(X_test_v6_imputed)[:, 1]
    y_pred_champ = xgb_champ.predict(X_test_v6_imputed)
    auroc_champ = roc_auc_score(y_test_v6, p_champ)
    auprc_champ = average_precision_score(y_test_v6, p_champ)

    tn, fp, fn, tp = confusion_matrix(y_test_v6, y_pred_champ).ravel()
    sens_champ = tp / max((tp + fn), 1)
    fpr_champ = fp / max((fp + tn), 1)

    out_text = "V6 Advanced Wearable-Derivable Features Benchmark\\n"
    out_text += "="*65 + "\\n"
    out_text += "[XGBoost: V3 Standard Features (Baseline)]\\n"
    out_text += f"  ROC-AUC: {auroc_base:.4f}\\n"
    out_text += f"  AUPRC:   {auprc_base:.4f}\\n"
    out_text += f"  Sens:    {sens_base:.4f} | FPR: {fpr_base:.4f}\\n\\n"
    out_text += "[XGBoost: V6 Advanced Wearable Features (Champion)]\\n"
    out_text += f"  ROC-AUC: {auroc_champ:.4f}\\n"
    out_text += f"  AUPRC:   {auprc_champ:.4f}\\n"
    out_text += f"  Sens:    {sens_champ:.4f} | FPR: {fpr_champ:.4f}\\n\\n"

    print("\\n=====================================================================")
    print(out_text.strip())
    print("=====================================================================")

    with open('32_v6_xgb_advanced_features_results.txt', 'w') as f:  # EDIT: results output filename
        f.write(out_text)

    print("✅ V6 Feature Testing completed successfully. Saved to 32_v6_xgb_advanced_features_results.txt")

if __name__ == '__main__':
    run_v6_advanced_features()
