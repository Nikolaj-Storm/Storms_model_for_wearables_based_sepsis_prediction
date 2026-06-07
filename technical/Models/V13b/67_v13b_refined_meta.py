# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  67_v13b_refined_meta.py
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Quick meta-learner refinement. Rebuilds the V13 OOF base-learner stack
#    (NOSE Random Forest, etiology-specific feature subsets) and compares two
#    alternative meta-learners, a balanced logistic regression and an
#    aggressive scale_pos_weight XGBoost, to maximise True Positives. Reports
#    natural-prevalence, threshold-optimised, and balanced 1:1 metrics.
#
#  INPUTS
#    v12_dataset_1h_train.parquet             (V12 engineered train features)
#    v12_dataset_1h_test.parquet              (V12 engineered test features)
#    v8_dataset_1h_etiology_train.parquet     (stay-level etiology labels, train)
#    v8_dataset_1h_etiology_test.parquet      (stay-level etiology labels, test)
#  OUTPUTS
#    67_v13b_benchmark_results.txt            (metrics report)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    Assumed working directory  -  run from the folder holding the four input
#      parquet files; reads/writes use bare relative filenames.
#    vitals list                -  the six wearable-proxy vital columns.
#    NOSE RF base learner       -  n_estimators=200, max_depth=10,
#                                  min_samples_leaf=5, max_features='sqrt',
#                                  class_weight='balanced'.
#    NOSE num_subsets           -  5.
#    CV folds                   -  StratifiedGroupKFold n_splits=5.
#    LR meta-learner            -  class_weight='balanced', max_iter=1000.
#    XGBoost meta-learner       -  n_estimators=50, max_depth=3,
#                                  scale_pos_weight=15, learning_rate=0.1.
#    Threshold sweep            -  np.arange(0.10, 0.90, 0.01); max-TP gate
#                                  FPR < 0.30; default threshold 0.50.
#    Random seed                -  42.
#
#  REQUIRES: pandas, numpy, scikit-learn, xgboost, lightgbm
# ============================================================================
"""
V13b: Quick Meta-Learner Refinement
Reuses the V13 OOF outputs but tests alternative meta-learners
to maximize True Positives while leveraging V13's improved base models.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
import time

warnings.filterwarnings('ignore')

# V13 features (same as 67_v13)
def apply_v13_features(df):
    # EDIT: vitals - the six wearable-proxy vital-sign columns
    vitals = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']
    for v in vitals:
        first_diff = df[v] - df.groupby('stay_id')[v].shift(1)
        df[f'accel_{v}'] = first_diff - first_diff.groupby(df['stay_id']).shift(1)
    for v in vitals:
        patient_baseline = df.groupby('stay_id')[v].transform(lambda x: x.expanding().mean())
        deviation = df[v] - patient_baseline
        df[f'cusum_pos_{v}'] = deviation.clip(lower=0).groupby(df['stay_id']).cumsum()
        df[f'cusum_neg_{v}'] = deviation.clip(upper=0).abs().groupby(df['stay_id']).cumsum()
    df['pulse_pressure'] = df['sbp'] - df['dbp']
    df['rpp'] = df['heart_rate'] * df['sbp']
    df['msi'] = df['heart_rate'] / ((df['sbp'] + 2 * df['dbp']) / 3 + 1e-6)
    df['resp_distress'] = df['resprate'] * (100 - df['spo2'])
    df['fever_tachycardia'] = df['heart_rate'] * np.maximum(df['temp_c'] - 37, 0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    return df

def get_etiology_features(stream_name, all_features):
    base_keywords = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp',
                     'shock_index', 'map', 'age', 'weight_kg', 'time_since',
                     'news_score', 'partial_qsofa', 'news_delta', 'qsofa_delta']
    if 'resp' in stream_name:
        priority_keywords = ['resprate', 'spo2', 'ventilation', 'cardiorespiratory',
            'resp_distress', 'news', 'qsofa', 'exp_min_spo2', 'exp_std_spo2', 'spo2_sd']
    elif 'uri' in stream_name:
        priority_keywords = ['temp_c', 'fever_tachycardia', 'tachycardia_excess',
            'news', 'heart_rate', 'exp_max_temp', 'exp_std_temp', 'temp_c_sd',
            'cusum_pos_temp', 'cusum_neg_temp', 'slope_4h_temp', 'accel_temp']
    elif 'other' in stream_name:
        priority_keywords = ['shock_index', 'map', 'perfusion', 'sbp', 'dbp',
            'pulse_pressure', 'rpp', 'msi', 'exp_std_sbp', 'sbp_sd', 'dbp_sd',
            'cusum_pos_sbp', 'cusum_neg_sbp', 'cusum_pos_dbp',
            'slope_4h_sbp', 'slope_4h_dbp', 'accel_sbp', 'accel_dbp']
    else:
        return all_features
    selected = [f for f in all_features if any(kw in f for kw in base_keywords + priority_keywords)]
    # EDIT: minimum padded feature count (30)
    if len(selected) < 30:
        remaining = [f for f in all_features if f not in selected]
        selected.extend(remaining[:30 - len(selected)])
    return selected


# EDIT: NOSE num_subsets default (5)
def train_nose_ensemble(df, features, target_col, num_subsets=5):
    """NOSE with RF only (like V12) but tuned hyperparams and V13 features."""
    pos_stays = df[df[target_col] == 1]['stay_id'].unique()
    neg_stays = np.setdiff1d(df['stay_id'].unique(), pos_stays)
    # EDIT: random seed (42)
    np.random.seed(42)
    neg_stays_shuffled = np.random.permutation(neg_stays)
    chunk_size = len(pos_stays)
    models = []
    for i in range(num_subsets):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(neg_stays_shuffled))
        subset_neg = neg_stays_shuffled[start_idx:end_idx]
        if len(subset_neg) == 0:
            break
        subset_stays = np.concatenate([pos_stays, subset_neg])
        df_subset = df[df['stay_id'].isin(subset_stays)]
        X_sub = df_subset[features].values
        y_sub = df_subset[target_col].values
        # EDIT: NOSE RF base-learner hyperparameters
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced', random_state=42+i, n_jobs=-1)
        model.fit(X_sub, y_sub)
        models.append(model)
    return models


def predict_nose(models, df_eval, features):
    X = df_eval[features].values
    preds = np.zeros(len(df_eval))
    for model in models:
        preds += model.predict_proba(X)[:, 1]
    return preds / len(models)


def run_v13b():
    print("=" * 70)
    print("V13b: NOSE RF + V13 Features + LogReg Meta + Thresh Optimization")
    print("=" * 70)
    t0 = time.time()

    # Load data
    print("\n⚙️ Loading datasets...")
    # EDIT: input train/test parquet paths (relative to working directory)
    df_train = pd.read_parquet('v12_dataset_1h_train.parquet').reset_index()
    df_test = pd.read_parquet('v12_dataset_1h_test.parquet').reset_index()
    if 'stay_id' not in df_train.columns and 'level_0' in df_train.columns:
        df_train = df_train.rename(columns={'level_0': 'stay_id', 'level_1': 'charttime'})
        df_test = df_test.rename(columns={'level_0': 'stay_id', 'level_1': 'charttime'})

    # V13 features
    print("\n🧬 Applying V13 Features...")
    df_train = apply_v13_features(df_train)
    df_test = apply_v13_features(df_test)

    # Etiology targets
    print("\n⚙️ Mapping Etiology Targets...")
    # EDIT: etiology label parquet paths (relative to working directory)
    df_v8_train = pd.read_parquet('v8_dataset_1h_etiology_train.parquet',
                                  columns=['stay_id', 'target_resp', 'target_uri', 'target_other'])
    df_v8_test = pd.read_parquet('v8_dataset_1h_etiology_test.parquet',
                                 columns=['stay_id', 'target_resp', 'target_uri', 'target_other'])
    train_e = df_v8_train.groupby('stay_id').max()
    test_e = df_v8_test.groupby('stay_id').max()
    df_train = df_train.merge(train_e, on='stay_id', how='left').fillna(0)
    df_test = df_test.merge(test_e, on='stay_id', how='left').fillna(0)

    etiologies = ['target_resp', 'target_uri', 'target_other']
    for col in etiologies:
        df_train[f'{col}_6h'] = ((df_train['is_sepsis_6h'] == 1) & (df_train[col] == 1)).astype(int)
        df_train[f'{col}_12h'] = ((df_train['is_sepsis_12h'] == 1) & (df_train[col] == 1)).astype(int)
        df_test[f'{col}_6h'] = ((df_test['is_sepsis_6h'] == 1) & (df_test[col] == 1)).astype(int)
        df_test[f'{col}_12h'] = ((df_test['is_sepsis_12h'] == 1) & (df_test[col] == 1)).astype(int)

    streams = ['is_sepsis_6h', 'is_sepsis_12h',
               'target_resp_6h', 'target_resp_12h',
               'target_uri_6h', 'target_uri_12h',
               'target_other_6h', 'target_other_12h']

    exclude_cols = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h',
                    'stay_id', 'charttime', 'intime', 'sepsis3_time',
                    'target_resp', 'target_uri', 'target_other'] + \
                   [f"{c}_{h}" for c in etiologies for h in ["6h", "12h"]]
    all_features = [c for c in df_train.columns if c not in exclude_cols]

    imputer = SimpleImputer(strategy='median')
    df_train[all_features] = imputer.fit_transform(df_train[all_features])
    df_test[all_features] = imputer.transform(df_test[all_features])

    print(f"\n📊 Features: {len(all_features)}")

    # OOF
    print(f"\n🧱 Building OOF Meta-Features (5-Fold, 5 NOSE subsets, RF only)...")
    df_train_meta = df_train[['stay_id', 'is_sepsis_6h', 'is_sepsis_stay']].copy()
    for s in streams:
        df_train_meta[f'meta_{s}'] = 0.0

    # EDIT: CV folds (n_splits=5) and random seed (42)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_train, df_train['is_sepsis_stay'], groups=df_train['stay_id'])):
        print(f"  -> Fold {fold+1}/5...")
        X_tr = df_train.iloc[train_idx]
        X_va = df_train.iloc[val_idx]
        for stream in streams:
            sf = get_etiology_features(stream, all_features)
            # EDIT: NOSE num_subsets (5)
            nose = train_nose_ensemble(X_tr, sf, stream, num_subsets=5)
            df_train_meta.loc[val_idx, f'meta_{stream}'] = predict_nose(nose, X_va, sf)

    # Variance features
    df_train_meta['var_6h'] = df_train_meta[
        ['meta_is_sepsis_6h', 'meta_target_resp_6h', 'meta_target_uri_6h', 'meta_target_other_6h']].var(axis=1)
    df_train_meta['var_12h'] = df_train_meta[
        ['meta_is_sepsis_12h', 'meta_target_resp_12h', 'meta_target_uri_12h', 'meta_target_other_12h']].var(axis=1)

    meta_features = [f'meta_{s}' for s in streams] + ['var_6h', 'var_12h']

    # ---- META-LEARNER A: LogReg (balanced) ----
    print("\n🧠 Training LogReg Meta-Learner (balanced)...")
    # EDIT: LR meta-learner hyperparameters
    lr_meta = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr_meta.fit(df_train_meta[meta_features], df_train_meta['is_sepsis_6h'])

    # ---- META-LEARNER B: XGBoost (aggressive sensitivity) ----
    print("🧠 Training XGBoost Meta-Learner (scale_pos_weight=15)...")
    # EDIT: XGBoost meta-learner hyperparameters (aggressive scale_pos_weight=15)
    xgb_meta = XGBClassifier(
        n_estimators=50, max_depth=3, scale_pos_weight=15,
        learning_rate=0.1, use_label_encoder=False, eval_metric='logloss',
        random_state=42, verbosity=0)
    xgb_meta.fit(df_train_meta[meta_features], df_train_meta['is_sepsis_6h'])

    # FINAL ENSEMBLES
    print(f"\n🛠️ Training 8 Final NOSE Ensembles...")
    final_ensembles = {}
    final_features = {}
    for stream in streams:
        sf = get_etiology_features(stream, all_features)
        final_features[stream] = sf
        # EDIT: NOSE num_subsets (5)
        final_ensembles[stream] = train_nose_ensemble(df_train, sf, stream, num_subsets=5)

    # TEST
    print(f"\n🧪 Evaluating on Test Set...")
    df_test_meta = df_test[['stay_id', 'is_sepsis_6h', 'is_sepsis_stay']].copy()
    for stream in streams:
        df_test_meta[f'meta_{stream}'] = predict_nose(
            final_ensembles[stream], df_test, final_features[stream])

    df_test_meta['var_6h'] = df_test_meta[
        ['meta_is_sepsis_6h', 'meta_target_resp_6h', 'meta_target_uri_6h', 'meta_target_other_6h']].var(axis=1)
    df_test_meta['var_12h'] = df_test_meta[
        ['meta_is_sepsis_12h', 'meta_target_resp_12h', 'meta_target_uri_12h', 'meta_target_other_12h']].var(axis=1)

    y_test = df_test_meta['is_sepsis_6h']

    out_text = "V13b Improved NOSE Architecture Results\n"
    out_text += "=" * 65 + "\n"
    out_text += f"Features: {len(all_features)} | NOSE: 5 subsets, RF only | Etiology-specific features\n"
    out_text += "=" * 65 + "\n\n"

    # --- Evaluate both meta-learners ---
    for name, meta, label in [
        ('LogReg (balanced)', lr_meta, 'A'),
        ('XGBoost (spw=15)', xgb_meta, 'B')
    ]:
        p_test = meta.predict_proba(df_test_meta[meta_features])[:, 1]
        auroc = roc_auc_score(y_test, p_test)
        auprc = average_precision_score(y_test, p_test)

        # Standard threshold 0.5
        # EDIT: default decision threshold (0.5)
        y_pred = (p_test >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sens = tp / max((tp + fn), 1)
        fpr = fp / max((fp + tn), 1)

        out_text += f"{label}. Meta-Learner: {name} @ threshold 0.50\n"
        out_text += f"  ROC-AUC:     {auroc:.4f}  (V12 RF was 0.7547)\n"
        out_text += f"  AUPRC:       {auprc:.4f}  (V12 RF was 0.0614)\n"
        out_text += f"  Sensitivity: {sens:.4f}  (V12 RF was 0.6545)\n"
        out_text += f"  FPR:         {fpr:.4f}  (V12 RF was 0.2692)\n"
        out_text += f"  TP: {tp} | FP: {fp} | FN: {fn}\n\n"

        # Threshold sweep for best TP with FPR < 30%
        best_tp = 0
        best_t = 0.5
        best_r = {}
        # EDIT: threshold sweep grid (0.10 to 0.90, step 0.01) and max-TP FPR ceiling (0.30)
        for t in np.arange(0.10, 0.90, 0.01):
            yp = (p_test >= t).astype(int)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, yp).ravel()
            s_t = tp_t / max((tp_t + fn_t), 1)
            f_t = fp_t / max((fp_t + tn_t), 1)
            if tp_t > best_tp and f_t < 0.30:
                best_tp = tp_t; best_t = t
                best_r = {'tp': tp_t, 'fp': fp_t, 'fn': fn_t, 'sensitivity': s_t, 'fpr': f_t}

        out_text += f"  Threshold Opt (max TP, FPR<30%): t={best_t:.2f}\n"
        out_text += f"    Sensitivity: {best_r['sensitivity']:.4f}\n"
        out_text += f"    FPR:         {best_r['fpr']:.4f}\n"
        out_text += f"    TP: {best_r['tp']} | FP: {best_r['fp']} | FN: {best_r['fn']}\n\n"

    # Balanced evaluation using LogReg (likely better for TP)
    p_lr = lr_meta.predict_proba(df_test_meta[meta_features])[:, 1]
    test_pos_stays = df_test_meta[df_test_meta['is_sepsis_stay'] == 1]['stay_id'].unique()
    test_neg_stays = df_test_meta[df_test_meta['is_sepsis_stay'] == 0]['stay_id'].unique()
    # EDIT: random seed (42)
    np.random.seed(42)
    bal_neg = np.random.choice(test_neg_stays, size=len(test_pos_stays), replace=False)
    bal_stays = np.concatenate([test_pos_stays, bal_neg])
    mask = df_test_meta['stay_id'].isin(bal_stays)
    y_bal = df_test_meta.loc[mask, 'is_sepsis_6h']
    p_bal = p_lr[mask]
    y_pred_bal = (p_bal >= 0.5).astype(int)
    auroc_bal = roc_auc_score(y_bal, p_bal)
    tnb, fpb, fnb, tpb = confusion_matrix(y_bal, y_pred_bal).ravel()
    sens_bal = tpb / max((tpb + fnb), 1)
    fpr_bal = fpb / max((fpb + tnb), 1)

    out_text += "C. Balanced 1:1 (LogReg Meta)\n"
    out_text += f"  ROC-AUC:     {auroc_bal:.4f}  (V12 RF was 0.7420)\n"
    out_text += f"  Sensitivity: {sens_bal:.4f}  (V12 RF was 0.6545)\n"
    out_text += f"  FPR:         {fpr_bal:.4f}  (V12 RF was 0.2856)\n"

    print("\n" + "=" * 70)
    print(out_text.strip())
    print("=" * 70)

    # EDIT: output metrics report path (relative to working directory)
    with open('67_v13b_benchmark_results.txt', 'w') as f:
        f.write(out_text)

    t1 = time.time()
    print(f"\n✅ V13b completed in {(t1-t0)/60:.1f} minutes.")


if __name__ == '__main__':
    run_v13b()
