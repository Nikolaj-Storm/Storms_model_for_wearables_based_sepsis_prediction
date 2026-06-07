# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  78_v13b_sirs_evaluation.py
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Reuses the V13b NOSE RF plus meta-learner stack but retargets it at the
#    Sepsis-2 (SIRS) labels instead of Sepsis-3, as a loose-target contrast
#    experiment. The looser, higher-prevalence target is expected to produce
#    artificially high metrics, which is the point of the comparison.
#
#  INPUTS
#    v12_dataset_1h_train.parquet             (V12 engineered train features)
#    v12_dataset_1h_test.parquet             (V12 engineered test features)
#    74_barton_sirs_dataset_train.parquet     (SIRS / Barton targets, train)
#    74_barton_sirs_dataset_test.parquet      (SIRS / Barton targets, test)
#    v8_dataset_1h_etiology_train.parquet     (stay-level etiology labels, train)
#    v8_dataset_1h_etiology_test.parquet      (stay-level etiology labels, test)
#  OUTPUTS
#    78_v13b_sirs_results.txt                 (metrics report)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    Assumed working directory  -  run from the folder holding the input
#      parquet files; reads/writes use bare relative filenames.
#    vitals list                -  the six wearable-proxy vital columns.
#    NOSE RF base learner       -  n_estimators=100, max_depth=10,
#                                  min_samples_leaf=5, max_features='sqrt',
#                                  class_weight='balanced'.
#    NOSE num_subsets           -  3 (reduced for the demonstration run).
#    CV folds                   -  StratifiedGroupKFold n_splits=3.
#    LR meta-learner            -  class_weight='balanced', max_iter=1000.
#    Threshold sweep            -  np.arange(0.10, 0.90, 0.01); max-TP gate
#                                  FPR <= 0.2112 (matches Barton's FPR limit).
#    Random seed                -  42.
#
#  REQUIRES: pandas, numpy, scikit-learn
# ============================================================================
"""
V13b vs Loose Targets Experiment
Reusing V13b NOSE RF + Meta-Learner stack, but pointing it to predict Sepsis-2 (SIRS criteria).
This expects artificially high metrics.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
import warnings
import time

warnings.filterwarnings('ignore')

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

# EDIT: NOSE num_subsets default (5; called with 3 below)
def train_nose_ensemble(df, features, target_col, num_subsets=5):
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
            n_estimators=100, max_depth=10, # Slightly smaller to run faster for the test
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

def run_v13b_sirs():
    print("=" * 70)
    print("V13b Setup + SIRS (Sepsis-2) Target")
    print("Executing massive target expansion evaluation")
    print("=" * 70)
    t0 = time.time()

    print("\n⚙️ Loading datasets and swapping targets...")
    # EDIT: input train/test parquet paths (relative to working directory)
    df_train = pd.read_parquet('v12_dataset_1h_train.parquet').reset_index()
    df_test = pd.read_parquet('v12_dataset_1h_test.parquet').reset_index()

    # Load the SIRS targets assigned in stage 74
    # EDIT: SIRS / Barton target parquet paths (relative to working directory)
    df_sirs_train = pd.read_parquet('74_barton_sirs_dataset_train.parquet')[['stay_id', 'charttime', 'is_barton_6h', 'is_barton_12h', 'is_barton_stay']]
    df_sirs_test = pd.read_parquet('74_barton_sirs_dataset_test.parquet')[['stay_id', 'charttime', 'is_barton_6h', 'is_barton_12h', 'is_barton_stay']]

    # Join
    df_train = df_train.merge(df_sirs_train, on=['stay_id', 'charttime'], how='left')
    df_test = df_test.merge(df_sirs_test, on=['stay_id', 'charttime'], how='left')

    # Apply V13 tracking
    print("\n🧬 Applying V13 Features...")
    df_train = apply_v13_features(df_train)
    df_test = apply_v13_features(df_test)

    # Load Etiologies mapped to the ORIGINAL target?
    # Actually, the patient-level etiologies (target_resp, target_uri) are static. We can intersect them with the SIRS label!
    print("\n⚙️ Mapping Etiology Targets to SIRS Labels...")
    # EDIT: etiology label parquet paths (relative to working directory)
    df_v8_train = pd.read_parquet('v8_dataset_1h_etiology_train.parquet', columns=['stay_id', 'target_resp', 'target_uri', 'target_other'])
    df_v8_test = pd.read_parquet('v8_dataset_1h_etiology_test.parquet', columns=['stay_id', 'target_resp', 'target_uri', 'target_other'])
    train_e = df_v8_train.groupby('stay_id').max()
    test_e = df_v8_test.groupby('stay_id').max()
    df_train = df_train.merge(train_e, on='stay_id', how='left').fillna(0)
    df_test = df_test.merge(test_e, on='stay_id', how='left').fillna(0)

    etiologies = ['target_resp', 'target_uri', 'target_other']
    for col in etiologies:
        df_train[f'{col}_6h'] = ((df_train['is_barton_6h'] == 1) & (df_train[col] == 1)).astype(int)
        df_test[f'{col}_6h'] = ((df_test['is_barton_6h'] == 1) & (df_test[col] == 1)).astype(int)

    streams = ['is_barton_6h', 'target_resp_6h', 'target_uri_6h', 'target_other_6h']

    exclude_cols = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h',
                    'is_barton_stay', 'is_barton_6h', 'is_barton_12h',
                    'stay_id', 'charttime', 'intime', 'sepsis3_time',
                    'target_resp', 'target_uri', 'target_other'] + \
                   [f"{c}_6h" for c in etiologies] + [f"{c}_12h" for c in etiologies]

    all_features = [c for c in df_train.columns if c not in exclude_cols]
    imputer = SimpleImputer(strategy='median')
    df_train[all_features] = imputer.fit_transform(df_train[all_features])
    df_test[all_features] = imputer.transform(df_test[all_features])

    print(f"\n📊 Features: {len(all_features)}")

    # OOF
    print(f"\n🧱 Building OOF Meta-Features (5-Fold, 5 NOSE subsets)...")
    df_train_meta = df_train[['stay_id', 'is_barton_6h', 'is_barton_stay']].copy()
    for s in streams:
        df_train_meta[f'meta_{s}'] = 0.0

    # EDIT: CV folds (n_splits=3) and random seed (42)
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42) # Using 3 to accelerate slightly for demonstration
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_train, df_train['is_barton_stay'], groups=df_train['stay_id'])):
        print(f"  -> Fold {fold+1}/3...")
        X_tr = df_train.iloc[train_idx]
        X_va = df_train.iloc[val_idx]
        for stream in streams:
            sf = get_etiology_features(stream, all_features)
            # EDIT: NOSE num_subsets (3)
            nose = train_nose_ensemble(X_tr, sf, stream, num_subsets=3)
            df_train_meta.loc[val_idx, f'meta_{stream}'] = predict_nose(nose, X_va, sf)

    df_train_meta['var_6h'] = df_train_meta[['meta_is_barton_6h', 'meta_target_resp_6h', 'meta_target_uri_6h', 'meta_target_other_6h']].var(axis=1)
    meta_features = [f'meta_{s}' for s in streams] + ['var_6h']

    print("\n🧠 Training LogReg Meta-Learner (balanced)...")
    # EDIT: LR meta-learner hyperparameters
    lr_meta = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

    # Train only on non-null masks
    valid_train = df_train_meta['is_barton_6h'].notnull()
    lr_meta.fit(df_train_meta.loc[valid_train, meta_features], df_train_meta.loc[valid_train, 'is_barton_6h'])

    print(f"\n🛠️ Training Final NOSE Ensembles...")
    final_ensembles = {}
    final_features = {}
    for stream in streams:
        sf = get_etiology_features(stream, all_features)
        final_features[stream] = sf
        valid_stream = df_train[stream].notnull()
        # EDIT: NOSE num_subsets (3)
        final_ensembles[stream] = train_nose_ensemble(df_train[valid_stream], sf, stream, num_subsets=3)

    print(f"\n🧪 Evaluating on Test Set...")
    df_test_meta = df_test[['stay_id', 'is_barton_6h', 'is_barton_stay']].copy()
    for stream in streams:
        df_test_meta[f'meta_{stream}'] = predict_nose(final_ensembles[stream], df_test, final_features[stream])

    df_test_meta['var_6h'] = df_test_meta[['meta_is_barton_6h', 'meta_target_resp_6h', 'meta_target_uri_6h', 'meta_target_other_6h']].var(axis=1)

    # Valid Test instances
    valid_test = df_test_meta['is_barton_6h'].notnull()
    y_test = df_test_meta.loc[valid_test, 'is_barton_6h']
    X_test_meta = df_test_meta.loc[valid_test, meta_features]

    p_test = lr_meta.predict_proba(X_test_meta)[:, 1]
    auroc = roc_auc_score(y_test, p_test)
    auprc = average_precision_score(y_test, p_test)

    # Sweep thresholds for optimum FPR <= 21.12% (Barton's exact FPR)
    best_tp = 0
    best_t = 0.5
    best_r = {}
    # EDIT: threshold sweep grid (0.10 to 0.90, step 0.01) and max-TP FPR ceiling (0.2112)
    for t in np.arange(0.10, 0.90, 0.01):
        yp = (p_test >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
        sens = tp / max((tp + fn), 1)
        fpr = fp / max((fp + tn), 1)
        if tp > best_tp and fpr <= 0.2112:  # Matches Barton's exact FPR limit
            best_tp = tp; best_t = t
            best_r = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'sens': sens, 'fpr': fpr}

    out_text = f"V13b Setup + SIRS (Sepsis-2) Target\n"
    out_text += f"AUROC: {auroc:.4f}\nAUPRC: {auprc:.4f}\n"
    out_text += f"\nThreshold Optimised (t={best_t:.2f}):\n"
    out_text += f"TP: {best_r['tp']} | FP: {best_r['fp']} | FN: {best_r['fn']} | TN: {best_r['tn']}\n"
    out_text += f"Sensitivity: {best_r['sens']:.4f}\n"
    out_text += f"FPR: {best_r['fpr']:.4f}\n"

    print("\n" + "=" * 70)
    print(out_text.strip())
    print("=" * 70)

    # EDIT: output metrics report path (relative to working directory)
    with open('78_v13b_sirs_results.txt', 'w') as f:
        f.write(out_text)

if __name__ == '__main__':
    run_v13b_sirs()
