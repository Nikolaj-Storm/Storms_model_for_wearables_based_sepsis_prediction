# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  V13b_out_of_core_benchmark.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Out-of-core training and evaluation of the V13b NOSE (Negative One-Subset
#    Ensemble) stacking model for 6-hour sepsis prediction. Streams large
#    parquet train files in batches, builds 4 etiology-specific Random Forest
#    base ensembles per fold, stacks them with LogReg and XGBoost meta-learners,
#    and evaluates across 15-min, 1-hour and 4-hour sampling resolutions.
#
#  INPUTS
#    ../Data/All engineered features/Dataset_all_engineered_15min_train.parquet
#    ../Data/All engineered features/Dataset_all_engineered_15min_test.parquet
#    ../Data/All engineered features/Dataset_all_engineered_1h_train.parquet
#    ../Data/All engineered features/Dataset_all_engineered_1h_test.parquet
#    ../Data/All engineered features/Dataset_all_engineered_4h_train.parquet
#    ../Data/All engineered features/Dataset_all_engineered_4h_test.parquet
#    BigQuery: physionet-data.mimiciv_3_1_icu.icustays, .mimiciv_3_1_hosp.diagnoses_icd
#  OUTPUTS
#    ../Results/v13b_out_of_core_performance.csv
#    ../Results/v13b_base_model_performance.csv
#    (also writes/removes transient temp parquet and CSV files in the working dir)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    DATA_DIR           -  input parquet folder (relative ../Data/...; assumes
#                          you run from the original technical/Models/ directory)
#    RESULTS_DIR        -  output folder (relative ../Results)
#    DATASETS           -  resolution -> (train, test) parquet filenames
#    TARGET             -  label column, is_sepsis_6h
#    EXCLUDE_COLS       -  columns dropped from the feature matrix
#    ETIOLOGIES         -  the 4 per-etiology target columns to model
#    GCP project        -  BigQuery client project id (YOUR_GCP_PROJECT)
#    StratifiedGroupKFold n_splits=5, shuffle=True, random_state=42
#    NOSE subset count  -  5 negative subsets (range(5)) per fold and final
#    np.random.seed     -  42 (negative-stay shuffling)
#    Base RF            -  n_estimators=200, max_depth=10, min_samples_leaf=5,
#                          max_features='sqrt', class_weight='balanced',
#                          random_state=42+i
#    LogReg meta        -  class_weight='balanced', random_state=42, max_iter=1000
#    XGBoost meta       -  n_estimators=50, max_depth=3, scale_pos_weight=15,
#                          learning_rate=0.1, random_state=42
#    Decision threshold -  0.5 for hard-label metrics (base and meta)
#    iter_batches batch_size  -  200000 rows per streamed parquet batch
#
#  REQUIRES: scikit-learn, xgboost, pandas, numpy, pyarrow, google-cloud-bigquery
# ============================================================================

import pandas as pd
import numpy as np
import os
import time
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from google.cloud import bigquery
import pyarrow.parquet as pq
import pyarrow as pa
import warnings

warnings.filterwarnings('ignore')

# CONFIG
DATA_DIR = '../Data/All engineered features'  # EDIT: input parquet folder (assumes original technical/Models/ cwd)
RESULTS_DIR = '../Results'  # EDIT: output folder
DATASETS = {  # EDIT: resolution -> (train, test) parquet filenames
    '15_min': ('Dataset_all_engineered_15min_train.parquet', 'Dataset_all_engineered_15min_test.parquet'),
    '1_hour': ('Dataset_all_engineered_1h_train.parquet', 'Dataset_all_engineered_1h_test.parquet'),
    '4_hour': ('Dataset_all_engineered_4h_train.parquet', 'Dataset_all_engineered_4h_test.parquet')
}
TARGET = 'is_sepsis_6h'  # EDIT: label column
EXCLUDE_COLS = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h', 'stay_id', 'charttime', 'intime', 'sepsis3_time', 'time_since_ICU_admit_hours', 'target_resp', 'target_uri', 'target_other']  # EDIT: columns excluded from features
ETIOLOGIES = ['is_sepsis_6h', 'target_resp', 'target_uri', 'target_other']  # EDIT: per-etiology target columns

def get_etiology_mapping():
    print("Fetching Etiology Labels from BigQuery...")
    client = bigquery.Client(project="YOUR_GCP_PROJECT")  # EDIT: BigQuery project id
    query = """
    WITH cohort AS (
        SELECT hadm_id, stay_id FROM `physionet-data.mimiciv_3_1_icu.icustays`
    ), diagnoses AS (
        SELECT hadm_id, icd_code, icd_version FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    )
    SELECT c.stay_id, d.icd_code, d.icd_version
    FROM cohort c LEFT JOIN diagnoses d ON c.hadm_id = d.hadm_id
    """
    df_diag = client.query(query).to_dataframe()

    resp_codes_10 = ['J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J20', 'J21', 'J22', 'J69']
    resp_codes_9 = ['480', '481', '482', '483', '484', '485', '486', '487', '488', '507']
    uri_codes_10 = ['N390', 'N39', 'N30', 'N10', 'N11', 'N12']
    uri_codes_9 = ['5990', '599', '590', '595']

    def is_respiratory(row):
        code = str(row['icd_code'])
        v = row['icd_version']
        if pd.isna(v): return False
        if v == 10: return any(code.startswith(x) for x in resp_codes_10)
        else: return any(code.startswith(x) for x in resp_codes_9)

    def is_urinary(row):
        code = str(row['icd_code'])
        v = row['icd_version']
        if pd.isna(v): return False
        if v == 10: return any(code.startswith(x) for x in uri_codes_10)
        else: return any(code.startswith(x) for x in uri_codes_9)

    df_diag['is_resp'] = df_diag.apply(is_respiratory, axis=1)
    df_diag['is_uri'] = df_diag.apply(is_urinary, axis=1)

    df_etiology = df_diag.groupby('stay_id').agg({'is_resp': 'max', 'is_uri': 'max'}).reset_index()
    return df_etiology

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
    if len(selected) < 30:
        remaining = [f for f in all_features if f not in selected]
        selected.extend(remaining[:30 - len(selected)])
    return selected

def apply_targets(df, df_etiology):
    if df_etiology is not None:
        df = df.merge(df_etiology, on='stay_id', how='left')
        df['is_resp'] = df['is_resp'].fillna(False).astype(int)
        df['is_uri'] = df['is_uri'].fillna(False).astype(int)
    else:
        df['is_resp'] = 0
        df['is_uri'] = 0

    df['target_resp'] = 0
    df['target_uri'] = 0
    df['target_other'] = 0

    septic_mask = (df['is_sepsis_6h'] == 1)
    resp_mask = septic_mask & (df['is_resp'] == 1)
    df.loc[resp_mask, 'target_resp'] = 1
    uri_mask = septic_mask & (df['is_uri'] == 1) & (~resp_mask)
    df.loc[uri_mask, 'target_uri'] = 1
    other_mask = septic_mask & (~resp_mask) & (~uri_mask)
    df.loc[other_mask, 'target_other'] = 1

    if 'is_resp' in df.columns:
        df.drop(columns=['is_resp', 'is_uri'], inplace=True)
    return df

def get_metrics(name, res_name, y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    test_acc = (tp + tn) / total

    return {
        'Resolution': f"{res_name} ({name})",
        'Test Accuracy (%)': f"{test_acc*100:.2f}%",
        'Precision': f"{tp / max(tp+fp, 1):.4f}",
        'Recall': f"{tp / max(tp+fn, 1):.4f}",
        'F1 Score': f"{(2*tp) / max(2*tp+fp+fn, 1):.4f}",
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

def extract_single_subset(source_path, stays_set, out_path):
    """Stream source_path once, writing only rows for stays_set to out_path."""
    writer = None
    pf = pq.ParquetFile(source_path)
    for batch in pf.iter_batches(batch_size=200000):  # EDIT: streamed parquet batch size (rows)
        df_batch = batch.to_pandas()
        df_sub = df_batch[df_batch['stay_id'].isin(stays_set)]
        if len(df_sub) > 0:
            table = pa.Table.from_pandas(df_sub)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
    if writer is not None:
        writer.close()

def run_out_of_core_v13b():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, 'v13b_out_of_core_performance.csv')
    base_out_file = os.path.join(RESULTS_DIR, 'v13b_base_model_performance.csv')

    df_etiology = get_etiology_mapping()
    all_results = []
    base_model_results = []

    for res_name, (train_filename, test_filename) in DATASETS.items():
        print(f"\n=======================================================")
        print(f"Processing Resolution: {res_name} (Out-of-Core NOSE V13b)")
        print(f"=======================================================")
        train_path = os.path.join(DATA_DIR, train_filename)
        test_path = os.path.join(DATA_DIR, test_filename)

        # 1. Load ONLY metadata (stay_id, is_sepsis_stay, is_sepsis_6h)
        print("Loading metadata to compute CV folds and NOSE subsets...")
        df_meta = pd.read_parquet(train_path, columns=['stay_id', 'is_sepsis_stay', 'is_sepsis_6h'])

        # 2. Determine base feature columns
        schema = pq.read_schema(train_path)
        all_features = [c for c in schema.names if c not in EXCLUDE_COLS]
        feature_maps = {e: get_etiology_features(e, all_features) for e in ETIOLOGIES}

        meta_cols = [f'meta_{e}' for e in ETIOLOGIES]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)  # EDIT: CV folds / shuffle / seed

        # 3. K-Fold Out-Of-Core Loop
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_meta, df_meta['is_sepsis_stay'], groups=df_meta['stay_id'])):
            print(f"\n  --- Fold {fold+1}/5 ---")

            train_stays = df_meta.iloc[train_idx]['stay_id'].unique()
            val_stays = df_meta.iloc[val_idx]['stay_id'].unique()

            pos_stays = df_meta.iloc[train_idx][df_meta.iloc[train_idx]['is_sepsis_stay'] == 1]['stay_id'].unique()
            neg_stays = np.setdiff1d(train_stays, pos_stays)

            np.random.seed(42)  # EDIT: negative-stay shuffle seed
            neg_stays_shuffled = np.random.permutation(neg_stays)
            chunk_size = len(pos_stays)



            subset_models = {e: [] for e in ETIOLOGIES}

            for i in range(5):  # EDIT: NOSE negative-subset count per fold
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, len(neg_stays_shuffled))
                subset_neg = neg_stays_shuffled[start_idx:end_idx]
                if len(subset_neg) == 0: break
                stays_set = set(pos_stays).union(set(subset_neg))

                tmp_path = f"temp_subset_{i}.parquet"
                print(f"    -> Extracting NOSE Subset {i+1}/5 from disk...")
                t0 = time.time()
                extract_single_subset(train_path, stays_set, tmp_path)
                df_sub = pd.read_parquet(tmp_path)
                os.remove(tmp_path)  # free immediately
                df_sub = apply_targets(df_sub, df_etiology).fillna(0)

                for etiology in ETIOLOGIES:
                    X_sub = df_sub[feature_maps[etiology]].values
                    y_sub = df_sub[etiology].values
                    # EDIT: base Random Forest hyperparameters
                    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, max_features='sqrt', class_weight='balanced', random_state=42+i, n_jobs=-1)
                    model.fit(X_sub, y_sub)
                    subset_models[etiology].append(model)

                del df_sub
                gc.collect()
                print(f"       Trained 4 etiology models for subset {i+1} in {time.time()-t0:.1f}s")

            # Predict Validation — extract val stays sequentially too
            print(f"    -> Extracting Validation Fold from disk...")
            tmp_val_path = "temp_val.parquet"
            extract_single_subset(train_path, set(val_stays), tmp_val_path)
            df_val = pd.read_parquet(tmp_val_path)
            os.remove(tmp_val_path)
            df_val = apply_targets(df_val, df_etiology).fillna(0)

            fold_meta = df_val[['stay_id', 'is_sepsis_6h']].copy()
            for etiology in ETIOLOGIES:
                preds = np.zeros(len(df_val))
                for model in subset_models[etiology]:
                    preds += model.predict_proba(df_val[feature_maps[etiology]].values)[:, 1]
                fold_meta[f'meta_{etiology}'] = preds / len(subset_models[etiology])

            fold_meta.to_csv(f"temp_oof_fold_{fold}.csv", index=False)
            del df_val, fold_meta, subset_models
            gc.collect()

            # Clean up temp val chunk
            val_chunk_path = f"temp_val_fold_{fold}.parquet"
            if os.path.exists(val_chunk_path): os.remove(val_chunk_path)

        print("\n  Aggregating OOF Meta-Features...")
        df_train_meta = pd.concat([pd.read_csv(f"temp_oof_fold_{f}.csv") for f in range(5)])
        for f in range(5): os.remove(f"temp_oof_fold_{f}.csv")

        # 4. Train Meta-Learners
        print("  Training Meta-Learners...")
        # EDIT: Logistic Regression meta-learner hyperparameters
        lr_meta = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        lr_meta.fit(df_train_meta[meta_cols], df_train_meta['is_sepsis_6h'])

        # EDIT: XGBoost meta-learner hyperparameters
        xgb_meta = XGBClassifier(n_estimators=50, max_depth=3, scale_pos_weight=15, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_meta.fit(df_train_meta[meta_cols], df_train_meta['is_sepsis_6h'])

        # 5. Final Ensembles (Train on ALL train data in 5 subsets)
        print("\n  Training Final Ensembles on FULL train set...")
        pos_stays_full = df_meta[df_meta['is_sepsis_stay'] == 1]['stay_id'].unique()
        neg_stays_full = np.setdiff1d(df_meta['stay_id'].unique(), pos_stays_full)
        np.random.seed(42)  # EDIT: negative-stay shuffle seed (final ensembles)
        neg_stays_shuffled = np.random.permutation(neg_stays_full)
        chunk_size = len(pos_stays_full)

        final_models = {e: [] for e in ETIOLOGIES}
        for i in range(5):  # EDIT: NOSE negative-subset count (final ensembles)
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(neg_stays_shuffled))
            subset_neg = neg_stays_shuffled[start_idx:end_idx]
            if len(subset_neg) == 0: break
            stays_set = set(pos_stays_full).union(set(subset_neg))

            tmp_path = f"temp_final_{i}.parquet"
            print(f"    -> Extracting Final Subset {i+1}/5 (sequential)...")
            t0 = time.time()
            extract_single_subset(train_path, stays_set, tmp_path)
            df_sub = pd.read_parquet(tmp_path)
            os.remove(tmp_path)  # free disk immediately
            df_sub = apply_targets(df_sub, df_etiology).fillna(0)

            for etiology in ETIOLOGIES:
                X_sub = df_sub[feature_maps[etiology]].values
                y_sub = df_sub[etiology].values
                # EDIT: base Random Forest hyperparameters (final ensembles)
                model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, max_features='sqrt', class_weight='balanced', random_state=42+i, n_jobs=-1)
                model.fit(X_sub, y_sub)
                final_models[etiology].append(model)

            del df_sub
            gc.collect()
            print(f"       Final subset {i+1} done in {time.time()-t0:.1f}s")

        # 6. Evaluate on Test Set & Record Base Models
        print("\n  Evaluating on Test Set (and recording base model performance)...")
        # For simplicity in memory, test sets are smaller, we just load it
        df_test = pd.read_parquet(test_path)
        df_test = apply_targets(df_test, df_etiology).fillna(0)

        df_test_meta = df_test[['stay_id', 'is_sepsis_6h']].copy()
        y_test_true = df_test['is_sepsis_6h'].values

        # Evaluate Base Models Independently
        for etiology in ETIOLOGIES:
            preds_ensemble = np.zeros(len(df_test))
            for idx, model in enumerate(final_models[etiology]):
                # Predict for this specific base tree
                p_base = model.predict_proba(df_test[feature_maps[etiology]].values)[:, 1]
                preds_ensemble += p_base

                # Save base model metrics
                y_pred_base = (p_base >= 0.5).astype(int)  # EDIT: base-model decision threshold
                base_metrics = get_metrics(f"Base RF (Subset {idx+1}) - {etiology}", res_name, y_test_true, y_pred_base, p_base)
                base_model_results.append(base_metrics)

            df_test_meta[f'meta_{etiology}'] = preds_ensemble / len(final_models[etiology])

        # Save base model results incrementally
        pd.DataFrame(base_model_results).to_csv(base_out_file, index=False)
        print(f"  -> Base model performance appended to {base_out_file}")

        y_test = df_test_meta['is_sepsis_6h']
        p_lr = lr_meta.predict_proba(df_test_meta[meta_cols])[:, 1]
        p_xgb = xgb_meta.predict_proba(df_test_meta[meta_cols])[:, 1]

        y_pred_lr = (p_lr >= 0.5).astype(int)  # EDIT: meta-learner decision threshold (LogReg)
        metrics_lr = get_metrics('V13b NOSE LogReg Meta', res_name, y_test, y_pred_lr, p_lr)
        all_results.append(metrics_lr)

        y_pred_xgb = (p_xgb >= 0.5).astype(int)  # EDIT: meta-learner decision threshold (XGBoost)
        metrics_xgb = get_metrics('V13b NOSE XGBoost Meta', res_name, y_test, y_pred_xgb, p_xgb)
        all_results.append(metrics_xgb)

        pd.DataFrame(all_results).to_csv(out_file, index=False)
        print(f"  -> Final Meta-Learner results appended to {out_file}")
        del df_test, df_test_meta, final_models
        gc.collect()

if __name__ == '__main__':
    run_out_of_core_v13b()
