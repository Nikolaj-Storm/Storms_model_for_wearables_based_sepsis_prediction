# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  baseline_12h_benchmark.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Runs three baseline model tiers (raw vitals, engineered single-model,
#    dummy) against the 12-hour sepsis target (is_sepsis_12h) across the
#    15-min, 1-hour and 4-hour resolutions. Fills the 12h gap so the sheet
#    mirrors the 6-hour Excel layout. V13b is intentionally excluded.
#
#  INPUTS
#    ../Data/All engineered features/Dataset_all_engineered_{15min,1h,4h}_{train,test}.parquet
#    ../Data/v3_dataset_{15m,1h,4h}_{train,test}.parquet
#  OUTPUTS
#    ../Results/baseline_12h_results.csv
#    ../Results/baseline_12h_results.xlsx
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    ENG_DATA_DIR / RAW_DATA_DIR / RESULTS_DIR  -  relative paths; assumes you
#                          run from the original technical/Models/ directory
#    RESOLUTIONS        -  resolution -> filenames and per-resolution downsample
#    TARGET             -  label column, is_sepsis_12h
#    DROP_ALWAYS        -  columns dropped from the engineered feature matrix
#    RAW_VITALS         -  raw vital-sign feature list for Tier 1
#    15-min downsample  -  train/test sample(frac=0.5, random_state=42)
#    XGBoost (raw+eng)  -  n_estimators=100, max_depth=6, learning_rate=0.1,
#                          scale_pos_weight=10, random_state=42
#    Random Forest      -  n_estimators=100, max_depth=10,
#                          class_weight='balanced', random_state=42
#    Logistic Regression-  max_iter=500, solver='saga',
#                          class_weight='balanced', random_state=42
#    Dummy Classifier   -  strategy='stratified', random_state=42
#    Thresholds         -  0.5 (all tiers) plus 0.3 for engineered tier
#    SimpleImputer      -  strategy='median' (raw vitals)
#
#  REQUIRES: scikit-learn, xgboost, pandas, numpy, openpyxl
# ============================================================================

"""
baseline_12h_benchmark.py
=========================
Runs the three baseline model tiers against the 12-hour sepsis prediction
target (is_sepsis_12h) across all three sampling resolutions (15-min, 1-hour,
4-hour).  V13b is intentionally excluded -- this script fills the gap in the
existing results so the 12-hour sheet can show a full cross-tier comparison.

Tiers produced (matching the 6-hour Excel sheet exactly):
  1. Raw Vitals (No Feat. Eng.)        -- XGBoost, RF, LR @ threshold 0.5
  2. Engineered Features (Single Model) -- XGBoost, RF, LR @ 0.5 + 0.3
  3. Dummy Classifier                   -- Stratified @ 0.5

Output:
  ../Results/baseline_12h_results.csv   (full row-level CSV)
  ../Results/baseline_12h_results.xlsx  (formatted, matches 6h sheet layout)

Run from:  technical/Models/
  python baseline_12h_benchmark.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, accuracy_score, f1_score
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths  (all relative to technical/Models/ where you run the script)
# ---------------------------------------------------------------------------
ENG_DATA_DIR  = '../Data/All engineered features'  # EDIT: engineered-feature parquet folder
RAW_DATA_DIR  = '../Data'  # EDIT: raw v3 parquet folder
RESULTS_DIR   = '../Results'  # EDIT: output folder

RESOLUTIONS = {  # EDIT: resolution -> filenames + downsample flag
    '15-MINUTE': {
        'eng_train': 'Dataset_all_engineered_15min_train.parquet',
        'eng_test':  'Dataset_all_engineered_15min_test.parquet',
        'raw_train': 'v3_dataset_15m_train.parquet',
        'raw_test':  'v3_dataset_15m_test.parquet',
        'downsample': True,   # 15-min is very large; mirror what benchmark_models.py did
    },
    '1-HOUR': {
        'eng_train': 'Dataset_all_engineered_1h_train.parquet',
        'eng_test':  'Dataset_all_engineered_1h_test.parquet',
        'raw_train': 'v3_dataset_1h_train.parquet',
        'raw_test':  'v3_dataset_1h_test.parquet',
        'downsample': False,
    },
    '4-HOUR': {
        'eng_train': 'Dataset_all_engineered_4h_train.parquet',
        'eng_test':  'Dataset_all_engineered_4h_test.parquet',
        'raw_train': 'v3_dataset_4h_train.parquet',
        'raw_test':  'v3_dataset_4h_test.parquet',
        'downsample': False,
    },
}

TARGET      = 'is_sepsis_12h'  # EDIT: label column (12h horizon)
DROP_ALWAYS = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h',
               'stay_id', 'charttime', 'intime', 'sepsis3_time',
               'time_since_ICU_admit_hours']  # EDIT: columns dropped from engineered features

RAW_VITALS  = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp',
               'age', 'weight_kg']  # EDIT: raw vital-sign features (Tier 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def macro_f1(tp, fp, fn, tn):
    """Macro-averaged F1 = mean(F1_pos, F1_neg).  Matches how 6h sheet computed it."""
    f1_pos = 2*tp / max(2*tp + fp + fn, 1)
    f1_neg = 2*tn / max(2*tn + fn + fp, 1)
    return (f1_pos + f1_neg) / 2


def evaluate(y_true, y_prob, threshold, y_train_true, y_train_pred_hard):
    """Return a dict of all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1_pos    = 2 * precision * recall / max(precision + recall, 1e-9)
    mf1       = macro_f1(tp, fp, fn, tn)
    auroc     = roc_auc_score(y_true, y_prob)
    auprc     = average_precision_score(y_true, y_prob)
    train_acc = accuracy_score(y_train_true, y_train_pred_hard)
    test_acc  = accuracy_score(y_true, y_pred)

    return dict(
        Threshold=round(threshold, 4),
        AUPRC=round(auprc, 4),
        MacroF1=round(mf1, 4),
        Precision=round(precision, 4),
        Recall=round(recall, 4),
        F1_pos=round(f1_pos, 4),
        ROC_AUC=round(auroc, 4),
        TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn),
        TP_pct=round(tp/total, 4),
        FN_pct=round(fn/total, 4),
        FP_pct=round(fp/total, 4),
        TN_pct=round(tn/total, 4),
        Train_Acc=round(train_acc, 4),
        Test_Acc=round(test_acc, 4),
    )


def clean_target(df):
    """Drop rows where the 12h target is NaN, then cast to int."""
    return df.dropna(subset=[TARGET])


def load_eng(res_cfg):
    train = pd.read_parquet(os.path.join(ENG_DATA_DIR, res_cfg['eng_train']))
    test  = pd.read_parquet(os.path.join(ENG_DATA_DIR, res_cfg['eng_test']))
    train = clean_target(train)
    test  = clean_target(test)
    if res_cfg['downsample']:
        print("    [15-min] Downsampling engineered train/test by 50% to manage memory")
        train = train.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed
        test  = test.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed
    drop = [c for c in DROP_ALWAYS if c in train.columns]
    y_tr = train[TARGET].astype(int).values
    y_te = test[TARGET].astype(int).values
    X_tr = train.drop(columns=drop).fillna(0).astype('float32')
    X_te = test.drop(columns=drop).fillna(0).astype('float32')
    print(f"    After NaN-drop -- train: {len(y_tr)} rows  test: {len(y_te)} rows")
    print(f"    12h sepsis prevalence -- train: {y_tr.mean():.4f}  test: {y_te.mean():.4f}")
    return X_tr, y_tr, X_te, y_te


def load_raw(res_cfg):
    train = pd.read_parquet(os.path.join(RAW_DATA_DIR, res_cfg['raw_train']))
    test  = pd.read_parquet(os.path.join(RAW_DATA_DIR, res_cfg['raw_test']))
    train = clean_target(train)
    test  = clean_target(test)
    if res_cfg['downsample']:
        print("    [15-min] Downsampling raw vitals train/test by 50% to manage memory")
        train = train.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed
        test  = test.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed
    cols = [c for c in RAW_VITALS if c in train.columns]
    y_tr = train[TARGET].astype(int).values
    y_te = test[TARGET].astype(int).values
    print(f"    After NaN-drop -- train: {len(y_tr)} rows  test: {len(y_te)} rows")
    print(f"    12h sepsis prevalence -- train: {y_tr.mean():.4f}  test: {y_te.mean():.4f}")
    imp  = SimpleImputer(strategy='median')  # EDIT: raw-vitals imputation strategy
    X_tr = pd.DataFrame(imp.fit_transform(train[cols]), columns=cols).astype('float32')
    X_te = pd.DataFrame(imp.transform(test[cols]),  columns=cols).astype('float32')
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for res_label, res_cfg in RESOLUTIONS.items():
        print(f"\n{'='*65}")
        print(f"  RESOLUTION: {res_label}")
        print(f"{'='*65}")

        # ----------------------------------------------------------------
        # TIER 1 -- Raw Vitals (No Feat. Eng.)
        # ----------------------------------------------------------------
        print("\n  [Tier 1] Raw Vitals (No Feat. Eng.)")
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = load_raw(res_cfg)
        print(f"    Train: {X_tr_raw.shape}  |  Test: {X_te_raw.shape}")

        raw_models = [
            # EDIT: Tier 1 XGBoost hyperparameters
            ('XGBoost',           XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                                scale_pos_weight=10, random_state=42,
                                                use_label_encoder=False, eval_metric='logloss',
                                                verbosity=0, n_jobs=-1)),
            # EDIT: Tier 1 Random Forest hyperparameters
            ('Random Forest',     RandomForestClassifier(n_estimators=100, max_depth=10,
                                                         class_weight='balanced', random_state=42,
                                                         n_jobs=-1)),
            ('Logistic Regression', None),   # needs scaling -- handled below
        ]

        for algo_name, model in raw_models:
            print(f"    -> {algo_name}...")
            t0 = time.time()

            if algo_name == 'Logistic Regression':
                scaler = StandardScaler()
                Xtr_s = np.clip(scaler.fit_transform(X_tr_raw), -1e6, 1e6)
                Xte_s = np.clip(scaler.transform(X_te_raw),     -1e6, 1e6)
                Xtr_s = np.nan_to_num(Xtr_s)
                Xte_s = np.nan_to_num(Xte_s)
                # EDIT: Tier 1 Logistic Regression hyperparameters
                model = LogisticRegression(max_iter=500, solver='saga',
                                           class_weight='balanced', random_state=42, n_jobs=-1)
                model.fit(Xtr_s, y_tr_raw)
                y_prob   = model.predict_proba(Xte_s)[:, 1]
                y_tr_hard = model.predict(Xtr_s)
            else:
                model.fit(X_tr_raw, y_tr_raw)
                y_prob    = model.predict_proba(X_te_raw)[:, 1]
                y_tr_hard = model.predict(X_tr_raw)

            metrics = evaluate(y_te_raw, y_prob, 0.5, y_tr_raw, y_tr_hard)  # EDIT: Tier 1 decision threshold
            rows.append(dict(
                Resolution=res_label,
                ModelTier='Raw Vitals (No Feat. Eng.)',
                Algorithm=algo_name,
                OperatingPoint='Default threshold (0.50)',
                **metrics
            ))
            print(f"       ROC-AUC={metrics['ROC_AUC']}  Recall={metrics['Recall']}  done in {time.time()-t0:.0f}s")

        del X_tr_raw, X_te_raw, y_tr_raw, y_te_raw

        # ----------------------------------------------------------------
        # TIER 2 -- Engineered Features (Single Model)
        # ----------------------------------------------------------------
        print("\n  [Tier 2] Engineered Features (Single Model)")
        X_tr_eng, y_tr_eng, X_te_eng, y_te_eng = load_eng(res_cfg)

        eng_models = [
            # EDIT: Tier 2 XGBoost hyperparameters
            ('XGBoost',           XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                                scale_pos_weight=10, random_state=42,
                                                use_label_encoder=False, eval_metric='logloss',
                                                verbosity=0, n_jobs=-1)),
            # EDIT: Tier 2 Random Forest hyperparameters
            ('Random Forest',     RandomForestClassifier(n_estimators=100, max_depth=10,
                                                         class_weight='balanced', random_state=42,
                                                         n_jobs=-1)),
            ('Logistic Regression', None),
        ]

        for algo_name, model in eng_models:
            print(f"    -> {algo_name}...")
            t0 = time.time()

            if algo_name == 'Logistic Regression':
                scaler = StandardScaler()
                Xtr_s = np.clip(scaler.fit_transform(X_tr_eng), -1e6, 1e6)
                Xte_s = np.clip(scaler.transform(X_te_eng),     -1e6, 1e6)
                Xtr_s = np.nan_to_num(Xtr_s)
                Xte_s = np.nan_to_num(Xte_s)
                # EDIT: Tier 2 Logistic Regression hyperparameters
                model = LogisticRegression(max_iter=500, solver='saga',
                                           class_weight='balanced', random_state=42, n_jobs=-1)
                model.fit(Xtr_s, y_tr_eng)
                y_prob    = model.predict_proba(Xte_s)[:, 1]
                y_tr_hard = model.predict(Xtr_s)
                X_eval = Xte_s
            else:
                model.fit(X_tr_eng, y_tr_eng)
                y_prob    = model.predict_proba(X_te_eng)[:, 1]
                y_tr_hard = model.predict(X_tr_eng)

            for thresh, label in [(0.5, 'Default threshold (0.50)'),
                                   (0.3, 'Custom threshold (0.300)')]:  # EDIT: Tier 2 thresholds (0.5 and 0.3)
                metrics = evaluate(y_te_eng, y_prob, thresh, y_tr_eng, y_tr_hard)
                rows.append(dict(
                    Resolution=res_label,
                    ModelTier='Engineered Features (Single Model)',
                    Algorithm=algo_name,
                    OperatingPoint=label,
                    **metrics
                ))

            print(f"       ROC-AUC={metrics['ROC_AUC']}  Recall(0.3)={rows[-1]['Recall']}  done in {time.time()-t0:.0f}s")

        # ----------------------------------------------------------------
        # TIER 3 -- Dummy Classifier
        # ----------------------------------------------------------------
        print("\n  [Tier 3] Dummy Classifier")
        t0 = time.time()
        dummy = DummyClassifier(strategy='stratified', random_state=42)  # EDIT: dummy strategy / seed
        dummy.fit(X_tr_eng, y_tr_eng)
        y_prob_d  = dummy.predict_proba(X_te_eng)[:, 1]
        y_tr_hard_d = dummy.predict(X_tr_eng)
        metrics_d = evaluate(y_te_eng, y_prob_d, 0.5, y_tr_eng, y_tr_hard_d)  # EDIT: Tier 3 decision threshold
        rows.append(dict(
            Resolution=res_label,
            ModelTier='Dummy Classifier',
            Algorithm='Stratified Dummy',
            OperatingPoint='Default threshold (0.50)',
            **metrics_d
        ))
        print(f"       done in {time.time()-t0:.0f}s")

        del X_tr_eng, X_te_eng, y_tr_eng, y_te_eng

    # -------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'baseline_12h_results.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"\n\nCSV saved -> {csv_path}")

    # -------------------------------------------------------------------
    # Save Excel -- formatted to match the 6h sheet layout
    # -------------------------------------------------------------------
    xlsx_path = os.path.join(RESULTS_DIR, 'baseline_12h_results.xlsx')

    HEADER = ['Model Tier', 'Algorithm / Model', 'Variant / Operating Point',
              'Threshold', 'AUPRC', 'Macro F1', 'Precision', 'Recall',
              'F1 (pos class)', 'ROC AUC',
              'TP', 'FN', 'FP', 'TN',
              'TP %', 'FN %', 'FP %', 'TN %',
              'Train Acc', 'Test Acc']

    col_map = {
        'ModelTier':      'Model Tier',
        'Algorithm':      'Algorithm / Model',
        'OperatingPoint': 'Variant / Operating Point',
        'Threshold':      'Threshold',
        'AUPRC':          'AUPRC',
        'MacroF1':        'Macro F1',
        'Precision':      'Precision',
        'Recall':         'Recall',
        'F1_pos':         'F1 (pos class)',
        'ROC_AUC':        'ROC AUC',
        'TP':             'TP',
        'FN':             'FN',
        'FP':             'FP',
        'TN':             'TN',
        'TP_pct':         'TP %',
        'FN_pct':         'FN %',
        'FP_pct':         'FP %',
        'TN_pct':         'TN %',
        'Train_Acc':      'Train Acc',
        'Test_Acc':       'Test Acc',
    }

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for res_label in RESOLUTIONS:
            sheet_rows = []
            sheet_rows.append([f'{res_label} RESOLUTION  --  12-Hour Sepsis-3 Prediction  --  Baseline Tiers'] + ['']*(len(HEADER)-1))
            sheet_rows.append([''] * len(HEADER))
            sheet_rows.append(HEADER)

            subset = df_out[df_out['Resolution'] == res_label]
            for _, r in subset.iterrows():
                row = [r.get(k, '') for k in col_map]
                sheet_rows.append(row)

            sheet_df = pd.DataFrame(sheet_rows)
            sheet_name = res_label.replace('-', '').replace(' ', '_')[:31]
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

    print(f"Excel saved -> {xlsx_path}")

    # -------------------------------------------------------------------
    # Quick summary table in console
    # -------------------------------------------------------------------
    print("\n\n" + "="*65)
    print("SUMMARY  (12-hour horizon, all tiers, primary metric ROC-AUC)")
    print("="*65)
    summary = df_out[df_out['OperatingPoint'].str.contains('0.50')][
        ['Resolution','ModelTier','Algorithm','ROC_AUC','Recall','AUPRC']
    ].sort_values(['Resolution','ModelTier','Algorithm'])
    print(summary.to_string(index=False))
    print("="*65)


if __name__ == '__main__':
    run()
