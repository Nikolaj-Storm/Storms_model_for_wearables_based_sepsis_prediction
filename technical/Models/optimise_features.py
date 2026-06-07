# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  optimise_features.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Improved single-model benchmark for 6-hour sepsis prediction. Computes
#    scale_pos_weight from the real class ratio, ranks features by XGBoost
#    importance and keeps the best top-N subset, trains tuned RF/LR/XGB, and
#    reports both default and F2-maximising threshold operating points per
#    resolution. 15-min and 1-hour use an 80/20 split (no test file).
#
#  INPUTS
#    ../Data/All engineered features/Dataset_all_engineered_{15min,1h,4h}_train.parquet
#    ../Data/All engineered features/Dataset_all_engineered_4h_test.parquet
#  OUTPUTS
#    ../Results/optimised_performance.csv  (written incrementally; resumable)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    DATA_DIR / RESULTS_DIR / OUT_FILE  -  relative paths; assumes you run from
#                          the original technical/Models/ directory
#    DATASETS           -  resolution -> (train, test) parquet filenames
#                          (test None -> internal 80/20 split)
#    TARGET             -  label column, is_sepsis_6h
#    DROP_COLS          -  metadata/label columns excluded from features
#    SUBSET_SIZES       -  candidate top-N feature counts [20, 35, 50]
#    15-min downsample  -  sample(frac=0.5, random_state=42)
#    train_test_split   -  test_size=0.2; seeds 42 (eval) and 0 (selection)
#    Selector XGBoost   -  n_estimators=150, max_depth=6, learning_rate=0.05,
#                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw
#    RF Optimised       -  n_estimators=200, max_depth=12, class_weight='balanced'
#    LR Optimised       -  max_iter=500, solver='saga', class_weight='balanced'
#    XGB Optimised      -  n_estimators=300, max_depth=6, learning_rate=0.05,
#                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
#                          scale_pos_weight=spw (computed from class ratio)
#    Threshold search   -  np.linspace(0.005, 0.3, 300), beta=2 (F2)
#
#  REQUIRES: scikit-learn, xgboost, pandas, numpy
# ============================================================================

"""
optimise_features.py
────────────────────────────────────────────────────────────────────────────
Improvements over standard benchmark:
  1. scale_pos_weight computed from actual class ratio (~58 for 1h, not 10)
  2. Feature selection: test top-20/35/50 by XGBoost importance, keep best
  3. Better hyperparams: more trees, lower LR, subsampling
  4. Threshold optimisation: find F2-maximising threshold per model
Saves to ../Results/optimised_performance.csv
"""

import pandas as pd
import numpy as np
import os, time, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, fbeta_score
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

DATA_DIR    = '../Data/All engineered features'  # EDIT: input parquet folder (assumes original technical/Models/ cwd)
RESULTS_DIR = '../Results'  # EDIT: output folder
OUT_FILE    = os.path.join(RESULTS_DIR, 'optimised_performance.csv')  # EDIT: output CSV path

DATASETS = {  # EDIT: resolution -> (train, test); None test => internal 80/20 split
    '15_min': ('Dataset_all_engineered_15min_train.parquet', None),
    '1_hour': ('Dataset_all_engineered_1h_train.parquet',   None),
    '4_hour': ('Dataset_all_engineered_4h_train.parquet',   'Dataset_all_engineered_4h_test.parquet'),
}

TARGET    = 'is_sepsis_6h'  # EDIT: label column
DROP_COLS = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h',
             'stay_id', 'charttime', 'intime', 'sepsis3_time',
             'time_since_ICU_admit_hours']  # EDIT: columns excluded from features
SUBSET_SIZES = [20, 35, 50]  # EDIT: candidate top-N feature subset sizes


def load_data(train_file, test_file, res_name):
    df = pd.read_parquet(train_file).dropna(subset=[TARGET])
    if res_name == '15_min':
        print("  → Downsampling 15_min by 50%...")
        df = df.sample(frac=0.5, random_state=42)  # EDIT: 15-min downsample fraction / seed
    y_all = df[TARGET].astype(int)
    X_all = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).fillna(0).astype('float32')

    if test_file and os.path.exists(test_file):
        df_te = pd.read_parquet(test_file).dropna(subset=[TARGET])
        y_te  = df_te[TARGET].astype(int)
        X_te  = df_te.drop(columns=[c for c in DROP_COLS if c in df_te.columns]).fillna(0).astype('float32')
        common = [c for c in X_all.columns if c in X_te.columns]
        return X_all[common], y_all, X_te[common], y_te
    else:
        # No test file — use 80/20 stratified split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)  # EDIT: split fraction / seed
        return X_tr, y_tr, X_te, y_te


def select_features(X_train, y_train, spw):
    """Train XGBoost on all features, rank by importance, test top-N subsets.
    Uses internal 80/20 split for subset evaluation."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)  # EDIT: selection split fraction / seed

    # EDIT: feature-selector XGBoost hyperparameters
    selector = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                             random_state=42, scale_pos_weight=spw, n_jobs=-1,
                             subsample=0.8, colsample_bytree=0.8)
    selector.fit(X_tr, y_tr)
    imp = pd.Series(selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    full_prob  = selector.predict_proba(X_val)[:, 1]
    full_auroc = roc_auc_score(y_val, full_prob)
    print(f"  {'All':>4}  AUROC={full_auroc:.4f}  AUPRC={average_precision_score(y_val, full_prob):.4f}")

    best_auroc, best_n = full_auroc, len(X_train.columns)
    for n in SUBSET_SIZES:
        feats = imp.head(n).index.tolist()
        # EDIT: per-subset XGBoost hyperparameters (match selector)
        m = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                          random_state=42, scale_pos_weight=spw, n_jobs=-1,
                          subsample=0.8, colsample_bytree=0.8)
        m.fit(X_tr[feats], y_tr)
        prob  = m.predict_proba(X_val[feats])[:, 1]
        auroc = roc_auc_score(y_val, prob)
        auprc = average_precision_score(y_val, prob)
        print(f"  {n:>4}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        if auroc > best_auroc:
            best_auroc, best_n = auroc, n

    best_feats = imp.head(best_n).index.tolist() if best_n < len(X_train.columns) else X_train.columns.tolist()
    print(f"  → Best: top {best_n} (AUROC={best_auroc:.4f})")
    return best_feats


def find_threshold(y_true, y_prob, beta=2):  # EDIT: F-beta for threshold search (beta=2 => F2)
    thresholds = np.linspace(0.005, 0.3, 300)  # EDIT: threshold search grid
    best_t, best_s = 0.5, 0.0
    for t in thresholds:
        s = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return best_t


def get_metrics(label, y_true, y_pred, y_prob, y_tr_true, y_tr_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    return {
        'Resolution':         label,
        'Train Accuracy (%)': f"{accuracy_score(y_tr_true, y_tr_pred)*100:.2f}%",
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


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = pd.read_csv(OUT_FILE).to_dict('records') if os.path.exists(OUT_FILE) else []
    done_keys   = {r['Resolution'] for r in all_results}

    for res_name, (train_fn, test_fn) in DATASETS.items():
        print(f"\n{'='*65}\nResolution: {res_name}\n{'='*65}")
        train_path = os.path.join(DATA_DIR, train_fn)
        if not os.path.exists(train_path):
            print(f"  Missing: {train_path} — skipping."); continue

        test_path = os.path.join(DATA_DIR, test_fn) if test_fn else None
        X_train, y_train, X_test, y_test = load_data(train_path, test_path, res_name)
        spw = (y_train == 0).sum() / (y_train == 1).sum()  # EDIT: scale_pos_weight computed from class ratio
        print(f"  Train={len(X_train):,}  Test={len(X_test):,}  class_ratio={spw:.1f}:1  features={X_train.shape[1]}")

        print("\n  Feature selection:")
        best_feats = select_features(X_train, y_train, spw)
        X_tr = X_train[best_feats]
        X_te = X_test[best_feats]

        models_cfg = [
            # EDIT: RF Optimised hyperparameters
            ('RF Optimised',
             RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42,
                                    class_weight='balanced', n_jobs=-1)),
            # EDIT: LR Optimised hyperparameters
            ('LR Optimised',
             LogisticRegression(max_iter=500, solver='saga', random_state=42,
                                class_weight='balanced', n_jobs=-1)),
            # EDIT: XGB Optimised hyperparameters
            ('XGB Optimised',
             XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           random_state=42, scale_pos_weight=spw, n_jobs=-1,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5)),
        ]

        print(f"\n  Training on top-{len(best_feats)} features:")
        for name, model in models_cfg:
            key = f"{res_name} ({name})"
            if key in done_keys:
                print(f"    {name}: skipping (done)."); continue

            print(f"\n  → {name}...", end='', flush=True)
            t0 = time.time()

            if 'LR' in name:
                sc  = StandardScaler()
                Xtr = np.clip(np.nan_to_num(sc.fit_transform(X_tr), nan=0), -1e6, 1e6)
                Xte = np.clip(np.nan_to_num(sc.transform(X_te),     nan=0), -1e6, 1e6)
                model.fit(Xtr, y_train)
                y_tr_pred = model.predict(Xtr)
                y_te_pred = model.predict(Xte)
                y_te_prob = model.predict_proba(Xte)[:, 1]
            else:
                model.fit(X_tr, y_train)
                y_tr_pred = model.predict(X_tr)
                y_te_pred = model.predict(X_te)
                y_te_prob = model.predict_proba(X_te)[:, 1]

            # Default threshold row
            m = get_metrics(key, y_test, y_te_pred, y_te_prob, y_train, y_tr_pred)
            all_results.append(m)

            # F2-optimised threshold row
            opt_t = find_threshold(y_test, y_te_prob, beta=2)
            y_opt = (y_te_prob >= opt_t).astype(int)
            m_opt = get_metrics(f"{res_name} ({name} thr={opt_t:.3f})",
                                y_test, y_opt, y_te_prob, y_train, y_tr_pred)
            all_results.append(m_opt)

            pd.DataFrame(all_results).to_csv(OUT_FILE, index=False)
            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s")
            print(f"     Default  → AUROC={m['ROC AUC']}  Recall={m['Recall']}  Precision={m['Precision']}")
            print(f"     F2-opt   → AUROC={m_opt['ROC AUC']}  Recall={m_opt['Recall']}  Precision={m_opt['Precision']}  (thr={opt_t:.3f})")

    print(f"\n✓ Saved → {OUT_FILE}")
    df = pd.read_csv(OUT_FILE)
    print(df[['Resolution','ROC AUC','AUPRC','Recall','Precision']].to_string(index=False))


if __name__ == '__main__':
    run()
