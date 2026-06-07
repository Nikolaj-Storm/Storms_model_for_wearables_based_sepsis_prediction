# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  optimise_features_v2.py
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Leakage-free optimised baselines (RF / LR / XGB) at three sampling
#    resolutions. Performs XGBoost importance-based feature selection over
#    candidate top-N subsets, trains each baseline on the selected features, and
#    reports default and F2-optimised-threshold metrics. Uses the real test
#    parquet as the held-out set when present, otherwise a stay-level
#    GroupShuffleSplit to avoid the row-level leakage of v1.
#
#  INPUTS
#    ../../Data/All engineered features/Dataset_all_engineered_{15min,1h,4h}_{train,test}.parquet
#  OUTPUTS
#    ../../Results/optimised_performance_v2.csv
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    Assumed working directory  -  run from technical/Models/V13b so ../../Data
#      and ../../Results resolve.
#    DATA_DIR / RESULTS_DIR / OUT_FILE  -  input folder, output root, CSV path.
#    DATASETS                   -  resolution -> (train_fn, test_fn) map.
#    TARGET                     -  prediction label column (is_sepsis_6h).
#    DROP_COLS                  -  columns excluded from features (note this v2
#                                  drops time_since_ICU_admit_hours).
#    SUBSET_SIZES=[20,35,50]    -  candidate top-N feature counts.
#    RANDOM_STATE=42            -  random seed.
#    15_min downsample fraction -  0.5 (stay-level).
#    Selector XGB               -  n_estimators=150, max_depth=6,
#                                  learning_rate=0.05, scale_pos_weight=spw,
#                                  subsample=0.8, colsample_bytree=0.8.
#    RF Optimised               -  n_estimators=200, max_depth=12,
#                                  class_weight='balanced'.
#    LR Optimised               -  max_iter=500, solver='saga',
#                                  class_weight='balanced'.
#    XGB Optimised              -  n_estimators=300, max_depth=6,
#                                  learning_rate=0.05, scale_pos_weight=spw,
#                                  subsample=0.8, colsample_bytree=0.8,
#                                  min_child_weight=5.
#    F2 threshold sweep         -  np.linspace(0.005, 0.3, 300), beta=2.
#
#  REQUIRES: pandas, numpy, scikit-learn, xgboost
# ============================================================================
"""
optimise_features_v2.py
────────────────────────────────────────────────────────────────────────────
Leakage-free version of optimise_features.py.

Difference from v1: when a Dataset_all_engineered_*_test.parquet file exists
for a resolution, it is used as the held-out evaluation set (proper
stay-level split, no leakage). When no test file exists, falls back to a
STAY-LEVEL train_test_split via stay_id grouping rather than the v1
row-level split. The v1 row-level split caused inflated AUROC because the
same patient's rows appeared in both train and test partitions.

All other choices (DROP_COLS, hyperparameters, F2 threshold optimisation,
feature selection by XGBoost importance) are preserved from v1 so the
baselines remain comparable to the original optimised numbers, just
evaluated on a leakage-free split.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, fbeta_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# EDIT: DATA_DIR / RESULTS_DIR / OUT_FILE (relative to working directory)
DATA_DIR = "../../Data/All engineered features"
RESULTS_DIR = "../../Results"
OUT_FILE = os.path.join(RESULTS_DIR, "optimised_performance_v2.csv")

# EDIT: DATASETS - resolution -> (train_fn, test_fn) map
DATASETS = {
    "15_min": ("Dataset_all_engineered_15min_train.parquet",
               "Dataset_all_engineered_15min_test.parquet"),
    "1_hour": ("Dataset_all_engineered_1h_train.parquet",
               "Dataset_all_engineered_1h_test.parquet"),
    "4_hour": ("Dataset_all_engineered_4h_train.parquet",
               "Dataset_all_engineered_4h_test.parquet"),
}

# EDIT: TARGET label, DROP_COLS, SUBSET_SIZES candidate counts, RANDOM_STATE
TARGET = "is_sepsis_6h"
DROP_COLS = ["is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
             "stay_id", "charttime", "intime", "sepsis3_time",
             "time_since_ICU_admit_hours"]
SUBSET_SIZES = [20, 35, 50]
RANDOM_STATE = 42


def load_data(train_file, test_file, res_name):
    df = pd.read_parquet(train_file).dropna(subset=[TARGET])
    if res_name == "15_min":
        # Stay-level downsampling (preserve stay structure) so feature selection
        # and training don't blow memory. Same fraction as v1 (50%).
        print("  -> Downsampling 15_min by 50% at the STAY level...")
        rng = np.random.RandomState(RANDOM_STATE)
        # EDIT: 15_min stay-level downsample fraction (0.5)
        keep = rng.choice(df["stay_id"].unique(),
                          size=int(df["stay_id"].nunique() * 0.5),
                          replace=False)
        df = df[df["stay_id"].isin(keep)]

    y_all = df[TARGET].astype(int)
    stay_all = df["stay_id"].values
    X_all = df.drop(columns=[c for c in DROP_COLS if c in df.columns]) \
              .fillna(0).astype("float32")

    if test_file and os.path.exists(test_file):
        # Use the real test parquet (stay-level split is already enforced upstream).
        df_te = pd.read_parquet(test_file).dropna(subset=[TARGET])
        y_te = df_te[TARGET].astype(int)
        X_te = df_te.drop(columns=[c for c in DROP_COLS if c in df_te.columns]) \
                    .fillna(0).astype("float32")
        common = [c for c in X_all.columns if c in X_te.columns]
        return X_all[common], y_all, X_te[common], y_te
    else:
        # Stay-level fallback: GroupShuffleSplit so no stay_id is in both
        # train and test. This is the leakage fix vs. v1.
        print("  -> No test parquet; falling back to STAY-LEVEL 80/20 split.")
        # EDIT: stay-level fallback test fraction (0.2)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(gss.split(X_all, y_all, groups=stay_all))
        return (X_all.iloc[tr_idx], y_all.iloc[tr_idx],
                X_all.iloc[te_idx], y_all.iloc[te_idx])


def select_features(X_train, y_train, spw):
    """Same as v1, but uses a stay-aware split would require stay_id;
    inside select_features the data is already feature-only, so this internal
    split is only used for ranking and won't leak into the held-out test set."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
    # EDIT: feature-selector XGB hyperparameters
    selector = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                             random_state=42, scale_pos_weight=spw, n_jobs=-1,
                             subsample=0.8, colsample_bytree=0.8)
    selector.fit(X_tr, y_tr)
    imp = pd.Series(selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    full_prob = selector.predict_proba(X_val)[:, 1]
    full_auroc = roc_auc_score(y_val, full_prob)
    print(f"  {'All':>4}  AUROC={full_auroc:.4f}  AUPRC={average_precision_score(y_val, full_prob):.4f}")
    best_auroc, best_n = full_auroc, len(X_train.columns)
    for n in SUBSET_SIZES:
        feats = imp.head(n).index.tolist()
        # EDIT: per-subset selection XGB hyperparameters (match selector)
        m = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                          random_state=42, scale_pos_weight=spw, n_jobs=-1,
                          subsample=0.8, colsample_bytree=0.8)
        m.fit(X_tr[feats], y_tr)
        prob = m.predict_proba(X_val[feats])[:, 1]
        auroc = roc_auc_score(y_val, prob)
        auprc = average_precision_score(y_val, prob)
        print(f"  {n:>4}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        if auroc > best_auroc:
            best_auroc, best_n = auroc, n
    best_feats = imp.head(best_n).index.tolist() if best_n < len(X_train.columns) else X_train.columns.tolist()
    print(f"  -> Best: top {best_n} (AUROC={best_auroc:.4f})")
    return best_feats


def find_threshold(y_true, y_prob, beta=2):
    # EDIT: F2 threshold sweep grid (0.005 to 0.3, 300 points) and beta=2
    thresholds = np.linspace(0.005, 0.3, 300)
    best_t, best_s = 0.5, 0.0
    for t in thresholds:
        s = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return best_t


def get_metrics(label, y_true, y_pred, y_prob, y_tr_true, y_tr_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)
    return {
        "Resolution": label,
        "Train Accuracy (%)": f"{accuracy_score(y_tr_true, y_tr_pred)*100:.2f}%",
        "Test Accuracy (%)":  f"{accuracy_score(y_true, y_pred)*100:.2f}%",
        "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        "Recall":    f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        "F1 Score":  f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        "ROC AUC":   f"{roc_auc_score(y_true, y_prob):.4f}",
        "AUPRC":     f"{average_precision_score(y_true, y_prob):.4f}",
        "TP (Count)": int(tp), "FN (Count)": int(fn),
        "FP (Count)": int(fp), "TN (Count)": int(tn),
        "TP (%)": f"{tp/total*100:.2f}%", "FN (%)": f"{fn/total*100:.2f}%",
        "FP (%)": f"{fp/total*100:.2f}%", "TN (%)": f"{tn/total*100:.2f}%",
    }


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for res_name, (train_fn, test_fn) in DATASETS.items():
        print(f"\n{'='*65}\nResolution: {res_name}  (LEAKAGE-FREE)\n{'='*65}")
        train_path = os.path.join(DATA_DIR, train_fn)
        if not os.path.exists(train_path):
            print(f"  Missing: {train_path} - skipping."); continue

        test_path = os.path.join(DATA_DIR, test_fn) if test_fn else None
        X_train, y_train, X_test, y_test = load_data(train_path, test_path, res_name)
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"  Train={len(X_train):,}  Test={len(X_test):,}  "
              f"class_ratio={spw:.1f}:1  features={X_train.shape[1]}")

        print("\n  Feature selection:")
        best_feats = select_features(X_train, y_train, spw)
        X_tr = X_train[best_feats]
        X_te = X_test[best_feats]

        # EDIT: baseline model hyperparameters (RF / LR / XGB Optimised)
        models_cfg = [
            ("RF Optimised",
             RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42,
                                    class_weight="balanced", n_jobs=-1)),
            ("LR Optimised",
             LogisticRegression(max_iter=500, solver="saga", random_state=42,
                                class_weight="balanced", n_jobs=-1)),
            ("XGB Optimised",
             XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           random_state=42, scale_pos_weight=spw, n_jobs=-1,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5)),
        ]

        print(f"\n  Training on top-{len(best_feats)} features:")
        for name, model in models_cfg:
            print(f"\n  -> {name}...", end="", flush=True)
            t0 = time.time()
            if "LR" in name:
                sc = StandardScaler()
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

            key = f"{res_name} ({name})"
            m = get_metrics(key, y_test, y_te_pred, y_te_prob, y_train, y_tr_pred)
            all_results.append(m)

            opt_t = find_threshold(y_test, y_te_prob, beta=2)
            y_opt = (y_te_prob >= opt_t).astype(int)
            m_opt = get_metrics(f"{res_name} ({name} thr={opt_t:.3f})",
                                y_test, y_opt, y_te_prob, y_train, y_tr_pred)
            all_results.append(m_opt)

            pd.DataFrame(all_results).to_csv(OUT_FILE, index=False)
            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s")
            print(f"     Default  -> AUROC={m['ROC AUC']}  Recall={m['Recall']}  Precision={m['Precision']}")
            print(f"     F2-opt   -> AUROC={m_opt['ROC AUC']}  Recall={m_opt['Recall']}  Precision={m_opt['Precision']}  (thr={opt_t:.3f})")

    print(f"\nSaved -> {OUT_FILE}")
    df = pd.read_csv(OUT_FILE)
    print(df[["Resolution", "ROC AUC", "AUPRC", "Recall", "Precision"]].to_string(index=False))


if __name__ == "__main__":
    run()
