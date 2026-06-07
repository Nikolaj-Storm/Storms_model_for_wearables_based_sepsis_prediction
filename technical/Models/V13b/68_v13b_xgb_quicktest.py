# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  68_v13b_xgb_quicktest.py
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Quick test of XGBoost versus Random Forest as the NOSE base learner. Runs
#    a slimmed V13b pipeline on a 5% stay-level subsample of the 1-hour "all
#    engineered" dataset, once with RF base learners and once with XGB, and
#    compares AUROC, AUPRC, sensitivity, FPR and wall time at the F1-optimal
#    threshold.
#
#  INPUTS
#    ../../Data/All engineered features/Dataset_all_engineered_1h_train.parquet
#    ../../Data/All engineered features/Dataset_all_engineered_1h_test.parquet
#      (or the white-labeled TEST_FILE_FALLBACK if the primary test is missing)
#    Etiology train/test parquet (first existing of the _ETIO_CANDIDATES list)
#  OUTPUTS
#    ../../Results/V13b_final/xgb_vs_rf_quicktest.txt
#    ../../Results/V13b_final/xgb_vs_rf_quicktest.csv
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    Assumed working directory  -  run from technical/Models/V13b so ../../Data
#      and ../../Results resolve.
#    DATA_DIR / RESULTS_DIR     -  engineered-feature folder and output root.
#    _ETIO_CANDIDATES           -  ordered list of etiology train parquet paths.
#    TRAIN_FILE / TEST_FILE_*   -  1-hour engineered train/test parquet paths.
#    DROP_COLS                  -  columns excluded from the feature set.
#    NUM_NOSE_SUBSETS=5         -  NOSE negative-undersampling subsets.
#    NUM_FOLDS=3                -  OOF folds (cheaper than 5 for the quicktest).
#    SAMPLE_FRAC=0.05           -  stay-level subsample fraction.
#    RANDOM_STATE=42            -  random seed.
#    RF base learner            -  n_estimators=200, max_depth=12,
#                                  min_samples_leaf=5, max_features='sqrt',
#                                  class_weight='balanced'.
#    XGB base learner           -  n_estimators=200, max_depth=6,
#                                  learning_rate=0.05, subsample=0.8,
#                                  colsample_bytree=0.8, min_child_weight=5,
#                                  scale_pos_weight=neg/pos.
#    LR meta-learner            -  class_weight='balanced', max_iter=1000.
#    F1-max sweep grid          -  np.linspace(0.005, 0.95, 950).
#
#  REQUIRES: pandas, numpy, scikit-learn, xgboost
# ============================================================================
"""
V13b Quick Test: XGBoost vs Random Forest as the NOSE base learner.

Runs a slimmed V13b pipeline on a 5% stay-level subsample of the 1-hour
"all engineered" dataset, twice: once with RF base learners, once with XGB.
Compares AUROC, AUPRC, sensitivity, FPR, and wall time at the F1-optimal
threshold (recall-weighted).

Outputs:
  ../../Results/V13b_final/xgb_vs_rf_quicktest.txt
  ../../Results/V13b_final/xgb_vs_rf_quicktest.csv

If XGB beats RF by >0.01 AUROC at 5%, consider running V13b end-to-end with
XGB base. Otherwise stick with RF.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, fbeta_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# EDIT: DATA_DIR / RESULTS_DIR (relative to working directory)
DATA_DIR = Path("../../Data/All engineered features")
RESULTS_DIR = Path("../../Results/V13b_final")
# EDIT: _ETIO_CANDIDATES - ordered list of etiology train parquet paths to try
_ETIO_CANDIDATES = [
    Path("../../Data/Etiology/v8_dataset_1h_etiology_train.parquet"),
    Path("/PATH/TO/PROJECT/technical/Data/Etiology/v8_dataset_1h_etiology_train.parquet"),
    Path("/PATH/TO/PROJECT/technical/v8_dataset_1h_etiology_train.parquet"),
]
ETIO_TRAIN = next((p for p in _ETIO_CANDIDATES if p.exists()), _ETIO_CANDIDATES[0])
ETIO_TEST = Path(str(ETIO_TRAIN).replace("_train.parquet", "_test.parquet"))
# EDIT: TRAIN_FILE / TEST_FILE primary and fallback paths
TRAIN_FILE = DATA_DIR / "Dataset_all_engineered_1h_train.parquet"
TEST_FILE_PRIMARY = DATA_DIR / "Dataset_all_engineered_1h_test.parquet"
TEST_FILE_FALLBACK = Path("/PATH/TO/PROJECT/technical/Dataset_all_engineered_1h_test.parquet")

# EDIT: DROP_COLS - columns excluded from the feature set
DROP_COLS = [
    "is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
    "stay_id", "charttime", "intime", "sepsis3_time",
    "time_since_ICU_admit_hours",
]

# EDIT: subsample / NOSE / fold / seed settings
NUM_NOSE_SUBSETS = 5
NUM_FOLDS = 3            # cheaper than 5 for the quicktest
SAMPLE_FRAC = 0.05
RANDOM_STATE = 42


def load():
    df_train = pd.read_parquet(TRAIN_FILE)
    test_path = TEST_FILE_PRIMARY if TEST_FILE_PRIMARY.exists() else TEST_FILE_FALLBACK
    df_test = pd.read_parquet(test_path)

    cols = ["stay_id", "target_resp", "target_uri", "target_other"]
    etio_tr = pd.read_parquet(ETIO_TRAIN, columns=cols).groupby("stay_id").max().reset_index()
    etio_te = pd.read_parquet(ETIO_TEST, columns=cols).groupby("stay_id").max().reset_index()
    df_train = df_train.merge(etio_tr, on="stay_id", how="left").fillna({"target_resp": 0, "target_uri": 0, "target_other": 0})
    df_test = df_test.merge(etio_te, on="stay_id", how="left").fillna({"target_resp": 0, "target_uri": 0, "target_other": 0})

    for c in ["target_resp", "target_uri", "target_other"]:
        for h in ["6h", "12h"]:
            df_train[f"{c}_{h}"] = ((df_train[f"is_sepsis_{h}"] == 1) & (df_train[c] == 1)).astype(int)
            df_test[f"{c}_{h}"] = ((df_test[f"is_sepsis_{h}"] == 1) & (df_test[c] == 1)).astype(int)
    return df_train, df_test


def stay_subsample(df, frac, seed):
    stays = df["stay_id"].unique()
    rng = np.random.RandomState(seed)
    keep = rng.choice(stays, size=max(1, int(len(stays) * frac)), replace=False)
    return df[df["stay_id"].isin(keep)].copy()


def features(df):
    excl = set(DROP_COLS) | {"target_resp", "target_uri", "target_other"} | {
        f"{c}_{h}" for c in ["target_resp", "target_uri", "target_other"] for h in ["6h", "12h"]
    }
    return [c for c in df.columns if c not in excl]


def build_base(kind, seed_offset, spw):
    if kind == "rf":
        # EDIT: RF base-learner hyperparameters
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE + seed_offset, n_jobs=-1,
        )
    # EDIT: XGB base-learner hyperparameters (scale_pos_weight dynamic = neg/pos)
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE + seed_offset, n_jobs=-1, verbosity=0,
    )


def train_nose(df, feats, target, kind):
    pos = df[df[target] == 1]["stay_id"].unique()
    neg = np.setdiff1d(df["stay_id"].unique(), pos)
    rng = np.random.RandomState(RANDOM_STATE)
    neg = rng.permutation(neg)
    chunk = max(len(pos), 1)
    models = []
    for i in range(NUM_NOSE_SUBSETS):
        s = i * chunk
        e = min(s + chunk, len(neg))
        if e <= s:
            break
        stays = np.concatenate([pos, neg[s:e]])
        sub = df[df["stay_id"].isin(stays)]
        X = sub[feats].values
        y = sub[target].values
        spw = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
        m = build_base(kind, i, spw)
        m.fit(X, y)
        models.append(m)
    return models


def predict_nose(models, df, feats):
    X = df[feats].values
    p = np.zeros(len(df))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / max(len(models), 1)


def f1_max_threshold(y, p):
    # EDIT: F1-max sweep grid (0.005 to 0.95, 950 points)
    ts = np.linspace(0.005, 0.95, 950)
    best_t, best = 0.5, -1
    for t in ts:
        s = fbeta_score(y, (p >= t).astype(int), beta=1, zero_division=0)
        if s > best:
            best, best_t = s, t
    return best_t


def metrics_at(y, p, t):
    yp = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "threshold": float(t),
        "sensitivity": float(sens),
        "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def run_one(kind, df_train, df_test, feats, streams):
    print(f"\n=== {kind.upper()} base learner ===")
    t0 = time.time()
    df_train_meta = df_train[["stay_id", "is_sepsis_6h", "is_sepsis_stay"]].copy()
    for s in streams:
        df_train_meta[f"meta_{s}"] = 0.0

    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df_train, df_train["is_sepsis_stay"], groups=df_train["stay_id"])):
        print(f"  fold {fold + 1}/{NUM_FOLDS}")
        X_tr = df_train.iloc[tr_idx]
        X_va = df_train.iloc[va_idx]
        for stream in streams:
            models = train_nose(X_tr, feats, stream, kind)
            df_train_meta.loc[df_train_meta.index[va_idx], f"meta_{stream}"] = predict_nose(models, X_va, feats)

    df_train_meta["var_6h"] = df_train_meta[
        ["meta_is_sepsis_6h", "meta_target_resp_6h", "meta_target_uri_6h", "meta_target_other_6h"]
    ].var(axis=1)
    meta_features = [f"meta_{s}" for s in streams] + ["var_6h"]
    # EDIT: LR meta-learner hyperparameters
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(df_train_meta[meta_features], df_train_meta["is_sepsis_6h"])

    final = {s: train_nose(df_train, feats, s, kind) for s in streams}
    df_test_meta = df_test[["stay_id", "is_sepsis_6h", "is_sepsis_stay"]].copy()
    for s in streams:
        df_test_meta[f"meta_{s}"] = predict_nose(final[s], df_test, feats)
    df_test_meta["var_6h"] = df_test_meta[
        ["meta_is_sepsis_6h", "meta_target_resp_6h", "meta_target_uri_6h", "meta_target_other_6h"]
    ].var(axis=1)
    p_test = lr.predict_proba(df_test_meta[meta_features])[:, 1]
    y = df_test_meta["is_sepsis_6h"]
    t_star = f1_max_threshold(y, p_test)
    m = metrics_at(y, p_test, t_star)
    m["wall_time_sec"] = round(time.time() - t0, 1)
    m["base_learner"] = kind.upper()
    return m


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading 1-hour engineered data + etiologies...")
    df_train_full, df_test_full = load()
    print(f"  full train: {len(df_train_full):,} rows, {df_train_full['stay_id'].nunique():,} stays")
    print(f"  full test:  {len(df_test_full):,} rows, {df_test_full['stay_id'].nunique():,} stays")

    print(f"\nSubsampling {SAMPLE_FRAC * 100:.0f}% of stays (seed={RANDOM_STATE})")
    df_train = stay_subsample(df_train_full, SAMPLE_FRAC, RANDOM_STATE)
    df_test = stay_subsample(df_test_full, SAMPLE_FRAC, RANDOM_STATE + 1)
    print(f"  train sub:  {len(df_train):,} rows, {df_train['stay_id'].nunique():,} stays")
    print(f"  test sub:   {len(df_test):,} rows, {df_test['stay_id'].nunique():,} stays")

    feats = features(df_train)
    imputer = SimpleImputer(strategy="median")
    df_train[feats] = imputer.fit_transform(df_train[feats])
    df_test[feats] = imputer.transform(df_test[feats])

    streams = [
        "is_sepsis_6h", "is_sepsis_12h",
        "target_resp_6h", "target_resp_12h",
        "target_uri_6h", "target_uri_12h",
        "target_other_6h", "target_other_12h",
    ]

    rows = []
    for kind in ["rf", "xgb"]:
        rows.append(run_one(kind, df_train, df_test, feats, streams))

    df = pd.DataFrame(rows)
    print("\n========== HEAD-TO-HEAD (5% subsample, F1-max threshold) ==========")
    print(df.to_string(index=False))

    out_csv = RESULTS_DIR / "xgb_vs_rf_quicktest.csv"
    df.to_csv(out_csv, index=False)
    out_txt = RESULTS_DIR / "xgb_vs_rf_quicktest.txt"
    with open(out_txt, "w") as f:
        f.write("V13b base-learner quicktest (5% stay-level subsample, F1-max threshold)\n")
        f.write("=" * 75 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n")
        delta = df.set_index("base_learner")["AUROC"]
        if len(delta) == 2:
            f.write(f"\nAUROC delta (XGB - RF) = {delta['XGB'] - delta['RF']:+.4f}\n")
            f.write("Recommendation: switch to XGB if delta > +0.01, else stay with RF.\n")
    print(f"\n✓ Saved {out_csv} and {out_txt}")


if __name__ == "__main__":
    main()
