# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  68b_v13b_base_learner_only.py
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Lean base-learner swap test (RF vs XGB inside NOSE). Designed to finish in
#    a single short execution slot. Trains one NOSE ensemble per base learner on
#    a stay-level subsample of the 1-hour engineered dataset for target
#    is_sepsis_6h, skipping the full OOF stack and the etiology streams, to
#    compare how RF and XGB discriminate under the NOSE 1:1 balancing scheme.
#
#  INPUTS
#    ../../Data/All engineered features/Dataset_all_engineered_1h_train.parquet
#    ../../Data/All engineered features/Dataset_all_engineered_1h_test.parquet
#  OUTPUTS
#    ../../Results/V13b_final/xgb_vs_rf_lean.csv
#    ../../Results/V13b_final/xgb_vs_rf_lean.txt
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    Assumed working directory  -  run from technical/Models/V13b so ../../Data
#      and ../../Results resolve.
#    DATA_DIR / RESULTS_DIR     -  engineered-feature folder and output root.
#    TRAIN_FILE / TEST_FILE     -  1-hour engineered train/test parquet paths.
#    DROP_COLS                  -  columns excluded from the feature set.
#    SAMPLE_FRAC=0.05           -  stay-level subsample fraction.
#    NUM_NOSE_SUBSETS=3         -  NOSE negative-undersampling subsets.
#    RANDOM_STATE=42            -  random seed.
#    RF_N_TREES=100 / XGB_N_TREES=100  -  base-learner estimator counts.
#    RF base learner            -  max_depth=12, min_samples_leaf=5,
#                                  max_features='sqrt', class_weight='balanced'.
#    XGB base learner           -  max_depth=6, learning_rate=0.05,
#                                  subsample=0.8, colsample_bytree=0.8,
#                                  min_child_weight=5, scale_pos_weight=neg/pos,
#                                  tree_method='hist'.
#    F1-max sweep grid          -  np.linspace(0.05, 0.95, 91).
#
#  REQUIRES: pandas, numpy, scikit-learn, xgboost
# ============================================================================
"""
V13b base-learner swap: lean test (RF vs XGB inside NOSE).

Designed to fit in a single short execution slot (< 45s) so it can run
inside a sandboxed shell. Trains ONE NOSE ensemble per base learner on a
stay-level subsample of the 1-hour engineered dataset, target = is_sepsis_6h.
Skips the full OOF stack and the etiology streams, because we only need to
compare how well RF vs XGB discriminate under the NOSE 1:1 balancing scheme.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, fbeta_score, roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# EDIT: DATA_DIR / RESULTS_DIR / TRAIN_FILE / TEST_FILE (relative to working dir)
DATA_DIR = Path("../../Data/All engineered features")
RESULTS_DIR = Path("../../Results/V13b_final")
TRAIN_FILE = DATA_DIR / "Dataset_all_engineered_1h_train.parquet"
TEST_FILE = DATA_DIR / "Dataset_all_engineered_1h_test.parquet"

# EDIT: DROP_COLS - columns excluded from the feature set
DROP_COLS = [
    "is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
    "stay_id", "charttime", "intime", "sepsis3_time",
    "time_since_ICU_admit_hours",
]

# Lean settings: 5% stays, 3 NOSE subsets, 100 trees
# EDIT: subsample / NOSE / seed / estimator-count settings
SAMPLE_FRAC = 0.05
NUM_NOSE_SUBSETS = 3
RANDOM_STATE = 42
RF_N_TREES = 100
XGB_N_TREES = 100


def stay_subsample(df, frac, seed):
    stays = df["stay_id"].unique()
    rng = np.random.RandomState(seed)
    keep = rng.choice(stays, size=max(1, int(len(stays) * frac)), replace=False)
    return df[df["stay_id"].isin(keep)].copy()


def features(df):
    excl = set(DROP_COLS)
    return [c for c in df.columns if c not in excl and df[c].dtype != "O"]


def build_base(kind, seed_offset, spw):
    if kind == "rf":
        # EDIT: RF base-learner hyperparameters
        return RandomForestClassifier(
            n_estimators=RF_N_TREES, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=RANDOM_STATE + seed_offset, n_jobs=-1,
        )
    # EDIT: XGB base-learner hyperparameters (scale_pos_weight dynamic = neg/pos)
    return XGBClassifier(
        n_estimators=XGB_N_TREES, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE + seed_offset, n_jobs=-1, verbosity=0,
        tree_method="hist",
    )


def train_nose(df, feats, target, kind):
    pos = df[df[target] == 1]["stay_id"].unique()
    neg = np.setdiff1d(df["stay_id"].unique(), pos)
    rng = np.random.RandomState(RANDOM_STATE)
    neg = rng.permutation(neg)
    chunk = max(len(pos), 1)
    models = []
    for i in range(NUM_NOSE_SUBSETS):
        s, e = i * chunk, min((i + 1) * chunk, len(neg))
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


def predict_nose(models, X):
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / max(len(models), 1)


def f1_max(y, p):
    # EDIT: F1-max sweep grid (0.05 to 0.95, 91 points)
    ts = np.linspace(0.05, 0.95, 91)
    best_t, best = 0.5, -1
    for t in ts:
        s = fbeta_score(y, (p >= t).astype(int), beta=1, zero_division=0)
        if s > best:
            best, best_t = s, t
    return best_t, best


def metrics_at(y, p, t):
    yp = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "threshold": float(t),
        "F1": float(fbeta_score(y, yp, beta=1, zero_division=0)),
        "sensitivity": float(tp) / max(tp + fn, 1),
        "fpr": float(fp) / max(fp + tn, 1),
        "precision": float(tp) / max(tp + fp, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def run_kind(kind, df_tr, df_te, feats, target):
    t0 = time.time()
    models = train_nose(df_tr, feats, target, kind)
    Xte = df_te[feats].values
    Xtr = df_tr[feats].values
    p_te = predict_nose(models, Xte)
    p_tr = predict_nose(models, Xtr)
    y_te = df_te[target].astype(int).values
    y_tr = df_tr[target].astype(int).values
    t_star, _ = f1_max(y_te, p_te)
    m = metrics_at(y_te, p_te, t_star)
    m["base_learner"] = kind.upper()
    m["wall_time_sec"] = round(time.time() - t0, 1)
    m["train_AUROC"] = float(roc_auc_score(y_tr, p_tr))
    return m


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[t={time.strftime('%H:%M:%S')}] Loading 1h dataset...", flush=True)
    t0 = time.time()
    df_train_full = pd.read_parquet(TRAIN_FILE)
    df_test_full = pd.read_parquet(TEST_FILE)
    print(f"  full train: {len(df_train_full):,} rows, "
          f"{df_train_full['stay_id'].nunique():,} stays  "
          f"({time.time() - t0:.1f}s)", flush=True)

    df_train = stay_subsample(df_train_full, SAMPLE_FRAC, RANDOM_STATE)
    df_test = stay_subsample(df_test_full, SAMPLE_FRAC, RANDOM_STATE + 1)
    pos_train = int((df_train.groupby("stay_id")["is_sepsis_6h"].max() == 1).sum())
    pos_test = int((df_test.groupby("stay_id")["is_sepsis_6h"].max() == 1).sum())
    print(f"  train sub:  {len(df_train):,} rows, {df_train['stay_id'].nunique():,} stays  "
          f"({pos_train} positive stays)", flush=True)
    print(f"  test  sub:  {len(df_test):,} rows, {df_test['stay_id'].nunique():,} stays  "
          f"({pos_test} positive stays)", flush=True)

    feats = features(df_train)
    print(f"  features: {len(feats)}", flush=True)
    imp = SimpleImputer(strategy="median")
    df_train[feats] = imp.fit_transform(df_train[feats])
    df_test[feats] = imp.transform(df_test[feats])

    rows = []
    for kind in ["rf", "xgb"]:
        print(f"\n[t={time.strftime('%H:%M:%S')}] Training {kind.upper()} NOSE...", flush=True)
        m = run_kind(kind, df_train, df_test, feats, "is_sepsis_6h")
        print(f"  {kind.upper()}: AUROC={m['AUROC']:.4f}  AUPRC={m['AUPRC']:.4f}  "
              f"F1={m['F1']:.4f}  Recall={m['sensitivity']:.4f}  FPR={m['fpr']:.4f}  "
              f"t*={m['threshold']:.3f}  ({m['wall_time_sec']}s)", flush=True)
        rows.append(m)

    df = pd.DataFrame(rows)
    print("\n========== HEAD-TO-HEAD ==========")
    print(df[[
        "base_learner", "AUROC", "AUPRC", "F1",
        "sensitivity", "fpr", "precision", "threshold", "wall_time_sec",
    ]].to_string(index=False))

    auroc = df.set_index("base_learner")["AUROC"]
    if {"RF", "XGB"}.issubset(auroc.index):
        delta = auroc["XGB"] - auroc["RF"]
        verdict = ("XGB substantially better — consider full XGB run"
                   if delta > 0.01
                   else "RF and XGB roughly tied — keep RF for narrative continuity"
                   if abs(delta) <= 0.01
                   else "RF better — keep RF")
        print(f"\nAUROC delta (XGB - RF) = {delta:+.4f}  ->  {verdict}")

    out_csv = RESULTS_DIR / "xgb_vs_rf_lean.csv"
    df.to_csv(out_csv, index=False)
    out_txt = RESULTS_DIR / "xgb_vs_rf_lean.txt"
    with open(out_txt, "w") as f:
        f.write("V13b lean base-learner test (5% stays, 3 NOSE subsets, target=is_sepsis_6h)\n")
        f.write("=" * 75 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n")
        if {"RF", "XGB"}.issubset(auroc.index):
            f.write(f"\nAUROC delta (XGB - RF) = {delta:+.4f}\n")
            f.write(f"Verdict: {verdict}\n")
    print(f"\nSaved {out_csv} and {out_txt}")


if __name__ == "__main__":
    main()
