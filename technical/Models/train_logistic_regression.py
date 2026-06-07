# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  train_logistic_regression.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Trains a Logistic Regression on the V12 1-hour engineered training set for
#    the 6-hour sepsis target, with StandardScaler applied first, and serialises
#    both the fitted model and the scaler to disk. No evaluation.
#
#  INPUTS
#    ../Data/v12_dataset_1h_train.parquet
#  OUTPUTS
#    saved_models/logistic_regression.joblib
#    saved_models/lr_scaler.joblib
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    DATA_DIR / TRAIN_FILE  -  relative input path; assumes you run from the
#                          original technical/Models/ directory
#    TARGET             -  label column, is_sepsis_6h
#    DROP_COLS          -  metadata/label columns excluded from features
#    Logistic Regression-  max_iter=1000, class_weight='balanced', random_state=42
#    Output model paths -  saved_models/logistic_regression.joblib, lr_scaler.joblib
#
#  REQUIRES: scikit-learn, joblib, pandas
# ============================================================================

import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib

# Paths relative to Models folder
DATA_DIR = '../Data'  # EDIT: input parquet folder (assumes original technical/Models/ cwd)
TRAIN_FILE = os.path.join(DATA_DIR, 'v12_dataset_1h_train.parquet')  # EDIT: training parquet filename
TARGET = 'is_sepsis_6h'  # EDIT: label column
DROP_COLS = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h', 'stay_id', 'charttime', 'intime', 'sepsis3_time']  # EDIT: columns excluded from features

print("Loading data...")
df = pd.read_parquet(TRAIN_FILE)
y = df[TARGET]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).fillna(0)

print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1)  # EDIT: Logistic Regression hyperparameters
model.fit(X_scaled, y)

print("Saving model and scaler...")
os.makedirs('saved_models', exist_ok=True)
joblib.dump(model, 'saved_models/logistic_regression.joblib')  # EDIT: output model path
joblib.dump(scaler, 'saved_models/lr_scaler.joblib')  # EDIT: output scaler path

print("Training complete.")
