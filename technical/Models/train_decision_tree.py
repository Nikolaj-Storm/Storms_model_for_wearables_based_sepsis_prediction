# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  train_decision_tree.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Trains a single Decision Tree on the V12 1-hour engineered training set
#    for the 6-hour sepsis target and serialises the fitted model to disk.
#    No evaluation; this is a fit-and-save utility.
#
#  INPUTS
#    ../Data/v12_dataset_1h_train.parquet
#  OUTPUTS
#    saved_models/decision_tree.joblib
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    DATA_DIR / TRAIN_FILE  -  relative input path; assumes you run from the
#                          original technical/Models/ directory
#    TARGET             -  label column, is_sepsis_6h
#    DROP_COLS          -  metadata/label columns excluded from features
#    Decision Tree      -  max_depth=10, class_weight='balanced', random_state=42
#    Output model path  -  saved_models/decision_tree.joblib
#
#  REQUIRES: scikit-learn, joblib, pandas
# ============================================================================

import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
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

print("Training Decision Tree...")
model = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')  # EDIT: Decision Tree hyperparameters

model.fit(X, y)

print("Saving model...")
os.makedirs('saved_models', exist_ok=True)
joblib.dump(model, 'saved_models/decision_tree.joblib')  # EDIT: output model path

print("Training complete.")
