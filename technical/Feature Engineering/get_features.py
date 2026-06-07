# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  get_features.py
#  Stage: 3 - Feature Engineering (feature-name enumerator)
#
#  PURPOSE
#    Builds an empty DataFrame with the full V12 + V13 column schema, applies
#    the feature stubs, then prints the sorted list of model-facing feature
#    names (everything except identifiers and target/label columns). Used to
#    confirm the exact feature count fed to the model.
#
#  INPUTS
#    none
#  OUTPUTS
#    none / prints the feature list to console
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    exclude_cols  -  identifier and target/label column names to omit from
#                     the printed feature list
#
#  REQUIRES: pandas, numpy
# ============================================================================
import pandas as pd
import numpy as np

def apply_v12_features(df):
    vitals = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']
    df['shock_index'] = 0
    df['map'] = 0
    df['partial_qsofa'] = 0
    df['news_score'] = 0
    df['qsofa_delta'] = 0
    df['news_delta'] = 0

    for v in vitals:
        df[f'exp_min_{v}'] = 0
        df[f'exp_max_{v}'] = 0
        df[f'exp_mean_{v}'] = 0
        df[f'exp_std_{v}'] = 0
        df[f'{v}_sd_4h'] = 0
        df[f'{v}_ewma_3h'] = 0
        df[f'slope_4h_{v}'] = 0
        df[f'lag_diff_1h_{v}'] = 0
        df[f'lag_ratio_1h_{v}'] = 0

    df['ventilation_perfusion_proxy'] = 0
    df['tachycardia_excess'] = 0
    df['perfusion_adequacy'] = 0
    df['cardiorespiratory_coupling_4h'] = 0
    return df

def apply_v13_features(df):
    vitals = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']
    for v in vitals:
        df[f'accel_{v}'] = 0
        df[f'cusum_pos_{v}'] = 0
        df[f'cusum_neg_{v}'] = 0
    df['pulse_pressure'] = 0
    df['rpp'] = 0
    df['msi'] = 0
    df['resp_distress'] = 0
    df['fever_tachycardia'] = 0
    return df

df = pd.DataFrame(columns=['stay_id', 'charttime', 'heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp', 'weight_kg', 'age', 'time_since', 'is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h', 'intime', 'sepsis3_time'])
df = apply_v12_features(df)
df = apply_v13_features(df)

# EDIT: identifier and target/label columns to exclude from the feature list
exclude_cols = ['is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h',
                'stay_id', 'charttime', 'intime', 'sepsis3_time',
                'target_resp', 'target_uri', 'target_other']
all_features = sorted([c for c in df.columns if c not in exclude_cols])
print(f"Features: {len(all_features)}")
for f in all_features:
    print(f)
