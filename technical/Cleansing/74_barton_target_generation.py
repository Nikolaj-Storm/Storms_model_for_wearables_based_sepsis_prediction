# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  74_barton_target_generation.py
#  Stage: 2 - Preprocessing (labelling + feature engineering)
#
#  PURPOSE
#    Builds the Barton-style SIRS baseline. Pulls suspicion-of-infection times
#    from BigQuery, computes a proxy SIRS score (HR, RR, temp) per hour, flags
#    the first onset that intersects the infection window, derives multi-horizon
#    Barton labels (0/6/12/24/48h), truncates after onset, and engineers the
#    Barton 31-feature set (current, lag-1h, lag-2h, and two diff terms per vital,
#    plus age) for train and test.
#
#  INPUTS
#    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection  (your derived table)
#    v12_dataset_1h_{split}.parquet  (preferred) or v3_dataset_1h_{split}.parquet
#  OUTPUTS
#    74_barton_sirs_dataset_train.parquet
#    74_barton_sirs_dataset_test.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID            -  your GCP project id used to run the query
#    suspicion table       -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection
#    input file pattern    -  v12_dataset_1h_{split} preferred, v3_dataset_1h_{split} fallback
#    SIRS thresholds       -  HR > 90, RR > 20, temp > 38 or < 36
#    SIRS score cutoff     -  onset requires sirs_score >= 2
#    infection window      -  intersection window, currently -24h to +48h
#    prediction HORIZONs   -  Barton label windows 0/6/12/24/48 hours
#    target_vitals         -  the 6 vitals used for Barton lag/diff features
#    output file pattern   -  74_barton_sirs_dataset_{split}.parquet
#
#  REQUIRES: pandas, numpy, google-cloud-bigquery, pyarrow | MIMIC-IV BigQuery access
# ============================================================================

import pandas as pd
import numpy as np
from google.cloud import bigquery
import os

print("--- Barton 2 (SIRS Baseline) Target Generation ---")

# 1. Acquire Suspicion of Infection
print("\n[1/3] Querying Suspicion of Infection from BigQuery...")
client = bigquery.Client(project="YOUR_GCP_PROJECT")  # EDIT: your GCP project id
query = """
SELECT stay_id, MIN(suspected_infection_time) as infection_time
FROM `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection`
WHERE suspected_infection_time IS NOT NULL
GROUP BY stay_id
"""  # EDIT: derived suspicion_of_infection table path
df_inf = client.query(query).to_dataframe()
df_inf['infection_time'] = pd.to_datetime(df_inf['infection_time'])
df_inf.set_index('stay_id', inplace=True)

def process_split(split='train'):
    file_path = f'v12_dataset_1h_{split}.parquet'  # EDIT: preferred input file pattern
    if not os.path.exists(file_path):
        file_path = f'v3_dataset_1h_{split}.parquet'  # EDIT: fallback input file pattern

    print(f"\n--- Processing {split.upper()} Split: {file_path} ---")
    df_raw = pd.read_parquet(file_path).reset_index()

    # 2. Map Infection Times to our Dataset
    df_raw['charttime'] = pd.to_datetime(df_raw['charttime'])
    df_raw = df_raw.join(df_inf, on='stay_id')

    # 3. Calculate proxy SIRS Criteria (3 basic signs)
    df_raw['sirs_hr'] = (df_raw['heart_rate'] > 90).astype(int)  # EDIT: SIRS heart rate threshold
    df_raw['sirs_rr'] = (df_raw['resprate'] > 20).astype(int)  # EDIT: SIRS respiratory rate threshold
    df_raw['sirs_temp'] = ((df_raw['temp_c'] > 38) | (df_raw['temp_c'] < 36)).astype(int)  # EDIT: SIRS temperature thresholds
    df_raw['sirs_score'] = df_raw['sirs_hr'] + df_raw['sirs_rr'] + df_raw['sirs_temp']

    valid_infection = (df_raw['infection_time'].notnull())
    infection_delta = (df_raw['charttime'] - df_raw['infection_time']).dt.total_seconds() / 3600

    is_valid_intersection = valid_infection & (infection_delta >= -24) & (infection_delta <= 48)  # EDIT: infection intersection window (hours)

    df_raw['is_barton_onset'] = ((df_raw['sirs_score'] >= 2) & is_valid_intersection).astype(int)  # EDIT: SIRS score cutoff for onset

    barton_onset_times = df_raw[df_raw['is_barton_onset'] == 1].groupby('stay_id')['charttime'].min().rename("barton_sepsis_time")

    df_raw = df_raw.join(barton_onset_times, on='stay_id')

    # Now, label targets
    is_barton_stay = df_raw['barton_sepsis_time'].notnull()
    df_raw['is_barton_stay'] = is_barton_stay.astype(int)

    df_raw['is_barton_0h'] = 0
    df_raw['is_barton_6h'] = 0
    df_raw['is_barton_12h'] = 0
    df_raw['is_barton_24h'] = 0
    df_raw['is_barton_48h'] = 0

    if is_barton_stay.any():
        hours_until_barton = (df_raw.loc[is_barton_stay, 'barton_sepsis_time'] - df_raw.loc[is_barton_stay, 'charttime']).dt.total_seconds() / 3600
        df_raw.loc[is_barton_stay & (hours_until_barton <= 0), 'is_barton_0h'] = 1   # EDIT: prediction HORIZON (0 hours)
        df_raw.loc[is_barton_stay & (hours_until_barton <= 6), 'is_barton_6h'] = 1   # EDIT: prediction HORIZON (6 hours)
        df_raw.loc[is_barton_stay & (hours_until_barton <= 12), 'is_barton_12h'] = 1  # EDIT: prediction HORIZON (12 hours)
        df_raw.loc[is_barton_stay & (hours_until_barton <= 24), 'is_barton_24h'] = 1  # EDIT: prediction HORIZON (24 hours)
        df_raw.loc[is_barton_stay & (hours_until_barton <= 48), 'is_barton_48h'] = 1  # EDIT: prediction HORIZON (48 hours)

    # Truncate dataset
    valid_rows_barton = ~is_barton_stay | (df_raw['charttime'] <= df_raw['barton_sepsis_time'])
    df_b = df_raw[valid_rows_barton].copy()

    # Feature mapping (Barton 31 features)
    target_vitals = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']  # EDIT: vitals used for Barton lag/diff features

    df_b['stay_id_shifted'] = df_b['stay_id'].shift(1)
    df_b['stay_id_shifted_2'] = df_b['stay_id'].shift(2)

    barton_features = []
    for v in target_vitals:
        bare_name = f"{v}_T"
        df_b[bare_name] = df_b[v]
        barton_features.append(bare_name)

    for v in target_vitals:
        lag_name = f"{v}_T-1h"
        df_b[lag_name] = np.where(df_b['stay_id'] == df_b['stay_id_shifted'], df_b[v].shift(1), np.nan)
        barton_features.append(lag_name)

    for v in target_vitals:
        lag_name = f"{v}_T-2h"
        df_b[lag_name] = np.where(df_b['stay_id'] == df_b['stay_id_shifted_2'], df_b[v].shift(2), np.nan)
        barton_features.append(lag_name)

    for v in target_vitals:
        diff1_name = f"{v}_diff_T_T1"
        df_b[diff1_name] = df_b[f"{v}_T"] - df_b[f"{v}_T-1h"]
        barton_features.append(diff1_name)

    for v in target_vitals:
        diff2_name = f"{v}_diff_T1_T2"
        df_b[diff2_name] = df_b[f"{v}_T-1h"] - df_b[f"{v}_T-2h"]
        barton_features.append(diff2_name)

    df_b['age_numeric'] = df_b['age'].astype(float)
    barton_features.append('age_numeric')

    target_cols = ['is_barton_0h', 'is_barton_6h', 'is_barton_12h', 'is_barton_24h', 'is_barton_48h', 'is_barton_stay', 'is_sepsis_stay']
    index_cols = ['stay_id', 'charttime', 'barton_sepsis_time', 'sepsis3_time', 'time_since_ICU_admit_hours'] if 'sepsis3_time' in df_b.columns else ['stay_id', 'charttime', 'barton_sepsis_time', 'time_since_ICU_admit_hours']

    final_cols = index_cols + barton_features + target_cols
    df_final = df_b[final_cols].copy()

    sepsis_stays_s3 = len(df_final[df_final['is_sepsis_stay'] == 1]['stay_id'].unique())
    sepsis_stays_s2 = len(df_final[df_final['is_barton_stay'] == 1]['stay_id'].unique())
    total_stays = len(df_final['stay_id'].unique())

    print(f"Total Stays: {total_stays}")
    print(f"Sepsis-3: {sepsis_stays_s3} stays ({sepsis_stays_s3/total_stays*100:.1f}%)")
    print(f"Sepsis-2/SIRS: {sepsis_stays_s2} stays ({sepsis_stays_s2/total_stays*100:.1f}%)")

    out_file = f'74_barton_sirs_dataset_{split}.parquet'  # EDIT: output file pattern
    df_final.to_parquet(out_file, engine='pyarrow')
    print(f"Saved: {out_file}")

process_split('train')
process_split('test')
