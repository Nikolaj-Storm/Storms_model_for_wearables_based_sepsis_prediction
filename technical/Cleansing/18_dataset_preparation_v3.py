# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  18_dataset_preparation_v3.py
#  Stage: 2 - Preprocessing
#
#  PURPOSE
#    V3 data engineering. Cleanses raw vitals, merges static demographics, and
#    computes multi-horizon sepsis labels (6h, 12h, full-stay) relative to the
#    Sepsis-3 onset anchor. Truncates rows after onset, splits patients 75/25,
#    and resamples each split to 4h, 1h, and 15m grids with horizon targets.
#
#  INPUTS
#    mimic_derived_vitals_cohort.parquet            (from 05_bq_to_python.py)
#    demographics_anchors.parquet                   (from 17_download_demographics.py)
#  OUTPUTS
#    v3_dataset_4h_train.parquet / v3_dataset_4h_test.parquet
#    v3_dataset_1h_train.parquet / v3_dataset_1h_test.parquet
#    v3_dataset_15m_train.parquet / v3_dataset_15m_test.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    input vitals path   -  mimic_derived_vitals_cohort.parquet
#    demographics path   -  demographics_anchors.parquet
#    itemid_map          -  itemid to vital-name mapping
#    bounds              -  biological plausibility ranges per vital
#    min stay duration   -  minimum valid stay hours, currently 7
#    prediction HORIZONs -  label windows is_sepsis_6h / is_sepsis_12h (6 and 12 hours)
#    train_size          -  train fraction in GroupShuffleSplit, currently 0.75
#    random_state        -  split seed, currently 42
#    RESOLUTION          -  resample grids 4h / 1h / 15min
#    output paths        -  the six v3 train/test parquet filenames
#
#  REQUIRES: pandas, numpy, scikit-learn
# ============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# V3 Data Engineering: Multi-Resolution Horizon Labelling & Demographic Merging
# ---------------------------------------------------------------------------

def load_vitals():
    print(f"Loading continuous Vitals array...")
    df = pd.read_parquet("mimic_derived_vitals_cohort.parquet")  # EDIT: input vitals path

    itemid_map = {  # EDIT: itemid to vital-name mapping
        220045: 'heart_rate',
        220210: 'resprate', 224690: 'resprate',
        220277: 'spo2',
        223761: 'temp_f', 223762: 'temp_c',
        220179: 'sbp', 220050: 'sbp',
        220180: 'dbp', 220051: 'dbp'
    }
    df['vital_name'] = df['itemid'].map(itemid_map)
    df = df.dropna(subset=['vital_name'])

    f_mask = df['vital_name'] == 'temp_f'
    df.loc[f_mask, 'valuenum'] = (df.loc[f_mask, 'valuenum'] - 32) * 5.0 / 9.0
    df.loc[f_mask, 'vital_name'] = 'temp_c'
    return df

def apply_biological_filters(df):
    print("Applying Biological Artifact Filters...")
    bounds = {  # EDIT: biological plausibility ranges per vital
        'heart_rate': (10, 300),
        'resprate': (1, 100),
        'spo2': (20, 100),
        'temp_c': (25, 45),
        'sbp': (40, 300),
        'dbp': (20, 200)
    }

    valid_masks = []
    for vital, (low, high) in bounds.items():
        mask = (df['vital_name'] == vital) & (df['valuenum'] >= low) & (df['valuenum'] <= high)
        valid_masks.append(mask)

    combined_mask = np.logical_or.reduce(valid_masks)
    df_clean = df[combined_mask].copy()
    return df_clean

def pivot_and_filter_stays(df):
    print("Pivoting data to Wide format...")
    df_wide = df.groupby(['stay_id', 'charttime', 'vital_name'])['valuenum'].mean().unstack("vital_name")
    df_wide = df_wide.reset_index()
    return df_wide

def merge_v3_demographics_and_horizons(df_wide):
    print("\n[V3 ARCHITECTURE] Merging Static Demographics & Computing Predictive Horizons...")
    df_demo = pd.read_parquet("demographics_anchors.parquet")  # EDIT: demographics path

    median_weight = df_demo['weight_kg'].median()
    df_demo['weight_kg'] = df_demo['weight_kg'].fillna(median_weight)

    df_merged = pd.merge(df_wide, df_demo, on='stay_id', how='left')

    df_merged['charttime'] = pd.to_datetime(df_merged['charttime'])
    df_merged['intime'] = pd.to_datetime(df_merged['intime'])
    df_merged['time_since_ICU_admit_hours'] = (df_merged['charttime'] - df_merged['intime']).dt.total_seconds() / 3600

    valid_stays = df_merged.groupby('stay_id')['time_since_ICU_admit_hours'].max()
    valid_stays = valid_stays[valid_stays >= 7].index  # EDIT: minimum valid stay duration (hours)
    df_merged = df_merged[df_merged['stay_id'].isin(valid_stays)]

    is_sepsis_stay = df_merged['sepsis3_time'].notnull()

    df_merged['sepsis3_time'] = pd.to_datetime(df_merged['sepsis3_time'])
    valid_rows = ~is_sepsis_stay | (df_merged['charttime'] <= df_merged['sepsis3_time'])
    df_merged = df_merged[valid_rows].copy()

    df_merged['is_sepsis_6h'] = 0
    df_merged['is_sepsis_12h'] = 0
    df_merged['is_sepsis_stay'] = 0

    # Recalculate mask for the truncated dataframe
    is_sepsis_stay = df_merged['sepsis3_time'].notnull()

    if is_sepsis_stay.any():
        df_merged.loc[is_sepsis_stay, 'is_sepsis_stay'] = 1
        hours_until_sepsis = (df_merged.loc[is_sepsis_stay, 'sepsis3_time'] - df_merged.loc[is_sepsis_stay, 'charttime']).dt.total_seconds() / 3600
        df_merged.loc[is_sepsis_stay & (hours_until_sepsis <= 6), 'is_sepsis_6h'] = 1   # EDIT: prediction HORIZON (6 hours)
        df_merged.loc[is_sepsis_stay & (hours_until_sepsis <= 12), 'is_sepsis_12h'] = 1  # EDIT: prediction HORIZON (12 hours)

    print("V3 Targets successfully mapped! (is_sepsis_6h, is_sepsis_12h)")
    return df_merged

def split_and_resample(df_merged):
    print("\nSplitting 75/25 Train/Test (Patient-Level GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)  # EDIT: train_size fraction and random_state seed
    train_idx, test_idx = next(gss.split(df_merged, groups=df_merged['stay_id']))

    df_train = df_merged.iloc[train_idx].copy()
    df_test = df_merged.iloc[test_idx].copy()

    def generate_resolutions(data, split_name):
        print(f"\n[{split_name.upper()}] Building V3 Multi-Resolution Arrays...")

        demo_cols = ['age', 'weight_kg', 'is_sepsis_stay']
        static_data = data.groupby('stay_id')[demo_cols].first()

        data.set_index('charttime', inplace=True)
        vitals_cols = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp', 'time_since_ICU_admit_hours']
        targets_col = ['is_sepsis_6h', 'is_sepsis_12h']

        # ---------------- A (4-Hour) ----------------
        print(" -> Generating V3 Dataset A (Standard Care 4-Hour Baseline)...")
        dataset_a = data.groupby('stay_id')[vitals_cols].resample('4h').ffill()  # EDIT: RESOLUTION (4h grid)
        targets_a = data.groupby('stay_id')[targets_col].resample('4h').max()  # EDIT: RESOLUTION (4h grid)
        dataset_a = pd.merge(dataset_a, targets_a, left_index=True, right_index=True)
        dataset_a = dataset_a.join(static_data, on='stay_id')

        # ---------------- B (1-Hour) ----------------
        print(" -> Generating V3 Dataset B (Primary Wearable 1-Hour Proxy)...")
        dataset_b = data.groupby('stay_id')[vitals_cols].resample('1h').mean().interpolate(method='linear', limit_direction='both')  # EDIT: RESOLUTION (1h grid)
        targets_b = data.groupby('stay_id')[targets_col].resample('1h').max()  # EDIT: RESOLUTION (1h grid)
        dataset_b = pd.merge(dataset_b, targets_b, left_index=True, right_index=True)
        dataset_b = dataset_b.join(static_data, on='stay_id')

        # ---------------- C (15-Min) ----------------
        print(" -> Generating V3 Dataset C (High-Frequency 15-Minute Sub-Cohort)...")
        dataset_c = data.groupby('stay_id')[vitals_cols].resample('15min').mean().interpolate(method='linear', limit_direction='both')  # EDIT: RESOLUTION (15min grid)
        targets_c = data.groupby('stay_id')[targets_col].resample('15min').max()  # EDIT: RESOLUTION (15min grid)
        dataset_c = pd.merge(dataset_c, targets_c, left_index=True, right_index=True)
        dataset_c = dataset_c.join(static_data, on='stay_id')

        return dataset_a, dataset_b, dataset_c

    train_a, train_b, train_c = generate_resolutions(df_train, "Train")
    test_a, test_b, test_c = generate_resolutions(df_test, "Test")
    print(f"Dataset B Shape: {train_b.shape}")

    return (train_a, train_b, train_c), (test_a, test_b, test_c)

if __name__ == "__main__":
    df_raw = load_vitals()
    df_clean = apply_biological_filters(df_raw)
    df_wide = pivot_and_filter_stays(df_clean)

    df_v3 = merge_v3_demographics_and_horizons(df_wide)

    train_datasets, test_datasets = split_and_resample(df_v3)
    train_a, train_b, train_c = train_datasets
    test_a, test_b, test_c = test_datasets

    print("\nSaving final V3 Multi-Resolution datasets to disk...")
    train_a.to_parquet("v3_dataset_4h_train.parquet", engine='pyarrow')  # EDIT: output path
    test_a.to_parquet("v3_dataset_4h_test.parquet", engine='pyarrow')  # EDIT: output path

    train_b.to_parquet("v3_dataset_1h_train.parquet", engine='pyarrow')  # EDIT: output path
    test_b.to_parquet("v3_dataset_1h_test.parquet", engine='pyarrow')  # EDIT: output path

    train_c.to_parquet("v3_dataset_15m_train.parquet", engine='pyarrow')  # EDIT: output path
    test_c.to_parquet("v3_dataset_15m_test.parquet", engine='pyarrow')  # EDIT: output path

    print("\nV3 Multi-Resolution Architecture Successfully Engineered!")
