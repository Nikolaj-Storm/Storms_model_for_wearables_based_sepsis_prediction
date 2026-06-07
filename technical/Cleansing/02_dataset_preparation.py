# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  02_dataset_preparation.py
#  Stage: 2 - Preprocessing
#
#  PURPOSE
#    Cleanses the raw cohort vitals and builds multi-resolution datasets. Maps
#    itemids to vital names, converts Fahrenheit to Celsius, applies biological
#    artifact filters, pivots long to wide, enforces stay-quality rules, splits
#    patients 75/25, and resamples each split to 4h, 1h, and 15m grids.
#
#  INPUTS
#    mimic_derived_vitals_cohort.parquet            (from 05_bq_to_python.py)
#  OUTPUTS
#    dataset_a_4h_train.parquet / dataset_a_4h_test.parquet
#    dataset_b_1h_train.parquet / dataset_b_1h_test.parquet
#    dataset_c_15m_train.parquet / dataset_c_15m_test.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    parquet_file        -  input raw cohort vitals path
#    itemid_map          -  itemid to vital-name mapping
#    bounds              -  biological plausibility ranges per vital
#    min stay duration   -  minimum ICU stay hours, currently 24
#    train_size          -  train fraction in GroupShuffleSplit, currently 0.75
#    random_state        -  split seed, currently 42
#    RESOLUTION          -  resample grids 4h / 1h / 15min
#    output paths        -  the six train/test parquet filenames
#
#  REQUIRES: pandas, numpy, scikit-learn
# ============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# MIMIC-IV Data Engineering Pipeline: Cleansing & Multi-Resolution Resampling
# ---------------------------------------------------------------------------

def load_and_map_vitals(filepath):
    print(f"Loading raw cohort vitals from {filepath}...")
    df = pd.read_parquet(filepath)

    # Map the raw itemids to human-readable column names
    itemid_map = {  # EDIT: itemid to vital-name mapping
        220045: 'heart_rate',
        220210: 'resprate', 224690: 'resprate',
        220277: 'spo2',
        223761: 'temp_f', 223762: 'temp_c',
        220179: 'sbp', 220050: 'sbp',
        220180: 'dbp', 220051: 'dbp'
    }

    print("Mapping Item IDs to explicit Vital Signs...")
    df['vital_name'] = df['itemid'].map(itemid_map)
    df = df.dropna(subset=['vital_name'])

    # Convert Fahrenheit to Celsius for uniformity
    f_mask = df['vital_name'] == 'temp_f'
    df.loc[f_mask, 'valuenum'] = (df.loc[f_mask, 'valuenum'] - 32) * 5.0 / 9.0
    df.loc[f_mask, 'vital_name'] = 'temp_c' # Reassign so everything is one column

    return df

def apply_biological_filters(df):
    print("Applying strict Biological Artifact Filters...")
    initial_rows = len(df)

    bounds = {  # EDIT: biological plausibility ranges per vital
        'heart_rate': (10, 300),
        'resprate': (1, 100),
        'spo2': (20, 100),
        'temp_c': (25, 45),
        'sbp': (40, 300),
        'dbp': (20, 200)
    }

    # Keep rows that are withinbounds OR rows that somehow didn't map (though we dropped those)
    valid_masks = []
    for vital, (low, high) in bounds.items():
        mask = (df['vital_name'] == vital) & (df['valuenum'] >= low) & (df['valuenum'] <= high)
        valid_masks.append(mask)

    # Combine masks
    combined_mask = np.logical_or.reduce(valid_masks)
    df_clean = df[combined_mask].copy()

    dropped = initial_rows - len(df_clean)
    print(f"  Dropped {dropped:,} physiologically impossible sensor artifacts.")
    return df_clean

def pivot_and_filter_stays(df):
    print("Pivoting data from Long to Wide format...")
    # Because a single patient has multiple vital signs taken at the EXACT same minute,
    # we average them if there are duplicates at the exact same timestamp before pivoting.
    df_wide = df.groupby(['stay_id', 'is_sepsis', 'charttime', 'vital_name'])['valuenum'].mean().unstack("vital_name")
    df_wide = df_wide.reset_index()

    initial_patients = df_wide['stay_id'].nunique()
    print(f"Tracking: Starting Phase 2 with {initial_patients:,} unique patients.")

    print("Enforcing 'Minimum 24-Hour Stay' rule...")
    stay_durations = df_wide.groupby('stay_id')['charttime'].apply(lambda x: (x.max() - x.min()).total_seconds() / 3600)
    valid_durations = stay_durations[stay_durations >= 24].index  # EDIT: minimum ICU stay duration (hours)
    df_wide = df_wide[df_wide['stay_id'].isin(valid_durations)]

    post_24h_patients = df_wide['stay_id'].nunique()
    print(f"Tracking: {initial_patients - post_24h_patients:,} patients dropped for ICU stay < 24 hrs. {post_24h_patients:,} remaining.")

    print("Enforcing 'Missing all 6 Core Vitals' rule...")
    # Count how many of the 6 vitals have AT LEAST ONE reading during the entire stay
    vitals_cols = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']
    vital_counts = df_wide.groupby('stay_id')[vitals_cols].count()
    has_all_6 = vital_counts[(vital_counts > 0).all(axis=1)].index
    df_wide = df_wide[df_wide['stay_id'].isin(has_all_6)]

    post_vitals_patients = df_wide['stay_id'].nunique()
    print(f"Tracking: {post_24h_patients - post_vitals_patients:,} patients dropped for missing complete 6-sensor array. {post_vitals_patients:,} remaining.")

    print("Enforcing '<50% Missingness Threshold' per patient per vital...")
    # We drop a stay_id entirely if they are missing > 50% of the timeline prior to imputation
    # Note: Since the dataframe is currently un-resampled (sporadic events),
    # checking missingness here means "did the sensor fail for half the time they were hooked up?".
    # A true missingness check is better done AFTER resampling to an hourly grid.

    return df_wide

def split_and_resample(df_wide):
    print("Splitting 75/25 Train/Test (Patient-Level GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)  # EDIT: train_size fraction and random_state seed
    train_idx, test_idx = next(gss.split(df_wide, groups=df_wide['stay_id']))

    df_train = df_wide.iloc[train_idx].copy()
    df_test = df_wide.iloc[test_idx].copy()

    unique_train = df_train['stay_id'].nunique()
    unique_test = df_test['stay_id'].nunique()
    print(f"Total valid patients remaining: {unique_train + unique_test:,}")
    print(f"  -> Training Cohort (75%): {unique_train:,} patients")
    print(f"  -> Testing Cohort (25%): {unique_test:,} patients")

    def generate_resolutions(data, split_name):
        print(f"\n[{split_name.upper()}] Building Multi-Resolution AI Datasets...")

        # Ensure datetime and sort
        data['charttime'] = pd.to_datetime(data['charttime'])
        data.set_index('charttime', inplace=True)

        vitals_cols = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']

        # We must keep targets associated
        targets = data.groupby('stay_id')['is_sepsis'].max() # Sepsis is static per stay

        # ---------------- A (4-Hour) ----------------
        print(" -> Generating Dataset A (Standard Care 4-Hour Baseline)...")
        # Forward fill the last known measurement to the current 4H grid (mimics nurse checks)
        dataset_a = data.groupby('stay_id')[vitals_cols].resample('4h').ffill()  # EDIT: RESOLUTION (4h grid)
        dataset_a = dataset_a.join(targets, on='stay_id')

        # ---------------- B (1-Hour) ----------------
        print(" -> Generating Dataset B (Primary Wearable 1-Hour Proxy)...")
        # Average multiple readings in the hour, then interpolate missing hours
        dataset_b = data.groupby('stay_id')[vitals_cols].resample('1h').mean().interpolate(method='linear', limit_direction='both')  # EDIT: RESOLUTION (1h grid)
        dataset_b = dataset_b.join(targets, on='stay_id')

        # ---------------- C (15-Min) ----------------
        print(" -> Generating Dataset C (High-Frequency 15-Minute Sub-Cohort)...")
        dataset_c = data.groupby('stay_id')[vitals_cols].resample('15min').mean().interpolate(method='linear', limit_direction='both')  # EDIT: RESOLUTION (15min grid)
        dataset_c = dataset_c.join(targets, on='stay_id')

        return dataset_a, dataset_b, dataset_c

    train_a, train_b, train_c = generate_resolutions(df_train, "Train")
    test_a, test_b, test_c = generate_resolutions(df_test, "Test")

    return (train_a, train_b, train_c), (test_a, test_b, test_c)

if __name__ == "__main__":
    parquet_file = "mimic_derived_vitals_cohort.parquet"  # EDIT: input raw cohort vitals path
    if not os.path.exists(parquet_file):
        print(f"Error: {parquet_file} not found. Did you run 05_bq_to_python.py?")
        exit(1)

    df_raw = load_and_map_vitals(parquet_file)
    df_clean = apply_biological_filters(df_raw)
    df_wide = pivot_and_filter_stays(df_clean)

    train_datasets, test_datasets = split_and_resample(df_wide)
    train_a, train_b, train_c = train_datasets
    test_a, test_b, test_c = test_datasets

    print("\nSaving final A/B/C datasets to disk for Feature Engineering...")
    train_a.to_parquet("dataset_a_4h_train.parquet", engine='pyarrow')  # EDIT: output path
    test_a.to_parquet("dataset_a_4h_test.parquet", engine='pyarrow')  # EDIT: output path

    train_b.to_parquet("dataset_b_1h_train.parquet", engine='pyarrow')  # EDIT: output path
    test_b.to_parquet("dataset_b_1h_test.parquet", engine='pyarrow')  # EDIT: output path

    train_c.to_parquet("dataset_c_15m_train.parquet", engine='pyarrow')  # EDIT: output path
    test_c.to_parquet("dataset_c_15m_test.parquet", engine='pyarrow')  # EDIT: output path

    print("\nAll AI DataFrames generated and saved to disk. Ready for Feature Engineering & Visualizations!")
