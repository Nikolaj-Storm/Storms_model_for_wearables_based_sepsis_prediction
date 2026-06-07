# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  pull_wbc_data.py
#  Stage: Exploratory
#
#  PURPOSE
#    Pulls white blood cell (WBC) lab values from MIMIC-IV via BigQuery for the
#    stays in the 4-hour test parquet, aligns the most recent prior lab to each
#    grid point, derives a SIRS WBC abnormality flag, and writes an augmented
#    "_full" parquet. Ends with a freshness / distribution sanity check.
#
#  INPUTS
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_with_GCS_O2.parquet
#    BigQuery tables physionet-data.mimiciv_3_1_hosp.labevents and
#      physionet-data.mimiciv_3_1_icu.icustays
#  OUTPUTS
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_full.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID                 -  GCP / BigQuery project used for billing.
#    input_path                 -  the 4h test parquet that is read.
#    output_path                -  the "_full" parquet that is written.
#    labevents itemids          -  WBC item ids (51301, 51300).
#    freshness window           -  4h cutoff for fresh vs forward-filled WBC.
#    SIRS_wbc thresholds        -  WBC >= 12 or <= 4 marks abnormal.
#
#  REQUIRES: pandas, numpy, google-cloud-bigquery, pyarrow
# ============================================================================
import pandas as pd
import numpy as np
from google.cloud import bigquery
import time
import os

# EDIT: PROJECT_ID - GCP / BigQuery project used for billing
PROJECT_ID = "YOUR_GCP_PROJECT"
client = bigquery.Client(project=PROJECT_ID)

# 1. Load data
print("Loading parquet...")
# EDIT: input_path - the 4h test parquet that is read
input_path = '/PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_with_GCS_O2.parquet'
df_4h = pd.read_parquet(input_path)

# Check lengths
print(f"4h length: {len(df_4h)}")

grid_4h = df_4h.reset_index()

stay_ids = grid_4h['stay_id'].unique()
print(f"Unique stay_ids: {len(stay_ids)}")

stay_ids_str = ",".join([str(x) for x in stay_ids])

# 2. Query Labevents and ICU stays joined
print("Querying Labevents and aligning to stays...")
# EDIT: labevents itemids - WBC item ids (51301, 51300)
query = f"""
SELECT i.stay_id, l.charttime, l.valuenum as WBC
FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
JOIN `physionet-data.mimiciv_3_1_icu.icustays` i
  ON l.subject_id = i.subject_id
WHERE i.stay_id IN ({stay_ids_str})
  AND l.itemid IN (51301, 51300)
  AND l.charttime BETWEEN i.intime AND i.outtime
  AND l.valuenum > 0
"""
df_lab = client.query(query).to_dataframe()

print(f"Fetched {len(df_lab)} labevents.")

# Standardize timestamps and types
print("Standardizing timestamps and types...")
df_lab['stay_id'] = df_lab['stay_id'].astype('int64')
df_lab['charttime'] = pd.to_datetime(df_lab['charttime']).dt.tz_localize(None).astype('datetime64[us]')
grid_4h['stay_id'] = grid_4h['stay_id'].astype('int64')
grid_4h['charttime'] = pd.to_datetime(grid_4h['charttime']).dt.tz_localize(None).astype('datetime64[us]')

# Keep most recent lab if multiple exist at the exact same charttime
df_lab = df_lab.sort_values(['stay_id', 'charttime']).drop_duplicates(subset=['stay_id', 'charttime'], keep='last')

# 3. Align Data
print("Aligning WBC to grid...")
grid_sorted = grid_4h[['stay_id', 'charttime']].copy().sort_values('charttime')
lab_sorted = df_lab.sort_values('charttime')

merged = pd.merge_asof(
    grid_sorted,
    lab_sorted.rename(columns={'charttime': 'lab_time'}),
    left_on='charttime',
    right_on='lab_time',
    by='stay_id',
    direction='backward'
)

# Calculate fresh vs forward-filled
merged['time_diff'] = merged['charttime'] - merged['lab_time']
# EDIT: freshness window (4h cutoff for fresh vs forward-filled WBC)
merged['is_fresh'] = merged['time_diff'] <= pd.Timedelta(hours=4)

# Restore index and join back to main dataframe
df_res = pd.merge(grid_4h, merged[['stay_id', 'charttime', 'WBC', 'is_fresh']], on=['stay_id', 'charttime'], how='left')

# SIRS_wbc logic: 1 if WBC >= 12 OR WBC <= 4, else 0
# EDIT: SIRS_wbc thresholds (WBC >= 12 or <= 4 marks abnormal)
df_res['SIRS_wbc'] = ((df_res['WBC'] >= 12) | (df_res['WBC'] <= 4)).astype(int)
# Ensure NaN means 0
df_res.loc[df_res['WBC'].isna(), 'SIRS_wbc'] = 0

df_res = df_res.set_index(['stay_id', 'charttime'])

# Restore original parquet state exactly with new columns
final_out = df_4h.copy()
final_out['WBC'] = df_res['WBC']
final_out['SIRS_wbc'] = df_res['SIRS_wbc']

# Save Outputs
# EDIT: output_path - the "_full" parquet that is written
output_path = '/PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_full.parquet'
print(f"Saving outputs to {output_path}...")
final_out.to_parquet(output_path)

# Sanity Check
print("\n--- Sanity Check ---")
print(f"Stays: {final_out.index.get_level_values('stay_id').nunique()}")
print(f"Rows: {len(final_out)}")

fresh_frac = df_res['is_fresh'].mean()
any_frac = df_res['WBC'].notna().mean()
print(f"Fraction with fresh (non-ffill) WBC: {fresh_frac:.4f}")
print(f"Fraction with any WBC (after ffill): {any_frac:.4f}")

wbc_valid = final_out['WBC'].dropna()
print(f"WBC Distribution:")
print(f"  Min: {wbc_valid.min():.2f}")
print(f"  25%: {wbc_valid.quantile(0.25):.2f}")
print(f"  Median: {wbc_valid.median():.2f}")
print(f"  75%: {wbc_valid.quantile(0.75):.2f}")
print(f"  Max: {wbc_valid.max():.2f}")

sirs_pos = final_out['SIRS_wbc'].mean()
print(f"SIRS_wbc positive rate: {sirs_pos:.4f}")

abnormal_frac = ((wbc_valid < 4) | (wbc_valid >= 12)).mean()
print(f"Fraction of valid WBCs in abnormal range (<4 or >=12): {abnormal_frac:.4f}")

print("Done!")
