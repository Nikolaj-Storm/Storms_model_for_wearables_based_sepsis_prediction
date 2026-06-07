# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  pull_toks_data.py
#  Stage: Exploratory
#
#  PURPOSE
#    Pulls GCS components, supplemental-oxygen status, and urine output from
#    MIMIC-IV via BigQuery for the stays in the 1-hour and 15-minute test
#    parquets, aligns them to each sampling grid, derives GCS_total, on_O2 and
#    a low-urine flag, and writes augmented "_full" parquets. Ends with a
#    coverage / distribution sanity check.
#
#  INPUTS
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_1h_test.parquet
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_15m_test.parquet
#    BigQuery tables physionet-data.mimiciv_3_1_icu.{icustays,chartevents,outputevents}
#  OUTPUTS
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_1h_test_full.parquet
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_15m_test_full.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID                 -  GCP / BigQuery project used for billing.
#    input parquet paths        -  the two v3 test parquets that are read.
#    output parquet paths       -  the two "_full" parquets that are written.
#    chartevents itemids        -  GCS (220739/223900/223901) + O2 (226732).
#    outputevents itemids       -  the urine-output item id list.
#    urine window / threshold   -  4h rolling sum, < 50 mL low-urine flag.
#    GCS default fill           -  15 when no GCS measured.
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
print("Loading parquets...")
# EDIT: input parquet paths (the two v3 test parquets)
df_1h = pd.read_parquet('/PATH/TO/PROJECT/technical/Data/v3_dataset_1h_test.parquet')
df_15m = pd.read_parquet('/PATH/TO/PROJECT/technical/Data/v3_dataset_15m_test.parquet')

# Check lengths
print(f"1h length: {len(df_1h)}, 15m length: {len(df_15m)}")

grid_1h = df_1h.reset_index()[['stay_id', 'charttime', 'is_sepsis_6h', 'is_sepsis_12h']]
grid_15m = df_15m.reset_index()[['stay_id', 'charttime', 'is_sepsis_6h', 'is_sepsis_12h']]

stay_ids = grid_1h['stay_id'].unique()
print(f"Unique stay_ids: {len(stay_ids)}")

stay_ids_str = ",".join([str(x) for x in stay_ids])

# 2. Query ICU stays
print("Querying icustays...")
query_icu = f"""
SELECT stay_id, subject_id, intime, outtime
FROM `physionet-data.mimiciv_3_1_icu.icustays`
WHERE stay_id IN ({stay_ids_str})
"""
df_icu = client.query(query_icu).to_dataframe()

subject_ids = df_icu['subject_id'].unique()
subject_ids_str = ",".join([str(x) for x in subject_ids])

# 3. Query Chartevents (GCS + O2)
print("Querying chartevents...")
# EDIT: chartevents itemids - GCS components (220739/223900/223901) + O2 (226732)
query_chart = f"""
SELECT stay_id, charttime, itemid, valuenum, value
FROM `physionet-data.mimiciv_3_1_icu.chartevents`
WHERE subject_id IN ({subject_ids_str})
  AND itemid IN (220739, 223900, 223901, 226732)
  AND stay_id IN ({stay_ids_str})
"""
df_chart = client.query(query_chart).to_dataframe()

# 4. Query Outputevents (Urine)
print("Querying outputevents...")
# EDIT: outputevents itemids - urine-output item id list
query_output = f"""
SELECT stay_id, charttime, itemid, value
FROM `physionet-data.mimiciv_3_1_icu.outputevents`
WHERE subject_id IN ({subject_ids_str})
  AND itemid IN (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)
  AND stay_id IN ({stay_ids_str})
"""
df_output = client.query(query_output).to_dataframe()

print("Standardizing timestamps and types...")
for df in [df_icu, df_chart, df_output, grid_1h, grid_15m]:
    if 'stay_id' in df.columns:
        df['stay_id'] = df['stay_id'].astype('int64')
    for col in df.columns:
        if 'time' in col:
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).astype('datetime64[us]')

print("Processing Data...")

# Process GCS
df_gcs = df_chart[df_chart['itemid'].isin([220739, 223900, 223901])].copy()
df_gcs = df_gcs[df_gcs['valuenum'].notnull() & (df_gcs['valuenum'] > 0)]
gcs_agg = df_gcs.groupby(['stay_id', 'charttime']).agg(
    GCS_total=('valuenum', 'sum'),
    n_components=('valuenum', 'count')
).reset_index()
# "Sum the three components per (stay_id, charttime) into GCS_total. Drop rows where any component is missing or zero."
gcs_agg = gcs_agg[gcs_agg['n_components'] == 3].drop(columns=['n_components'])

# Process O2
df_o2 = df_chart[df_chart['itemid'] == 226732].copy()
def parse_o2(val):
    if pd.isnull(val): return 0
    v = str(val).lower().strip()
    if v in ['none', 'room air']: return 0
    return 1
df_o2['on_O2'] = df_o2['value'].apply(parse_o2)
o2_agg = df_o2.groupby(['stay_id', 'charttime'])['on_O2'].max().reset_index()

# Process Urine
df_urine = df_output.copy()
df_urine['value'] = pd.to_numeric(df_urine['value'], errors='coerce').fillna(0)
urine_agg = df_urine.groupby(['stay_id', 'charttime'])['value'].sum().reset_index()

def align_data(grid_df, res_str):
    print(f"Aligning {res_str} data...")
    res = pd.Timedelta(res_str)

    # Merge intime
    df_res = pd.merge(grid_df, df_icu[['stay_id', 'intime']], on='stay_id', how='left')
    df_res = df_res.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    grid_sorted = df_res[['stay_id', 'charttime']].copy().sort_values('charttime')

    # Align GCS
    gcs_sorted = gcs_agg.sort_values('charttime')
    merged_gcs = pd.merge_asof(
        gcs_sorted,
        grid_sorted.rename(columns={'charttime': 'grid_time'}),
        left_on='charttime',
        right_on='grid_time',
        by='stay_id',
        direction='forward'
    )
    merged_gcs = merged_gcs.dropna(subset=['grid_time'])
    gcs_window = merged_gcs.groupby(['stay_id', 'grid_time'])['GCS_total'].min().reset_index()

    df_res = pd.merge(df_res, gcs_window, left_on=['stay_id', 'charttime'], right_on=['stay_id', 'grid_time'], how='left')
    # EDIT: GCS default fill (15 when no GCS measured)
    df_res['GCS_total'] = df_res.groupby('stay_id')['GCS_total'].ffill().fillna(15).astype(int)
    df_res = df_res.drop(columns=['grid_time'], errors='ignore')

    # Align O2
    o2_sorted = o2_agg.sort_values('charttime')
    merged_o2 = pd.merge_asof(
        o2_sorted,
        grid_sorted.rename(columns={'charttime': 'grid_time'}),
        left_on='charttime',
        right_on='grid_time',
        by='stay_id',
        direction='forward'
    )
    merged_o2 = merged_o2.dropna(subset=['grid_time'])
    o2_window = merged_o2.groupby(['stay_id', 'grid_time'])['on_O2'].max().reset_index()

    df_res = pd.merge(df_res, o2_window, left_on=['stay_id', 'charttime'], right_on=['stay_id', 'grid_time'], how='left')
    df_res['on_O2'] = df_res.groupby('stay_id')['on_O2'].ffill().fillna(0).astype(int)
    df_res = df_res.drop(columns=['grid_time'], errors='ignore')

    # Align Urine
    grid_points = df_res[['stay_id', 'charttime']].copy()
    grid_points['is_grid'] = True
    grid_points['value'] = 0.0

    urine_events = urine_agg[['stay_id', 'charttime', 'value']].copy()
    urine_events['is_grid'] = False

    combined = pd.concat([grid_points, urine_events], ignore_index=True)
    combined = combined.sort_values(['stay_id', 'charttime', 'is_grid']).reset_index(drop=True)

    # EDIT: urine window (4h rolling sum)
    rolled = combined.set_index('charttime').groupby('stay_id')['value'].rolling('4h', closed='right').sum()
    combined['urine_4h_sum'] = rolled.values

    grid_urine = combined[combined['is_grid']][['stay_id', 'charttime', 'urine_4h_sum']]

    df_res = pd.merge(df_res, grid_urine, on=['stay_id', 'charttime'], how='left')
    df_res['urine_4h_sum'] = df_res['urine_4h_sum'].fillna(0.0).astype(float)

    four_hours = pd.Timedelta(hours=4)
    # EDIT: urine threshold (< 50 mL low-urine flag, after first 4h of stay)
    df_res['urine_low'] = (
        (df_res['urine_4h_sum'] < 50) &
        (df_res['charttime'] >= df_res['intime'] + four_hours)
    ).astype(int)

    df_res = df_res.drop(columns=['intime'])
    df_res = df_res.set_index(['stay_id', 'charttime'])
    return df_res

final_1h = align_data(grid_1h, '1h')
final_15m = align_data(grid_15m, '15m')

# Re-attach to original parquets to ensure all existing features are kept
final_1h_out = df_1h.copy()
final_1h_out['GCS_total'] = final_1h['GCS_total']
final_1h_out['on_O2'] = final_1h['on_O2']
final_1h_out['urine_4h_sum'] = final_1h['urine_4h_sum']
final_1h_out['urine_low'] = final_1h['urine_low']

final_15m_out = df_15m.copy()
final_15m_out['GCS_total'] = final_15m['GCS_total']
final_15m_out['on_O2'] = final_15m['on_O2']
final_15m_out['urine_4h_sum'] = final_15m['urine_4h_sum']
final_15m_out['urine_low'] = final_15m['urine_low']

print("Saving outputs...")
# EDIT: output parquet paths (the two "_full" parquets)
final_1h_out.to_parquet('/PATH/TO/PROJECT/technical/Data/v3_dataset_1h_test_full.parquet')
final_15m_out.to_parquet('/PATH/TO/PROJECT/technical/Data/v3_dataset_15m_test_full.parquet')

# Sanity Check
def sanity_check(df, name):
    print(f"\n--- Sanity Check: {name} ---")
    print(f"Stays: {df.index.get_level_values('stay_id').nunique()}")
    print(f"Rows: {len(df)}")
    print(f"is_sepsis_6h prevalence: {df['is_sepsis_6h'].mean():.4f}")
    print(f"is_sepsis_12h prevalence: {df['is_sepsis_12h'].mean():.4f}")

    unique_stays = pd.Series(df.index.get_level_values('stay_id').unique())
    gcs_cov = unique_stays.isin(gcs_agg['stay_id']).mean()
    o2_cov = unique_stays.isin(o2_agg['stay_id']).mean()
    urine_cov = unique_stays.isin(urine_agg['stay_id']).mean()

    print(f"GCS Coverage (Stays): {gcs_cov:.4f}")
    print(f"O2 Coverage (Stays): {o2_cov:.4f}")
    print(f"Urine Coverage (Stays): {urine_cov:.4f}")

    print(f"urine_low flag rate: {df['urine_low'].mean():.4f}")

    print("GCS_total Distribution:")
    print(df['GCS_total'].value_counts().sort_index())

    print("on_O2 Distribution:")
    print(df['on_O2'].value_counts())

sanity_check(final_1h_out, "1-Hour Dataset")
sanity_check(final_15m_out, "15-Minute Dataset")

print("Done!")
