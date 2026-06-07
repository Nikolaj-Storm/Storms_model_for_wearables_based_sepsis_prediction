# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  37_extract_etiology_labels.py
#  Stage: 2 - Preprocessing (labelling)
#
#  PURPOSE
#    V8 etiology labelling. Pulls ICD-9/10 diagnosis codes from BigQuery,
#    classifies each stay as respiratory and/or urinary by code prefix, merges
#    those flags into the V3 1-hour train/test sets, and derives mutually
#    exclusive sepsis subtype targets (respiratory, urinary, other) gated on the
#    6-hour sepsis label.
#
#  INPUTS
#    physionet-data.mimiciv_3_1_icu.icustays          (public MIMIC-IV)
#    physionet-data.mimiciv_3_1_hosp.diagnoses_icd    (public MIMIC-IV)
#    v3_dataset_1h_train.parquet / v3_dataset_1h_test.parquet  (from 18_..._v3.py)
#  OUTPUTS
#    v8_dataset_1h_etiology_train.parquet
#    v8_dataset_1h_etiology_test.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID          -  your GCP project id used to run the query
#    resp_codes_10 / _9  -  ICD-10 / ICD-9 prefixes counted as respiratory
#    uri_codes_10 / _9   -  ICD-10 / ICD-9 prefixes counted as urinary
#    sepsis gate column  -  label gating the subtype targets, currently is_sepsis_6h
#    input v3 paths      -  v3_dataset_1h_train / test parquet paths
#    output paths        -  v8 etiology train/test parquet paths
#
#  REQUIRES: pandas, numpy, google-cloud-bigquery, pyarrow | MIMIC-IV BigQuery access
# ============================================================================

import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
import warnings

warnings.filterwarnings('ignore')

def extract_etiology_labels():
    print("---------------------------------------------------------------------")
    print("PHASE 15: V8 Etiology-Specific Diagnosis Extraction (Google BigQuery)")
    print("---------------------------------------------------------------------")

    PROJECT_ID = "YOUR_GCP_PROJECT"  # EDIT: your GCP project id
    try:
        client = bigquery.Client(project=PROJECT_ID)
        print(f"Authenticated successfully with BigQuery ({PROJECT_ID})")
    except Exception as e:
        print("BigQuery authentication failed.")
        return

    # The SQL query specifically joins icustays to diagnoses_icd to extract the disease causes
    query = """
    WITH cohort AS (
        SELECT subject_id, hadm_id, stay_id
        FROM `physionet-data.mimiciv_3_1_icu.icustays`
    ),
    diagnoses AS (
        SELECT subject_id, hadm_id, icd_code, icd_version
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    )
    SELECT
        c.stay_id,
        d.icd_code,
        d.icd_version
    FROM cohort c
    LEFT JOIN diagnoses d ON c.hadm_id = d.hadm_id
    """

    print("\nStreaming ICD-9 and ICD-10 diagnosis codes from hospital server...")
    df_diag = client.query(query).to_dataframe(create_bqstorage_client=True)
    print(f"Extracted {len(df_diag):,} diagnostic rows.")

    print("\nClassifying Sepsis Etiologies...")

    # ---------------------------------------------------------------------------
    # Mapping the specific ICD-10 and ICD-9 Codes to Biological Subtypes
    # ---------------------------------------------------------------------------
    # Respiratory / Pneumonia Sepsis codes (ICD-10: J13-J18, J20-J22; ICD-9: 480-488)
    resp_codes_10 = ['J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J20', 'J21', 'J22', 'J69']  # EDIT: respiratory ICD-10 prefixes
    resp_codes_9 = ['480', '481', '482', '483', '484', '485', '486', '487', '488', '507']  # EDIT: respiratory ICD-9 prefixes

    # Urinary Sepsis codes (ICD-10: N39, N30; ICD-9: 599, 590)
    uri_codes_10 = ['N390', 'N39', 'N30', 'N10', 'N11', 'N12']  # EDIT: urinary ICD-10 prefixes
    uri_codes_9 = ['5990', '599', '590', '595']  # EDIT: urinary ICD-9 prefixes

    # Check if a code belongs to a subtype (doing startswith for broader ICD matching)
    def is_respiratory(row):
        code = str(row['icd_code'])
        v = row['icd_version']
        if pd.isna(v):
            return False
        if v == 10:
            return any(code.startswith(x) for x in resp_codes_10)
        else:
            return any(code.startswith(x) for x in resp_codes_9)

    def is_urinary(row):
        code = str(row['icd_code'])
        v = row['icd_version']
        if pd.isna(v):
            return False
        if v == 10:
            return any(code.startswith(x) for x in uri_codes_10)
        else:
            return any(code.startswith(x) for x in uri_codes_9)

    # Apply conditions
    df_diag['is_resp'] = df_diag.apply(is_respiratory, axis=1)
    df_diag['is_uri'] = df_diag.apply(is_urinary, axis=1)

    # Group back by stay_id (a stay_id has many diagnoses, we just need ANY True)
    print("  -> Aggregating patient stays...")
    df_etiology = df_diag.groupby('stay_id').agg({
        'is_resp': 'max',
        'is_uri': 'max'
    }).reset_index()

    print(f"Classified {len(df_etiology):,} unique ICU patient stays.")
    print(f"   * Respiratory Present: {df_etiology['is_resp'].sum():,}")
    print(f"   * Urinary Present: {df_etiology['is_uri'].sum():,}")

    print("\nMerging Etiology Labels with V3 1-Hour Training & Test datasets...")

    # Load the proxy dataset to merge the labels
    df_train = pd.read_parquet('v3_dataset_1h_train.parquet')  # EDIT: input v3 train path
    df_test = pd.read_parquet('v3_dataset_1h_test.parquet')  # EDIT: input v3 test path

    original_train_rows = len(df_train)
    original_test_rows = len(df_test)

    df_train = df_train.merge(df_etiology, on='stay_id', how='left')
    df_test = df_test.merge(df_etiology, on='stay_id', how='left')

    # Fill Nans with False (0) for patients without those specific diagnoses
    df_train['is_resp'] = df_train['is_resp'].fillna(False).astype(int)
    df_train['is_uri'] = df_train['is_uri'].fillna(False).astype(int)

    df_test['is_resp'] = df_test['is_resp'].fillna(False).astype(int)
    df_test['is_uri'] = df_test['is_uri'].fillna(False).astype(int)

    # Create the Mutually Exclusive Subtype Labels
    # If the patient IS Septic (is_sepsis_6h == 1), and has a Respiratory code, they are Respiratory Sepsis
    print("  -> Generating Mutually Exclusive Mathematical Target Columns (`target_resp`, `target_uri`, `target_other`)")

    def apply_sepsis_targets(df):
        # Default all zeros
        df['target_resp'] = 0
        df['target_uri'] = 0
        df['target_other'] = 0

        # Only assign 1 if the core Sepsis boundary is crossed
        septic_mask = (df['is_sepsis_6h'] == 1)  # EDIT: sepsis gate column for subtype targets

        # Respiratory gets first priority
        resp_mask = septic_mask & (df['is_resp'] == 1)
        df.loc[resp_mask, 'target_resp'] = 1

        # Urinary gets second priority (if not already resp)
        uri_mask = septic_mask & (df['is_uri'] == 1) & (~resp_mask)
        df.loc[uri_mask, 'target_uri'] = 1

        # Other Sepsis is anyone septic who isn't Resp or Uri
        other_mask = septic_mask & (~resp_mask) & (~uri_mask)
        df.loc[other_mask, 'target_other'] = 1
        return df

    df_train = apply_sepsis_targets(df_train)
    df_test = apply_sepsis_targets(df_test)

    print("\nLabel Generation Complete!")
    print(f"  [TRAIN] Respiratory Sepsis points: {df_train['target_resp'].sum():,}")
    print(f"  [TRAIN] Urinary Sepsis points:     {df_train['target_uri'].sum():,}")
    print(f"  [TRAIN] Other Sepsis points:       {df_train['target_other'].sum():,}")

    print("\nOverwriting Proxy Datasets with V8 Etiology Vectors...")
    df_train.to_parquet('v8_dataset_1h_etiology_train.parquet', engine='pyarrow', index=False)  # EDIT: output train path
    df_test.to_parquet('v8_dataset_1h_etiology_test.parquet', engine='pyarrow', index=False)  # EDIT: output test path

    print("Success! The Parallel Datasets are fully constructed and highly compressed.")

if __name__ == '__main__':
    extract_etiology_labels()
