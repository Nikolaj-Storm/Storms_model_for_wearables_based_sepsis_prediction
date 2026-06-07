# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  05_bq_to_python.py
#  Stage: 1 - Extraction
#
#  PURPOSE
#    Runs the master extraction SQL on BigQuery, streams the result into a local
#    pandas DataFrame via the Arrow backend, downcasts numeric columns to halve
#    memory, and saves a compressed Parquet file for the preprocessing stage.
#
#  INPUTS
#    04_master_data_extraction.sql                  (SQL query file read from disk)
#    BigQuery tables referenced inside that SQL     (MIMIC-IV + your derived dataset)
#  OUTPUTS
#    mimic_derived_vitals_cohort.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID     -  your GCP project id used to bill/run the query
#    SQL file path  -  query file to execute, currently 04_master_data_extraction.sql
#    save_path      -  output parquet path
#
#  REQUIRES: pandas, google-cloud-bigquery, pyarrow | MIMIC-IV BigQuery access
# ============================================================================

# 05_bq_to_python.py
# This script executes the massive SQL query on the cloud servers,
# securely streams the result into local RAM,
# and explicitly compresses the float arrays to save 50% memory.

import os
import pandas as pd
from google.cloud import bigquery

# === CRITICAL SECURITY STEP ===
# Ensure you are authenticated with Google Cloud locally before running this.
# You can authenticate by running: gcloud auth application-default login

PROJECT_ID = "YOUR_GCP_PROJECT"  # EDIT: your GCP project id
print(f"Initializing BigQuery Client for project: {PROJECT_ID}")

try:
    client = bigquery.Client(project=PROJECT_ID)
except Exception as e:
    print("Failed to initialize client. Did you run 'gcloud auth application-default login' in terminal?")
    raise e

# Read the SQL file we just created
with open('04_master_data_extraction.sql', 'r') as file:  # EDIT: SQL query file path
    query = file.read()

print("\nSending query to Google BigQuery (this may take a couple minutes).")
print("The cloud servers are doing the 300-million row filtering for you...")

try:
    # Execute query and download to Pandas Dataframe via the fast Arrow backend
    df_raw = client.query(query).to_dataframe(create_bqstorage_client=True)

    print("\nDownload Complete!")
    print(f"Row count: {len(df_raw):,}")

    # --- Memory Optimization ---
    print("\nCompressing memory footprint (float64 -> float32)...")
    memory_before = df_raw.memory_usage(deep=True).sum() / (1024**2)

    # Cast float64 to float32 to instantly halve the memory usage of the numeric values
    df_raw['valuenum'] = df_raw['valuenum'].astype('float32')

    # Downcast integers representing item_ids and stay_ids to standard 32-bit int instead of 64
    df_raw['stay_id'] = df_raw['stay_id'].astype('int32')
    df_raw['itemid'] = df_raw['itemid'].astype('int32')

    # Optimize string objects if category is applicable (is_sepsis is a binary int)
    df_raw['is_sepsis'] = df_raw['is_sepsis'].astype('int8')

    memory_after = df_raw.memory_usage(deep=True).sum() / (1024**2)
    print(f"Memory reduction: {memory_before:.1f} MB -> {memory_after:.1f} MB (Saved {((memory_before - memory_after) / memory_before) * 100:.1f}%)")

    # Save to local disk as a highly-compressed Parquet file (.parquet)
    # Parquet is ~70% smaller than CSV and loads 10x faster into Pandas!
    save_path = "mimic_derived_vitals_cohort.parquet"  # EDIT: output parquet path
    print(f"\nSaving highly compressed dataset to {save_path}...")
    df_raw.to_parquet(save_path, engine='pyarrow', index=False)

    print("\nSuccess! You can now load this .parquet file into your 02_dataset_preparation script!")

except Exception as e:
    print(f"Error during extraction: {e}")
