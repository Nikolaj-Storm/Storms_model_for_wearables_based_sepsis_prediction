# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  17_download_demographics.py
#  Stage: 1 - Extraction
#
#  PURPOSE
#    Runs the demographics extraction SQL on BigQuery, downloads age, weight and
#    anchor times into pandas, downcasts the columns to save memory, and saves
#    an auxiliary Parquet file used by the V3 preprocessing stage.
#
#  INPUTS
#    16_extract_demographics.sql                    (SQL query file read from disk)
#    BigQuery tables referenced inside that SQL     (MIMIC-IV + your derived dataset)
#  OUTPUTS
#    demographics_anchors.parquet
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID     -  your GCP project id used to bill/run the query
#    SQL file path  -  query file to execute, currently 16_extract_demographics.sql
#    save_path      -  output parquet path
#
#  REQUIRES: pandas, google-cloud-bigquery, pyarrow | MIMIC-IV BigQuery access
# ============================================================================

import os
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "YOUR_GCP_PROJECT"  # EDIT: your GCP project id
print(f"Initializing V3 Demographics Extraction for project: {PROJECT_ID}")

try:
    client = bigquery.Client(project=PROJECT_ID)
except Exception as e:
    print("Failed to initialize client. Did you run 'gcloud auth application-default login'?")
    raise e

with open('16_extract_demographics.sql', 'r') as file:  # EDIT: SQL query file path
    query = file.read()

print("\nExtracting Age, Weight, and Anchor Times from BigQuery...")

try:
    df_demo = client.query(query).to_dataframe(create_bqstorage_client=True)

    print("\nDownload Complete!")
    print(f"Row count: {len(df_demo):,} patients")

    # Compress standard variables
    df_demo['stay_id'] = df_demo['stay_id'].astype('int32')
    df_demo['age'] = df_demo['age'].astype('int16')
    df_demo['weight_kg'] = df_demo['weight_kg'].astype('float32')

    # Save auxiliary demographics file
    save_path = "demographics_anchors.parquet"  # EDIT: output parquet path
    print(f"Saving auxiliary V3 dataset to {save_path}...")
    df_demo.to_parquet(save_path, engine='pyarrow', index=False)

except Exception as e:
    print(f"Error during extraction: {e}")
