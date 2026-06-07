# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  query_soi_stats.py
#  Stage: 1 - Extraction (diagnostic / stats)
#
#  PURPOSE
#    Quick probes on the derived suspicion_of_infection table. Counts total rows
#    versus rows carrying a stay_id, and prints the table column schema from the
#    information schema.
#
#  INPUTS
#    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection      (your derived table)
#    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.INFORMATION_SCHEMA.COLUMNS  (dataset metadata)
#  OUTPUTS
#    none / prints two small result sets to console
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID       -  your GCP project id used to run the query
#    suspicion table  -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection
#    derived dataset  -  YOUR_DERIVED_DATASET used for the INFORMATION_SCHEMA probe
#
#  REQUIRES: pandas, google-cloud-bigquery | MIMIC-IV BigQuery access
# ============================================================================

import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="YOUR_GCP_PROJECT")  # EDIT: your GCP project id

def run_query(sql):
    return client.query(sql).to_dataframe()

sql_soi = """
SELECT COUNT(1) as total, COUNT(stay_id) as with_icu
FROM `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection`
"""  # EDIT: derived suspicion_of_infection table path
print("SOI counts:")
print(run_query(sql_soi))

sql_schema = """
SELECT column_name, data_type
FROM `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'suspicion_of_infection'
"""  # EDIT: derived dataset for INFORMATION_SCHEMA probe
try:
    print("\nSOI Schema:")
    print(run_query(sql_schema))
except Exception as e:
    print("Schema error:", e)
