# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  query_sepsis_stats.py
#  Stage: 1 - Extraction (diagnostic / stats)
#
#  PURPOSE
#    Sanity-check probes on the sepsis definitions. Checks whether the derived
#    sepsis3 table has rows without an ICU stay, lists example sepsis/septicemia
#    ICD codes, and compares total hospital admissions against those with an ICU
#    stay.
#
#  INPUTS
#    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3        (your derived table)
#    physionet-data.mimiciv_3_1_icu.icustays              (public MIMIC-IV)
#    physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses      (public MIMIC-IV)
#    physionet-data.mimiciv_3_1_hosp.admissions           (public MIMIC-IV)
#  OUTPUTS
#    none / prints three small result sets to console
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID     -  your GCP project id used to run the query
#    sepsis3 table  -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3
#    ICD LIMIT      -  number of sepsis ICD codes to preview, currently 10
#
#  REQUIRES: pandas, google-cloud-bigquery | MIMIC-IV BigQuery access
# ============================================================================

import pandas as pd
from google.cloud import bigquery
import sys

# Initialize client
try:
    client = bigquery.Client(project="YOUR_GCP_PROJECT")  # EDIT: your GCP project id
except Exception as e:
    print(f"Auth error: {e}")
    sys.exit(1)

def run_query(sql):
    try:
        return client.query(sql).to_dataframe()
    except Exception as e:
        print(f"Error querying: {e}")
        return None

print("Checking sepsis definitions and locations...")

# 1. Check if sepsis3 table has non-ICU stays
sql_sepsis3 = """
SELECT COUNT(1) as cnt
FROM `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3` s
LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` i ON s.stay_id = i.stay_id
WHERE i.stay_id IS NULL
"""  # EDIT: derived sepsis3 table path
df_s3 = run_query(sql_sepsis3)
print("Sepsis3 rows without ICU stay:")
if df_s3 is not None:
    print(df_s3)

# 2. Check ICD diagnoses for sepsis
sql_icd = """
SELECT icd_version, icd_code, long_title
FROM `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses`
WHERE LOWER(long_title) LIKE '%sepsis%' OR LOWER(long_title) LIKE '%septicemia%'
LIMIT 10
"""  # EDIT: ICD preview LIMIT
df_icd = run_query(sql_icd)
print("\nSepsis ICD codes:")
if df_icd is not None:
    print(df_icd)

# 3. Get overall hospital admissions vs ICU
sql_hosp = """
SELECT
  COUNT(DISTINCT a.hadm_id) as total_admissions,
  COUNT(DISTINCT i.hadm_id) as admissions_with_icu
FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` i ON a.hadm_id = i.hadm_id
"""
df_hosp = run_query(sql_hosp)
print("\nHospital admissions vs ICU:")
if df_hosp is not None:
    print(df_hosp)
