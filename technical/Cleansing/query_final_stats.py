# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  query_final_stats.py
#  Stage: 1 - Extraction (diagnostic / stats)
#
#  PURPOSE
#    Classifies every hospital admission into a sepsis-onset location group
#    (ward-only, ICU-acquired, post-ICU ward, pre-ICU ward, or no sepsis) using
#    infection time relative to ICU in/out times, then reports patient counts,
#    deaths, and mortality rate per group.
#
#  INPUTS
#    physionet-data.mimiciv_3_1_hosp.admissions      (public MIMIC-IV)
#    physionet-data.mimiciv_3_1_icu.icustays         (public MIMIC-IV)
#    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection  (your derived table)
#  OUTPUTS
#    none / prints the grouped mortality result set to console
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID       -  your GCP project id used to run the query
#    suspicion table  -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection
#
#  REQUIRES: pandas, google-cloud-bigquery | MIMIC-IV BigQuery access
# ============================================================================

import pandas as pd
from google.cloud import bigquery

client = bigquery.Client(project="YOUR_GCP_PROJECT")  # EDIT: your GCP project id

sql = """
WITH hosp_admissions AS (
    SELECT hadm_id, subject_id, hospital_expire_flag
    FROM `physionet-data.mimiciv_3_1_hosp.admissions`
),
icu_stays AS (
    SELECT hadm_id, MIN(intime) as first_icu_in, MAX(outtime) as last_icu_out
    FROM `physionet-data.mimiciv_3_1_icu.icustays`
    GROUP BY hadm_id
),
infections AS (
    SELECT hadm_id, MIN(suspected_infection_time) as infection_time
    FROM `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection`
    GROUP BY hadm_id
),
classified AS (
    SELECT
        h.hadm_id,
        h.hospital_expire_flag,
        CASE
            WHEN i.hadm_id IS NULL AND inf.hadm_id IS NOT NULL THEN 'Group_1_WardOnly_Sepsis'
            WHEN i.hadm_id IS NOT NULL AND inf.hadm_id IS NOT NULL THEN
                CASE
                    WHEN inf.infection_time >= i.first_icu_in AND inf.infection_time <= i.last_icu_out THEN 'Group_2_ICU_Acquired_Sepsis'
                    WHEN inf.infection_time > i.last_icu_out THEN 'Group_3_Post_ICU_Ward_Sepsis'
                    WHEN inf.infection_time < i.first_icu_in THEN 'Group_4_Pre_ICU_Ward_Sepsis'
                    ELSE 'Unknown'
                END
            ELSE 'No_Sepsis'
        END as sepsis_group
    FROM hosp_admissions h
    LEFT JOIN icu_stays i ON h.hadm_id = i.hadm_id
    LEFT JOIN infections inf ON h.hadm_id = inf.hadm_id
)
SELECT
    sepsis_group,
    COUNT(hadm_id) as total_patients,
    SUM(hospital_expire_flag) as total_deaths,
    ROUND(SUM(hospital_expire_flag) * 100.0 / COUNT(hadm_id), 2) as mortality_rate_pct
FROM classified
GROUP BY sepsis_group
ORDER BY sepsis_group
"""  # EDIT: derived suspicion_of_infection table path is inside this query

print("Executing classification query on BigQuery. Please wait...")
try:
    df = client.query(sql).to_dataframe()
    print("\n--- RESULTS ---")
    print(df.to_string(index=False))
except Exception as e:
    print(e)
