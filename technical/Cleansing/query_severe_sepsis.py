# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  query_severe_sepsis.py
#  Stage: 1 - Extraction (diagnostic / stats)
#
#  PURPOSE
#    Classifies admissions into suspected-sepsis groups by onset location
#    (ward-only, pre-ICU ward, ICU-acquired, post-ICU ward) and flags severe
#    sepsis from Sepsis-3 (ICU) plus severe-sepsis ICD codes (ward). Reports
#    suspected counts, severe cases, and severe deaths per group, with derived
#    incidence and mortality percentages.
#
#  INPUTS
#    physionet-data.mimiciv_3_1_hosp.admissions       (public MIMIC-IV)
#    physionet-data.mimiciv_3_1_icu.icustays          (public MIMIC-IV)
#    physionet-data.mimiciv_3_1_hosp.diagnoses_icd    (public MIMIC-IV)
#    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection  (your derived table)
#    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3                 (your derived table)
#  OUTPUTS
#    none / prints the grouped severe-sepsis result set to console
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    PROJECT_ID         -  your GCP project id used to run the query
#    suspicion table    -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.suspicion_of_infection
#    sepsis3 table      -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3
#    severe ICD codes   -  ward severe-sepsis ICD-9 (99592, 78552) / ICD-10 (R6520, R6521)
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
sepsis3_icu AS (
    SELECT stay_id, sepsis3
    FROM `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3`
    WHERE sepsis3 IS TRUE
),
icu_severe AS (
    SELECT DISTINCT i.hadm_id
    FROM `physionet-data.mimiciv_3_1_icu.icustays` i
    JOIN sepsis3_icu s3 ON i.stay_id = s3.stay_id
),
ward_severe_icd AS (
    SELECT DISTINCT hadm_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE
       (icd_version = 9 AND icd_code IN ('99592', '78552')) OR
       (icd_version = 10 AND icd_code IN ('R6520', 'R6521'))
),
classified AS (
    SELECT
        h.hadm_id,
        h.hospital_expire_flag,
        CASE
            WHEN i.hadm_id IS NULL AND inf.hadm_id IS NOT NULL THEN 'Group 1: Ward-Only Sepsis'
            WHEN i.hadm_id IS NOT NULL AND inf.hadm_id IS NOT NULL THEN
                CASE
                    WHEN inf.infection_time >= i.first_icu_in AND inf.infection_time <= i.last_icu_out THEN 'Group 3: ICU-Acquired Sepsis'
                    WHEN inf.infection_time > i.last_icu_out THEN 'Group 4: Post-ICU Ward Sepsis'
                    WHEN inf.infection_time < i.first_icu_in THEN 'Group 2: Pre-ICU Ward Sepsis'
                    ELSE 'Unknown'
                END
            ELSE 'No Sepsis'
        END as suspected_sepsis_group,
        CASE
            WHEN (i.hadm_id IS NOT NULL AND isev.hadm_id IS NOT NULL) OR wsev.hadm_id IS NOT NULL THEN 1
            ELSE 0
        END as is_severe_sepsis
    FROM hosp_admissions h
    LEFT JOIN icu_stays i ON h.hadm_id = i.hadm_id
    LEFT JOIN infections inf ON h.hadm_id = inf.hadm_id
    LEFT JOIN icu_severe isev ON h.hadm_id = isev.hadm_id
    LEFT JOIN ward_severe_icd wsev ON h.hadm_id = wsev.hadm_id
)
SELECT
    suspected_sepsis_group,
    COUNT(hadm_id) as total_suspected_sepsis,
    SUM(is_severe_sepsis) as severe_sepsis_cases,
    SUM(CASE WHEN is_severe_sepsis = 1 THEN hospital_expire_flag ELSE 0 END) as severe_sepsis_deaths
FROM classified
WHERE suspected_sepsis_group != 'No Sepsis'
GROUP BY suspected_sepsis_group
ORDER BY suspected_sepsis_group
"""  # EDIT: derived suspicion_of_infection + sepsis3 table paths and ward severe ICD codes are inside this query

print("Executing severe sepsis classification...")
try:
    df = client.query(sql).to_dataframe()
    df['severe_incidence_pct'] = (df['severe_sepsis_cases'] / df['total_suspected_sepsis'] * 100).round(2)
    df['severe_mortality_pct'] = (df['severe_sepsis_deaths'] / df['severe_sepsis_cases'] * 100).round(2)
    print("\n--- RESULTS ---")
    print(df.to_string(index=False))
except Exception as e:
    print(e)
