-- Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
-- Non-commercial use only. If you use or adapt this code, please cite the author.
-- See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

-- ============================================================================
--  01_cohort_extraction.sql
--  Stage: 1 - Extraction
--
--  PURPOSE
--    Defines the adult ICU sepsis cohort from MIMIC-IV. Part A prints the
--    flowchart attrition counts (initial stays, adults, valid prediction
--    window). Part B emits the actual filtered cohort rows for download.
--
--  INPUTS
--    physionet-data.mimiciv_3_1_icu.icustays        (public MIMIC-IV)
--    physionet-data.mimiciv_3_1_hosp.patients       (public MIMIC-IV)
--    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3   (your derived Sepsis-3 table)
--  OUTPUTS
--    none / prints query result sets (flowchart counts + cohort rows)
--
--  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
--    derived dataset    -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3 table path
--    age threshold      -  adults only filter, currently >= 18
--    prediction window  -  valid sepsis window, currently 7 to 2000 hours from intime
--
--  REQUIRES: BigQuery SQL | MIMIC-IV BigQuery access
-- ============================================================================

-- 01_cohort_extraction.sql
-- Part A: FLOWCHART TRACKER
-- Run this query first to get the exact numbers for your thesis flowchart!

WITH all_icu AS (
    SELECT stay_id, subject_id, hadm_id, intime FROM `physionet-data.mimiciv_3_1_icu.icustays`
),
adults AS (
    SELECT i.stay_id
    FROM all_icu i
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON i.subject_id = p.subject_id
    WHERE (p.anchor_age + (EXTRACT(YEAR FROM i.intime) - p.anchor_year)) >= 18  -- EDIT: age threshold (adults only)
),
with_sepsis AS (
    SELECT a.stay_id, s.sepsis3, s.sofa_time AS sepsis3_time, i.intime
    FROM adults a
    JOIN all_icu i ON a.stay_id = i.stay_id
    LEFT JOIN `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3` s ON a.stay_id = s.stay_id  -- EDIT: derived sepsis3 table
),
time_filtered AS (
    SELECT stay_id
    FROM with_sepsis
    WHERE (sepsis3 IS NULL)
       OR (sepsis3 IS TRUE
           AND TIMESTAMP_DIFF(sepsis3_time, intime, HOUR) >= 7   -- EDIT: prediction window lower bound (hours)
           AND TIMESTAMP_DIFF(sepsis3_time, intime, HOUR) <= 2000)  -- EDIT: prediction window upper bound (hours)
)
SELECT
    '1. Initial ICU Stays' as step, COUNT(stay_id) as remaining_patients FROM all_icu
UNION ALL
SELECT '2. Adults Only (Age >= 18)', COUNT(stay_id) FROM adults
UNION ALL
SELECT '3. Valid Prediction Window (7-2000 hrs)', COUNT(stay_id) FROM time_filtered
ORDER BY remaining_patients DESC;

-- ==========================================
-- Part B: THE ACTUAL DATA EXTRACTION
-- Once you have your flowchart numbers, run this query to get the actual data to download.

WITH adult_icu AS (
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime,
        ie.outtime,
        pat.anchor_age + (EXTRACT(YEAR FROM ie.intime) - pat.anchor_year) AS age
    FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` pat
        ON ie.subject_id = pat.subject_id
    WHERE (pat.anchor_age + (EXTRACT(YEAR FROM ie.intime) - pat.anchor_year)) >= 18  -- EDIT: age threshold (adults only)
),
cohort_with_sepsis AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.stay_id,
        a.intime,
        a.outtime,
        a.age,
        IF(s3.sepsis3 IS TRUE, 1, 0) AS is_sepsis,
        s3.sofa_time AS sepsis3_time
    FROM adult_icu a
    LEFT JOIN `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3` s3  -- EDIT: derived sepsis3 table
        ON a.stay_id = s3.stay_id
)
SELECT *
FROM cohort_with_sepsis
WHERE (is_sepsis = 0)
   OR (is_sepsis = 1
       AND TIMESTAMP_DIFF(sepsis3_time, intime, HOUR) >= 7   -- EDIT: prediction window lower bound (hours)
       AND TIMESTAMP_DIFF(sepsis3_time, intime, HOUR) <= 2000)  -- EDIT: prediction window upper bound (hours)
