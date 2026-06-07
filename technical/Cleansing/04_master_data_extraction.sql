-- Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
-- Non-commercial use only. If you use or adapt this code, please cite the author.
-- See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

-- ============================================================================
--  04_master_data_extraction.sql
--  Stage: 1 - Extraction
--
--  PURPOSE
--    The heavy-lifting extraction. Builds the filtered adult ICU sepsis cohort
--    and joins it to chartevents to pull only the 6 core vital signs, restricted
--    to the window from ICU admission up to the sepsis onset time. Runs server
--    side in BigQuery to avoid pulling 300M+ rows locally.
--
--  INPUTS
--    physionet-data.mimiciv_3_1_icu.icustays        (public MIMIC-IV)
--    physionet-data.mimiciv_3_1_hosp.patients       (public MIMIC-IV)
--    physionet-data.mimiciv_3_1_icu.chartevents     (public MIMIC-IV)
--    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3   (your derived Sepsis-3 table)
--  OUTPUTS
--    none / returns the raw vitals cohort result set for download
--
--  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
--    GCP project/dataset  -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3 table path
--    age threshold        -  adults only filter, currently >= 18
--    prediction window    -  valid sepsis window, currently 7 to 2000 hours from intime
--    itemid set           -  the chartevents itemids mapped to each of the 6 vitals
--
--  REQUIRES: BigQuery SQL | MIMIC-IV BigQuery access
-- ============================================================================

-- 04_master_data_extraction.sql
-- This is the heavy-lifting query. We run this strictly in BigQuery
-- so that the cloud servers handle the 300+ million row joins,
-- saving the local 16GB laptop from crashing.

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
    LEFT JOIN `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3` s3  -- EDIT: GCP project/dataset + derived sepsis3 table
        ON a.stay_id = s3.stay_id
),
filtered_cohort AS (
    SELECT *
    FROM cohort_with_sepsis
    WHERE (is_sepsis = 0)  -- Control patient
       OR (is_sepsis = 1
           AND TIMESTAMP_DIFF(sepsis3_time, intime, HOUR) >= 7   -- EDIT: prediction window lower bound (hours)
           AND TIMESTAMP_DIFF(sepsis3_time, intime, HOUR) <= 2000)  -- EDIT: prediction window upper bound (hours)
),
vitals_raw AS (
    -- Extract ONLY the 6 core vital signs for the strictly filtered cohort
    SELECT
        c.stay_id,
        c.is_sepsis,
        TIMESTAMP_DIFF(ce.charttime, c.intime, HOUR) AS hours_since_admission,
        ce.charttime,
        ce.itemid,
        ce.valuenum
    FROM filtered_cohort c
    INNER JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ce
        ON c.stay_id = ce.stay_id
    WHERE ce.itemid IN (  -- EDIT: itemid set for the 6 core vitals
        220045, -- Heart Rate
        220210, 224690, -- Respiratory Rate
        220277, -- SpO2
        223761, 223762, -- Temperature (F and C)
        220179, 220050, -- Systolic Blood Pressure
        220180, 220051  -- Diastolic Blood Pressure
    )
    AND ce.valuenum IS NOT NULL
    -- Restrict to the vital window: From ICU admission up until the sepsis diagnosis
    AND ce.charttime >= c.intime
    AND (c.is_sepsis = 0 OR ce.charttime <= c.sepsis3_time)
)
SELECT * FROM vitals_raw;
