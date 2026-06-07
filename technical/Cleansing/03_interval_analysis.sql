-- Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
-- Non-commercial use only. If you use or adapt this code, please cite the author.
-- See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

-- ============================================================================
--  03_interval_analysis.sql
--  Stage: 1 - Extraction (diagnostic)
--
--  PURPOSE
--    Measures the real measurement frequency of the 6 core vital signs across
--    the cohort. For each patient and vital it finds the median gap between
--    readings, then counts how many patients are tracked at 5 min, 15 min,
--    hourly, 4 hourly, or worse than 4 hourly cadence.
--
--  INPUTS
--    physionet-data.mimiciv_3_1_icu.chartevents     (public MIMIC-IV)
--  OUTPUTS
--    none / prints one row per vital_category with the frequency buckets
--
--  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
--    itemid set         -  the chartevents itemids mapped to each of the 6 vitals
--    interval buckets   -  median interval thresholds (5, 15, 60, 240 minutes)
--
--  REQUIRES: BigQuery SQL | MIMIC-IV BigQuery access
-- ============================================================================

-- 03_interval_analysis.sql
-- Calculate the actual measurement frequency for ALL 6 core vital signs across the cohort.
-- This will tell us our "weakest link" down to the minute.

WITH vital_readings AS (
    -- Map all the different itemids to our 6 core vital sign categories
    SELECT
        stay_id,
        charttime,
        CASE
            WHEN itemid IN (220045) THEN '1_HeartRate'
            WHEN itemid IN (220277) THEN '2_SpO2'
            WHEN itemid IN (220210, 224690) THEN '3_RespRate'
            WHEN itemid IN (223761, 223762) THEN '4_Temperature'
            WHEN itemid IN (220179, 220050) THEN '5_Systolic_BP'
            WHEN itemid IN (220180, 220051) THEN '6_Diastolic_BP'
            ELSE 'Other'
        END AS vital_category
    FROM `physionet-data.mimiciv_3_1_icu.chartevents`
    WHERE itemid IN (  -- EDIT: itemid set for the 6 core vitals
        220045, -- HR
        220277, -- SpO2
        220210, 224690, -- RR
        223761, 223762, -- Temp (F & C)
        220179, 220050, -- SBP (Non-invasive & Invasive)
        220180, 220051  -- DBP (Non-invasive & Invasive)
    )
    AND valuenum IS NOT NULL -- Ignore blank clicks
),
time_ordered AS (
    -- Get the time gap to the *previous* reading of the EXACT same vital sign for the EXACT same patient
    SELECT
        stay_id,
        vital_category,
        charttime,
        LAG(charttime) OVER (PARTITION BY stay_id, vital_category ORDER BY charttime) as prev_charttime
    FROM vital_readings
),
time_diffs AS (
    -- Calculate minutes between readings
    SELECT
        stay_id,
        vital_category,
        TIMESTAMP_DIFF(charttime, prev_charttime, MINUTE) as diff_mins
    FROM time_ordered
    WHERE prev_charttime IS NOT NULL
      AND TIMESTAMP_DIFF(charttime, prev_charttime, MINUTE) > 0
),
patient_medians AS (
    -- Find the median tracking interval per patient, per vital sign
    SELECT
        stay_id,
        vital_category,
        APPROX_QUANTILES(diff_mins, 2)[OFFSET(1)] as median_interval_mins
    FROM time_diffs
    GROUP BY stay_id, vital_category
)
-- Aggregate the final results to see how many total patients meet the measurement frequency!
SELECT
    vital_category,
    COUNT(stay_id) AS total_patients_measured,
    COUNTIF(median_interval_mins <= 5) AS tracked_at_least_every_5_mins,      -- EDIT: interval bucket (minutes)
    COUNTIF(median_interval_mins <= 15) AS tracked_at_least_every_15_mins,    -- EDIT: interval bucket (minutes)
    COUNTIF(median_interval_mins <= 60) AS tracked_at_least_hourly,           -- EDIT: interval bucket (minutes)
    COUNTIF(median_interval_mins <= 240) AS tracked_at_least_every_4_hours,   -- EDIT: interval bucket (minutes)
    COUNTIF(median_interval_mins > 240) AS tracked_worse_than_4_hours         -- EDIT: interval bucket (minutes)
FROM patient_medians
GROUP BY vital_category
ORDER BY vital_category;
