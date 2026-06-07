-- Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
-- Non-commercial use only. If you use or adapt this code, please cite the author.
-- See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

-- ============================================================================
--  16_extract_demographics.sql
--  Stage: 1 - Extraction
--
--  PURPOSE
--    Extracts static demographics (age, admission weight) plus the sepsis onset
--    anchor time for every adult ICU stay. Weight is taken as the mean of the
--    admission/daily weight readings, bounded to plausible adult values. Used
--    later to build the 6 hour prediction horizon proxy.
--
--  INPUTS
--    physionet-data.mimiciv_3_1_icu.icustays        (public MIMIC-IV)
--    physionet-data.mimiciv_3_1_hosp.patients       (public MIMIC-IV)
--    physionet-data.mimiciv_3_1_icu.chartevents     (public MIMIC-IV)
--    YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3   (your derived Sepsis-3 table)
--  OUTPUTS
--    none / returns one row per stay (stay_id, age, weight_kg, intime, sepsis3_time)
--
--  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
--    derived dataset    -  YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3 table path
--    age threshold      -  adults only filter, currently >= 18
--    weight itemids     -  chartevents itemids for admission/daily weight (226512, 224639)
--    weight bounds      -  plausible adult weight range, currently > 30 and < 350 kg
--
--  REQUIRES: BigQuery SQL | MIMIC-IV BigQuery access
-- ============================================================================

-- 16_extract_demographics.sql
-- We need to extract the static demographics (Age, Weight) and
-- the crucial Sepsis Onset Anchor time to build the 6-Hour Horizon proxy.

WITH adult_icu AS (
    SELECT
        ie.stay_id,
        ie.intime,
        s3.sofa_time AS sepsis3_time,
        pat.anchor_age + (EXTRACT(YEAR FROM ie.intime) - pat.anchor_year) AS age
    FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
    INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` pat
        ON ie.subject_id = pat.subject_id
    LEFT JOIN `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.sepsis3` s3  -- EDIT: derived sepsis3 table
        ON ie.stay_id = s3.stay_id
    WHERE (pat.anchor_age + (EXTRACT(YEAR FROM ie.intime) - pat.anchor_year)) >= 18  -- EDIT: age threshold (adults only)
),
-- Weight is tricky. We'll pull the closest weight (itemid 226512 or 224639) to admission
weight_data AS (
    SELECT
        c.stay_id,
        AVG(c.valuenum) as weight_kg
    FROM `physionet-data.mimiciv_3_1_icu.chartevents` c
    WHERE c.itemid IN (226512, 224639) -- EDIT: weight itemids (Admission Weight kg / Daily Weight)
      AND c.valuenum IS NOT NULL
      AND c.valuenum > 30 -- EDIT: weight lower bound (kg) - drop impossible adult weights
      AND c.valuenum < 350 -- EDIT: weight upper bound (kg)
    GROUP BY c.stay_id
)
SELECT
    a.stay_id,
    a.age,
    w.weight_kg,
    a.intime,
    a.sepsis3_time
FROM adult_icu a
LEFT JOIN weight_data w ON a.stay_id = w.stay_id
ORDER BY a.stay_id;
