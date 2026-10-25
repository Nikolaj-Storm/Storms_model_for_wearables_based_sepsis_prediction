-- Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
-- Non-commercial use only. If you use or adapt this code, please cite the author.
-- See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

-- Inputs the full bedside scores need beyond the six vital signs, per ICU stay of the model cohort:
--   gcs_s  Glasgow Coma Scale total at every change ("minutes since ICU admission:GCS"; intubated = 15, MIMIC Code convention)
--   o2_s   supplemental-oxygen flag at every change (device other than None / Room Air, or O2 flow > 0)
--   wbc_s  white cell count values from 24 h before ICU admission to ICU discharge
-- Export the result to comparator_events.parquet (columns stay_id, gcs_s, o2_s, wbc_s) for full_clinical_comparators.py.
CREATE OR REPLACE TABLE `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.comparator_events` AS   -- EDIT: your project / dataset
WITH c AS (SELECT stay_id, hadm_id, intime, outtime FROM `YOUR_GCP_PROJECT.YOUR_DERIVED_DATASET.model_cohort`),  -- EDIT
ce AS (
  SELECT e.stay_id, e.charttime, e.itemid, e.value, e.valuenum
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` e JOIN c USING (stay_id)
  WHERE e.itemid IN (220739, 223900, 223901, 226732, 223834)),
g0 AS (
  SELECT stay_id, charttime,
    MAX(IF(itemid=220739, valuenum, NULL)) eye, MAX(IF(itemid=223901, valuenum, NULL)) motor,
    MAX(IF(itemid=223900, valuenum, NULL)) verbal, MAX(IF(itemid=223900 AND value='No Response-ETT', 1, 0)) ett
  FROM ce WHERE itemid IN (220739, 223900, 223901) GROUP BY 1, 2),
g1 AS (
  SELECT stay_id, charttime,
    LAST_VALUE(eye IGNORE NULLS) OVER w eye, LAST_VALUE(motor IGNORE NULLS) OVER w motor,
    LAST_VALUE(verbal IGNORE NULLS) OVER w verbal, ett
  FROM g0 WINDOW w AS (PARTITION BY stay_id ORDER BY charttime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)),
g2 AS (
  SELECT stay_id, charttime,
    CASE WHEN ett = 1 THEN 15 WHEN eye IS NULL OR motor IS NULL OR verbal IS NULL THEN NULL ELSE eye + motor + verbal END gcs
  FROM g1),
g3 AS (SELECT *, LAG(gcs) OVER (PARTITION BY stay_id ORDER BY charttime) prev FROM g2 WHERE gcs IS NOT NULL),
gs AS (
  SELECT stay_id, STRING_AGG(FORMAT('%d:%d', DATETIME_DIFF(charttime, c.intime, MINUTE), CAST(gcs AS INT64)), ';' ORDER BY charttime) gcs_s
  FROM g3 JOIN c USING (stay_id) WHERE prev IS NULL OR prev != gcs GROUP BY stay_id),
o0 AS (
  SELECT stay_id, charttime,
    MAX(CASE WHEN itemid = 226732 AND value NOT IN ('None', 'Room Air') THEN 1
             WHEN itemid = 223834 AND valuenum > 0 THEN 1 ELSE 0 END) o2
  FROM ce WHERE itemid IN (226732, 223834) GROUP BY 1, 2),
o1 AS (SELECT *, LAG(o2) OVER (PARTITION BY stay_id ORDER BY charttime) prev FROM o0),
os AS (
  SELECT stay_id, STRING_AGG(FORMAT('%d:%d', DATETIME_DIFF(charttime, c.intime, MINUTE), o2), ';' ORDER BY charttime) o2_s
  FROM o1 JOIN c USING (stay_id) WHERE prev IS NULL OR prev != o2 GROUP BY stay_id),
w AS (
  SELECT c.stay_id, STRING_AGG(FORMAT('%d:%.1f', DATETIME_DIFF(l.charttime, c.intime, MINUTE), l.valuenum), ';' ORDER BY l.charttime) wbc_s
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` l JOIN c ON l.hadm_id = c.hadm_id
  WHERE l.itemid IN (51301, 51300) AND l.valuenum IS NOT NULL
    AND l.charttime BETWEEN DATETIME_SUB(c.intime, INTERVAL 24 HOUR) AND c.outtime
  GROUP BY c.stay_id)
SELECT c.stay_id, gs.gcs_s, os.o2_s, w.wbc_s
FROM c LEFT JOIN gs USING (stay_id) LEFT JOIN os USING (stay_id) LEFT JOIN w USING (stay_id);
