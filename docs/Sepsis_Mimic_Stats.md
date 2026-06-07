# Sepsis Incidence and Mortality in MIMIC-IV 

To evaluate the incidence and mortality of suspected sepsis across different hospital settings, we extracted a retrospective cohort from the MIMIC-IV dataset, incorporating derived clinical tables (`YOUR_DERIVED_DATASET`). 

Because the strict Sepsis-3 definition heavily relies on the Sequential Organ Failure Assessment (SOFA) score—which is typically only calculated using high-frequency physiological data available during intensive care unit (ICU) stays—we utilized the onset time of **Suspicion of Infection** (concurrent antibiotic administration and blood culture) to define suspected sepsis. This allowed us to accurately determine the timing of sepsis onset relative to a patient's ICU stay timeline across the entire hospitalization. 

To eliminate post-hoc outcome bias, the cohorts are strictly defined by the physical location of the patient *at the time of suspected sepsis onset*:

### Comprehensive Master Table

| Onset Location | Suspected Sepsis (n) | Suspected Mortality (%) | Severe Sepsis / Shock (n) | Progression Rate (%) | Severe Mortality (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Cohort** (All Hospitalizations) | 231,022 | **3.89%** | 42,018 | **18.19%** | **17.12%** |
| **1. Ward-Onset** | 212,774 | **3.11%** | 28,283 | **13.29%** | **17.38%** |
| **2. ICU-Onset** | 16,449 | **14.13%** | 13,668 | **83.09%** | **16.43%** |
| **3. Post-ICU Onset** | 1,799 | **2.56%** | 19 | **1.06%** | **10.53%** |

*\* Progression Rate indicates the percentage of patients with suspected infection who subsequently met code-based or Sepsis-3 criteria for severe organ dysfunction or shock.*

### 1. Ward-onset
- **Definition:** Sepsis occurred when the patient was on the ward.
- **n (Suspected Sepsis):** 212,774 (All-Cause Mortality: 3.11%)
- **n (Severe Sepsis):** 28,283 (All-Cause Mortality: 17.38%)
- **Progression:** 13.29% of suspected infections progressed to severe organ dysfunction or shock.

### 2. ICU-onset
- **Definition:** Sepsis happened when the patient was in the ICU.
- **n (Suspected Sepsis):** 16,449 (All-Cause Mortality: 14.13%)
- **n (Severe Sepsis):** 13,668 (All-Cause Mortality: 16.43%)
- **Progression:** 83.09% of suspected infections progressed to severe organ dysfunction or shock.

### 3. Post-ICU onset
- **Definition:** Sepsis happened when the patient was no longer in ICU - but had been previously.
- **n (Suspected Sepsis):** 1,799 (All-Cause Mortality: 2.56%)
- **n (Severe Sepsis):** 19 (All-Cause Mortality: 10.53%)
- **Progression:** 1.06% of suspected infections progressed to severe organ dysfunction or shock.

---

### Data Extraction Protocol
The cohort classification and extraction was performed in Google BigQuery using the following logic grouping strictly by onset timestamp relative to ICU timeline:

```sql
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
            WHEN i.hadm_id IS NULL AND inf.hadm_id IS NOT NULL THEN '1. Ward-Onset'
            WHEN i.hadm_id IS NOT NULL AND inf.hadm_id IS NOT NULL THEN
                CASE 
                    WHEN inf.infection_time < i.first_icu_in THEN '1. Ward-Onset'
                    WHEN inf.infection_time >= i.first_icu_in AND inf.infection_time <= i.last_icu_out THEN '2. ICU-Onset'
                    WHEN inf.infection_time > i.last_icu_out THEN '3. Post-ICU Onset'
                    ELSE 'Unknown'
                END
            ELSE 'No Sepsis'
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
```
