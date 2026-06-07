# data/

This folder holds datasets the pipeline reads. **No real patient data is, or may
be, stored here in the public repository** (MIMIC-IV is credentialed-access — see
`../DATA_ACCESS.md`). The `.gitignore` blocks every `*.parquet` except the
synthetic sample.

## `synthetic/`

A tiny, fully **synthetic** dataset that imitates the *schema* of the real
MIMIC-IV-derived files so the modeling and figure code can run without any real
data. Generate it with:

```bash
python tools/make_synthetic_data.py
```

Files produced and the real schema each imitates:

| File | Cols | Schema it mirrors |
|---|---|---|
| `v3_dataset_4h_train/_test.parquet` | 12 | base vitals dataset at 4-hour resolution |
| `v3_dataset_1h_train/_test.parquet` | 12 | base vitals dataset at 1-hour resolution |
| `v8_dataset_1h_etiology_train/_test.parquet` | 18 | etiology-labelled dataset (adds `stay_id`, `is_resp`, `is_uri`, `target_resp/uri/other`) |

**Base 12-column schema:** `heart_rate, resprate, spo2, temp_c, sbp, dbp,
time_since_ICU_admit_hours, is_sepsis_6h, is_sepsis_12h, age, weight_kg,
is_sepsis_stay`.

Every value is randomly drawn from plausible physiological distributions. The
results are meaningless and must not be used for any clinical or scientific
claim — the sample exists only to confirm the code runs and the file formats
line up. A weak deterioration signal is injected near simulated onset so a model
can technically fit, but discrimination on this data is essentially chance.

## Where real data goes

When you have MIMIC-IV access, the Stage-1/2 scripts write the real datasets
(`mimic_derived_vitals_cohort.parquet`, `demographics_anchors.parquet`,
`v3_dataset_*`, `v12_dataset_*`, the engineered sets, etc.). Point the later
stages at those files. They will be git-ignored automatically.
