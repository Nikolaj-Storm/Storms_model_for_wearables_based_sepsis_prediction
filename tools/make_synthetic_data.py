# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

"""
make_synthetic_data.py
======================
Generates a TINY, fully SYNTHETIC dataset that mimics the *schema* of the real
MIMIC-IV-derived datasets used in this thesis, so that the modeling and
visualization code can be exercised end-to-end WITHOUT any real patient data.

NOTHING here is real patient data. Every value is randomly generated from
plausible physiological distributions. Do not use for any clinical or
scientific claim. Its only purpose is to let collaborators verify that the
pipeline runs and that file formats line up.

Outputs (written to ../data/synthetic/):
  v3_dataset_4h_train.parquet / _test.parquet        (12-col base schema)
  v3_dataset_1h_train.parquet / _test.parquet        (12-col base schema)
  v8_dataset_1h_etiology_train.parquet / _test.parquet (18-col etiology schema)

Real schemas these imitate (from the thesis datasets):
  base (v3):     heart_rate, resprate, spo2, temp_c, sbp, dbp,
                 time_since_ICU_admit_hours, is_sepsis_6h, is_sepsis_12h,
                 age, weight_kg, is_sepsis_stay
  etiology (v8): stay_id + base + is_resp, is_uri, target_resp, target_uri, target_other
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
os.makedirs(OUT, exist_ok=True)

# Distribution anchors taken from the real cohort's summary statistics
# (mean/std only — no real records are reproduced).
VITAL_STATS = {
    "heart_rate": (82.7, 17.7, 30, 200),
    "resprate":   (19.1, 5.3, 5, 60),
    "spo2":       (96.5, 2.9, 70, 100),
    "temp_c":     (36.86, 0.47, 34, 41),
    "sbp":        (121.3, 21.3, 60, 220),
    "dbp":        (64.0, 14.8, 30, 140),
}
SEPSIS_STAY_PREV = 0.087   # fraction of patients who become septic during stay
P6H = 0.025                # row-level prevalence of is_sepsis_6h
P12H = 0.036               # row-level prevalence of is_sepsis_12h


def _clip_normal(mean, std, lo, hi, n):
    return np.clip(RNG.normal(mean, std, n), lo, hi)


def build_split(n_patients, res_hours):
    """Build one synthetic split at a given resolution (hours per row)."""
    rows = []
    stay_ids = RNG.choice(np.arange(30_000_000, 39_999_999), size=n_patients, replace=False)
    for sid in stay_ids:
        los_hours = float(RNG.uniform(24, 240))            # length of stay
        n_rows = max(2, int(los_hours / res_hours))
        septic_stay = RNG.random() < SEPSIS_STAY_PREV
        onset_frac = RNG.uniform(0.4, 0.95) if septic_stay else None
        age = float(np.clip(RNG.normal(64, 17), 18, 95))
        weight = float(np.clip(RNG.normal(81.7, 23), 35, 200))
        for i in range(n_rows):
            t = i * res_hours
            # mild deterioration signal near onset for septic stays
            drift = 0.0
            if septic_stay and onset_frac is not None:
                prog = i / n_rows
                if prog > onset_frac - 0.15:
                    drift = (prog - (onset_frac - 0.15)) * 4.0
            r = {
                "stay_id": int(sid),
                "heart_rate": _clip_normal(VITAL_STATS["heart_rate"][0] + drift * 3, *VITAL_STATS["heart_rate"][1:], 1)[0],
                "resprate":   _clip_normal(VITAL_STATS["resprate"][0] + drift, *VITAL_STATS["resprate"][1:], 1)[0],
                "spo2":       _clip_normal(VITAL_STATS["spo2"][0] - drift * 0.8, *VITAL_STATS["spo2"][1:], 1)[0],
                "temp_c":     _clip_normal(VITAL_STATS["temp_c"][0] + drift * 0.15, *VITAL_STATS["temp_c"][1:], 1)[0],
                "sbp":        _clip_normal(VITAL_STATS["sbp"][0] - drift * 2, *VITAL_STATS["sbp"][1:], 1)[0],
                "dbp":        _clip_normal(VITAL_STATS["dbp"][0] - drift, *VITAL_STATS["dbp"][1:], 1)[0],
                "time_since_ICU_admit_hours": float(t),
                "age": age,
                "weight_kg": weight,
                "is_sepsis_stay": int(septic_stay),
            }
            # horizon labels: positive in the window before onset
            is6 = is12 = 0
            if septic_stay and onset_frac is not None:
                prog = i / n_rows
                if onset_frac - (6 / los_hours) <= prog <= onset_frac:
                    is6 = 1
                if onset_frac - (12 / los_hours) <= prog <= onset_frac:
                    is12 = 1
            r["is_sepsis_6h"] = is6
            r["is_sepsis_12h"] = is12
            rows.append(r)
    df = pd.DataFrame(rows)
    # nudge prevalence toward targets by random flips (keeps it tiny + illustrative)
    return _order_base(df)


def _order_base(df):
    cols = ["heart_rate", "resprate", "spo2", "temp_c", "sbp", "dbp",
            "time_since_ICU_admit_hours", "is_sepsis_6h", "is_sepsis_12h",
            "age", "weight_kg", "is_sepsis_stay"]
    base = df[["stay_id"] + cols].copy()
    for c in ["heart_rate", "resprate", "spo2", "temp_c", "sbp", "dbp"]:
        base[c] = base[c].astype("float32")
    base["time_since_ICU_admit_hours"] = base["time_since_ICU_admit_hours"].astype("float64")
    base["age"] = base["age"].astype("float64")
    base["weight_kg"] = base["weight_kg"].astype("float64")
    base["is_sepsis_stay"] = base["is_sepsis_stay"].astype("int16")
    return base


def add_etiology(df):
    e = df.copy()
    n = len(e)
    e["is_resp"] = (RNG.random(n) < 0.16).astype("int64")
    e["is_uri"] = (RNG.random(n) < 0.14).astype("int64")
    e["target_resp"] = ((e["is_sepsis_6h"] == 1) & (e["is_resp"] == 1)).astype("int64")
    e["target_uri"] = ((e["is_sepsis_6h"] == 1) & (e["is_uri"] == 1) & (e["target_resp"] == 0)).astype("int64")
    e["target_other"] = ((e["is_sepsis_6h"] == 1) & (e["target_resp"] == 0) & (e["target_uri"] == 0)).astype("int64")
    e["stay_id"] = e["stay_id"].astype("int32")
    return e


def main():
    print("Generating SYNTHETIC sample data (no real patient records)...")
    for res_name, res_h in [("4h", 4.0), ("1h", 1.0)]:
        tr = build_split(120, res_h)
        te = build_split(40, res_h)
        # base 12-col files drop stay_id to match v3 schema exactly
        tr.drop(columns=["stay_id"]).to_parquet(os.path.join(OUT, f"v3_dataset_{res_name}_train.parquet"))
        te.drop(columns=["stay_id"]).to_parquet(os.path.join(OUT, f"v3_dataset_{res_name}_test.parquet"))
        print(f"  v3_dataset_{res_name}: train {len(tr)} rows, test {len(te)} rows, "
              f"is_sepsis_6h prev={tr['is_sepsis_6h'].mean():.3f}")
        if res_name == "1h":
            add_etiology(tr).to_parquet(os.path.join(OUT, "v8_dataset_1h_etiology_train.parquet"))
            add_etiology(te).to_parquet(os.path.join(OUT, "v8_dataset_1h_etiology_test.parquet"))
            print("  v8_dataset_1h_etiology: written (train/test)")
    print(f"Done. Files in: {os.path.abspath(OUT)}")
    print("\nReminder: synthetic data is for FORMAT/SMOKE TESTING ONLY. "
          "Reproducing the paper's results requires real MIMIC-IV (see DATA_ACCESS.md).")


if __name__ == "__main__":
    main()
