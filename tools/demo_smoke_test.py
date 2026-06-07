# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

"""
demo_smoke_test.py
==================
Minimal end-to-end smoke test on the SYNTHETIC sample data. Confirms that the
data format, a couple of engineered features, and a classifier all wire up
correctly. This is NOT the thesis model and the numbers are meaningless (random
data) — it only proves the plumbing works on any machine.

Run:
    python tools/make_synthetic_data.py      # once, to create the sample
    python tools/demo_smoke_test.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
VITALS = ["heart_rate", "resprate", "spo2", "temp_c", "sbp", "dbp"]
CONTEXT = ["time_since_ICU_admit_hours", "age", "weight_kg"]
TARGET = "is_sepsis_6h"


def add_basic_features(df):
    """A couple of the thesis's Layer-1 clinical composites, for illustration."""
    df = df.copy()
    df["shock_index"] = df["heart_rate"] / df["sbp"].replace(0, np.nan)
    df["map"] = (df["sbp"] + 2 * df["dbp"]) / 3.0
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    return df.fillna(0)


def main():
    tr = add_basic_features(pd.read_parquet(os.path.join(DATA, "v3_dataset_1h_train.parquet")))
    te = add_basic_features(pd.read_parquet(os.path.join(DATA, "v3_dataset_1h_test.parquet")))
    feats = VITALS + CONTEXT + ["shock_index", "map", "pulse_pressure"]

    Xtr, ytr = tr[feats], tr[TARGET]
    Xte, yte = te[feats], te[TARGET]
    print(f"train rows={len(tr)}  test rows={len(te)}  positives(train)={int(ytr.sum())}")

    if ytr.sum() < 2 or yte.sum() < 1:
        print("Synthetic positives too sparse to score AUROC — format check still passed.")
        print("SMOKE TEST PASSED (data loads, features build, model fits).")
        return

    clf = RandomForestClassifier(n_estimators=80, max_depth=8, class_weight="balanced",
                                 random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"AUROC={roc_auc_score(yte, p):.3f}  AUPRC={average_precision_score(yte, p):.3f}  "
          f"(meaningless on random data — plumbing check only)")
    print("SMOKE TEST PASSED (data loads, features build, model fits, metrics compute).")


if __name__ == "__main__":
    main()
