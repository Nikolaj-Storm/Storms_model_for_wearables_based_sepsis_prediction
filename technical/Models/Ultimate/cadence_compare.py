#!/usr/bin/env python3
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  cadence_compare.py
#  Stage 5 - Evaluation (sampling cadence)
#
#  PURPOSE
#    Paired patient-bootstrap comparison of the 1-hour and 4-hour grids for the
#    same models and the same test patients (paper Table 2).
#
#  INPUTS
#    Results/ultimate_full/preds_<partition>_{1h,4h}.parquet
#  OUTPUTS
#    Results/ultimate_full/cadence_comparison.csv
#
#  USER-EDITABLE SETTINGS
#    --B (bootstrap replicates)
#    Paths resolve relative to technical/ (ROOT = parents[2] of this file).
# ============================================================================

"""
cadence_compare.py  -  1-hour (continuous / wearable) vs 4-hour (ward-check) grid, same models, same held-out patients.

Reads preds_<partition>_1h.parquet and preds_<partition>_4h.parquet written by run_ultimate_full.py.
Both grids use the same patient-level split (same seed, same 58,970 stays), so the test PATIENTS are identical,
while the scored rows differ (each grid is scored on its own rows).
Delta AUROC (1h - 4h) uses a PAIRED patient-cluster bootstrap: the same resampled patients weight both grids.
Output: cadence_comparison.csv (+ DONE_cadence).
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
T = "is_sepsis_6h"


class WScorer:
    def __init__(self, p, y, code):
        u, inv = np.unique(-p, return_inverse=True)
        self.inv, self.y, self.code, self.nu = inv, y.astype(bool), code, len(u)

    def auc(self, sw):
        w = sw[self.code]
        pw = np.bincount(self.inv[self.y], w[self.y], self.nu); nw = np.bincount(self.inv[~self.y], w[~self.y], self.nu)
        return 1 - np.sum(pw * (np.cumsum(nw) - nw + 0.5 * nw)) / (pw.sum() * nw.sum())


def recall_at_fa(p, y, fa):
    o = np.argsort(-p); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    ok = fp <= fa * np.maximum(tp, 1)
    return float(tp[ok].max() / yy.sum()) if ok.any() else 0.0


def main(a):
    out = Path(a.out); rows = []
    for part in ("etiology", "kmeans", "both"):
        f1, f4 = out / f"preds_{part}_1h.parquet", out / f"preds_{part}_4h.parquet"
        if not (f1.exists() and f4.exists()):
            continue
        d1, d4 = pd.read_parquet(f1), pd.read_parquet(f4)
        s1, s4 = set(d1.subject_id), set(d4.subject_id)
        assert s1 == s4, f"test patients differ between grids ({len(s1 ^ s4)} differ)"
        pt = pd.concat([d1[["subject_id", T]], d4[["subject_id", T]]]).groupby("subject_id")[T].max()
        idx = {s: i for i, s in enumerate(pt.index)}
        c1, c4 = d1.subject_id.map(idx).values, d4.subject_id.map(idx).values
        pos, neg = np.where(pt.values == 1)[0], np.where(pt.values == 0)[0]; rng = np.random.default_rng(0)
        W = [np.bincount(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]), minlength=len(pt)).astype(float)
             for _ in range(a.B)]
        y1, y4 = d1[T].values, d4[T].values
        models = [c for c in d1.columns if c in d4.columns and c not in ("stay_id", "subject_id", "charttime", T)]
        for m in models:
            p1, p4 = d1[m].values, d4[m].values
            S1, S4 = WScorer(p1, y1, c1), WScorer(p4, y4, c4)
            A1 = np.array([S1.auc(w) for w in W]); A4 = np.array([S4.auc(w) for w in W]); D = A1 - A4
            rows.append(dict(partition=part, model=m,
                             auroc_1h=roc_auc_score(y1, p1), auroc_4h=roc_auc_score(y4, p4),
                             d_auroc_1h_minus_4h=float(np.median(D)), d_lo=float(np.quantile(D, .025)), d_hi=float(np.quantile(D, .975)),
                             lift_1h=average_precision_score(y1, p1) / y1.mean(), lift_4h=average_precision_score(y4, p4) / y4.mean(),
                             prev_1h=y1.mean(), prev_4h=y4.mean(),
                             sens10FA_1h=recall_at_fa(p1, y1, 10), sens10FA_4h=recall_at_fa(p4, y4, 10),
                             rows_1h=len(d1), rows_4h=len(d4), patients=len(pt)))
    r = pd.DataFrame(rows)
    r.to_csv(out / "cadence_comparison.csv", index=False)
    (out / "DONE_cadence").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    pd.set_option("display.width", 250, "display.max_colwidth", 55)
    print("CADENCE 1h vs 4h (paired patient-cluster bootstrap, B=%d)\n" % a.B +
          r[["partition", "model", "auroc_1h", "auroc_4h", "d_auroc_1h_minus_4h", "d_lo", "d_hi", "lift_1h", "lift_4h"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "Results" / "ultimate_full"))
    main(ap.parse_args())
