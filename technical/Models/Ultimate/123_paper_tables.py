#!/usr/bin/env python3
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  123_paper_tables.py
#  Stage 6 - Paper tables
#
#  PURPOSE
#    Operating points, encounter-protocol comparators and feature importance
#    used in earlier drafts of the journal paper.
#
#  INPUTS
#    Results/ultimate_full/
#  OUTPUTS
#    Results/paper/
#
#  USER-EDITABLE SETTINGS
#    stage argument (enc, importance, hourly, tables)
#    Paths resolve relative to technical/ (ROOT = parents[2] of this file).
# ============================================================================

"""
123_paper_tables.py  -  numbers for the Barton-format paper (Tables 2-4, Figure 2) from the finished full runs.

Stage 'enc <partition>': rebuilds the Barton-protocol encounters (identical code + seeds as barton_protocol_full.py),
    rebuilds the 3-layer predictions from the cached layer-1 outputs (ckpt_barton_1h), checks the AUROC against the
    logged value, adds vital-sign-only comparator scores (SIRS, qSOFA, NEWS2, MEWS) at the index time, and stores
    per (H, draw) arrays in Results/paper/enc_<partition>_H<H>_d<draw>.npz.
Stage 'importance': single XGBoost (196 features) refitted per horizon and draw on the encounter training rows;
    XGBoost 'weight' importance (number of splits, as in Barton et al. Table 3) averaged over the 5 draws.
Stage 'hourly': operating points for the hourly 6-h-ahead task (1 h and 4 h grids) incl. comparators.
Stage 'tables': writes table2_barton_protocol.csv, table4_hourly.csv and figure data.

Comparators use only the six vitals (no WBC / mentation / supplemental O2 available to a wearable):
    SIRS-vitals  HR > 90, RR > 20, T > 38 or < 36                      (0-3)
    qSOFA-vitals RR >= 22, SBP <= 100                                 (0-2)
    NEWS2-vitals RR, SpO2 (scale 1), SBP, HR, temperature sub-scores  (consciousness and air assumed normal)
    MEWS-vitals  SBP, HR, RR, temperature sub-scores                   (AVPU assumed alert)
Operating point: the threshold whose sensitivity is closest to 0.80 (as in Barton et al. Table 2).
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "Results" / "ultimate_full"; OUT = ROOT / "Results" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
VITALS = ["heart_rate", "resprate", "spo2", "temp_c", "sbp", "dbp"]
NONFEAT = {"stay_id", "subject_id", "charttime", "intime", "sepsis3_time", "is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
           "target_resp", "target_uri", "target_other", "is_test", "time_since_ICU_admit_hours"}
HOR, DRAWS = (0, 24, 48), 5


# ------------------------------------------------------------------ copied from barton_protocol_full.py (must stay identical)
def encounters(d, H, seed):
    d = d.copy()
    d["pos"] = d.groupby("stay_id").cumcount()
    rng = np.random.default_rng(seed)
    last = d.groupby("stay_id").charttime.transform("max")
    sep = d.is_sepsis_stay.values == 1
    elig_neg = (~sep) & (d.time_since_ICU_admit_hours.values >= 7)
    neg_rows = d[elig_neg].groupby("stay_id").apply(lambda g: g.charttime.values[rng.integers(len(g))])
    onset = pd.Series(np.where(sep, last.values, np.datetime64("NaT")), index=d.index)
    onset[~sep] = d.loc[~sep, "stay_id"].map(neg_rows).values
    d["index_time"] = pd.to_datetime(onset.values) - pd.Timedelta(hours=H)
    e = d[(d.charttime == d.index_time) & (d.pos >= 2)].copy()
    e["y"] = e.is_sepsis_stay.astype(int)
    return e


def xgb(y, depth=4, lr=0.05, trees=2000, es=True):
    spw = float((y == 0).sum()) / max(float(y.sum()), 1.0)
    return XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, tree_method="hist", eval_metric="auc", early_stopping_rounds=50 if es else None,
                         n_jobs=-1, random_state=0, verbosity=0)


def fit_xgb_es(X, y, groups, depth=4, lr=0.05):
    a, v = next(GroupShuffleSplit(1, test_size=0.15, random_state=1).split(X, y, groups))
    m = xgb(y[a], depth, lr); m.fit(X[a], y[a], eval_set=[(X[v], y[v])], verbose=False); return m


def lr_(): return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
def addvar(X): return np.column_stack([X, X.var(axis=1)])


# ------------------------------------------------------------------ comparators (vitals only)
def comparators(df):
    hr, rr, sp, t, sb = (df[c].values for c in ("heart_rate", "resprate", "spo2", "temp_c", "sbp"))
    n = lambda x: np.nan_to_num(x, nan=-1e9)          # missing -> condition false
    sirs = (n(hr) > 90).astype(int) + (n(rr) > 20) + ((n(t) > 38) | ((t < 36) & ~np.isnan(t)))
    qsofa = (n(rr) >= 22).astype(int) + ((sb <= 100) & ~np.isnan(sb))
    def band(x, edges, pts):                          # edges ascending, pts len = len(edges)+1 ; nan -> 0
        out = np.array(pts)[np.searchsorted(edges, x, side="right")]
        return np.where(np.isnan(x), 0, out)
    # NEWS2 (RCP 2017): RR <=8:3, 9-11:1, 12-20:0, 21-24:2, >=25:3 ; SpO2 scale1 <=91:3, 92-93:2, 94-95:1, >=96:0
    # SBP <=90:3, 91-100:2, 101-110:1, 111-219:0, >=220:3 ; HR <=40:3, 41-50:1, 51-90:0, 91-110:1, 111-130:2, >=131:3
    # T <=35.0:3, 35.1-36.0:1, 36.1-38.0:0, 38.1-39.0:1, >=39.1:2
    news = (band(np.round(rr), [8, 11, 20, 24], [3, 1, 0, 2, 3]) + band(np.round(sp), [91, 93, 95], [3, 2, 1, 0])
            + band(np.round(sb), [90, 100, 110, 219], [3, 2, 1, 0, 3]) + band(np.round(hr), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
            + band(np.round(t, 1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2]))
    # MEWS (Subbe 2001): SBP <=70:3, 71-80:2, 81-100:1, 101-199:0, >=200:2 ; HR <=40:2, 41-50:1, 51-100:0, 101-110:1, 111-129:2, >=130:3
    # RR <9:2, 9-14:0, 15-20:1, 21-29:2, >=30:3 ; T <35:2, 35-38.4:0, >=38.5:2
    mews = (band(np.round(sb), [70, 80, 100, 199], [3, 2, 1, 0, 2]) + band(np.round(hr), [40, 50, 100, 110, 129], [2, 1, 0, 1, 2, 3])
            + band(np.round(rr), [8, 14, 20, 29], [2, 0, 1, 2, 3]) + band(np.round(t, 1), [34.9, 38.4], [2, 0, 2]))
    return {"SIRS (vitals)": sirs.astype(float), "qSOFA (vitals)": qsofa.astype(float), "NEWS2 (vitals)": news.astype(float),
            "MEWS (vitals)": mews.astype(float)}


def op_point(y, p, target=0.80):
    fpr, tpr, thr = roc_curve(y, p)
    ok = (fpr < 1) & (tpr > 0)                      # exclude the trivial 'flag nobody' / 'flag everybody' points
    fpr, tpr = fpr[ok], tpr[ok]; i = int(np.argmin(np.abs(tpr - target)))
    se, sp = tpr[i], 1 - fpr[i]
    lrp = se / max(1 - sp, 1e-12); lrn = (1 - se) / max(sp, 1e-12)
    return dict(sens=se, spec=sp, DOR=lrp / max(lrn, 1e-12), LRpos=lrp, LRneg=lrn, alerts_per_TP=((1 - sp) * (y == 0).sum()) / max(se * y.sum(), 1))


def load_light():
    cols = ["stay_id", "charttime", "time_since_ICU_admit_hours", "is_sepsis_stay", "is_test", "subject_id"] + VITALS
    d = pd.read_parquet(RES / "ckpt_1h" / "prepared.parquet", columns=cols)
    return d.sort_values(["stay_id", "charttime"]).reset_index(drop=True)


def stage_enc(part):
    d = load_light(); ck = RES / "ckpt_barton_1h"
    nk = {"etiology": [("global", None)] + [("etiology", None)] * 3,
          "kmeans": [("global", None)] + [("kmeans5", None)] * 5 + [("kmeans6", None)] * 6}[part]
    logged = {(r["H"], r["draw"], r["model"]): r["auroc"] for f in ck.glob(f"rows_{part}_H*_d*.json") for r in json.load(open(f))}
    for H in HOR:
        for s in range(DRAWS):
            f = OUT / f"enc_{part}_H{H}_d{s}.npz"
            if f.exists():
                continue
            e = encounters(d, H, s); tr, te = e[e.is_test == 0].reset_index(drop=True), e[e.is_test == 1].reset_index(drop=True)
            y, yt = tr.y.values, te.y.values; tag = f"H{H}_d{s}"
            fo = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
            L1o, L1t, fam = [], [], []
            for si, (name, _) in enumerate(nk):
                for learner in ("rf", "xgb"):
                    key = f"{tag}_{part}_s{si}_{name}_{learner}"
                    L1o.append(np.load(ck / f"{key}_oof.npy")); L1t.append(np.load(ck / f"{key}_test.npy")); fam.append(name)
            L1o, L1t, fam = np.column_stack(L1o), np.column_stack(L1t), np.array(fam)
            L2o, L2t = [], []
            for fm in [x for x in dict.fromkeys(fam) if x != "global"] or ["global"]:
                cols = np.where((fam == "global") | (fam == fm))[0]; Xo, Xt = addvar(L1o[:, cols]), addvar(L1t[:, cols])
                for mk in (lr_, lambda: xgb(y, 3, 0.05, 200, es=False)):
                    o = np.zeros(len(tr))
                    for a, b in fo: o[b] = mk().fit(Xo[a], y[a]).predict_proba(Xo[b])[:, 1]
                    L2o.append(o); L2t.append(mk().fit(Xo, y).predict_proba(Xt)[:, 1])
            ult = lr_().fit(addvar(np.column_stack(L2o)), y).predict_proba(addvar(np.column_stack(L2t)))[:, 1]
            single, bart = np.load(ck / f"{tag}_single.npy"), np.load(ck / f"{tag}_barton31.npy")
            chk = {m: (roc_auc_score(yt, p), logged.get((H, s, m))) for m, p in
                   ((f"ultimate_{part} (3L)", ult), ("single_xgb (196 feats)", single), ("barton_lag31 (XGB)", bart))}
            for m, (a, b) in chk.items():
                assert b is None or abs(a - b) < 5e-3, f"reproduction mismatch {m} H{H} d{s}: {a} vs {b}"
            comp = comparators(te)
            np.savez(f, y=yt, stay_id=te.stay_id.values, ultimate=ult, single=single, barton31=bart, **{k: v for k, v in comp.items()})
            print(f"{part} H={H} d={s}: n={len(te)} pos={yt.sum()} reproduced " + ", ".join(f"{m.split(chr(32))[0]} {a:.4f} (logged {b:.4f})" for m, (a, b) in chk.items()), flush=True)


def stage_importance():
    import pyarrow.parquet as pq
    d = load_light(); rows = []
    pf = pq.ParquetFile(RES / "ckpt_1h" / "prepared.parquet")
    F_all = [c for c in pf.schema_arrow.names if c not in NONFEAT]
    for H in HOR:
        todo = [s for s in range(DRAWS) if not (OUT / f"imp_H{H}_d{s}.json").exists()]
        if todo:
            E = {s: encounters(d, H, s)[["stay_id", "charttime", "y"]] for s in todo}
            keys = pd.concat(E.values()).drop_duplicates(["stay_id", "charttime"])[["stay_id", "charttime"]]
            parts = [b.to_pandas().merge(keys, on=["stay_id", "charttime"])
                     for b in pf.iter_batches(batch_size=100_000, columns=["stay_id", "charttime", "is_test", "subject_id"] + F_all)]
            X = pd.concat(parts)
            for s in todo:
                x = X.merge(E[s], on=["stay_id", "charttime"]); tr = x[x.is_test == 0]
                Xtr = tr[F_all].fillna(tr[F_all].median()).values
                mdl = fit_xgb_es(Xtr, tr.y.values, tr.subject_id.values)
                sc = mdl.get_booster().get_score(importance_type="weight")
                json.dump({F_all[int(k[1:])]: v for k, v in sc.items()}, open(OUT / f"imp_H{H}_d{s}.json", "w"))
                print(f"importance H={H} d={s} done (train rows {len(tr)})", flush=True)
        imp = {}
        for s in range(DRAWS):
            for k, v in json.load(open(OUT / f"imp_H{H}_d{s}.json")).items(): imp[k] = imp.get(k, 0) + v / DRAWS
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:10]
        rows += [dict(H=H, rank=i + 1, feature=k, mean_splits=round(v, 1)) for i, (k, v) in enumerate(top)]
    pd.DataFrame(rows).to_csv(OUT / "table3_feature_importance.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))


def stage_hourly():
    rows = []
    for g in ("1h", "4h"):
        p = pd.read_parquet(RES / f"preds_kmeans_{g}.parquet"); e = pd.read_parquet(RES / f"preds_etiology_{g}.parquet")
        v = pd.read_parquet(RES / f"ckpt_{g}" / "prepared.parquet", columns=["stay_id", "charttime"] + VITALS)
        p = p.merge(v, on=["stay_id", "charttime"], how="left"); assert len(p) == len(e) and p[VITALS[0]].notna().mean() > 0.5
        y = p["is_sepsis_6h"].values
        M = {"Stacked ensemble (k-means, 3 layers)": p["ULTIMATE 3L [kmeans] LR meta"].values,
             "Single XGBoost (196 features)": p["single XGBoost (global stream, 196 feats)"].values,
             "Previous design (etiology NOSE-RF, XGB meta)": e["CURRENT design (global+etiology NOSE-RF, XGB meta)"].values}
        M.update(comparators(p))
        for k, s in M.items():
            rows.append(dict(grid=g, model=k, auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), lift=average_precision_score(y, s) / y.mean(),
                             prevalence=y.mean(), **op_point(y, s)))
    r = pd.DataFrame(rows); r.round(4).to_csv(OUT / "table4_hourly.csv", index=False); print(r.round(3).to_string(index=False))


def stage_tables():
    rows, roc = [], {}
    for part in ("kmeans", "etiology"):
        for H in HOR:
            for s in range(DRAWS):
                z = np.load(OUT / f"enc_{part}_H{H}_d{s}.npz"); y = z["y"]
                models = {"ultimate": z["ultimate"], "single": z["single"], "barton31": z["barton31"]}
                if part == "kmeans" and H == 0:
                    models.update({k: z[k] for k in ("SIRS (vitals)", "qSOFA (vitals)", "NEWS2 (vitals)", "MEWS (vitals)")})
                for m, p in models.items():
                    if part == "etiology" and m != "ultimate":
                        continue
                    name = f"ultimate_{part}" if m == "ultimate" else m
                    rows.append(dict(H=H, draw=s, model=name, n=len(y), pos=int(y.sum()), auroc=roc_auc_score(y, p), **op_point(y, p)))
                    if s == 0 and part == "kmeans":
                        fpr, tpr, _ = roc_curve(y, p); idx = np.linspace(0, len(fpr) - 1, min(400, len(fpr))).astype(int)
                        roc[f"{name}|H{H}"] = (fpr[idx].tolist(), tpr[idx].tolist())
    r = pd.DataFrame(rows)
    s = r.groupby(["model", "H"]).agg(n=("n", "mean"), pos=("pos", "mean"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                      sens=("sens", "mean"), spec=("spec", "mean"), DOR=("DOR", "mean"), LRpos=("LRpos", "mean"),
                                      LRneg=("LRneg", "mean"), alerts_per_TP=("alerts_per_TP", "mean")).round(4)
    s.to_csv(OUT / "table2_barton_protocol.csv"); json.dump(roc, open(OUT / "fig2_roc_data.json", "w"))
    print(s.to_string())


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "enc": stage_enc(sys.argv[2])
    elif st == "importance": stage_importance()
    elif st == "hourly": stage_hourly()
    elif st == "tables": stage_tables()
