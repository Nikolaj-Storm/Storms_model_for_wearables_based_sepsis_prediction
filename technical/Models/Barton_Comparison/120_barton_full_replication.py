#!/usr/bin/env python3
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  120_barton_full_replication.py
#  Stage 5 - Evaluation (faithful re-implementation)
#
#  PURPOSE
#    Faithful re-implementation of Barton et al. (2019) on MIMIC-IV from raw
#    chart events (paper Appendix A, Supplementary Table S2). Details in the
#    docstring below.
#
#  INPUTS
#    Data/mimic_derived_vitals_cohort.parquet, Data/demographics_anchors.parquet, Data/stay_subject_map.parquet
#  OUTPUTS
#    Results/barton_replication/
#
#  USER-EDITABLE SETTINGS
#    --draws, --sample, --budget, DEPTHS / LRS grid
#    Paths resolve relative to technical/ (ROOT = parents[2] of this file).
# ============================================================================

"""
120_barton_full_replication.py  -  faithful re-implementation of Barton et al. (2019, Comput Biol Med)
"Evaluation of a machine learning algorithm for up to 48-hour advance prediction of sepsis using six vital signs"
on MIMIC-IV, starting from RAW chartevents (not from our engineered grids).

What is replicated (manuscript section in brackets)
  Cohort    adult ICU stays; >= 1 recording of EACH of the 6 vitals; stay 7-2000 h; septic onset (Sepsis-3, Seymour)
            >= 7 h and <= 2000 h after admission                                           [Data; Fig 1]
            (our extraction 04_master_data_extraction.sql applies exactly these filters; stay length = last vital time)
  Binning   hourly bins; mean of multiple measurements in an hour; empty hour -> carry-forward of the last value   [Outcomes]
            bins are anchored at the prediction time t:  (t-1,t], (t-2,t-1], (t-3,t-2]  -> strictly causal
  Labels    septic: onset = Sepsis-3 time.  non-septic: onset drawn from a continuous uniform distribution over the stay
            prediction time t = onset - H, H in {0, 24, 48}; encounter enters horizon H only if onset >= H
  Features  6 vitals x {t, t-1, t-2, (t)-(t-1), (t-1)-(t-2)} = 30 features                         [MLA]
  Model     XGBoost, early stopping; max depth and learning rate by 5-fold CV grid search on the training set
            (depth 3-8, lr 0.05-0.12 step 0.01, best mean AUROC)                                  [MLA]
  Split     80/20 random by ENCOUNTER; 10-fold CV on the 80%; each of the 10 models is also scored on the 20% hold-out;
            reported = mean (SD) over the 10 models                                                [MLA]
Not replicable: UCSF data (wards/ED, 91k encounters) - MIMIC-IV is ICU-only. Barton's BIDMC (MIMIC-III ICU) result is
external validation of a UCSF-trained model at onset only (0.89).

Variants (all re-use the hyper-parameters chosen for A):
  A  literal       as above (30 features, encounter split, negative onset ~ U[0, stay length])
  B  +age          31 features (age is reported as the most important feature in Barton's Table 3, but the text says 30)
  C  patient split 80/20 and CV folds grouped by PATIENT (MIMIC has repeat ICU stays of the same patient)
  D  symmetric     negative pseudo-onset ~ U[7 h, stay length], mirroring the 7 h rule applied to septic stays
Draws: the random negative onset is redrawn --draws times (seed 0..); grid search uses draw 0 of A.

Stages (checkpointed in Results/barton_replication/):  prep | grid <H> | final | report   (default: all)
"""
import argparse, json, os, sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "Data", ROOT / "Results" / "barton_replication"
SAMPLE = 1.0
T0 = time.time()
VIT = {"hr": [220045], "rr": [220210, 224690], "spo2": [220277], "temp": [223761, 223762], "sbp": [220179, 220050], "dbp": [220180, 220051]}
RANGE = {"hr": (20, 300), "rr": (1, 80), "spo2": (50, 100), "temp": (25, 45), "sbp": (40, 300), "dbp": (10, 200)}  # data-entry errors only
HORIZONS = (0, 24, 48)
DEPTHS, LRS = range(3, 9), [round(0.05 + 0.01 * i, 2) for i in range(8)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:5.1f}m] {m}", flush=True)


def save_json(f, obj):
    tmp = Path(str(f) + ".tmp"); json.dump(obj, open(tmp, "w")); os.replace(tmp, f)


# ----------------------------------------------------------------------------------------------------------- prep
def load_obs():
    import pyarrow.parquet as pq
    a = pd.read_parquet(DATA / "demographics_anchors.parquet")[["stay_id", "age", "intime", "sepsis3_time"]].set_index("stay_id")
    intime_ns = a.intime.astype("datetime64[ns]").astype("int64")
    inv = {i: k for k, (v, ids) in enumerate(VIT.items()) for i in ids}
    lo = np.array([RANGE[v][0] for v in VIT]); hi = np.array([RANGE[v][1] for v in VIT])
    parts, n0 = [], 0
    keep = None
    if SAMPLE < 1.0:
        ids = pq.read_table(DATA / "mimic_derived_vitals_cohort.parquet", columns=["stay_id"]).column(0).to_numpy()
        u = np.unique(ids); keep = set(np.random.default_rng(0).choice(u, int(len(u) * SAMPLE), replace=False))
    for b in pq.ParquetFile(DATA / "mimic_derived_vitals_cohort.parquet").iter_batches(batch_size=2_000_000):
        x = b.to_pandas(); n0 += len(x)
        if keep is not None:
            x = x[x.stay_id.isin(keep)]
        h = (x.charttime.astype("datetime64[ns]").astype("int64").values - x.stay_id.map(intime_ns).values) / 3.6e12
        vi = x.itemid.map(inv).values.astype("int8"); val = x.valuenum.values.astype("float64")
        f = x.itemid.values == 223761; val[f] = (val[f] - 32) * 5 / 9                # Fahrenheit -> Celsius
        k = (val >= lo[vi]) & (val <= hi[vi]) & (h >= 0)
        parts.append(pd.DataFrame({"stay_id": x.stay_id.values[k].astype("int64"), "is_sepsis": x.is_sepsis.values[k].astype("int8"),
                                   "h": h[k].astype("float64"), "vi": vi[k], "val": val[k].astype("float32")}))
    d = pd.concat(parts, ignore_index=True); del parts
    d["vital"] = pd.Categorical.from_codes(d.vi, list(VIT)); d = d.drop(columns="vi")
    log(f"observations {n0:,} -> {len(d):,} after unit conversion / range check")
    return d, a


def cohort(d, a):
    st = d.groupby("stay_id").agg(sep=("is_sepsis", "max"), los=("h", "max"))
    st = st.join(a.rename(columns={"sepsis3_time": "s3"}))
    nv = d.groupby("stay_id", observed=True).vital.nunique()
    st["onset_sep"] = (st.s3 - st.intime).dt.total_seconds() / 3600.0
    flow = {"stays with any vital (04_master_data_extraction)": len(st)}
    st = st[nv.reindex(st.index) == 6]; flow[">= 1 recording of each of the 6 vitals"] = len(st)
    keep = np.where(st.sep == 1, (st.onset_sep >= 7) & (st.onset_sep <= 2000), (st.los >= 7) & (st.los <= 2000))
    st = st[keep]; flow["stay / onset within 7-2000 h"] = len(st)
    flow["septic"] = int(st.sep.sum())
    subj = pd.read_parquet(DATA / "stay_subject_map.parquet").set_index("stay_id").subject_id
    miss = ~st.index.isin(subj.index)
    st["subject_id"] = st.index.map(subj)
    st.loc[miss, "subject_id"] = -st.index[miss]                       # (not in our map) -> own group
    return st, flow


def window_feats(d, enc):
    """enc: DataFrame(stay_id, t). Returns 6 vitals x {t0,t1,t2} hourly-bin means with carry-forward, strictly <= t."""
    out = {}
    sid = {s: i for i, s in enumerate(np.unique(enc.stay_id))}
    for v in VIT:
        x = d[(d.vital == v) & d.stay_id.isin(list(sid))]
        # integer seconds -> exact window boundaries (an observation at exactly t belongs to (t-1, t])
        si = x.stay_id.map(sid).values.astype("int64"); key = si * 10**8 + np.round(x.h.values * 3600).astype("int64")
        o = np.argsort(key, kind="stable"); key, val, si = key[o], x.val.values[o], si[o]
        cs = np.concatenate([[0.0], np.cumsum(val.astype("float64"))])
        qs = enc.stay_id.map(sid).values.astype("int64"); q = qs * 10**8 + np.round(enc.t.values * 3600).astype("int64")
        for lag in (0, 1, 2):
            e_ = q - lag * 3600; s_ = e_ - 3600
            i1 = np.searchsorted(key, s_, "right"); i2 = np.searchsorted(key, e_, "right")
            n = i2 - i1; m = np.where(n > 0, (cs[i2] - cs[i1]) / np.maximum(n, 1), np.nan)
            j = i1 - 1                                                            # last value at or before window start
            ok = (j >= 0) & (si[np.maximum(j, 0)] == qs)
            cf = np.where(ok, val[np.maximum(j, 0)], np.nan)
            out[f"{v}_t{lag}"] = np.where(n > 0, m, cf)
        out[f"{v}_d01"] = out[f"{v}_t0"] - out[f"{v}_t1"]; out[f"{v}_d12"] = out[f"{v}_t1"] - out[f"{v}_t2"]
    return pd.DataFrame(out, index=enc.index)


def build_encounters(d, st, H, seed, neg_from):
    rng = np.random.default_rng(seed)
    sep = st.sep.values == 1
    lo = np.full(len(st), float(neg_from)); hi = st.los.values.astype(float)
    neg_on = lo + rng.random(len(st)) * np.maximum(hi - lo, 0)
    onset = np.where(sep, st.onset_sep.values, neg_on)
    e = pd.DataFrame({"stay_id": st.index.values, "subject_id": st.subject_id.values, "y": sep.astype(int),
                      "age": st.age.values, "onset": onset})
    e = e[e.onset >= H].reset_index(drop=True)
    e["t"] = e.onset - H
    return pd.concat([e, window_feats(d, e)], axis=1)


def prep(a):
    d, a_ = load_obs(); st, flow = cohort(d, a_); save_json(OUT / "flow.json", flow); log(f"cohort {flow}")
    for var, neg_from in (("lit", 0.0), ("sym", 7.0)):
        for H in HORIZONS:
            for s in range(a.draws):
                f = OUT / f"enc_{var}_H{H}_d{s}.parquet"
                if f.exists():
                    continue
                e = build_encounters(d, st, H, s, neg_from); tmp = Path(str(f) + ".tmp"); e.to_parquet(tmp); os.replace(tmp, f)
                log(f"{var} H={H} draw={s}: {len(e):,} encounters, {e.y.sum():,} septic ({e.y.mean():.3%}); "
                    f"all-NaN feature rows {e.filter(regex='_t0$').isna().all(axis=1).mean():.2%}")


# ------------------------------------------------------------------------------------------------------- modelling
F30 = [f"{v}_{k}" for v in VIT for k in ("t0", "t1", "t2", "d01", "d12")]


def split(e, by_patient):
    if by_patient:
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(e, e.y, e.subject_id))
    else:
        a, b = train_test_split(np.arange(len(e)), test_size=0.2, random_state=0)
    return np.sort(a), np.sort(b)


def fit(X, y, depth, lr, groups=None, seed=0):
    """early stopping on an inner 10% of the training data (Barton: 'early stopping to prevent model overfit')."""
    if groups is None:
        a, v = train_test_split(np.arange(len(y)), test_size=0.1, random_state=seed, stratify=y)
    else:
        a, v = next(GroupShuffleSplit(1, test_size=0.1, random_state=seed).split(X, y, groups))
    m = XGBClassifier(n_estimators=3000, max_depth=depth, learning_rate=lr, tree_method="hist", eval_metric="auc",
                      early_stopping_rounds=50, n_jobs=a_jobs, random_state=seed, verbosity=0)
    m.fit(X[a], y[a], eval_set=[(X[v], y[v])], verbose=False)
    return m


def grid(H):
    e = pd.read_parquet(OUT / f"enc_lit_H{H}_d0.parquet"); tr, _ = split(e, False)
    X, y = e.loc[tr, F30].values, e.loc[tr, "y"].values
    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    f = OUT / f"grid_H{H}.json"; res = json.load(open(f)) if f.exists() else {}
    t_end = time.time() + a_budget
    for dep, lr in itertools.product(DEPTHS, LRS):
        k = f"{dep}_{lr}"
        if k in res:
            continue
        if time.time() > t_end:
            log(f"grid H={H}: {len(res)}/{len(DEPTHS)*len(LRS)} done (time budget) - run again"); return False
        res[k] = float(np.mean([roc_auc_score(y[b], fit(X[a], y[a], dep, lr).predict_proba(X[b])[:, 1]) for a, b in folds]))
        save_json(f, res)
    best = max(res, key=res.get); log(f"grid H={H} complete: best depth/lr {best}  CV AUROC {res[best]:.4f}")
    return True


def best_params(H):
    res = json.load(open(OUT / f"grid_H{H}.json")); assert len(res) == len(DEPTHS) * len(LRS), f"grid H={H} incomplete"
    dep, lr = max(res, key=res.get).split("_"); return int(dep), float(lr)


VARIANTS = {"A_literal": ("lit", F30, False), "B_plus_age": ("lit", F30 + ["age"], False),
            "C_patient_split": ("lit", F30, True), "D_symmetric_neg": ("sym", F30, False)}


def final():
    f = OUT / "final_rows.json"; rows = json.load(open(f)) if f.exists() else []
    done = {(r["variant"], r["H"], r["draw"], r["cv_model"]) for r in rows}
    t_end = time.time() + a_budget
    for var, (ev, F, byp) in VARIANTS.items():
        for H in HORIZONS:
            dep, lr = best_params(H)
            for s in range(a_draws):
                e = pd.read_parquet(OUT / f"enc_{ev}_H{H}_d{s}.parquet"); tr, te = split(e, byp)
                X, y, g = e.loc[tr, F].values, e.loc[tr, "y"].values, e.loc[tr, "subject_id"].values
                Xt, yt = e.loc[te, F].values, e.loc[te, "y"].values
                cv = GroupKFold(10).split(X, y, g) if byp else KFold(10, shuffle=True, random_state=0).split(X)
                for i, (a, b) in enumerate(cv):
                    if (var, H, s, i) in done:
                        continue
                    if time.time() > t_end:
                        log(f"final: {len(rows)} model fits done (time budget) - run again"); return False
                    m = fit(X[a], y[a], dep, lr, g[a] if byp else None, seed=i)
                    pv, pt = m.predict_proba(X[b])[:, 1], m.predict_proba(Xt)[:, 1]
                    rows.append(dict(variant=var, H=H, draw=s, cv_model=i, depth=dep, lr=lr, n_trees=int(m.best_iteration + 1),
                                     auroc_cvfold=float(roc_auc_score(y[b], pv)), auroc_test=float(roc_auc_score(yt, pt)),
                                     auprc_test=float(average_precision_score(yt, pt)), prev_test=float(yt.mean()),
                                     n_train=int(len(a)), n_test=int(len(yt)), pos_test=int(yt.sum())))
                    save_json(f, rows)
    log("final complete"); return True


def report():
    r = pd.DataFrame(json.load(open(OUT / "final_rows.json"))); r["lift_test"] = r.auprc_test / r.prev_test
    per_draw = r.groupby(["variant", "H", "draw"]).agg(auroc_test=("auroc_test", "mean"), sd_10models=("auroc_test", "std"),
                                                       auroc_cv=("auroc_cvfold", "mean"), lift=("lift_test", "mean"),
                                                       n_test=("n_test", "first"), pos_test=("pos_test", "first"))
    s = per_draw.groupby(["variant", "H"]).agg(auroc_test=("auroc_test", "mean"), sd_10models=("sd_10models", "mean"),
                                              sd_across_draws=("auroc_test", "std"), auroc_cv=("auroc_cv", "mean"),
                                              lift=("lift", "mean"), n_test=("n_test", "mean"), pos_test=("pos_test", "mean"))
    hp = {H: best_params(H) for H in HORIZONS}
    s["depth_lr"] = [f"{hp[h][0]} / {hp[h][1]}" for h in s.index.get_level_values("H")]
    s["barton_reported_UCSF"] = [{0: 0.88, 24: 0.84, 48: 0.83}[h] for h in s.index.get_level_values("H")]
    s.round(4).to_csv(OUT / "barton_replication_summary.csv"); per_draw.round(4).to_csv(OUT / "barton_replication_by_draw.csv")
    pd.set_option("display.width", 250)
    log("cohort flow " + json.dumps(json.load(open(OUT / "flow.json"))))
    log("BARTON FULL REPLICATION on MIMIC-IV (mean of 10 CV models on the 20% hold-out; Barton depth/lr chosen: 4/0.05, 3/0.12, 6/0.12)\n"
        + s.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["prep", "grid", "final", "report", "all"])
    ap.add_argument("H", nargs="?", type=int)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of stays (smoke test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=1e9, help="seconds per call before checkpoint-and-exit")
    a = ap.parse_args(); a_jobs, a_budget, a_draws = a.jobs, a.budget, a.draws
    SAMPLE = a.sample
    if a.out: OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("prep", "all"): prep(a)
    if a.stage == "grid": sys.exit(0 if grid(a.H) else 3)
    if a.stage == "all":
        for H in HORIZONS: grid(H)
    if a.stage in ("final", "all"):
        if not final(): sys.exit(3)
    if a.stage in ("report", "all"): report()
