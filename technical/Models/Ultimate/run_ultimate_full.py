#!/usr/bin/env python3
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  run_ultimate_full.py
#  Stage 4 - Modeling (journal paper: three-layer stacks)
#
#  PURPOSE
#    Main pipeline of the journal paper. Builds the 196 wearable features, the
#    patient-level 80/20 split, and the three-layer stack with etiology streams
#    (--partition etiology) or k-means physiology streams (--partition kmeans),
#    on the 1-hour or 4-hour grid (--grid). Every fit is checkpointed, so an
#    interrupted run resumes.
#
#  INPUTS
#    technical/Data/All engineered features/*.parquet, Data/stay_subject_map.parquet, Data/Etiology/v8_dataset_1h_etiology_*.parquet
#  OUTPUTS
#    Results/ultimate_full/ (ckpt_<grid>/, preds_<partition>_<grid>.parquet, results_*.csv, comparison_<grid>.csv)
#
#  USER-EDITABLE SETTINGS
#    --out, --jobs, --sample (smoke tests), model hyper-parameters in fit_nose_rf / fit_xgb / xgb_small
#    Paths resolve relative to technical/ (ROOT = parents[2] of this file).
# ============================================================================

"""
run_ultimate_full.py  -  full-scale run of the "ultimate" stacked ensemble (POC sections 10-16)
=================================================================================================

What it builds (all choices fixed in advance, from the 10-40% POCs):

  Data      MIMIC-IV model cohort (58,970 ICU stays), 1-hour grid (or --grid 4h), 6-hour target.
            NEW PATIENT-LEVEL SPLIT: train and test files are pooled and re-split 80/20 by subject_id
            (the original split was by stay; 29% of test stays had the same patient in train).
  Features  96 engineered features (production)  +  24h history statistics (6/12/24h rolling mean/SD/min/max/
            slope, deviation from first-6h baseline)  +  shock proxies (hours MAP<65, hours shock index>1,
            min MAP, max shock index over 24h). All computed causally from the six vitals = wearable-only.
  Layer 1   Specialist streams, each with TWO base learners:
              * NOSE random forest (production V13b config: 5 balanced subsets x RF 150 trees, depth 8, leaf 35)
                on the 96 production features
              * XGBoost (hist, depth 8, lr 0.03, early stopping on an internal 10% patient-grouped split)
                on all 196 features
            Streams:  --partition etiology : global + respiratory/urinary/other
                      --partition kmeans   : global + K-means-5 + K-means-6 physiology clusters (fitted on the
                                             training septic stays' pre-onset profile only)
                      --partition both     : global + etiology + K-means-5 + K-means-6
            Out-of-fold predictions on train (StratifiedGroupKFold by patient), refit-on-all for test.
  Layer 2   Per partition family: {logistic regression, small XGBoost} on [global + family streams, both
            learners] + cross-stream variance, fitted out-of-fold (patient-grouped) -> no in-sample leakage.
  Layer 3   Logistic-regression meta (primary) and XGBoost meta (secondary).
  References on the SAME split: current design (global + etiology NOSE-RF streams -> production XGB meta,
            only available once the etiology run exists) and a single XGBoost (the global XGB stream).

Everything is checkpointed under --out/ckpt, so an interrupted run resumes where it stopped.

Usage (from the b2 folder, in the venv that has xgboost/scikit-learn/pandas/pyarrow):
    python technical/Models/Ultimate/run_ultimate_full.py --partition etiology
    python technical/Models/Ultimate/run_ultimate_full.py --partition kmeans
    python technical/Models/Ultimate/run_ultimate_full.py --compare          # after both runs
Quick smoke test (2% of patients, minutes):
    python technical/Models/Ultimate/run_ultimate_full.py --partition etiology --sample 0.02 --out /tmp/ult_smoke
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]          # .../b2/technical
DATA = ROOT / "Data"
VITALS = ["heart_rate", "resprate", "spo2", "temp_c", "sbp", "dbp"]
NONFEAT = {"stay_id", "subject_id", "charttime", "intime", "sepsis3_time", "is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
           "target_resp", "target_uri", "target_other"}
T = "is_sepsis_6h"
T0 = time.time()


def save_npy(f, arr):
    """atomic save: write to a temp file, then rename (a crash never leaves a half-written checkpoint)."""
    tmp = Path(str(f) + ".tmp.npy"); np.save(tmp, arr); os.replace(tmp, f)


def save_npz(f, **kw):
    tmp = Path(str(f) + ".tmp.npz"); np.savez(tmp, **kw); os.replace(tmp, f)


def log(msg):
    el = time.time() - T0
    print(f"[{time.strftime('%H:%M:%S')} +{el/3600:5.2f}h] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------- data
def read_engineered(grid, keep_subjects=None, stay_map=None):
    parts = []
    for split in ("train", "test"):
        f = DATA / "All engineered features" / f"Dataset_all_engineered_{grid}_{split}.parquet"
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=250_000):
            x = b.to_pandas()
            if "stay_id" not in x.columns:
                x = x.reset_index()
            x["stay_id"] = x["stay_id"].astype("int64")
            if keep_subjects is not None:
                x = x[x.stay_id.map(stay_map).isin(keep_subjects)]
            for c in x.select_dtypes("float64").columns:
                x[c] = x[c].astype("float32")
            parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    d["charttime"] = pd.to_datetime(d["charttime"])
    return d.sort_values(["stay_id", "charttime"]).reset_index(drop=True)


def add_features(d, rows_per_hour):
    """24h history stats + first-6h baseline deviation + shock proxies, causal, within stay."""
    g = d.groupby("stay_id", sort=False)
    new = {}
    for v in VITALS:
        for hrs in (6, 12, 24):
            w = max(int(round(hrs * rows_per_hour)), 1)
            r = g[v].rolling(w, min_periods=1)
            new[f"h{hrs}_mean_{v}"] = r.mean().values
            new[f"h{hrs}_sd_{v}"] = r.std().fillna(0).values
            new[f"h{hrs}_min_{v}"] = r.min().values
            new[f"h{hrs}_max_{v}"] = r.max().values
            new[f"h{hrs}_slope_{v}"] = (d[v] - g[v].shift(w)).values
        n = g.cumcount().values
        k = max(int(round(6 * rows_per_hour)), 1)
        expm = g[v].expanding().mean().values
        frozen = pd.Series(np.where(n == k - 1, expm, np.nan), index=d.index).groupby(d.stay_id).ffill().values
        base = np.where(n < k, expm, frozen)
        base = np.where(np.isnan(base), expm, base)
        new[f"dev_baseline_{v}"] = (d[v].values - base)
    mp = (d.sbp + 2 * d.dbp) / 3
    si = d.heart_rate / d.sbp
    w24 = max(int(round(24 * rows_per_hour)), 1)
    tmp = pd.DataFrame({"stay_id": d.stay_id, "lo": (mp < 65).astype("float32"), "hi": (si > 1).astype("float32"), "mp": mp, "si": si})
    r = tmp.groupby("stay_id", sort=False)[["lo", "hi", "mp", "si"]].rolling(w24, min_periods=1)
    new["shk_hours_map_lt65_24h"] = (r.lo.sum() / rows_per_hour).values
    new["shk_hours_si_gt1_24h"] = (r.hi.sum() / rows_per_hour).values
    new["shk_min_map_24h"] = r.mp.min().values
    new["shk_max_si_24h"] = r.si.max().values
    add = pd.DataFrame(new, index=d.index).astype("float32")
    return pd.concat([d, add], axis=1)


def etiology_labels():
    e = pd.concat([pd.read_parquet(DATA / "Etiology" / f"v8_dataset_1h_etiology_{s}.parquet",
                                   columns=["stay_id", "target_resp", "target_uri", "target_other"]).groupby("stay_id").max()
                   for s in ("train", "test")])
    return e[~e.index.duplicated()]


def build(args, ck):
    """Returns tr, te (DataFrames with ids/labels), Xr/Xx (96 RF features) and Xg/Xgt (all features), groups."""
    f = ck / "prepared.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        log(f"loaded prepared data {d.shape}")
    else:
        mp = pd.read_parquet(DATA / "stay_subject_map.parquet").set_index("stay_id").subject_id
        keep = None
        if args.sample < 1.0:
            subj = np.sort(mp.unique()); rng = np.random.default_rng(0)
            keep = set(rng.choice(subj, int(len(subj) * args.sample), replace=False))
        d = read_engineered(args.grid, keep, mp)
        d["subject_id"] = d.stay_id.map(mp).astype("int64")
        log(f"read {len(d):,} rows, {d.stay_id.nunique():,} stays, {d.subject_id.nunique():,} patients")
        d = add_features(d, rows_per_hour=1.0 if args.grid == "1h" else 0.25)
        e = etiology_labels()
        for c in ["target_resp", "target_uri", "target_other"]:
            d[c] = d.stay_id.map(e[c]).fillna(0).astype("int8")
        # patient-level 80/20 split, fixed seed
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
        subj = d.groupby("subject_id").is_sepsis_stay.max()
        a, b = next(gss.split(subj.index.values, subj.values, groups=subj.index.values))
        test_subj = set(subj.index.values[b])
        d["is_test"] = d.subject_id.isin(test_subj).astype("int8")
        tmp = Path(str(f) + ".tmp"); d.to_parquet(tmp, index=False); os.replace(tmp, f)
        log(f"features built, saved {f}")
    feats_all = [c for c in d.columns if c not in NONFEAT and c != "is_test"]
    feats_rf = [c for c in feats_all if not c.startswith(("h6_", "h12_", "h24_", "dev_baseline_", "shk_"))]
    tr = d[d.is_test == 0].reset_index(drop=True); te = d[d.is_test == 1].reset_index(drop=True)
    del d; gc.collect()
    med = tr[feats_all].median()
    tr[feats_all] = tr[feats_all].fillna(med); te[feats_all] = te[feats_all].fillna(med)
    log(f"train {len(tr):,} rows / {tr.stay_id.nunique():,} stays / {tr.subject_id.nunique():,} patients "
        f"(septic stays {tr.groupby('stay_id').is_sepsis_stay.max().sum():,}); test {len(te):,} rows / "
        f"{te.stay_id.nunique():,} stays (septic {te.groupby('stay_id').is_sepsis_stay.max().sum():,}); "
        f"features RF {len(feats_rf)}, XGB {len(feats_all)}; prevalence train {tr[T].mean():.4f} test {te[T].mean():.4f}")
    assert not (set(tr.subject_id) & set(te.subject_id)), "patient overlap between train and test"
    return tr, te, feats_rf, feats_all


# ------------------------------------------------------------------------------------------------------------- streams
def cluster_groups(tr, feats_rf, k, ck):
    f = ck / f"groups_kmeans{k}.json"
    if f.exists():
        return {int(a): b for a, b in json.load(open(f)).items()}
    sep = tr[(tr.is_sepsis_stay == 1) & (tr[T] == 1)]
    prof = sep.groupby("stay_id")[feats_rf].mean()
    Z = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(prof))
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    g = {int(s): int(l) for s, l in zip(prof.index, lab)}
    tmp = Path(str(f) + ".tmp"); json.dump({str(a): b for a, b in g.items()}, open(tmp, "w")); os.replace(tmp, f)
    log(f"K-means-{k} groups (septic train stays): {pd.Series(list(g.values())).value_counts().sort_index().to_dict()}")
    return g


def stream_list(args, tr, feats_rf, ck):
    S = [("global", 0, None)]
    if args.partition in ("etiology", "both"):
        for k, c in enumerate(["target_resp", "target_uri", "target_other"]):
            S.append(("etiology", k, c))
    if args.partition in ("kmeans", "both"):
        for kk in (5, 6):
            g = cluster_groups(tr, feats_rf, kk, ck)
            for k in range(kk):
                S.append((f"kmeans{kk}", k, g))
    return S


def stream_y(df, spec):
    fam, k, info = spec
    y = df[T].values
    if fam == "global":
        return y
    if fam == "etiology":
        return ((y == 1) & (df[info].values == 1)).astype(int)
    gid = df.stay_id.map(info).fillna(-1).values
    return ((y == 1) & (gid == k)).astype(int)


def fit_nose_rf(X, rows, y, stays, args):
    """X = full train matrix; rows = training row indices (memory: only the NOSE subset rows are ever copied)."""
    y, stays = y[rows], stays[rows]
    pos = np.unique(stays[y == 1]); neg = np.random.RandomState(42).permutation(np.setdiff1d(np.unique(stays), pos))
    ms = []
    for i in range(args.nose):
        sub = neg[i * len(pos):(i + 1) * len(pos)]
        if len(sub) == 0 or len(pos) == 0:
            break
        msk = np.isin(stays, np.concatenate([pos, sub]))
        ms.append(RandomForestClassifier(n_estimators=args.rf_trees, max_depth=8, min_samples_leaf=35, max_features="sqrt",
                                         random_state=42 + i, n_jobs=args.jobs).fit(X[rows[msk]], y[msk]))
        gc.collect()
    return ms


def pred_nose(ms, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0) if ms else np.zeros(len(X))


def fit_xgb(X, rows, y, groups, args):
    """X = full train matrix; rows = training row indices (memory: rows are copied once, not twice)."""
    y, groups = y[rows], groups[rows]
    if y.sum() == 0:
        return None
    a, v = next(GroupShuffleSplit(1, test_size=0.1, random_state=1).split(rows, y, groups))   # split depends on groups only
    spw = float((y[a] == 0).sum()) / max(float(y[a].sum()), 1.0)
    m = XGBClassifier(n_estimators=args.xgb_trees, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5,
                      min_child_weight=5, reg_lambda=10, scale_pos_weight=spw, tree_method="hist", eval_metric="auc",
                      early_stopping_rounds=50, n_jobs=args.jobs, random_state=42, verbosity=0)
    Xa, Xv = X[rows[a]], X[rows[v]]
    m.fit(Xa, y[a], eval_set=[(Xv, y[v])], verbose=False)
    del Xa, Xv; gc.collect()
    return m


def layer1(args, tr, te, feats_rf, feats_all, streams, folds, ck):
    Xg = tr[feats_all].to_numpy(dtype="float32"); Xgt = te[feats_all].to_numpy(dtype="float32")
    ri = [feats_all.index(c) for c in feats_rf]
    Xr = np.ascontiguousarray(Xg[:, ri]); Xrt = np.ascontiguousarray(Xgt[:, ri])
    # free the feature columns held in the DataFrames (only ids/labels are needed after this point)
    tr.drop(columns=feats_all, inplace=True); te.drop(columns=feats_all, inplace=True); gc.collect()
    log(f"feature matrices built (float32): train {Xg.nbytes/1e9:.2f} GB + RF {Xr.nbytes/1e9:.2f} GB; DataFrame features freed")
    stays, subj = tr.stay_id.values, tr.subject_id.values
    allrows = np.arange(len(tr))
    n_jobs_total = len(streams) * 2 * (len(folds) + 1); done = 0
    out = {}
    for spec in streams:
        y = stream_y(tr, spec)
        for learner in ("rf", "xgb"):
            key = f"{spec[0]}_{spec[1]}_{learner}"
            oof = np.zeros(len(tr), dtype="float32")
            for i, ab in enumerate(folds + [None]):
                a, b = ab if ab is not None else (None, None)
                f = ck / f"L1_{key}_{'final' if b is None else i}.npy"
                if not f.exists():
                    t0 = time.time()
                    rows = allrows if b is None else a
                    if learner == "rf":
                        ms = fit_nose_rf(Xr, rows, y, stays, args)
                        p = pred_nose(ms, Xrt if b is None else Xr[b]); del ms
                    else:
                        m = fit_xgb(Xg, rows, y, subj, args)
                        Xe = Xgt if b is None else Xg[b]
                        p = m.predict_proba(Xe)[:, 1] if m is not None else np.zeros(len(Xe))
                        del m, Xe
                    gc.collect()
                    save_npy(f, p.astype("float32"))
                    done += 1
                    log(f"L1 {key} {'final' if b is None else 'fold ' + str(i)}  ({time.time()-t0:.0f}s)  [{done}/{n_jobs_total} new fits]")
                else:
                    done += 1
                if b is not None:
                    oof[b] = np.load(f)
            out[key] = (oof, np.load(ck / f"L1_{key}_final.npy"))
            gc.collect()
    return out


def addvar(X):
    return np.column_stack([X, X.var(axis=1)])


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))


def xgb_small(y):
    spw = float((y == 0).sum()) / max(float(y.sum()), 1.0)
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=50, scale_pos_weight=spw, tree_method="hist", n_jobs=-1, random_state=0, verbosity=0)


def layer2(L1, tr, args, ck, families):
    y = tr[T].values
    folds2 = list(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(tr, tr.is_sepsis_stay, tr.subject_id))
    O, Tt, names = [], [], []
    for fam in families:
        cols = [k for k in L1 if k.startswith("global_")] + [k for k in L1 if k.startswith(fam + "_")]
        Xo = addvar(np.column_stack([L1[k][0] for k in cols])); Xt = addvar(np.column_stack([L1[k][1] for k in cols]))
        for name, mk in (("lr", lr), ("xgb", lambda: xgb_small(y))):
            f = ck / f"L2_{fam}_{name}.npz"
            if not f.exists():
                o = np.zeros(len(tr))
                for a, b in folds2:
                    o[b] = mk().fit(Xo[a], y[a]).predict_proba(Xo[b])[:, 1]
                t = mk().fit(Xo, y).predict_proba(Xt)[:, 1]
                save_npz(f, o=o, t=t); log(f"L2 {fam} {name} done")
            z = np.load(f); O.append(z["o"]); Tt.append(z["t"]); names.append(f"{fam}:{name}")
    return np.column_stack(O), np.column_stack(Tt), names


# ---------------------------------------------------------------------------------------------------------- evaluation
def recall_at_fa(p, y, fa):
    o = np.argsort(-p); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    ok = fp <= fa * np.maximum(tp, 1)
    return float(tp[ok].max() / yy.sum()) if ok.any() else 0.0


def metrics(y, p):
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), lift=average_precision_score(y, p) / y.mean(),
                sens_at_10FA=recall_at_fa(p, y, 10), sens_at_20FA=recall_at_fa(p, y, 20))


class WScorer:
    """AUROC under integer cluster (patient) weights, ties handled like sklearn - fast bootstrap."""
    def __init__(self, p, y, code):
        u, inv = np.unique(-p, return_inverse=True)
        self.inv, self.y, self.code, self.nu = inv, y.astype(bool), code, len(u)

    def auc(self, sw):
        w = sw[self.code]
        pw = np.bincount(self.inv[self.y], w[self.y], self.nu); nw = np.bincount(self.inv[~self.y], w[~self.y], self.nu)
        return 1 - np.sum(pw * (np.cumsum(nw) - nw + 0.5 * nw)) / (pw.sum() * nw.sum())


def bootstrap_table(te, P, ref_name, B):
    y = te[T].values
    pt = te.groupby("subject_id").is_sepsis_stay.max()
    code = te.subject_id.map({s: i for i, s in enumerate(pt.index)}).values
    pos, neg = np.where(pt.values == 1)[0], np.where(pt.values == 0)[0]; rng = np.random.default_rng(0)
    W = [np.bincount(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]), minlength=len(pt)).astype(float)
         for _ in range(B)]
    S = {k: WScorer(p, y, code) for k, p in P.items()}
    A = {k: np.array([s.auc(w) for w in W]) for k, s in S.items()}
    rows = []
    for k, p in P.items():
        r = dict(model=k, **metrics(y, p), auroc_lo=np.quantile(A[k], .025), auroc_hi=np.quantile(A[k], .975))
        if ref_name in P and k != ref_name:
            d = A[k] - A[ref_name]
            r.update(d_auroc_vs_ref=float(np.median(d)), d_lo=float(np.quantile(d, .025)), d_hi=float(np.quantile(d, .975)))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------- main
def run(args):
    out = Path(args.out); ck = out / f"ckpt_{args.grid}"; ck.mkdir(parents=True, exist_ok=True)
    tr, te, feats_rf, feats_all = build(args, ck)
    folds = list(StratifiedGroupKFold(args.folds, shuffle=True, random_state=42).split(tr, tr.is_sepsis_stay, tr.subject_id))
    streams = stream_list(args, tr, feats_rf, ck)
    log(f"partition={args.partition}: {len(streams)} streams x 2 learners x {len(folds)+1} fits")
    L1 = layer1(args, tr, te, feats_rf, feats_all, streams, folds, ck)
    fams = [f for f in ("etiology", "kmeans5", "kmeans6") if any(k.startswith(f + "_") for k in L1)]
    L2o, L2t, names = layer2(L1, tr, args, ck, fams)
    y = tr[T].values
    P = {}
    P["single XGBoost (global stream, 196 feats)"] = L1["global_0_xgb"][1]
    P[f"ULTIMATE 3L [{args.partition}] LR meta"] = lr().fit(addvar(L2o), y).predict_proba(addvar(L2t))[:, 1]
    P[f"ULTIMATE 3L [{args.partition}] XGB meta"] = xgb_small(y).fit(addvar(L2o), y).predict_proba(addvar(L2t))[:, 1]
    if all(f"etiology_{k}_rf" in L1 for k in range(3)):
        cols = ["global_0_rf"] + [f"etiology_{k}_rf" for k in range(3)]
        Xo = addvar(np.column_stack([L1[c][0] for c in cols])); Xt = addvar(np.column_stack([L1[c][1] for c in cols]))
        spw = float((y == 0).sum()) / max(float(y.sum()), 1.0)
        prod = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=spw, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0)
        P["CURRENT design (global+etiology NOSE-RF, XGB meta)"] = prod.fit(Xo, y).predict_proba(Xt)[:, 1]
    preds = te[["stay_id", "subject_id", "charttime", T]].copy()
    for k, p in P.items():
        preds[k] = p
    preds.to_parquet(out / f"preds_{args.partition}_{args.grid}.parquet", index=False)
    (out / f"DONE_main_{args.partition}_{args.grid}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    ref = "CURRENT design (global+etiology NOSE-RF, XGB meta)" if "CURRENT design (global+etiology NOSE-RF, XGB meta)" in P else "single XGBoost (global stream, 196 feats)"
    tab = bootstrap_table(te, P, ref, args.B)
    tab.to_csv(out / f"results_{args.partition}_{args.grid}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    log("RESULTS (test = held-out patients; CI = patient-cluster bootstrap)\n" + tab.round(4).to_string(index=False))
    log(f"reference for deltas: {ref}")


def compare(args):
    out = Path(args.out); P = {}; base = None
    for part in ("etiology", "kmeans", "both"):
        f = out / f"preds_{part}_{args.grid}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            if base is None:
                base = d[["stay_id", "subject_id", "charttime", T]]
            assert (d.stay_id.values == base.stay_id.values).all(), "runs used different test rows"
            for c in d.columns:
                if c not in ("stay_id", "subject_id", "charttime", T) and c not in P:
                    P[c] = d[c].values
    te = base.copy()
    st = pd.read_parquet(out / f"ckpt_{args.grid}" / "prepared.parquet", columns=["stay_id", "is_sepsis_stay"]).drop_duplicates("stay_id").set_index("stay_id").is_sepsis_stay
    te["is_sepsis_stay"] = te.stay_id.map(st).values
    ref = "CURRENT design (global+etiology NOSE-RF, XGB meta)"
    tab = bootstrap_table(te, P, ref if ref in P else list(P)[0], args.B)
    tab.to_csv(out / f"comparison_{args.grid}.csv", index=False)
    (out / f"DONE_compare_{args.grid}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    log("COMPARISON\n" + tab.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["etiology", "kmeans", "both"], default="etiology")
    ap.add_argument("--grid", choices=["1h", "4h"], default="1h")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nose", type=int, default=5)
    ap.add_argument("--rf-trees", type=int, default=150)
    ap.add_argument("--xgb-trees", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of PATIENTS (for smoke tests)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "Results" / "ultimate_full"))
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
    compare(a) if a.compare else run(a)
