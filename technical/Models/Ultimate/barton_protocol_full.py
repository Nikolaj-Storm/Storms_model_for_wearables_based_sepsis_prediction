#!/usr/bin/env python3
# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  barton_protocol_full.py
#  Stage 5 - Evaluation (encounter protocol)
#
#  PURPOSE
#    Evaluates the stacks, a single XGBoost and Barton et al.'s lag features
#    under the encounter protocol of Barton et al. (2019), one scored time per
#    stay at onset minus 0, 24 and 48 h, five draws of non-septic pseudo-onset
#    (paper Table 3).
#
#  INPUTS
#    Results/ultimate_full/ckpt_<grid>/prepared.parquet and groups_kmeans*.json (run run_ultimate_full.py first)
#  OUTPUTS
#    Results/ultimate_full/barton_protocol_<partition>_<grid>_*.csv
#
#  USER-EDITABLE SETTINGS
#    --draws, --grid, --partition
#    Paths resolve relative to technical/ (ROOT = parents[2] of this file).
# ============================================================================

"""
barton_protocol_full.py  -  the ultimate ensemble evaluated the way Barton et al. (2019) evaluate:
one scored row per ICU stay, at sepsis onset (0 h), 24 h and 48 h before onset.

Uses the prepared data + patient-level split + K-means groups written by run_ultimate_full.py (run that first).
Encounter protocol (Barton Sec. 2.4, as in 90_build_protocol_datasets.py):
  septic stay      index = onset - H   (onset = last charted grid row, rows are truncated at onset)
  non-septic stay  pseudo-onset drawn uniformly among rows >= 7 h after ICU admission, index = pseudo-onset - H
  a stay is eligible at horizon H only if the index row exists and has >= 2 earlier rows (Barton's t-1, t-2).
  The random negative draw is repeated over --draws seeds; results are mean +/- SD across draws.
Models are trained per horizon on TRAIN-patient encounters only and scored on TEST-patient encounters:
  barton_lag31     Barton's own representation: 6 vitals at t, t-1, t-2 + 2 hourly differences + age, XGBoost
  single_xgb       XGBoost on the 196 wearable features
  ultimate_<part>  streams x {NOSE-RF on 96 feats, XGBoost on 196 feats} -> {LR, XGB} per family -> LR meta
time_since_ICU_admit_hours is excluded from all encounter models (it is a design artefact under this protocol).
Usage:  python barton_protocol_full.py --partition etiology      (and --partition kmeans)
"""
import argparse, json, time, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
VITALS = ["heart_rate", "resprate", "spo2", "temp_c", "sbp", "dbp"]
NONFEAT = {"stay_id", "subject_id", "charttime", "intime", "sepsis3_time", "is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
           "target_resp", "target_uri", "target_other", "is_test", "time_since_ICU_admit_hours"}
T0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:5.1f}m] {m}", flush=True)


def encounters(d, H, seed):
    d = d.copy()
    d["pos"] = d.groupby("stay_id").cumcount()
    rng = np.random.default_rng(seed)
    last = d.groupby("stay_id").charttime.transform("max")
    sep = d.is_sepsis_stay.values == 1
    # septic: onset row = last row; non-septic: random pseudo-onset among rows >= 7h after admission
    elig_neg = (~sep) & (d.time_since_ICU_admit_hours.values >= 7)
    neg_rows = d[elig_neg].groupby("stay_id").apply(lambda g: g.charttime.values[rng.integers(len(g))])
    onset = pd.Series(np.where(sep, last.values, np.datetime64("NaT")), index=d.index)
    onset[~sep] = d.loc[~sep, "stay_id"].map(neg_rows).values
    d["index_time"] = pd.to_datetime(onset.values) - pd.Timedelta(hours=H)
    e = d[(d.charttime == d.index_time) & (d.pos >= 2)].copy()
    e["y"] = e.is_sepsis_stay.astype(int)
    return e


def barton_feats(d):
    g = d.groupby("stay_id")
    out = {}
    for v in VITALS:
        v1, v2 = g[v].shift(1), g[v].shift(2)
        out[f"{v}_t0"], out[f"{v}_t1"], out[f"{v}_t2"] = d[v], v1, v2
        out[f"{v}_d01"], out[f"{v}_d12"] = d[v] - v1, v1 - v2
    X = pd.DataFrame(out, index=d.index); X["b_age"] = d["age"]
    return X


def xgb(y, depth=4, lr=0.05, trees=2000, es=True):
    spw = float((y == 0).sum()) / max(float(y.sum()), 1.0)
    return XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, tree_method="hist", eval_metric="auc", early_stopping_rounds=50 if es else None,
                         n_jobs=-1, random_state=0, verbosity=0)


def fit_xgb_es(X, y, groups, depth=4, lr=0.05):
    if y.sum() < 2:
        return None
    a, v = next(GroupShuffleSplit(1, test_size=0.15, random_state=1).split(X, y, groups))
    m = xgb(y[a], depth, lr); m.fit(X[a], y[a], eval_set=[(X[v], y[v])], verbose=False); return m


def nose_rf(X, y, k=5):
    pos = np.where(y == 1)[0]; neg = np.random.RandomState(42).permutation(np.where(y == 0)[0]); ms = []
    for i in range(k):
        sub = neg[i * len(pos):(i + 1) * len(pos)]
        if len(sub) == 0 or len(pos) == 0: break
        r = np.concatenate([pos, sub])
        ms.append(RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=35, max_features="sqrt",
                                         random_state=42 + i, n_jobs=-1).fit(X[r], y[r]))
    return ms


def pr(m, X):
    if m is None: return np.zeros(len(X))
    if isinstance(m, list): return np.mean([x.predict_proba(X)[:, 1] for x in m], axis=0) if m else np.zeros(len(X))
    return m.predict_proba(X)[:, 1]


def lr_(): return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
def addvar(X): return np.column_stack([X, X.var(axis=1)])


def recall_at_fa(p, y, fa):
    o = np.argsort(-p); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy); ok = fp <= fa * np.maximum(tp, 1)
    return float(tp[ok].max() / yy.sum()) if ok.any() else 0.0


def run(args):
    ck = Path(args.out) / f"ckpt_{args.grid}"
    d = pd.read_parquet(ck / "prepared.parquet").sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    F_all = [c for c in d.columns if c not in NONFEAT]
    F_rf = [c for c in F_all if not c.startswith(("h6_", "h12_", "h24_", "dev_baseline_", "shk_"))]
    B = barton_feats(d); BF = list(B.columns); d = pd.concat([d, B], axis=1)
    groups = {}
    if args.partition in ("kmeans", "both"):
        for kk in (5, 6):
            groups[kk] = {int(a): b for a, b in json.load(open(ck / f"groups_kmeans{kk}.json")).items()}
    rows = []
    ck2 = Path(args.out) / f"ckpt_barton_{args.grid}"; ck2.mkdir(parents=True, exist_ok=True)
    def cached(name, fn):
        f = ck2 / f"{name}.npy"
        if f.exists():
            return np.load(f)
        v = np.asarray(fn(), dtype="float64"); tmp = Path(str(f) + ".tmp.npy"); np.save(tmp, v); os.replace(tmp, f); return v
    for H in args.horizons:
        for seed in range(args.draws):
            done_f = ck2 / f"rows_{args.partition}_H{H}_d{seed}.json"
            if done_f.exists():
                rows += json.load(open(done_f)); log(f"H={H}h draw={seed}: loaded from checkpoint"); continue
            new_rows = []
            e = encounters(d, H, seed)
            tr, te = e[e.is_test == 0].reset_index(drop=True), e[e.is_test == 1].reset_index(drop=True)
            FF = list(dict.fromkeys(F_all + BF)); med = tr[FF].median(); tr[FF] = tr[FF].fillna(med); te[FF] = te[FF].fillna(med)
            y, yt = tr.y.values, te.y.values
            P = {}
            tag = f"H{H}_d{seed}"
            P["barton_lag31 (XGB)"] = cached(f"{tag}_barton31", lambda: pr(fit_xgb_es(tr[BF].values, y, tr.subject_id.values), te[BF].values))
            P["single_xgb (196 feats)"] = cached(f"{tag}_single", lambda: pr(fit_xgb_es(tr[F_all].values, y, tr.subject_id.values), te[F_all].values))
            # streams
            streams = [("global", y)]
            if args.partition in ("etiology", "both"):
                for c in ["target_resp", "target_uri", "target_other"]:
                    streams.append(("etiology", ((y == 1) & (tr[c].values == 1)).astype(int)))
            for kk, g in groups.items():
                gid = tr.stay_id.map(g).fillna(-1).values
                for k in range(kk):
                    streams.append((f"kmeans{kk}", ((y == 1) & (gid == k)).astype(int)))
            fo = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
            L1o, L1t, fam = [], [], []
            for si, (name, ys) in enumerate(streams):
                for learner in ("rf", "xgb"):
                    def _oof(ys=ys, learner=learner):
                        o = np.zeros(len(tr))
                        for a, b in fo:
                            if learner == "rf": o[b] = pr(nose_rf(tr[F_rf].values[a], ys[a]), tr[F_rf].values[b])
                            else: o[b] = pr(fit_xgb_es(tr[F_all].values[a], ys[a], tr.subject_id.values[a]), tr[F_all].values[b])
                        return o
                    def _tst(ys=ys, learner=learner):
                        return pr(nose_rf(tr[F_rf].values, ys), te[F_rf].values) if learner == "rf" else pr(fit_xgb_es(tr[F_all].values, ys, tr.subject_id.values), te[F_all].values)
                    key = f"{tag}_{args.partition}_s{si}_{name}_{learner}"
                    o = cached(key + "_oof", _oof); t = cached(key + "_test", _tst)
                    L1o.append(o); L1t.append(t); fam.append(name)
            L1o, L1t, fam = np.column_stack(L1o), np.column_stack(L1t), np.array(fam)
            L2o, L2t = [], []
            for f in [x for x in dict.fromkeys(fam) if x != "global"] or ["global"]:
                cols = np.where((fam == "global") | (fam == f))[0]
                Xo, Xt = addvar(L1o[:, cols]), addvar(L1t[:, cols])
                for mk in (lr_, lambda: xgb(y, 3, 0.05, 200, es=False)):
                    o = np.zeros(len(tr))
                    for a, b in fo: o[b] = mk().fit(Xo[a], y[a]).predict_proba(Xo[b])[:, 1]
                    L2o.append(o); L2t.append(mk().fit(Xo, y).predict_proba(Xt)[:, 1])
            L2o, L2t = np.column_stack(L2o), np.column_stack(L2t)
            P[f"ultimate_{args.partition} (3L)"] = lr_().fit(addvar(L2o), y).predict_proba(addvar(L2t))[:, 1]
            for k, p in P.items():
                new_rows.append(dict(H=H, draw=seed, model=k, n_test=len(te), pos_test=int(yt.sum()), n_train=len(tr),
                                     auroc=float(roc_auc_score(yt, p)), auprc=float(average_precision_score(yt, p)), prev=float(yt.mean()),
                                     sens_at_10FA=recall_at_fa(p, yt, 10)))
                log(f"H={H}h draw={seed} {k:28s} AUROC {new_rows[-1]['auroc']:.4f}  (test {len(te)}, pos {int(yt.sum())})")
            tmp = Path(str(done_f) + ".tmp"); json.dump(new_rows, open(tmp, "w")); os.replace(tmp, done_f)
            rows += new_rows
    r = pd.DataFrame(rows); r["lift"] = r.auprc / r.prev
    out = Path(args.out); r.to_csv(out / f"barton_protocol_{args.partition}_{args.grid}_by_draw.csv", index=False)
    s = r.groupby(["H", "model"]).agg(n_test=("n_test", "mean"), pos=("pos_test", "mean"), auroc=("auroc", "mean"),
                                      auroc_sd=("auroc", "std"), auprc=("auprc", "mean"), lift=("lift", "mean"),
                                      sens_at_10FA=("sens_at_10FA", "mean")).round(4)
    s.to_csv(out / f"barton_protocol_{args.partition}_{args.grid}_summary.csv")
    (out / f"DONE_barton_{args.partition}_{args.grid}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("BARTON-PROTOCOL SUMMARY (Barton et al. 2019 reported 0.88 / 0.84 / 0.83 at 0 / 24 / 48 h)\n" + s.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["etiology", "kmeans", "both"], default="etiology")
    ap.add_argument("--grid", choices=["1h", "4h"], default="1h")
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 24, 48])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "Results" / "ultimate_full"))
    run(ap.parse_args())
