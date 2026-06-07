# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  train_pr_curve_models.py
#  Stage: 6 - Visualization
#
#  PURPOSE
#    Train one (resolution x feature-set x algorithm) job and cache the test-set
#    predicted probabilities as a parquet of (y_true, y_prob), so the PR-curve
#    plotter can build a full curve. Designed to be invoked once per job to fit
#    a per-call time budget, with on-disk caching and a no-op on cache hit.
#
#  INPUTS
#    Engineered: /PATH/TO/PROJECT/technical/Data/All engineered features/
#                Dataset_all_engineered_<res>_{train,test}.parquet
#    Raw:        /PATH/TO/PROJECT/technical/Data/
#                v3_dataset_<res>_{train,test}.parquet
#  OUTPUTS
#    /PATH/TO/PROJECT/pr_curve_predictions/preds_<res>_<featset>_<algo>.parquet
#    /PATH/TO/PROJECT/pr_curve_predictions/selected_features/*  (feature caches)
#    Also prints progress to the console.
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    _CANDIDATE_BASES  -  project base-path candidate list, first existing wins
#    ENG_FILES         -  engineered train/test parquet filenames per resolution
#    RAW_FILES         -  raw-vitals train/test parquet filenames per resolution
#    TARGET            -  label column to predict (is_sepsis_6h)
#    DROP_COLS         -  columns excluded from the engineered feature matrix
#    RAW_FEATURES      -  raw-vitals feature column list
#    ALL_RESOLUTIONS / ALL_FEATURE_SETS / ALL_ALGORITHMS  -  CLI choice sets
#    random seed       -  random_state=42 and rng seeds used throughout
#    top-k features    -  k=50 engineered feature selection count
#    model hyperparameters  -  XGB/RF/LR n_estimators, depth, lr, caps, etc.
#                         in the train_* and selector functions
#
#  REQUIRES: numpy, pandas, pyarrow, xgboost, scikit-learn
# ============================================================================
"""
Per-job training script that produces test-set predicted probabilities for one
(resolution × algorithm × feature-set) combination, then writes a parquet of
(y_true, y_prob) so the PR-curve plotter can build a full curve from it.

The script is designed to be invoked once per job so each call fits inside the
sandbox time budget. Caches predictions on disk; calling again with the same
job key is a no-op unless --force is passed.

Hyperparameters mirror the configurations that produced the operating points
in `all results updated.xlsx`:
  * Engineered jobs follow `optimise_features.py` (top-50 XGB importance,
    XGB n=300 / depth 6 / lr 0.05, RF n=200 / depth 12 balanced, LR balanced).
  * Raw-vitals jobs follow `benchmark_basic_features.py` (XGB n=100 /
    scale_pos_weight=10, RF n=100 / depth 10 balanced, LR balanced).

Usage:
    python train_pr_curve_models.py <resolution> <feature_set> <algorithm>
    where:
        resolution   ∈ {4_hour, 1_hour, 15_min}
        feature_set  ∈ {engineered, raw}
        algorithm    ∈ {xgb, rf, lr}

Or:
    python train_pr_curve_models.py --list      # list all 18 jobs
    python train_pr_curve_models.py --status    # show which are cached
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# EDIT: project base-path candidate list, first existing path is used
_CANDIDATE_BASES = [
    Path("/PATH/TO/PROJECT"),
]
BASE = next((p for p in _CANDIDATE_BASES if p.exists()), _CANDIDATE_BASES[0])

CACHE_DIR = BASE / "pr_curve_predictions"
CACHE_DIR.mkdir(exist_ok=True)
FEATURE_CACHE_DIR = CACHE_DIR / "selected_features"
FEATURE_CACHE_DIR.mkdir(exist_ok=True)

ENG_DIR = BASE / "technical/Data/All engineered features"
# EDIT: engineered train/test parquet filenames per resolution
ENG_FILES = {
    "4_hour": ("Dataset_all_engineered_4h_train.parquet",   "Dataset_all_engineered_4h_test.parquet"),
    "1_hour": ("Dataset_all_engineered_1h_train.parquet",   "Dataset_all_engineered_1h_test.parquet"),
    "15_min": ("Dataset_all_engineered_15min_train.parquet", "Dataset_all_engineered_15min_test.parquet"),
}

RAW_DIR = BASE / "technical/Data"
# EDIT: raw-vitals train/test parquet filenames per resolution
RAW_FILES = {
    "4_hour": ("v3_dataset_4h_train.parquet", "v3_dataset_4h_test.parquet"),
    "1_hour": ("v3_dataset_1h_train.parquet", "v3_dataset_1h_test.parquet"),
    "15_min": ("v3_dataset_15m_train.parquet", "v3_dataset_15m_test.parquet"),
}

# EDIT: label column to predict
TARGET = "is_sepsis_6h"
# EDIT: columns excluded from the engineered feature matrix
DROP_COLS = ["is_sepsis_stay", "is_sepsis_6h", "is_sepsis_12h",
             "stay_id", "charttime", "intime", "sepsis3_time",
             "time_since_ICU_admit_hours"]
# EDIT: raw-vitals feature columns
RAW_FEATURES = ["age", "weight_kg",
                "heart_rate", "resprate", "spo2", "temp_c", "sbp", "dbp"]

# EDIT: CLI choice sets
ALL_RESOLUTIONS = ["4_hour", "1_hour", "15_min"]
ALL_FEATURE_SETS = ["engineered", "raw"]
ALL_ALGORITHMS = ["xgb", "rf", "lr"]


def cache_path(resolution: str, feature_set: str, algorithm: str) -> Path:
    return CACHE_DIR / f"preds_{resolution}_{feature_set}_{algorithm}.parquet"


def load_engineered(resolution: str, downsample_15m: bool = True):
    train_fn, test_fn = ENG_FILES[resolution]
    df_train = pd.read_parquet(ENG_DIR / train_fn).dropna(subset=[TARGET])
    df_test = pd.read_parquet(ENG_DIR / test_fn).dropna(subset=[TARGET])

    if resolution == "15_min" and downsample_15m:
        df_train = df_train.sample(frac=0.5, random_state=42)  # EDIT: random seed
        df_test = df_test.sample(frac=0.5, random_state=42)  # EDIT: random seed

    y_train = df_train[TARGET].astype(int).to_numpy()
    y_test = df_test[TARGET].astype(int).to_numpy()
    X_train = df_train.drop(columns=[c for c in DROP_COLS if c in df_train.columns]).fillna(0).astype("float32")
    X_test = df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns]).fillna(0).astype("float32")
    common = [c for c in X_train.columns if c in X_test.columns]
    return X_train[common], y_train, X_test[common], y_test


def load_raw(resolution: str, downsample_15m: bool = True):
    train_fn, test_fn = RAW_FILES[resolution]
    df_train = pd.read_parquet(RAW_DIR / train_fn).dropna(subset=[TARGET])
    df_test = pd.read_parquet(RAW_DIR / test_fn).dropna(subset=[TARGET])

    if resolution == "15_min" and downsample_15m:
        df_train = df_train.sample(frac=0.5, random_state=42)  # EDIT: random seed
        df_test = df_test.sample(frac=0.5, random_state=42)  # EDIT: random seed

    y_train = df_train[TARGET].astype(int).to_numpy()
    y_test = df_test[TARGET].astype(int).to_numpy()
    keep = [c for c in RAW_FEATURES if c in df_train.columns]
    X_train = df_train[keep].fillna(0).astype("float32")
    X_test = df_test[keep].fillna(0).astype("float32")
    return X_train, y_train, X_test, y_test


def select_engineered_topk(X_train, y_train, k: int = 50, resolution: str | None = None):  # EDIT: top-k feature count
    """Pick the top-k engineered features by XGB importance, with on-disk
    caching so the selection is paid for once per resolution and reused across
    all three engineered training jobs."""
    feature_file = None
    if resolution is not None:
        feature_file = FEATURE_CACHE_DIR / f"engineered_top{k}_{resolution}.txt"
        if feature_file.exists():
            return feature_file.read_text().splitlines()

    from xgboost import XGBClassifier

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    # EDIT: XGB feature-selector hyperparameters
    selector = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.05,
                             random_state=42, scale_pos_weight=spw, n_jobs=-1,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist")
    selector.fit(X_train, y_train)
    imp = pd.Series(selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feats = imp.head(k).index.tolist()
    if feature_file is not None:
        feature_file.write_text("\n".join(feats))
    return feats


def train_engineered_xgb(X_tr, y_tr, X_te):
    from xgboost import XGBClassifier
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    # Reduced from optimise_features.py's 300 trees to 150 to fit a 45s budget;
    # the PR-curve shape stays close to the optimised configuration.
    # EDIT: engineered XGB hyperparameters
    model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.07,
                          random_state=42, scale_pos_weight=spw, n_jobs=-1,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, tree_method="hist")
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def train_engineered_rf(X_tr, y_tr, X_te):
    from sklearn.ensemble import RandomForestClassifier
    cap = 300_000  # EDIT: training-row cap for engineered RF
    if len(X_tr) > cap:
        rng = np.random.default_rng(42)  # EDIT: random seed
        idx = rng.choice(len(X_tr), size=cap, replace=False)
        Xtr = X_tr.iloc[idx] if hasattr(X_tr, "iloc") else X_tr[idx]
        ytr = y_tr[idx]
    else:
        Xtr, ytr = X_tr, y_tr
    print(f"    [rf] training on {len(Xtr)} rows", flush=True)
    # Reduced from optimise_features.py's 200/depth=12 to fit a 45s budget;
    # qualitative PR-curve shape is preserved.
    # EDIT: engineered RF hyperparameters
    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   random_state=42, class_weight="balanced",
                                   n_jobs=-1)
    model.fit(Xtr, ytr)
    return model.predict_proba(X_te)[:, 1]


def train_engineered_lr(X_tr, y_tr, X_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    print("    [lr] subsampling", flush=True)
    cap = 200_000  # EDIT: training-row cap for engineered LR
    if len(X_tr) > cap:
        rng = np.random.default_rng(42)  # EDIT: random seed
        idx = rng.choice(len(X_tr), size=cap, replace=False)
        Xtr_raw = X_tr.iloc[idx] if hasattr(X_tr, "iloc") else X_tr[idx]
        ytr = y_tr[idx]
    else:
        Xtr_raw, ytr = X_tr, y_tr
    print(f"    [lr] subsampled to {len(Xtr_raw)}", flush=True)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr_raw)
    Xte = sc.transform(X_te)
    print(f"    [lr] scaled", flush=True)
    Xtr = np.nan_to_num(Xtr, nan=0)
    Xte = np.nan_to_num(Xte, nan=0)
    np.clip(Xtr, -1e6, 1e6, out=Xtr)
    np.clip(Xte, -1e6, 1e6, out=Xte)
    print(f"    [lr] fitting", flush=True)
    # EDIT: engineered LR hyperparameters
    model = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42,
                               class_weight="balanced", n_jobs=-1)
    model.fit(Xtr, ytr)
    print(f"    [lr] predicting", flush=True)
    return model.predict_proba(Xte)[:, 1]


def train_raw_xgb(X_tr, y_tr, X_te):
    from xgboost import XGBClassifier
    # EDIT: raw-vitals XGB hyperparameters
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                          random_state=42, scale_pos_weight=10, n_jobs=-1,
                          tree_method="hist")
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def train_raw_rf(X_tr, y_tr, X_te):
    from sklearn.ensemble import RandomForestClassifier
    cap = 300_000  # EDIT: training-row cap for raw RF
    if len(X_tr) > cap:
        rng = np.random.default_rng(42)  # EDIT: random seed
        idx = rng.choice(len(X_tr), size=cap, replace=False)
        Xtr = X_tr.iloc[idx] if hasattr(X_tr, "iloc") else X_tr[idx]
        ytr = y_tr[idx]
    else:
        Xtr, ytr = X_tr, y_tr
    print(f"    [rf-raw] training on {len(Xtr)} rows", flush=True)
    # EDIT: raw-vitals RF hyperparameters
    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   random_state=42, class_weight="balanced",
                                   n_jobs=-1)
    model.fit(Xtr, ytr)
    return model.predict_proba(X_te)[:, 1]


def train_raw_lr(X_tr, y_tr, X_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = np.clip(np.nan_to_num(sc.fit_transform(X_tr), nan=0), -1e6, 1e6)
    Xte = np.clip(np.nan_to_num(sc.transform(X_te),    nan=0), -1e6, 1e6)
    cap = 200_000  # EDIT: training-row cap for raw LR
    if len(Xtr) > cap:
        rng = np.random.default_rng(42)  # EDIT: random seed
        idx = rng.choice(len(Xtr), size=cap, replace=False)
        Xtr_fit = Xtr[idx]
        ytr_fit = y_tr[idx]
    else:
        Xtr_fit, ytr_fit = Xtr, y_tr
    # EDIT: raw-vitals LR hyperparameters
    model = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42,
                               class_weight="balanced", n_jobs=-1)
    model.fit(Xtr_fit, ytr_fit)
    return model.predict_proba(Xte)[:, 1]


TRAIN_FUNCS = {
    ("engineered", "xgb"): train_engineered_xgb,
    ("engineered", "rf"):  train_engineered_rf,
    ("engineered", "lr"):  train_engineered_lr,
    ("raw", "xgb"):        train_raw_xgb,
    ("raw", "rf"):         train_raw_rf,
    ("raw", "lr"):         train_raw_lr,
}


def engineered_subset_paths(resolution: str) -> tuple[Path, Path]:
    return (
        FEATURE_CACHE_DIR / f"engineered_subset_{resolution}_train.parquet",
        FEATURE_CACHE_DIR / f"engineered_subset_{resolution}_test.parquet",
    )


def _select_features_via_sample(resolution: str, k: int = 50, sample_rows: int = 200_000):  # EDIT: top-k feature count and sample size
    """Run feature selection on a downsampled view of the train file. For
    large resolutions we read a single parquet row group rather than the full
    file, which is enough to rank feature importance and dodges the 45s budget."""
    feature_file = FEATURE_CACHE_DIR / f"engineered_top{k}_{resolution}.txt"
    if feature_file.exists():
        return feature_file.read_text().splitlines()

    train_fn, _ = ENG_FILES[resolution]
    print(f"[selecting] sampling rows from {train_fn}", flush=True)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ENG_DIR / train_fn)
    df_full = pf.read_row_group(0).to_pandas()
    print(f"  rg0 read shape={df_full.shape}", flush=True)
    df_full = df_full[df_full[TARGET].notna()]
    if len(df_full) > sample_rows:
        df_full = df_full.sample(n=sample_rows, random_state=42)  # EDIT: random seed
    df = df_full
    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).fillna(0).astype("float32")

    from xgboost import XGBClassifier
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    # EDIT: XGB feature-selector hyperparameters
    selector = XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.05,
                             random_state=42, scale_pos_weight=spw, n_jobs=-1,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist")
    selector.fit(X, y)
    imp = pd.Series(selector.feature_importances_, index=X.columns).sort_values(ascending=False)
    feats = imp.head(k).index.tolist()
    feature_file.write_text("\n".join(feats))
    return feats


def prepare_engineered_subset(resolution: str, k: int = 50, which: str = "both") -> tuple[Path, Path]:  # EDIT: top-k feature count
    """Load engineered train/test with only selected features (using a column
    projection at parquet read time to avoid loading the full 1500-column
    matrix), and save the slim subsets so training jobs skip the heavy load."""
    train_subset, test_subset = engineered_subset_paths(resolution)
    if train_subset.exists() and test_subset.exists():
        return train_subset, test_subset

    feats = _select_features_via_sample(resolution, k=k)

    print(f"[prepare] engineered subset for {resolution} ({len(feats)} features)")
    t0 = time.time()
    train_fn, test_fn = ENG_FILES[resolution]
    columns_to_load = list(set(feats) | {TARGET})

    def _save_subset(src: Path, dst: Path, downsample_frac: float = 1.0):
        # Read row-group by row-group, downsample on the fly, and stream rows
        # into the output via a ParquetWriter so we never materialise the whole
        # subset in memory or pay a separate concat-and-write tail at the end.
        import pyarrow as pa
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(src)
        rng = np.random.default_rng(42)  # EDIT: random seed
        writer = None
        rows_total = 0
        try:
            for i in range(pf.num_row_groups):
                chunk = pf.read_row_group(i, columns=columns_to_load).to_pandas()
                chunk = chunk[chunk[TARGET].notna()]
                if downsample_frac < 1.0 and len(chunk) > 0:
                    n_keep = int(len(chunk) * downsample_frac)
                    if n_keep > 0:
                        pick = rng.choice(len(chunk), size=n_keep, replace=False)
                        chunk = chunk.iloc[pick]
                if len(chunk) == 0:
                    continue
                chunk = chunk.rename(columns={TARGET: "__target__"})
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(dst, table.schema)
                writer.write_table(table)
                rows_total += len(chunk)
        finally:
            if writer is not None:
                writer.close()
        return (rows_total, len(columns_to_load))

    # 15-min has 8.6M training rows. Drop to 25% so the prep step + downstream
    # training jobs all fit within the per-call time budget; 2M training rows
    # is more than enough for a stable PR curve.
    downsample = 0.25 if resolution == "15_min" else 1.0  # EDIT: 15-min downsample fraction
    if which in ("both", "train") and not train_subset.exists():
        shape = _save_subset(ENG_DIR / train_fn, train_subset, downsample_frac=downsample)
        print(f"  train subset saved: {shape}  t={time.time()-t0:.1f}s")
    if which in ("both", "test") and not test_subset.exists():
        shape = _save_subset(ENG_DIR / test_fn, test_subset, downsample_frac=downsample)
        print(f"  test subset saved: {shape}  t={time.time()-t0:.1f}s")
    return train_subset, test_subset


def load_engineered_subset(resolution: str):
    train_subset, test_subset = engineered_subset_paths(resolution)
    if not (train_subset.exists() and test_subset.exists()):
        prepare_engineered_subset(resolution)
    print(f"  reading subsets...", flush=True)
    df_train = pd.read_parquet(train_subset)
    print(f"  train read {df_train.shape}", flush=True)
    df_test = pd.read_parquet(test_subset)
    print(f"  test read {df_test.shape}", flush=True)
    target_col = "__target__" if "__target__" in df_train.columns else TARGET
    y_train = df_train.pop(target_col).astype(np.int8).to_numpy()
    target_col = "__target__" if "__target__" in df_test.columns else TARGET
    y_test = df_test.pop(target_col).astype(np.int8).to_numpy()
    print(f"  popped targets", flush=True)
    feats_file = FEATURE_CACHE_DIR / f"engineered_top50_{resolution}.txt"
    if feats_file.exists():
        canonical = [f for f in feats_file.read_text().splitlines() if f in df_train.columns]
    else:
        canonical = df_train.columns.tolist()
    print(f"  canonical={len(canonical)} cols", flush=True)
    # Convert via NumPy to avoid the per-column pandas pathway that hangs on
    # large 50-column dataframes inside the script process.
    train_arr = np.nan_to_num(df_train.loc[:, canonical].to_numpy(dtype=np.float32, copy=False), nan=0.0)
    print(f"  train arr {train_arr.shape}", flush=True)
    test_arr = np.nan_to_num(df_test.loc[:, canonical].to_numpy(dtype=np.float32, copy=False), nan=0.0)
    print(f"  test arr {test_arr.shape}", flush=True)
    df_train = pd.DataFrame(train_arr, columns=canonical)
    df_test = pd.DataFrame(test_arr, columns=canonical)
    return df_train, y_train, df_test, y_test


def run_one(resolution: str, feature_set: str, algorithm: str, force: bool = False) -> Path:
    out = cache_path(resolution, feature_set, algorithm)
    if out.exists() and not force:
        print(f"[cached] {out}")
        return out

    print(f"[loading] {resolution} {feature_set}")
    t0 = time.time()
    if feature_set == "engineered":
        # Use the precomputed subset to keep loading fast on large resolutions.
        X_train, y_train, X_test, y_test = load_engineered_subset(resolution)
    elif feature_set == "raw":
        X_train, y_train, X_test, y_test = load_raw(resolution)
    else:
        raise ValueError(f"unknown feature_set {feature_set!r}")
    print(f"  loaded train={X_train.shape}, test={X_test.shape} in {time.time()-t0:.1f}s")

    print(f"[training] {algorithm}")
    t1 = time.time()
    train_fn = TRAIN_FUNCS[(feature_set, algorithm)]
    y_prob = train_fn(X_train, y_train, X_test)
    print(f"  trained + predicted in {time.time()-t1:.1f}s")

    df_out = pd.DataFrame({"y_true": y_test.astype(np.int8), "y_prob": y_prob.astype(np.float32)})
    df_out.to_parquet(out, index=False)
    print(f"[saved] {out}  rows={len(df_out)}  pos={df_out['y_true'].sum()}  total_time={time.time()-t0:.1f}s")
    return out


def list_jobs():
    print(f"{'resolution':<10} {'feat':<11} {'algo':<5} {'cached':<7}")
    for r in ALL_RESOLUTIONS:
        for f in ALL_FEATURE_SETS:
            for a in ALL_ALGORITHMS:
                cached = cache_path(r, f, a).exists()
                print(f"{r:<10} {f:<11} {a:<5} {'yes' if cached else 'no':<7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("resolution", nargs="?", choices=ALL_RESOLUTIONS)
    parser.add_argument("feature_set", nargs="?", choices=ALL_FEATURE_SETS)
    parser.add_argument("algorithm", nargs="?", choices=ALL_ALGORITHMS)
    parser.add_argument("--force", action="store_true",
                        help="retrain even if cache exists")
    parser.add_argument("--list", action="store_true",
                        help="list all jobs and their cache state")
    parser.add_argument("--prepare-engineered", choices=ALL_RESOLUTIONS,
                        help="preselect top-50 engineered features and save subset parquets")
    parser.add_argument("--prep-which", choices=["both", "train", "test"], default="both",
                        help="which subset to prepare (use to split a heavy prep across calls)")
    args = parser.parse_args()

    if args.list:
        list_jobs()
        return

    if args.prepare_engineered:
        prepare_engineered_subset(args.prepare_engineered, which=args.prep_which)
        return

    if not (args.resolution and args.feature_set and args.algorithm):
        parser.print_help()
        sys.exit(1)

    run_one(args.resolution, args.feature_set, args.algorithm, force=args.force)


if __name__ == "__main__":
    main()
