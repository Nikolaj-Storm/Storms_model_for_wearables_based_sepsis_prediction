# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  79_v13b_meta_vs_base_figure.py
#  Stage: 4 - Modeling (final V13b ensemble)
#
#  PURPOSE
#    Extends Figure 18 (Effect of Base Layer Specialization) by adding a fifth
#    bar per target group, the XGBoost meta-learner's lift on each of the four
#    6-hour labels. Reloads the OOF and test base-learner predictions produced
#    by 67_v13b_refined_meta.py, retrains the XGB meta-learner, scores every
#    base learner plus the meta against the four labels, and renders a grouped
#    bar chart of AUPRC lift over prevalence.
#
#  INPUTS
#    <ROOT>/Results/V13b_final/1_hour/checkpoints/oof_meta_full.parquet
#    <ROOT>/Results/V13b_final/1_hour/checkpoints/test_meta_predictions.parquet
#    <ROOT>/Data/Etiology/v8_dataset_1h_etiology_test.parquet
#  OUTPUTS
#    <ROOT>/Visualizations/meta_vs_base_lift_1h.csv
#    <ROOT>/Visualizations/figure_18_meta_vs_base_1h.png
#    /PATH/TO/OUTPUT/figure_18_meta_vs_base_1h.png   (delivery copy)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    ROOT                       -  project technical/ root (absolute path).
#    CKPT / ETIO / OUT          -  checkpoint, etiology, and visualization dirs.
#    deliver_path               -  delivery-copy output PNG path.
#    XGBoost meta-learner       -  n_estimators=400, max_depth=5,
#                                  learning_rate=0.05, subsample=0.8,
#                                  colsample_bytree=0.8, reg_lambda=1.0,
#                                  scale_pos_weight=neg/pos, tree_method='hist',
#                                  eval_metric='aucpr', random_state=42.
#    Plot params                -  figsize, bar width, hatch.linewidth, dpi.
#
#  REQUIRES: pandas, numpy, scikit-learn, xgboost, matplotlib
# ============================================================================
"""
79_v13b_meta_vs_base_figure.py

Extends Figure 18 (Effect of Base Layer Specialization) by adding a fifth
bar per target group, namely the XGBoost meta-learner's lift on each of
the four 6-hour labels.

Inputs (already produced by 67_v13b_refined_meta.py):
  - oof_meta_full.parquet           : OOF base-learner predictions on train
  - test_meta_predictions.parquet   : test-set base-learner predictions
  - v8_dataset_1h_etiology_test.parquet : row-level etiology indicators

Outputs:
  - meta_vs_base_lift_1h.csv : 5 x 4 lift table
  - figure_18_meta_vs_base_1h.png : grouped bar chart (4 targets x 5 ensembles)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import xgboost as xgb
import matplotlib.pyplot as plt

# EDIT: ROOT - project technical/ root (absolute path)
ROOT = Path("/PATH/TO/PROJECT/technical")
# EDIT: CKPT / ETIO / OUT - checkpoint, etiology, and visualization dirs
CKPT = ROOT / "Results/V13b_final/1_hour/checkpoints"
ETIO = ROOT / "Data/Etiology/v8_dataset_1h_etiology_test.parquet"
OUT  = Path("/PATH/TO/PROJECT/technical/Visualizations")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load OOF + test base-learner predictions
# ---------------------------------------------------------------------------
print("Loading OOF and test predictions...")
oof  = pd.read_parquet(CKPT / "oof_meta_full.parquet")
test = pd.read_parquet(CKPT / "test_meta_predictions.parquet")

streams_6h = ["is_sepsis_6h", "target_resp_6h", "target_uri_6h", "target_other_6h"]
meta_cols_6h = [f"meta_{s}" for s in streams_6h]

# Use all 10 meta-features (8 base preds + 2 variance features) for the meta
meta_features = [c for c in oof.columns if c.startswith("meta_")] + ["var_6h", "var_12h"]
print(f"  meta_features ({len(meta_features)}): {meta_features}")

# ---------------------------------------------------------------------------
# 2. Reconstruct etiology labels on test set (target_X_6h = is_sepsis_6h AND target_X)
# ---------------------------------------------------------------------------
print("\nReconstructing etiology labels on test set...")
etio = pd.read_parquet(ETIO, columns=["stay_id", "target_resp", "target_uri", "target_other"])

# test_meta_predictions is in row order matching v8 test parquet — verify by index alignment
assert len(test) == len(etio), f"row mismatch test={len(test)} etio={len(etio)}"
test = test.reset_index(drop=True)
etio = etio.reset_index(drop=True)
test["target_resp"]  = etio["target_resp"].values
test["target_uri"]   = etio["target_uri"].values
test["target_other"] = etio["target_other"].values

for col in ["target_resp", "target_uri", "target_other"]:
    test[f"{col}_6h"] = ((test["is_sepsis_6h"] == 1) & (test[col] == 1)).astype(int)

prev = {label: float(test[label].mean()) for label in streams_6h}
print("\nPrevalences on test set:")
for k, v in prev.items():
    print(f"  {k:20s}  {v*100:.3f}%")

# ---------------------------------------------------------------------------
# 3. Train XGBoost meta-learner on OOF, predict on test
# ---------------------------------------------------------------------------
print("\nTraining XGBoost meta-learner on OOF features (target=is_sepsis_6h)...")
X_train = oof[meta_features].values
y_train = oof["is_sepsis_6h"].values

pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
spw = neg / max(pos, 1.0)
print(f"  train rows={len(y_train):,}  pos={int(pos):,}  scale_pos_weight={spw:.2f}")

# EDIT: XGBoost meta-learner hyperparameters (scale_pos_weight dynamic = neg/pos)
xgb_meta = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    tree_method="hist",
    scale_pos_weight=spw,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)
xgb_meta.fit(X_train, y_train, verbose=False)

X_test = test[meta_features].values
y_meta_test = xgb_meta.predict_proba(X_test)[:, 1]
print(f"  predictions generated for {len(y_meta_test):,} test rows")

# ---------------------------------------------------------------------------
# 4. Score each base learner + meta against the four labels, compute lift
# ---------------------------------------------------------------------------
print("\nScoring all 5 streams against the 4 labels...")
ensembles = {
    "Global":      test["meta_is_sepsis_6h"].values,
    "Respiratory": test["meta_target_resp_6h"].values,
    "Urinary":     test["meta_target_uri_6h"].values,
    "Other":       test["meta_target_other_6h"].values,
    "Meta (XGB)":  y_meta_test,
}
target_pretty = {
    "is_sepsis_6h":    "Global",
    "target_resp_6h":  "Respiratory",
    "target_uri_6h":   "Urinary",
    "target_other_6h": "Other",
}

rows = []
for ens_name, scores in ensembles.items():
    for label_col in streams_6h:
        y_true = test[label_col].values
        auprc = average_precision_score(y_true, scores)
        p = prev[label_col]
        lift = auprc / p
        rows.append({
            "ensemble":  ens_name,
            "target":    target_pretty[label_col],
            "AUPRC":     auprc,
            "prevalence": p,
            "lift":      lift,
        })

df = pd.DataFrame(rows)
print("\nFull table:")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

csv_path = OUT / "meta_vs_base_lift_1h.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved table to {csv_path}")

# ---------------------------------------------------------------------------
# 5. Plot grouped bar chart matching Figure 18 style
# ---------------------------------------------------------------------------
print("\nRendering figure...")
targets = ["Global", "Respiratory", "Urinary", "Other"]
ens_order = ["Global", "Respiratory", "Urinary", "Other", "Meta (XGB)"]
colors = {
    "Global":      "#E27D7B",
    "Respiratory": "#9CD5BD",
    "Urinary":     "#B7A4D5",
    "Other":       "#D0CB8E",
    "Meta (XGB)":  "#3C3C3C",
}

pivot = df.pivot(index="target", columns="ensemble", values="lift").loc[targets, ens_order]

# Emphasis rule: on every target except Global, only the matching specialist
# is drawn as a solid bar. Every other bar in that group is drawn as a
# diagonal-striped bar in its own colour (transparent fill + hatched stripes).
# On the Global target, every bar is solid.

def is_focus(target_name, ensemble_name):
    if target_name == "Global":
        return True
    return ensemble_name == target_name

# EDIT: plot params - figure size and bar width
fig, ax = plt.subplots(figsize=(11.5, 5.8))
x = np.arange(len(targets))
width = 0.16

# Matplotlib hatch density / linewidth
# EDIT: plot params - hatch linewidth
plt.rcParams["hatch.linewidth"] = 1.8

for i, ens in enumerate(ens_order):
    offset = (i - (len(ens_order) - 1) / 2) * width
    vals = pivot[ens].values
    for xi, val in enumerate(vals):
        focus = is_focus(targets[xi], ens)
        if focus:
            ax.bar(x[xi] + offset, val, width,
                   color=colors[ens],
                   edgecolor="#2b2b2b", linewidth=0.5,
                   label=ens if xi == 0 else None)
        else:
            # transparent fill, diagonal stripes in the ensemble's own colour
            ax.bar(x[xi] + offset, val, width,
                   facecolor=(1, 1, 1, 0),       # fully transparent fill
                   edgecolor=colors[ens],
                   hatch="///",
                   linewidth=1.2,
                   label=ens if xi == 0 else None)
        ax.text(x[xi] + offset, val + 0.04, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, color="#2b2b2b")

ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.set_ylabel("Lift over random  (AUPRC / prevalence)")
ax.set_ylim(0, max(pivot.values.max() * 1.15, 4.0))
ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
png_path = OUT / "figure_18_meta_vs_base_1h.png"
# EDIT: plot params - output PNG dpi
plt.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Saved figure to {png_path}")

# Also save a copy to the outputs directory for direct delivery
# EDIT: deliver_path - delivery-copy output PNG path
deliver_path = Path("/PATH/TO/OUTPUT/figure_18_meta_vs_base_1h.png")
plt.savefig(deliver_path, dpi=200, bbox_inches="tight")
print(f"Delivery copy: {deliver_path}")

plt.close(fig)
print("\nDONE.")
