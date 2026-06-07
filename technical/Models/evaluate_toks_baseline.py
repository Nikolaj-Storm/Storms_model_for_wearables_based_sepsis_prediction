# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  evaluate_toks_baseline.py
#  Stage: 4 - Modeling
#
#  PURPOSE
#    Builds and evaluates a rule-based TOKS-proxy early-warning score (a
#    7-variable NEWS-style scoring system) as a non-ML clinical baseline. Merges
#    GCS and O2-device data from a raw BigQuery extract onto the 4-hour test
#    grid, computes the additive TOKS score, sweeps thresholds, and appends the
#    chosen operating points to an existing results Excel workbook.
#
#  INPUTS
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test.parquet
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_bq_raw.parquet
#    /PATH/TO/PROJECT/all results updated.xlsx   (appended to, read first)
#  OUTPUTS
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_with_GCS_O2.parquet
#    /PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_TOKS.parquet
#    /PATH/TO/PROJECT/all results updated.xlsx   (rows appended to two sheets)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:  to find each)
#    parquet_path / bq_path  -  input 4-hour test parquet and raw BQ extract
#    out_with_gcs / out_toks -  intermediate output parquet paths
#    excel_path              -  results workbook appended to
#    Scoring tables          -  score_* functions encode the TOKS/NEWS cutoffs
#    GCS valid component count-  3 components required per charttime
#    Resample cadence        -  '4h' grid for GCS (min) and O2 (max)
#    GCS / O2 defaults       -  fillna 15 (GCS) and 0 (on_O2)
#    vitals list             -  6 vitals used for has_missing_vital
#    Threshold sweep         -  list(range(21)) thresholds; operating points 3,5,7
#    Sheet names             -  "6h Sepsis Prediction", "12h Sepsis Prediction"
#
#  REQUIRES: scikit-learn, openpyxl, pandas, numpy
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

# EDIT: TOKS/NEWS-style scoring cutoffs (score_* functions) below
def score_resprate(v):
    if pd.isnull(v): return 0
    if v <= 8: return 3
    elif v <= 11: return 1
    elif v <= 20: return 0
    elif v <= 24: return 2
    else: return 3

def score_spo2(v):
    if pd.isnull(v): return 0
    if v <= 91: return 3
    elif v <= 93: return 2
    elif v <= 95: return 1
    else: return 0

def score_o2(v):
    if pd.isnull(v) or v == 0: return 0
    return 2

def score_sbp(v):
    if pd.isnull(v): return 0
    if v <= 90: return 3
    elif v <= 100: return 2
    elif v <= 110: return 1
    elif v <= 219: return 0
    else: return 3

def score_hr(v):
    if pd.isnull(v): return 0
    if v <= 40: return 3
    elif v <= 50: return 1
    elif v <= 90: return 0
    elif v <= 110: return 1
    elif v <= 130: return 2
    else: return 3

def score_temp(v):
    if pd.isnull(v): return 0
    if v <= 35.0: return 3
    elif v <= 36.0: return 1
    elif v <= 38.0: return 0
    elif v <= 39.0: return 1
    else: return 2

def score_gcs(v):
    if pd.isnull(v): return 0
    if v < 15: return 3
    return 0

def compute_metrics_for_thresholds(y_true, y_pred_prob, thresholds, horizon):
    auroc = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)

    rows = []
    total_samples = len(y_true)
    for k in thresholds:
        y_pred = (y_pred_prob >= k).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        f1_pos = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1_neg = 2 * npv * tnr / (npv + tnr) if (npv + tnr) > 0 else 0

        f1_macro = (f1_pos + f1_neg) / 2
        f05 = (1 + 0.25) * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0
        f2 = (1 + 4) * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnp = fn / (fn + tp) if (fn + tp) > 0 else 0

        rows.append({
            'Version': 'Rule-Based Baseline',
            'Model': 'TOKS Proxy (full 7-variable, 4h cadence)',
            'Variant / Operating Point': f'Threshold >= {k}',
            'AUPRC': auprc,
            'Precision (PPV)': precision,
            'AUROC': auroc,
            'F0.5': f05,
            'F1 Macro': f1_macro,
            'F2': f2,
            'FPR': fpr,
            'FNP': fnp,
            'TP': tp,
            'FN': fn,
            'FP': fp,
            'TN': tn,
            'Threshold': k,
            'Recall': recall,
            'F1 (pos class)': f1_pos,
            'TP %': tp / total_samples * 100,
            'FN %': fn / total_samples * 100,
            'FP %': fp / total_samples * 100,
            'TN %': tn / total_samples * 100,
            'Train Acc': np.nan,
            'Test Acc': (tp + tn) / total_samples
        })

    df_metrics = pd.DataFrame(rows)

    # Mark max F1 Macro
    max_f1_idx = df_metrics['F1 Macro'].idxmax()
    max_f1_row = df_metrics.loc[max_f1_idx].copy()
    max_f1_row['Variant / Operating Point'] = f'Max F1-Macro (Threshold >= {max_f1_row["Threshold"]})'

    # Return explicit operating points + max F1
    ops = df_metrics[df_metrics['Threshold'].isin([3, 5, 7])].copy()  # EDIT: explicit TOKS operating-point thresholds

    # avoid duplicate if max_f1 is one of 3, 5, 7
    if max_f1_row['Threshold'] not in [3, 5, 7]:
        ops = pd.concat([ops, max_f1_row.to_frame().T], ignore_index=True)
    else:
        idx_to_replace = ops[ops['Threshold'] == max_f1_row['Threshold']].index[0]
        ops.loc[idx_to_replace, 'Variant / Operating Point'] += ' (Max F1-Macro)'

    return ops

def main():
    print("Loading test cohort parquet...")
    parquet_path = "/PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test.parquet"  # EDIT: 4-hour test parquet
    df_test = pd.read_parquet(parquet_path)

    print("Loading BQ raw extract...")
    bq_path = "/PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_bq_raw.parquet"  # EDIT: raw BigQuery extract parquet
    df_bq = pd.read_parquet(bq_path)

    # --- Task 1 & 2: Process GCS and O2 ---
    df_gcs = df_bq[df_bq['type'] == 'gcs'].copy()
    df_o2 = df_bq[df_bq['type'] == 'o2'].copy()

    df_gcs = df_gcs[df_gcs['valuenum'].notnull()]
    df_gcs = df_gcs[df_gcs['valuenum'] > 0]

    # To compute total, we must sum per charttime. We only keep charttimes that have exactly 3 components
    gcs_grouped = df_gcs.groupby(['stay_id', 'charttime'])
    gcs_counts = gcs_grouped['itemid'].nunique()
    valid_gcs_times = gcs_counts[gcs_counts == 3].index  # EDIT: required GCS component count per charttime

    df_gcs_valid = df_gcs.set_index(['stay_id', 'charttime']).loc[valid_gcs_times].reset_index()
    df_gcs_total = df_gcs_valid.groupby(['stay_id', 'charttime'])['valuenum'].sum().reset_index()
    df_gcs_total.rename(columns={'valuenum': 'GCS_total'}, inplace=True)

    def get_o2_flag(v):
        if pd.isnull(v) or v == "None" or v == "Room air":
            return 0
        return 1

    df_o2['on_O2'] = df_o2['value'].apply(get_o2_flag)
    df_o2_total = df_o2.groupby(['stay_id', 'charttime'])['on_O2'].max().reset_index()

    # --- Task 3: Align to 4-hour grid ---
    # df_test is multi-indexed by (stay_id, charttime).
    # Resample BigQuery data to '4h' to match.
    df_gcs_total.set_index('charttime', inplace=True)
    gcs_4h = df_gcs_total.groupby('stay_id')['GCS_total'].resample('4h').min()  # EDIT: resample cadence / GCS aggregation

    df_o2_total.set_index('charttime', inplace=True)
    o2_4h = df_o2_total.groupby('stay_id')['on_O2'].resample('4h').max()  # EDIT: resample cadence / O2 aggregation

    df_merged = df_test.join(gcs_4h, how='left')
    df_merged = df_merged.join(o2_4h, how='left')

    # Forward fill within stay_id
    df_merged['GCS_total'] = df_merged.groupby('stay_id')['GCS_total'].ffill()
    df_merged['on_O2'] = df_merged.groupby('stay_id')['on_O2'].ffill()

    # Defaults
    df_merged['GCS_total'] = df_merged['GCS_total'].fillna(15)  # EDIT: GCS default when missing
    df_merged['on_O2'] = df_merged['on_O2'].fillna(0)  # EDIT: O2 default when missing

    # Save the merged parquet
    out_with_gcs = "/PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_with_GCS_O2.parquet"  # EDIT: merged GCS/O2 output parquet
    df_merged.to_parquet(out_with_gcs)

    # --- Task 4: Compute TOKS score ---
    vitals = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']  # EDIT: vitals used for has_missing_vital
    df_merged['has_missing_vital'] = df_merged[vitals].isnull().any(axis=1).astype(int)

    df_merged['pts_resprate'] = df_merged['resprate'].apply(score_resprate)
    df_merged['pts_spo2'] = df_merged['spo2'].apply(score_spo2)
    df_merged['pts_o2'] = df_merged['on_O2'].apply(score_o2)
    df_merged['pts_sbp'] = df_merged['sbp'].apply(score_sbp)
    df_merged['pts_hr'] = df_merged['heart_rate'].apply(score_hr)
    df_merged['pts_temp'] = df_merged['temp_c'].apply(score_temp)
    df_merged['pts_gcs'] = df_merged['GCS_total'].apply(score_gcs)

    df_merged['TOKS_total'] = (df_merged['pts_resprate'] + df_merged['pts_spo2'] +
                               df_merged['pts_o2'] + df_merged['pts_sbp'] +
                               df_merged['pts_hr'] + df_merged['pts_temp'] +
                               df_merged['pts_gcs'])

    cols_to_save = ['TOKS_total', 'has_missing_vital', 'is_sepsis_6h', 'is_sepsis_12h',
                    'pts_resprate', 'pts_spo2', 'pts_o2', 'pts_sbp', 'pts_hr', 'pts_temp', 'pts_gcs']
    out_toks = "/PATH/TO/PROJECT/technical/Data/v3_dataset_4h_test_TOKS.parquet"  # EDIT: TOKS scores output parquet
    df_merged[cols_to_save].to_parquet(out_toks)

    # --- Task 5: Benchmarking ---
    # summary
    num_stays = df_merged.index.get_level_values('stay_id').nunique()
    num_rows = len(df_merged)
    prev_6h = df_merged['is_sepsis_6h'].mean()
    prev_12h = df_merged['is_sepsis_12h'].mean()

    gcs_coverage = (df_merged['GCS_total'] != 15).mean()
    o2_coverage = (df_merged['on_O2'] != 0).mean()
    missing_vital_frac = df_merged['has_missing_vital'].mean()

    mask_6h = df_merged['is_sepsis_6h'].notna()
    y_true_6h = df_merged.loc[mask_6h, 'is_sepsis_6h'].values
    y_pred_prob_6h = df_merged.loc[mask_6h, 'TOKS_total'].values

    mask_12h = df_merged['is_sepsis_12h'].notna()
    y_true_12h = df_merged.loc[mask_12h, 'is_sepsis_12h'].values
    y_pred_prob_12h = df_merged.loc[mask_12h, 'TOKS_total'].values

    auroc_6h = roc_auc_score(y_true_6h, y_pred_prob_6h)

    print("=== TOKS Baseline Summary ===")
    print(f"Test cohort stay count: {num_stays} (expect 14,743)")
    print(f"Total test rows scored: {num_rows} (expect 196,700)")
    print(f"is_sepsis_6h prevalence: {prev_6h:.4%}")
    print(f"is_sepsis_12h prevalence: {prev_12h:.4%}")
    print(f"GCS coverage rate: {gcs_coverage:.4%}")
    print(f"O2 device coverage rate: {o2_coverage:.4%}")
    print(f"Fraction of rows flagged has_missing_vital = 1: {missing_vital_frac:.4%}")
    print(f"Headline TOKS_total AUROC (6h): {auroc_6h:.4f}")

    ops_6h = compute_metrics_for_thresholds(y_true_6h, y_pred_prob_6h, list(range(21)), '6h')  # EDIT: threshold sweep range
    ops_12h = compute_metrics_for_thresholds(y_true_12h, y_pred_prob_12h, list(range(21)), '12h')  # EDIT: threshold sweep range

    # Append to Excel
    excel_path = "/PATH/TO/PROJECT/all results updated.xlsx"  # EDIT: results workbook appended to
    wb = openpyxl.load_workbook(excel_path)

    # Append to 6h sheet
    ws_6h = wb["6h Sepsis Prediction"]  # EDIT: 6h sheet name
    # We just append at the bottom
    # Need to match column ordering: Version, Model, Variant, AUPRC, Precision...
    # The columns in ops_6h are already in this order if we match the DataFrame columns to the sheet
    # We will just append the values in the order of ops_6h.
    # Let's ensure the column order perfectly matches row 43 of the excel which is the header
    header_row = None
    for row in ws_6h.iter_rows(min_row=1, max_row=50, values_only=True):
        if row[0] == 'Version':
            header_row = list(row)
            break

    if header_row:
        # reorder ops_6h columns
        valid_cols = [c for c in header_row if c is not None]
        ops_6h = ops_6h[valid_cols]
        ops_12h = ops_12h[valid_cols]

        for _, row in ops_6h.iterrows():
            ws_6h.append(row.tolist())

        ws_12h = wb["12h Sepsis Prediction"]  # EDIT: 12h sheet name
        for _, row in ops_12h.iterrows():
            ws_12h.append(row.tolist())

        wb.save(excel_path)
        print("Successfully appended to Excel.")
    else:
        print("Could not find header row in Excel to match columns.")

if __name__ == "__main__":
    main()
