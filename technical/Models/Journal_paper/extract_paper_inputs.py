# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  extract_paper_inputs.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Step 1. Extracts test/train ids, grid vitals and the hourly test feature
#    matrix from the checkpointed run.
#
#  INPUTS   (under Results/ unless stated)
#    ckpt_<grid>/prepared.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    paper_v3/_extract/*.parquet
#  RUN
#    python extract_paper_inputs.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

"""Extracts the small inputs the journal-paper analyses need from the checkpointed run of run_ultimate_full.py.

prepared.parquet (written by run_ultimate_full.py) holds every grid row with its features, labels and split flag.
This script writes, per grid, the test rows' ids, labels and raw grid vitals, the training rows' ids, the full vitals
table (train + test, for the hourly Barton comparator) and, for the hourly grid, the test feature matrix (Fig. 4).
Rows keep the order of prepared.parquet, which is the order of every prediction file of the run.
"""
import sys
import pandas as pd
import pyarrow.parquet as pq

RES = '/PATH/TO/PROJECT/technical/Results/'          # EDIT: results root of run_ultimate_full.py
OUT = RES + 'paper_v3/_extract/'                       # EDIT: where the extracts are written
IDS = ['stay_id', 'subject_id', 'charttime', 'is_sepsis_stay', 'is_sepsis_6h', 'is_sepsis_12h',
       'target_resp', 'target_uri', 'target_other', 'time_since_ICU_admit_hours']
VIT = ['heart_rate', 'resprate', 'spo2', 'temp_c', 'sbp', 'dbp']

for g in ('1h', '4h'):
    pf = pq.ParquetFile(RES + f'ultimate_full/ckpt_{g}/prepared.parquet')
    te, tr = [], []
    for b in pf.iter_batches(batch_size=200_000):
        x = b.to_pandas()
        for c in x.select_dtypes('float64').columns:
            x[c] = x[c].astype('float32')
        te.append(x[x.is_test == 1]); tr.append(x[x.is_test == 0][['stay_id', 'subject_id', 'is_sepsis_stay', 'is_sepsis_6h',
                                                                        'target_resp', 'target_uri', 'target_other']])
    te = pd.concat(te, ignore_index=True); tr = pd.concat(tr, ignore_index=True)
    te[IDS + VIT].to_parquet(OUT + f'test_{g}_ids.parquet', index=False, compression='zstd')
    tr.to_parquet(OUT + f'train_{g}_ids.parquet', index=False, compression='zstd')
    if g == '1h':
        feats = [c for c in te.columns if c not in IDS + ['intime', 'sepsis3_time', 'is_test']]
        h = len(feats) // 2
        te[feats[:h]].to_parquet(OUT + 'test_1h_featsA.parquet', index=False, compression='zstd')
        te[feats[h:]].to_parquet(OUT + 'test_1h_featsB.parquet', index=False, compression='zstd')
    pq.read_table(RES + f'ultimate_full/ckpt_{g}/prepared.parquet',
                  columns=['stay_id', 'subject_id', 'charttime', 'is_sepsis_6h', 'is_test', 'age'] + VIT
                  ).to_pandas().to_parquet(OUT + f'all_{g}_vitals.parquet', index=False, compression='zstd')
    print(g, 'test rows', len(te), 'train rows', len(tr))
