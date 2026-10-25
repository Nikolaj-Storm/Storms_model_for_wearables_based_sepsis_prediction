# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  univariate_lift.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Univariate AUPRC lift of every feature on the hourly test rows (paper Fig.
#    4).
#
#  INPUTS   (under Results/ unless stated)
#    _extract/test_1h_ids.parquet, _extract/test_1h_feats{A,B}.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    data/fig4_univariate_1h.csv
#  RUN
#    python univariate_lift.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
R='/PATH/TO/PROJECT/technical/Results/paper_v3/_extract/'   # EDIT: point to your project / output folders
ids=pd.read_parquet(R+'test_1h_ids.parquet'); y=ids.is_sepsis_6h.values; prev=y.mean()
X=pd.concat([pd.read_parquet(R+'test_1h_featsA.parquet'),pd.read_parquet(R+'test_1h_featsB.parquet')],axis=1)
X['time_since_ICU_admit_hours']=ids.time_since_ICU_admit_hours.values
rows=[]
for c in X.columns:
    x=X[c].astype('float64'); x=x.fillna(x.median()).values
    a=roc_auc_score(y,x); s=1 if a>=0.5 else -1
    rows.append(dict(feature=c,auroc=max(a,1-a),lift=average_precision_score(y,s*x)/prev,direction=s))
u=pd.DataFrame(rows).sort_values('lift',ascending=False); u.to_csv('/PATH/TO/OUTPUT/data/fig4_univariate_1h.csv',index=False)   # EDIT: point to your project / output folders
print(u.head(25).round(3).to_string()); print(u.tail(10).round(3).to_string()); print(len(u), prev)
