# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  barton_hourly_comparator.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Barton et al. (2019) features (each vital at the scored row and the two
#    rows before, two differences, age) retrained with XGBoost on the six-hour
#    hourly task (paper Table 1, Fig. 2, Appendix A).
#
#  INPUTS   (under Results/ unless stated)
#    _extract/all_<grid>_vitals.parquet, _extract/test_<grid>_ids.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    barton_hourly_<grid>.parquet
#  RUN
#    python barton_hourly_comparator.py 1h all_1h_vitals.parquet   (and 4h all_4h_vitals.parquet)
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import numpy as np, pandas as pd, sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
g=sys.argv[1]; fn=sys.argv[2]
R='/PATH/TO/PROJECT/technical/Results/'; V4='/PATH/TO/OUTPUT/'   # EDIT: point to your project / output folders
d=pd.concat([pd.read_parquet(R+'paper_v3/_extract/'+f) for f in fn.split(',')],ignore_index=True)
d['charttime']=pd.to_datetime(d.charttime)
order=np.arange(len(d)); d['_o']=order
d=d.sort_values(['stay_id','charttime'],kind='stable')
VIT=['heart_rate','resprate','spo2','temp_c','sbp','dbp']; F=[]
gb=d.groupby('stay_id',sort=False)
for v in VIT:
    d[f'{v}_t1']=gb[v].shift(1); d[f'{v}_t2']=gb[v].shift(2)
    d[f'{v}_d01']=d[v]-d[f'{v}_t1']; d[f'{v}_d12']=d[f'{v}_t1']-d[f'{v}_t2']
    F+= [v,f'{v}_t1',f'{v}_t2',f'{v}_d01',f'{v}_d12']
F.append('age')
d=d.sort_values('_o')
tr=d[d.is_test==0]; te=d[d.is_test==1]
X=tr[F].to_numpy('float32'); y=tr.is_sepsis_6h.values
a,v=next(GroupShuffleSplit(1,test_size=0.1,random_state=0).split(X,y,tr.subject_id.values))
spw=float((y[a]==0).sum())/y[a].sum()
m=XGBClassifier(n_estimators=3000,max_depth=3,learning_rate=0.1,tree_method='hist',eval_metric='auc',early_stopping_rounds=50,scale_pos_weight=spw,n_jobs=-1,random_state=0,verbosity=0)
m.fit(X[a],y[a],eval_set=[(X[v],y[v])],verbose=False)
p=m.predict_proba(te[F].to_numpy('float32'))[:,1]
ids=pd.read_parquet(R+f'paper_v3/_extract/test_{g}_ids.parquet')
assert (ids.stay_id.values==te.stay_id.values).all()
yt=te.is_sepsis_6h.values
print(g,'Barton lag-31 hourly task AUROC',round(roc_auc_score(yt,p),4),'lift',round(average_precision_score(yt,p)/yt.mean(),3),'trees',m.best_iteration)
pd.DataFrame({'stay_id':te.stay_id.values,'barton31':p}).to_parquet(V4+f'barton_hourly_{g}.parquet')
