# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  no_global_stream_ablation.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Refits layers 2 and 3 of both stacks with and without the global stream
#    from the cached layer-1 out-of-fold predictions, with a paired patient
#    bootstrap (paper section 'Why the cluster stack gains'). Use scikit-learn
#    1.6.1 for the fold assignment to reproduce the original folds.
#
#  INPUTS   (under Results/ unless stated)
#    ckpt_1h/L1_*_{0..4,final}.npy, _extract/train_1h_ids.parquet, _extract/test_1h_ids.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    noglobal_results_1h.csv, noglobal_deltas_1h.csv
#  RUN
#    python no_global_stream_ablation.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import numpy as np, pandas as pd, json, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
R='/PATH/TO/PROJECT/technical/Results/'; CK=R+'ultimate_full/ckpt_1h/'   # EDIT: point to your project / output folders
O='/PATH/TO/OUTPUT/'   # EDIT: point to your project / output folders
T='is_sepsis_6h'
tr=pd.read_parquet(R+'paper_v3/_extract/train_1h_ids.parquet'); te=pd.read_parquet(R+'paper_v3/_extract/test_1h_ids.parquet')
y=tr[T].values; yt=te[T].values
def mk_folds(f, seed):
    """Fold assignment of the original run. StratifiedGroupKFold's shuffling changed across scikit-learn versions,
    so the assignment is generated once (scikit-learn 1.6.1 reproduces the original folds) and cached."""
    import os
    if not os.path.exists(O+f):
        fl=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(tr,tr.is_sepsis_stay,tr.subject_id))
        a=np.empty(len(tr),dtype=int)
        for i,(_,b) in enumerate(fl): a[b]=i
        np.save(O+f,a)
    a=np.load(O+f); return [(np.where(a!=i)[0],np.where(a==i)[0]) for i in range(5)]
folds=mk_folds('f1.npy',42)
for i,(a,b) in enumerate(folds):
    n=len(np.load(CK+f'L1_global_0_rf_{i}.npy')); assert n==len(b),(i,n,len(b))
print('folds match sizes')
def L1(key):
    o=np.zeros(len(tr),dtype='float32')
    for i,(a,b) in enumerate(folds): o[b]=np.load(CK+f'L1_{key}_{i}.npy')
    return o, np.load(CK+f'L1_{key}_final.npy')
addvar=lambda X: np.column_stack([X,X.var(axis=1)])
lr=lambda: make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=3000))
def xs(y):
    spw=float((y==0).sum())/max(float(y.sum()),1.)
    return XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=50,scale_pos_weight=spw,tree_method='hist',n_jobs=-1,random_state=0,verbosity=0)
folds2=mk_folds('f2.npy',7)
fam={'etiology':[f'etiology_{k}' for k in range(3)],'kmeans5':[f'kmeans5_{k}' for k in range(5)],'kmeans6':[f'kmeans6_{k}' for k in range(6)]}
cache={}
def get(k):
    if k not in cache: cache[k]=L1(k)
    return cache[k]
def stack(families, with_global):
    O2,T2=[],[]
    for f in families:
        keys=(['global_0'] if with_global else [])+fam[f]
        cols=[f'{k}_{l}' for k in keys for l in ('rf','xgb')]
        Xo=addvar(np.column_stack([get(c)[0] for c in cols])); Xt=addvar(np.column_stack([get(c)[1] for c in cols]))
        for mk in (lr, lambda: xs(y)):
            o=np.zeros(len(tr))
            for a,b in folds2: o[b]=mk().fit(Xo[a],y[a]).predict_proba(Xo[b])[:,1]
            O2.append(o); T2.append(mk().fit(Xo,y).predict_proba(Xt)[:,1])
    return lr().fit(addvar(np.column_stack(O2)),y).predict_proba(addvar(np.column_stack(T2)))[:,1]
P={}
t0=time.time()
for name,fams,g in (('Etiology stack (with global)',['etiology'],True),('Etiology stack (no global)',['etiology'],False),
                    ('Cluster stack (with global)',['kmeans5','kmeans6'],True),('Cluster stack (no global)',['kmeans5','kmeans6'],False)):
    P[name]=stack(fams,g); print(name, round(roc_auc_score(yt,P[name]),4), f'{time.time()-t0:.0f}s', flush=True)
P['Single XGBoost (global stream)']=get('global_0_xgb')[1]
pd.DataFrame(P).assign(stay_id=te.stay_id.values).to_parquet(O+'preds_noglobal_1h.parquet')
# bootstrap
pt=te.groupby('subject_id').is_sepsis_stay.max(); code=te.subject_id.map({s:i for i,s in enumerate(pt.index)}).values
pos,neg=np.where(pt.values==1)[0],np.where(pt.values==0)[0]; rng=np.random.default_rng(0)
class WS:
    def __init__(s,p):
        u,inv=np.unique(-p,return_inverse=True); s.inv,s.nu=inv,len(u)
    def auc(s,sw):
        w=sw[code]; yb=yt.astype(bool)
        pw=np.bincount(s.inv[yb],w[yb],s.nu); nw=np.bincount(s.inv[~yb],w[~yb],s.nu)
        return 1-np.sum(pw*(np.cumsum(nw)-nw+0.5*nw))/(pw.sum()*nw.sum())
W=[np.bincount(np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]),minlength=len(pt)).astype(float) for _ in range(1000)]
S={k:WS(p) for k,p in P.items()}; A={k:np.array([S[k].auc(w) for w in W]) for k in P}
def recall_at_fa(p,fa):
    o=np.argsort(-p); yy=yt[o]; tp=np.cumsum(yy); fp=np.cumsum(1-yy); ok=fp<=fa*np.maximum(tp,1); return float(tp[ok].max()/yy.sum())
rows=[]
for k,p in P.items():
    rows.append(dict(model=k,auroc=roc_auc_score(yt,p),lo=np.quantile(A[k],.025),hi=np.quantile(A[k],.975),lift=average_precision_score(yt,p)/yt.mean(),sens10=recall_at_fa(p,10),sens20=recall_at_fa(p,20)))
res=pd.DataFrame(rows); print(res.round(4).to_string())
d=[]
for a_,b_ in (('Etiology stack (no global)','Etiology stack (with global)'),('Cluster stack (no global)','Cluster stack (with global)'),
              ('Etiology stack (no global)','Single XGBoost (global stream)'),('Cluster stack (no global)','Single XGBoost (global stream)'),
              ('Cluster stack (no global)','Etiology stack (no global)')):
    x=A[a_]-A[b_]; d.append(dict(comparison=f'{a_} - {b_}',delta=float(np.median(x)),lo=float(np.quantile(x,.025)),hi=float(np.quantile(x,.975))))
dd=pd.DataFrame(d); print(dd.round(4).to_string())
res.to_csv(O+'noglobal_results_1h.csv',index=False); dd.to_csv(O+'noglobal_deltas_1h.csv',index=False)
