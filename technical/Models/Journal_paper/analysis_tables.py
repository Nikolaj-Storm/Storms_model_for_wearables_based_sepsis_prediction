# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  analysis_tables.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Table 1 (vitals-only version), stream-by-label lifts, source attribution,
#    stream diversity and alert-burden numbers.
#
#  INPUTS   (under Results/ unless stated)
#    preds_*_<grid>.parquet, ckpt_1h/L1_*_final.npy, _extract/test_<grid>_ids.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    data/table1_bedside_<grid>.csv, data/fig5_stream_by_label.csv, data/misc.json
#  RUN
#    python analysis_tables.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
R='/PATH/TO/PROJECT/technical/Results/'   # EDIT: point to your project / output folders
O='/PATH/TO/OUTPUT/data/'   # EDIT: point to your project / output folders
T='is_sepsis_6h'
KM='ULTIMATE 3L [kmeans] LR meta'; ET='ULTIMATE 3L [etiology] LR meta'; SX='single XGBoost (global stream, 196 feats)'

def comparators(df):
    hr, rr, sp, t, sb = (df[c].values for c in ("heart_rate", "resprate", "spo2", "temp_c", "sbp"))
    n = lambda x: np.nan_to_num(x, nan=-1e9)
    sirs = (n(hr) > 90).astype(int) + (n(rr) > 20) + ((n(t) > 38) | ((t < 36) & ~np.isnan(t)))
    qsofa = (n(rr) >= 22).astype(int) + ((sb <= 100) & ~np.isnan(sb))
    def band(x, edges, pts):
        out = np.array(pts)[np.searchsorted(edges, x, side="right")]
        return np.where(np.isnan(x), 0, out)
    news = (band(np.round(rr), [8, 11, 20, 24], [3, 1, 0, 2, 3]) + band(np.round(sp), [91, 93, 95], [3, 2, 1, 0])
            + band(np.round(sb), [90, 100, 110, 219], [3, 2, 1, 0, 3]) + band(np.round(hr), [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
            + band(np.round(t, 1), [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2]))
    mews = (band(np.round(sb), [70, 80, 100, 199], [3, 2, 1, 0, 2]) + band(np.round(hr), [40, 50, 100, 110, 129], [2, 1, 0, 1, 2, 3])
            + band(np.round(rr), [8, 14, 20, 29], [2, 0, 1, 2, 3]) + band(np.round(t, 1), [34.9, 38.4], [2, 0, 2]))
    return {"SIRS": sirs.astype(float), "qSOFA": qsofa.astype(float), "NEWS2": news.astype(float), "MEWS": mews.astype(float)}
CUT={"SIRS":2,"qSOFA":2,"NEWS2":5,"MEWS":5}

def fa_per_tp(y,p,sens):
    o=np.argsort(-p); yy=y[o]; tp=np.cumsum(yy); fp=np.cumsum(1-yy)
    i=np.searchsorted(tp, sens*yy.sum()); return fp[i]/tp[i]
def recall_at_fa(p, y, fa):
    o = np.argsort(-p); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    ok = fp <= fa * np.maximum(tp, 1); return float(tp[ok].max() / yy.sum()) if ok.any() else 0.0

out={}
# ---------------- Fig 2 / Table 1 : bedside scores vs models, both grids
for g in ('4h','1h'):
    ids=pd.read_parquet(R+f'paper_v3/_extract/test_{g}_ids.parquet')
    pk=pd.read_parquet(R+f'ultimate_full/preds_kmeans_{g}.parquet'); pe=pd.read_parquet(R+f'ultimate_full/preds_etiology_{g}.parquet')
    y=ids[T].values; prev=y.mean()
    S={'Stacked ensemble, physiology clusters':pk[KM].values,'Stacked ensemble, etiology specialists':pe[ET].values,'Single XGBoost':pk[SX].values}
    S.update(comparators(ids))
    rows=[]; curves={}
    for k,p in S.items():
        au=roc_auc_score(y,p); ap=average_precision_score(y,p)
        r=dict(score=k,auroc=au,auprc=ap,lift=ap/prev,prev=prev, sens10=recall_at_fa(p,y,10), sens20=recall_at_fa(p,y,20))
        if k in CUT:
            f=p>=CUT[k]; r.update(cut=CUT[k],sens_cut=f[y==1].mean(),fire_rate=f.mean(),ppv_cut=y[f].mean(), fpr_cut=f[y==0].mean())
        else:
            r['fa_at_80']=fa_per_tp(y,p,0.8)
        rows.append(r)
        fpr,tpr,_=roc_curve(y,p); pr,rc,_=precision_recall_curve(y,p)
        step=max(1,len(fpr)//1500); s2=max(1,len(pr)//1500)
        curves[k]=dict(fpr=fpr[::step].tolist()+[1.0],tpr=tpr[::step].tolist()+[1.0],prec=pr[::s2].tolist(),rec=rc[::s2].tolist(),
                       cut=([r['fpr_cut'],r['sens_cut'],r['ppv_cut']] if k in CUT else None))
    t=pd.DataFrame(rows); t.to_csv(O+f'table1_bedside_{g}.csv',index=False); print(g, '\n', t.round(4).to_string())
    json.dump(dict(prev=prev,curves=curves),open(O+f'fig2_curves_{g}.json','w'))

# ---------------- Fig 5 / explainability : etiology streams x labels (1h)
ids=pd.read_parquet(R+'paper_v3/_extract/test_1h_ids.parquet'); y=ids[T].values
pe=pd.read_parquet(R+'ultimate_full/preds_etiology_1h.parquet'); pk=pd.read_parquet(R+'ultimate_full/preds_kmeans_1h.parquet')
ck=R+'ultimate_full/ckpt_1h/'
L=lambda k: np.load(ck+f'L1_{k}_final.npy')
streams={'Global':('global_0',),'Respiratory':('etiology_0',),'Urinary':('etiology_1',),'Other':('etiology_2',)}
labels={'Global':y,'Respiratory':y*ids.target_resp.values,'Urinary':y*ids.target_uri.values,'Other':y*ids.target_other.values}
res=[]
for learner in ('xgb','rf'):
    for sn,(key,) in streams.items():
        p=L(f'{key}_{learner}')
        for ln,yl in labels.items():
            ap=average_precision_score(yl,p); res.append(dict(learner=learner,stream=sn,label=ln,lift=ap/yl.mean(),auroc=roc_auc_score(yl,p),prev=yl.mean()))
for nm,p in (('Etiology stack (L3)',pe[ET].values),('Cluster stack (L3)',pk[KM].values),('Single XGBoost',pk[SX].values)):
    for ln,yl in labels.items():
        ap=average_precision_score(yl,p); res.append(dict(learner='meta',stream=nm,label=ln,lift=ap/yl.mean(),auroc=roc_auc_score(yl,p),prev=yl.mean()))
f5=pd.DataFrame(res); f5.to_csv(O+'fig5_stream_by_label.csv',index=False)
print(f5.pivot_table(index=['learner','stream'],columns='label',values='lift').round(2))
print(f5.pivot_table(index=['learner','stream'],columns='label',values='auroc').round(3))

# attribution: among positive rows with a known source, does the highest-ranked specialist name the source?
pct=lambda p: pd.Series(p).rank(pct=True).values
for learner in ('xgb','rf'):
    E=np.column_stack([pct(L(f'etiology_{k}_{learner}')) for k in range(3)])
    tru=np.argmax(ids[['target_resp','target_uri','target_other']].values,axis=1)
    m=(y==1)&(ids[['target_resp','target_uri','target_other']].sum(axis=1).values==1)
    pred=E[m].argmax(1); acc=(pred==tru[m]).mean()
    freq=np.bincount(tru[m],minlength=3)/m.sum()
    # stay-level: last positive row (closest to onset) per stay
    dd=pd.DataFrame(dict(stay=ids.stay_id.values[m],pred=pred,tru=tru[m]))
    last=dd.groupby('stay').tail(1); accs=(last.pred==last.tru).mean()
    cm=pd.crosstab(dd.tru,dd.pred,normalize='index')
    out[f'attr_{learner}']=dict(row_acc=acc, majority=freq.max(), chance_prop=float((freq**2).sum()), n_rows=int(m.sum()), stay_acc=accs, n_stays=int(len(last)), cm=cm.round(3).values.tolist())
    print(learner,out[f'attr_{learner}'])

# ---------------- Fig 6 : diversity among streams (test set) vs performance
fam={'etiology':[f'etiology_{k}' for k in range(3)],'kmeans5':[f'kmeans5_{k}' for k in range(5)],'kmeans6':[f'kmeans6_{k}' for k in range(6)]}
div={}
for learner in ('xgb','rf'):
    g0=L(f'global_0_{learner}')
    for f,ks in fam.items():
        M=np.column_stack([L(f'{k}_{learner}') for k in ks]); C=np.corrcoef(M.T); iu=np.triu_indices(len(ks),1)
        div[f'{f}_{learner}']=dict(mean_r_within=float(C[iu].mean()), min_r=float(C[iu].min()), max_r=float(C[iu].max()),
            mean_r_global=float(np.mean([np.corrcoef(M[:,i],g0)[0,1] for i in range(len(ks))])))
    # all specialists combined kmeans
    M=np.column_stack([L(f'{k}_{learner}') for k in fam['kmeans5']+fam['kmeans6']]); C=np.corrcoef(M.T); iu=np.triu_indices(M.shape[1],1)
    div[f'kmeans_all_{learner}']=dict(mean_r_within=float(C[iu].mean()))
out['diversity']=div; print(json.dumps(div,indent=1))
# per-stream standalone lift on global label (k-means)
ks=[]
for f in ('kmeans5','kmeans6'):
    for k in fam[f]:
        for learner in ('xgb',):
            p=L(f'{k}_{learner}'); ks.append(dict(stream=k,lift=average_precision_score(y,p)/y.mean(),auroc=roc_auc_score(y,p)))
out['kmeans_stream_global']=ks; print(pd.DataFrame(ks).round(3))
# subgroup AUROC of stacks: etiology positives vs all negatives
sub=[]
for ln,col in (('Respiratory','target_resp'),('Urinary','target_uri'),('Other','target_other')):
    m=(y==0)|((y==1)&(ids[col].values==1))
    for nm,p in (('etiology',pe[ET].values),('kmeans',pk[KM].values),('single',pk[SX].values)):
        sub.append(dict(label=ln,model=nm,auroc=roc_auc_score(y[m],p[m])))
out['subgroup_auroc']=sub; print(pd.DataFrame(sub).pivot(index='label',columns='model',values='auroc').round(4))
# alert burden curve (1h) for both stacks
ab={}
for nm,p in (('kmeans',pk[KM].values),('etiology',pe[ET].values),('single',pk[SX].values)):
    o=np.argsort(-p); yy=y[o]; tp=np.cumsum(yy); fp=np.cumsum(1-yy); sens=tp/yy.sum(); fa=fp/np.maximum(tp,1)
    ab[nm]={str(s):float(fa[np.searchsorted(sens,s)]) for s in (0.2,0.4,0.5,0.6,0.8)}
    ab[nm].update({f'sens@{k}FA':recall_at_fa(p,y,k) for k in (5,10,20,30)})
out['alert_burden_1h']=ab; print(json.dumps(ab,indent=1))
json.dump(out,open(O+'misc.json','w'),indent=1,default=float)
