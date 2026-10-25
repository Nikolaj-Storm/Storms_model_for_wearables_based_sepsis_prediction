# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  full_clinical_comparators.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Full clinical SIRS, qSOFA, NEWS2 and MEWS on exactly the scored rows
#    (paper Table 1, Fig. 2). Vitals are the last charted value; GCS,
#    supplemental oxygen and white cell count come from comparator_events.sql.
#    AVPU is mapped from GCS (A >= 14, V 11-13, P 7-10, U <= 6).
#
#  INPUTS   (under Results/ unless stated)
#    Data/mimic_derived_vitals_cohort.parquet, _extract/test_<grid>_ids.parquet, comparator_events.parquet, preds_<partition>_<grid>.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    table1_fullscores_<grid>.csv, curves_fullscores_<grid>.json, fullscores_<grid>.parquet
#  RUN
#    python full_clinical_comparators.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
R='/PATH/TO/PROJECT/technical/'; V4='/PATH/TO/OUTPUT/'   # EDIT: point to your project / output folders
ev=pd.read_parquet(V4+'comparator_events.parquet').set_index('stay_id')
def parse(s):
    if s is None or s!=s: return np.array([]),np.array([])
    a=np.array([x.split(':') for x in s.split(';')],dtype=float); o=np.argsort(a[:,0],kind='stable'); return a[o,0],a[o,1]
P={c:{sid:parse(v) for sid,v in ev[c].items()} for c in ('gcs_s','o2_s','wbc_s')}
# raw vitals LOCF (as in v3)
v=pd.read_parquet(R+'Data/mimic_derived_vitals_cohort.parquet',columns=['stay_id','charttime','itemid','valuenum'])
m={220045:'heart_rate',220210:'resprate',224690:'resprate',220277:'spo2',223761:'temp_f',223762:'temp_c',220179:'sbp',220050:'sbp',220180:'dbp',220051:'dbp'}
v['vn']=v.itemid.map(m); v=v.dropna(subset=['vn'])
f=v.vn=='temp_f'; v.loc[f,'valuenum']=(v.loc[f,'valuenum']-32)*5/9; v.loc[f,'vn']='temp_c'
B={'heart_rate':(10,300),'resprate':(1,100),'spo2':(20,100),'temp_c':(25,45),'sbp':(40,300),'dbp':(20,200)}
v=v[np.logical_or.reduce([(v.vn==k)&(v.valuenum>=a)&(v.valuenum<=b) for k,(a,b) in B.items()])]
v['charttime']=pd.to_datetime(v.charttime).astype('datetime64[ns]'); v['stay_id']=v.stay_id.astype('int64')
w=v.groupby(['stay_id','charttime','vn']).valuenum.mean().unstack('vn').reset_index().sort_values('charttime')
def band(x, edges, pts):
    out=np.array(pts)[np.searchsorted(edges, x, side='right')]; return np.where(np.isnan(x),0,out)
def lastval(stays, tmin, key, default=np.nan):
    out=np.full(len(stays),default,dtype=float)
    df=pd.DataFrame({'s':stays,'t':tmin,'i':np.arange(len(stays))})
    for sid,g in df.groupby('s'):
        tt,vv=P[key].get(sid,(np.array([]),np.array([])))
        if len(tt)==0: continue
        j=np.searchsorted(tt,g.t.values,side='right')-1; ok=j>=0
        out[g.i.values[ok]]=vv[j[ok]]
    return out
i1=pd.read_parquet(R+'Results/paper_v3/_extract/test_1h_ids.parquet')
_it=pd.to_datetime(i1.charttime).astype('datetime64[ns]')-pd.to_timedelta(i1.time_since_ICU_admit_hours,unit='h')
INTIME=pd.Series(_it.values).groupby(i1.stay_id.values).median()
CUT={"SIRS":2,"qSOFA":2,"NEWS2":5,"MEWS":5}
res={}
for g,off in (('4h',pd.Timedelta(0)),('1h',pd.Timedelta(minutes=59,seconds=59))):
    ids=pd.read_parquet(R+f'Results/paper_v3/_extract/test_{g}_ids.parquet'); y=ids.is_sepsis_6h.values
    ct=pd.to_datetime(ids.charttime).astype('datetime64[ns]')
    intime=ids.stay_id.map(INTIME)
    chk=pd.Series(intime.values).groupby(ids.stay_id.values).agg(lambda x:(x.max()-x.min()).total_seconds()).max(); print(g,'intime spread s',chk)
    tmin=((ct+off)-intime).dt.total_seconds().values/60
    q=pd.DataFrame({'i':np.arange(len(ids)),'stay_id':ids.stay_id.astype('int64').values,'t':(ct+off).values}).sort_values('t')
    ww=w[w.stay_id.isin(set(q.stay_id))]
    for vn in ['heart_rate','resprate','spo2','temp_c','sbp','dbp']:
        s=ww[['stay_id','charttime',vn]].dropna().rename(columns={'charttime':'t'}).sort_values('t')
        q=pd.merge_asof(q.sort_values('t'),s,on='t',by='stay_id',direction='backward')
    q=q.sort_values('i').reset_index(drop=True)
    st=ids.stay_id.astype('int64').values
    gcs=lastval(st,tmin,'gcs_s'); o2=lastval(st,tmin,'o2_s',0.0); wbc=lastval(st,tmin,'wbc_s')
    print(g,'missing gcs',np.isnan(gcs).mean().round(3),'wbc',np.isnan(wbc).mean().round(3),'o2 on',o2.mean().round(3))
    hr,rr,sp,t,sb=(q[c].values for c in ('heart_rate','resprate','spo2','temp_c','sbp'))
    n=lambda x: np.nan_to_num(x,nan=-1e9)
    # AVPU from GCS: midpoints between McNarry & Goldhill (2004) mean GCS per level (A15, V13, P8, U6)
    avpu=np.where(np.isnan(gcs),0,np.select([gcs>=14,gcs>=11,gcs>=7],[0,1,2],3))
    sirs=(n(hr)>90).astype(int)+(n(rr)>20)+((n(t)>38)|((t<36)&~np.isnan(t)))+((n(wbc)>12)|((wbc<4)&~np.isnan(wbc)))
    qsofa=(n(rr)>=22).astype(int)+((sb<=100)&~np.isnan(sb))+((gcs<15)&~np.isnan(gcs))
    news=(band(np.round(rr),[8,11,20,24],[3,1,0,2,3])+band(np.round(sp),[91,93,95],[3,2,1,0])+band(np.round(sb),[90,100,110,219],[3,2,1,0,3])
          +band(np.round(hr),[40,50,90,110,130],[3,1,0,1,2,3])+band(np.round(t,1),[35.0,36.0,38.0,39.0],[3,1,0,1,2])+np.where(avpu>0,3,0)+np.where(o2>0,2,0))
    mews=(band(np.round(sb),[70,80,100,199],[3,2,1,0,2])+band(np.round(hr),[40,50,100,110,129],[2,1,0,1,2,3])+band(np.round(rr),[8,14,20,29],[2,0,1,2,3])
          +band(np.round(t,1),[34.9,38.4],[2,0,2])+avpu)
    C={'SIRS':sirs.astype(float),'qSOFA':qsofa.astype(float),'NEWS2':news.astype(float),'MEWS':mews.astype(float)}
    rows=[]; curves={}; prev=y.mean()
    for k,p in C.items():
        fl=p>=CUT[k]; ap=average_precision_score(y,p)
        rows.append(dict(score=k,auroc=roc_auc_score(y,p),auprc=ap,lift=ap/prev,cut=CUT[k],sens_cut=fl[y==1].mean(),fire_rate=fl.mean(),ppv_cut=y[fl].mean(),fpr_cut=fl[y==0].mean()))
        fpr,tpr,_=roc_curve(y,p); pr,rc,_=precision_recall_curve(y,p)
        curves[k]=dict(fpr=fpr.tolist(),tpr=tpr.tolist(),prec=pr.tolist(),rec=rc.tolist(),cut=[rows[-1]['fpr_cut'],rows[-1]['sens_cut'],rows[-1]['ppv_cut']])
    tab=pd.DataFrame(rows); print(tab.round(4).to_string()); tab.to_csv(V4+f'table1_fullscores_{g}.csv',index=False)
    json.dump(dict(prev=prev,curves=curves),open(V4+f'curves_fullscores_{g}.json','w'))
    pd.DataFrame(C).assign(stay_id=st).to_parquet(V4+f'fullscores_{g}.parquet')
    for f_,c in (('kmeans','ULTIMATE 3L [kmeans] LR meta'),('etiology','ULTIMATE 3L [etiology] LR meta')):
        p=pd.read_parquet(R+f'Results/ultimate_full/preds_{f_}_{g}.parquet')[c].values
        for r in rows:
            thr=np.quantile(p,1-r['fire_rate']); fl=p>=thr; print(g,f_,'at',r['score'],'rate',round(r['fire_rate'],4),'sens',round(fl[y==1].mean(),3),'ppv',round(y[fl].mean(),4))
