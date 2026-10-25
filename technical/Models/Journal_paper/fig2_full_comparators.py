# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  fig2_full_comparators.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Draws Fig. 2 with the full bedside scores and the hourly Barton
#    comparator, and writes the matching Table 1 values.
#
#  INPUTS   (under Results/ unless stated)
#    fullscores_<grid>.parquet, barton_hourly_<grid>.parquet, preds_*_<grid>.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    fig2_full_comparators_<grid>.png, table1_v4_<grid>.csv
#  RUN
#    python fig2_full_comparators.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#eeeeee','axes.axisbelow':True})
R='/PATH/TO/PROJECT/technical/Results/'; V4='/PATH/TO/OUTPUT/'   # EDIT: point to your project / output folders
for g in ('4h','1h'):
    ids=pd.read_parquet(R+f'paper_v3/_extract/test_{g}_ids.parquet'); y=ids.is_sepsis_6h.values; prev=y.mean()
    S={'Stack, physiology clusters':(pd.read_parquet(R+f'ultimate_full/preds_kmeans_{g}.parquet')['ULTIMATE 3L [kmeans] LR meta'].values,'#1f3a5f',2.6,'-'),
       'Stack, etiology specialists':(pd.read_parquet(R+f'ultimate_full/preds_etiology_{g}.parquet')['ULTIMATE 3L [etiology] LR meta'].values,'#2a9d8f',1.8,'-'),
       'Barton et al. (2019), reconstructed':(pd.read_parquet(V4+f'barton_hourly_{g}.parquet').barton31.values,'#7b4fa0',1.6,'--')}
    fs=pd.read_parquet(V4+f'fullscores_{g}.parquet'); CUT={'NEWS2':5,'MEWS':5,'SIRS':2,'qSOFA':2}
    for k,c in (('NEWS2','#c0392b'),('MEWS','#e69124'),('SIRS','#6b7a8f'),('qSOFA','#4c9a4c')): S[k]=(fs[k].values,c,1.5,'-')
    rows=[]
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.6))
    for k,(p,c,lw,ls) in S.items():
        au=roc_auc_score(y,p); apv=average_precision_score(y,p); rows.append(dict(model=k,auroc=au,auprc=apv,lift=apv/prev))
        fpr,tpr,_=roc_curve(y,p); pr,rc,_=precision_recall_curve(y,p); s=max(1,len(pr)//3000)
        ax[0].plot(fpr,tpr,color=c,lw=lw,ls=ls,label=f'{k} ({au:.3f})'); ax[1].plot(rc[::s],pr[::s],color=c,lw=lw,ls=ls,label=f'{k} ({apv:.4f})')
        if k in CUT:
            f=p>=CUT[k]; se=f[y==1].mean(); fp=f[y==0].mean(); pp=y[f].mean()
            ax[0].plot(fp,se,'o',mfc='white',mec=c,mew=2,ms=7,zorder=5); ax[1].plot(se,pp,'o',mfc='white',mec=c,mew=2,ms=7,zorder=5)
            rows[-1].update(cut=CUT[k],sens_cut=se,flagged=f.mean(),ppv_cut=pp)
    ax[0].plot([0,1],[0,1],ls=':',color='#bbbbbb',lw=1); ax[1].axhline(prev,ls=':',color='#bbbbbb',lw=1)
    ax[0].set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='Sensitivity',title='Receiver operating characteristic')
    ax[1].set(xlim=(0,1),ylim=(0,0.2),xlabel='Recall',ylabel='Precision',title='Precision-recall')
    ax[0].legend(title='AUROC',fontsize=8,title_fontsize=8.5,loc='lower right',frameon=False); ax[1].legend(title='AUPRC',fontsize=8,title_fontsize=8.5,loc='upper right',frameon=False)
    fig.tight_layout(); fig.savefig(V4+f'fig2_full_comparators_{g}.png',dpi=220); plt.close(fig)
    t=pd.DataFrame(rows); t.to_csv(V4+f'table1_v4_{g}.csv',index=False); print(g); print(t.round(4).to_string())
