# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  figures_2_to_6.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Draws Figs. 3, 4 and 6 (and earlier versions of Figs. 2 and 5) from the
#    outputs of analysis_tables.py and univariate_lift.py.
#
#  INPUTS   (under Results/ unless stated)
#    data/*.csv, data/misc.json, preds_*.parquet
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    figures/fig*.png
#  RUN
#    python figures_2_to_6.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
B='/PATH/TO/OUTPUT/'; D=B+'data/'; F=B+'figures/'   # EDIT: point to your project / output folders
R='/PATH/TO/PROJECT/technical/Results/'   # EDIT: point to your project / output folders
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#eeeeee','grid.linewidth':0.8,'axes.axisbelow':True})
NAVY='#1f3a5f'; TEAL='#2a9d8f'; GREY='#8a8f98'
COL={'NEWS2':'#c0392b','MEWS':'#e69124','qSOFA':'#4c9a4c','SIRS':'#6b7a8f'}
KM='Stacked ensemble, physiology clusters'; ET='Stacked ensemble, etiology specialists'; SX='Single XGBoost'
# ---------- Fig 2
for g in ('4h',):
    d=json.load(open(D+f'fig2_curves_{g}_locf.json')); C=d['curves']; prev=d['prev']; t=pd.read_csv(D+f'table1_bedside_{g}.csv').set_index('score')
    tl=pd.read_csv(D+f'table1_bedside_locf_{g}.csv').set_index('score'); t.loc[tl.index,['auroc','auprc']]=tl[['auroc','auprc']]
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.6))
    order=[(KM,NAVY,2.6,'-'),(ET,TEAL,1.8,'-'),(SX,GREY,1.4,'--')]+[(k,COL[k],1.5,'-') for k in ('NEWS2','MEWS','SIRS','qSOFA')]
    for k,c,lw,ls in order:
        lab={KM:'Stack, physiology clusters',ET:'Stack, etiology specialists',SX:'Single XGBoost'}.get(k,k+' (vitals)')
        ax[0].plot(C[k]['fpr'],C[k]['tpr'],color=c,lw=lw,ls=ls,label=f"{lab} ({t.loc[k,'auroc']:.3f})")
        ax[1].plot(C[k]['rec'],C[k]['prec'],color=c,lw=lw,ls=ls,label=f"{lab} ({t.loc[k,'auprc']:.4f})")
        if C[k]['cut']:
            fpr,se,ppv=C[k]['cut']
            ax[0].plot(fpr,se,'o',mfc='white',mec=c,mew=2,ms=7,zorder=5); ax[1].plot(se,ppv,'o',mfc='white',mec=c,mew=2,ms=7,zorder=5)
    ax[0].plot([0,1],[0,1],ls=':',color='#bbbbbb',lw=1); ax[1].axhline(prev,ls=':',color='#bbbbbb',lw=1); ax[1].text(0.40,prev-0.009,f'chance {prev:.3f}',color='#999999',fontsize=8.5)
    ax[0].set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='Sensitivity',title='Receiver operating characteristic')
    ax[1].set(xlim=(0,1),ylim=(0,0.2),xlabel='Recall',ylabel='Precision',title='Precision-recall')
    ax[0].legend(title='AUROC',fontsize=8.3,title_fontsize=8.5,loc='lower right',frameon=False); ax[1].legend(title='AUPRC',fontsize=8.3,title_fontsize=8.5,loc='upper right',frameon=False)
    fig.tight_layout(); fig.savefig(F+'fig2_roc_pr_4h.png',dpi=220); plt.close(fig)
# ---------- Fig 3  cadence PR
fig,ax=plt.subplots(2,1,figsize=(4.8,6.2),sharex=True)
for i,(g,ttl) in enumerate((('4h','4-hour sampling (ward cadence)'),('1h','1-hour sampling (wearable proxy)'))):
    ids=pd.read_parquet(R+f'paper_v3/_extract/test_{g}_ids.parquet'); y=ids.is_sepsis_6h.values; prev=y.mean()
    pk=pd.read_parquet(R+f'ultimate_full/preds_kmeans_{g}.parquet')['ULTIMATE 3L [kmeans] LR meta'].values
    pe=pd.read_parquet(R+f'ultimate_full/preds_etiology_{g}.parquet')['ULTIMATE 3L [etiology] LR meta'].values
    for p,c,lw,lab in ((pe,TEAL,1.3,'Etiology specialists'),(pk,NAVY if g=='1h' else '#8fa9c8',2.2,'Physiology clusters')):
        pr,rc,_=precision_recall_curve(y,p); s=max(1,len(pr)//3000)
        ax[i].plot(rc[::s],pr[::s],color=c,lw=lw,label=lab)
        if lw>2: ax[i].fill_between(rc[::s],pr[::s],color=c,alpha=0.12)
    apk=average_precision_score(y,pk); ape=average_precision_score(y,pe)
    ax[i].axhline(prev,ls='--',color='#aaaaaa',lw=1)
    ax[i].text(0.98,0.24,f'clusters AUPRC {apk:.4f}, {apk/prev:.2f}× chance, AUROC {roc_auc_score(y,pk):.3f}\netiology AUPRC {ape:.4f}, {ape/prev:.2f}× chance, AUROC {roc_auc_score(y,pe):.3f}',ha='right',va='top',fontsize=7.6,color='#333')
    ax[i].set(ylim=(0,0.35),xlim=(0,1),ylabel='Precision'); ax[i].set_title(ttl,loc='left',fontsize=10)
ax[0].legend(fontsize=8,frameon=False,loc='upper right',bbox_to_anchor=(1,1.0)); ax[1].set_xlabel('Recall')
fig.tight_layout(); fig.savefig(F+'fig3_cadence_pr.png',dpi=220); plt.close(fig)
# ---------- Fig 4  univariate lift heatmap
u=pd.read_csv(D+'fig4_univariate_1h.csv').set_index('feature').lift
V=['heart_rate','resprate','sbp','dbp','temp_c','spo2']; VL=['HR','RR','SBP','DBP','Temp','SpO2']
T=[('Raw value','{v}'),('Expanding min','exp_min_{v}'),('Expanding max','exp_max_{v}'),('Expanding mean','exp_mean_{v}'),('Expanding SD','exp_std_{v}'),
   ('Rolling SD (4h)','{v}_sd_4h'),('EWMA (3h)','{v}_ewma_3h'),('Slope (4h)','slope_4h_{v}'),('Lag diff (1h)','lag_diff_1h_{v}'),('Lag ratio (1h)','lag_ratio_1h_{v}'),
   ('Acceleration','accel_{v}'),('CUSUM positive drift','cusum_pos_{v}'),('CUSUM negative drift','cusum_neg_{v}'),
   ('24-h mean','h24_mean_{v}'),('24-h SD','h24_sd_{v}'),('24-h min','h24_min_{v}'),('24-h max','h24_max_{v}'),('24-h slope','h24_slope_{v}'),('Deviation from first 6 h','dev_baseline_{v}')]
M=np.array([[u[f.format(v=v)] for v in V] for _,f in T])
P=[('Shock index','shock_index'),('Modified shock index','msi'),('Mean arterial pressure','map'),('Pulse pressure','pulse_pressure'),('Rate-pressure product','rpp'),
   ('NEWS (vitals)','news_score'),('NEWS delta','news_delta'),('Partial qSOFA','partial_qsofa'),('qSOFA delta','qsofa_delta'),
   ('Fever-driven tachycardia','fever_tachycardia'),('Tachycardia excess','tachycardia_excess'),('Perfusion adequacy','perfusion_adequacy'),('Resp distress (CRDI)','resp_distress'),
   ('Ventilation/perfusion proxy','ventilation_perfusion_proxy'),('Cardiorespiratory coupling','cardiorespiratory_coupling_4h'),
   ('Hours MAP < 65 (24 h)','shk_hours_map_lt65_24h'),('Hours SI > 1 (24 h)','shk_hours_si_gt1_24h'),('Min MAP (24 h)','shk_min_map_24h'),('Max SI (24 h)','shk_max_si_24h'),
   ('Time since ICU admission','time_since_ICU_admit_hours'),('Age','age'),('Weight','weight_kg')]
N=np.array([[u[f]] for _,f in P])
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('t',['#f3efe6','#c9dbd6','#6fa39f','#2f6b6b','#173f40'])
vmin,vmax=0.9,1.8
fig=plt.figure(figsize=(11.5,8.6)); a=fig.add_axes([0.13,0.07,0.36,0.85]); b=fig.add_axes([0.70,0.07,0.06,0.85]); cax=fig.add_axes([0.93,0.3,0.015,0.4])
for ax_,Mx,rl,cl in ((a,M,[t for t,_ in T],VL),(b,N,[p for p,_ in P],['Lift'])):
    im=ax_.imshow(Mx,cmap=cm,vmin=vmin,vmax=vmax,aspect='auto'); ax_.grid(False)
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            ax_.text(j,i,f'{Mx[i,j]:.2f}',ha='center',va='center',fontsize=7.5,color='white' if Mx[i,j]>1.45 else '#222')
    ax_.set_xticks(range(len(cl))); ax_.set_xticklabels(cl); ax_.set_yticks(range(len(rl)))
    for s in ax_.spines.values(): s.set_visible(True)
a.set_yticklabels([t for t,_ in T],fontsize=8.6); b.yaxis.tick_right(); b.set_yticklabels([p for p,_ in P],fontsize=8.6)
a.set_title('(a) Kinematic and temporal features\nrows: transformation, columns: vital sign',loc='left',fontsize=10)
b.set_title('(b) Composites, interactions,\nshock proxies and context',loc='left',fontsize=10)
for yy in (8.5,14.5,18.5): b.axhline(yy,color='black',lw=1.2)
a.axhline(12.5,color='black',lw=1.2)
cb=fig.colorbar(im,cax=cax); cb.set_label('univariate AUPRC lift (AUPRC / prevalence)'); cb.ax.axhline(1.0,color='#c0392b',lw=1.5)
fig.savefig(F+'fig4_univariate_lift.png',dpi=220); plt.close(fig)
# ---------- Fig 5  stream x label + attribution
f5=pd.read_csv(D+'fig5_stream_by_label.csv'); misc=json.load(open(D+'misc.json'))
labs=['Global','Respiratory','Urinary','Other']; ests=[('Global','xgb'),('Respiratory','xgb'),('Urinary','xgb'),('Other','xgb'),('Etiology stack (L3)','meta')]
EC={'Global':'#d9604c','Respiratory':'#5fbf9f','Urinary':'#9b8fd6','Other':'#c7b84f','Etiology stack (L3)':'#333333'}
fig=plt.figure(figsize=(12.5,4.6)); a=fig.add_axes([0.06,0.13,0.6,0.68]); b=fig.add_axes([0.74,0.14,0.22,0.72])
w=0.16
for j,(e,l) in enumerate(ests):
    for i,lab in enumerate(labs):
        v=f5[(f5.stream==e)&(f5.learner==l)&(f5.label==lab)].lift.iloc[0]
        own=(e==lab)
        a.bar(i+(j-2)*w,v,w*0.92,color=EC[e] if own else 'white',edgecolor=EC[e],hatch=None if own else '///',lw=1.4,label=(e if e!='Global' else 'Global stream') if i==0 else None)
        a.text(i+(j-2)*w,v+0.06,f'{v:.2f}',ha='center',fontsize=7.3)
a.axhline(1,ls='--',color='#aaaaaa',lw=1); a.set_xticks(range(4)); a.set_xticklabels(labs); a.set_xlabel('Evaluation label'); a.set_ylabel('Lift over chance (AUPRC / prevalence)')
a.set_ylim(0,5.9); a.legend(ncol=5,fontsize=8.3,frameon=False,loc='upper center',bbox_to_anchor=(0.5,1.11)); a.set_title('(a) Each estimator scored against each label',loc='left',fontsize=10,pad=30)
cmx=np.array(misc['attr_xgb']['cm']); im=b.imshow(cmx,cmap=cm,vmin=0,vmax=0.6); b.grid(False)
for i in range(3):
    for j in range(3): b.text(j,i,f'{cmx[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if cmx[i,j]>0.42 else '#222')
b.set_xticks(range(3)); b.set_xticklabels(['Resp','Urin','Other']); b.set_yticks(range(3)); b.set_yticklabels(['Resp','Urin','Other'])
b.set_xlabel('Highest-ranked specialist'); b.set_ylabel('Coded infection source')
b.set_title('(b) Source attribution, septic hours',loc='left',fontsize=10)
fig.savefig(F+'fig5_streams_by_label.png',dpi=220); plt.close(fig)
# ---------- Fig 6  diversity vs gain
dl=pd.read_csv(R+'ultimate_full/paired_deltas_stacking.csv'); dl=dl[dl.grid=='1h']
div=misc['diversity']
fig,ax=plt.subplots(1,2,figsize=(10.5,4.2))
rows=[('Etiology\nspecialists','ULTIMATE 3L etiology (LR meta) - single XGBoost',TEAL),('Physiology\nclusters','ULTIMATE 3L kmeans (LR meta) - single XGBoost',NAVY)]
for i,(lab,key,c) in enumerate(rows):
    r=dl[dl.comparison==key].iloc[0]
    ax[0].bar(i,r.d_auroc_median*1000,0.55,color=c); ax[0].errorbar(i,r.d_auroc_median*1000,yerr=[[ (r.d_auroc_median-r.ci_lo)*1000],[(r.ci_hi-r.d_auroc_median)*1000]],color='black',capsize=5,lw=1.2)
    ax[0].text(i+0.3,r.d_auroc_median*1000+0.2,f'+{r.d_auroc_median:.4f}',ha='left',fontsize=9)
ax[0].axhline(0,color='#555555',lw=1); ax[0].set_xticks(range(2)); ax[0].set_xticklabels([r[0] for r in rows]); ax[0].set_xlim(-0.6,1.9)
ax[0].set_ylabel('AUROC gain over single XGBoost (×10$^{-3}$)'); ax[0].set_title('(a) What the stack adds over its own global stream',loc='left',fontsize=10)
pts=[('Etiology specialists',div['etiology_xgb']['mean_r_within'],div['etiology_xgb']['mean_r_global'],TEAL),
     ('Clusters, k = 5',div['kmeans5_xgb']['mean_r_within'],div['kmeans5_xgb']['mean_r_global'],NAVY),
     ('Clusters, k = 6',div['kmeans6_xgb']['mean_r_within'],div['kmeans6_xgb']['mean_r_global'],'#5b7fa8')]
for lab,x,yv,c in pts:
    ax[1].scatter(x,yv,s=110,color=c,zorder=3); ax[1].annotate(lab,(x,yv),xytext=(8,-14 if 'k = 6' in lab else 8),textcoords='offset points',fontsize=9)
ax[1].set(xlim=(-0.3,1.0),ylim=(0,1.0),xlabel='Mean pairwise correlation between specialist streams',ylabel='Mean correlation of specialists with global stream')
ax[1].set_title('(b) Diversity of the specialist streams (test set)',loc='left',fontsize=10)
fig.tight_layout(); fig.savefig(F+'fig6_diversity.png',dpi=220); plt.close(fig)
print('done')
