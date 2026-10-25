# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  fig5_specialisation_rf.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Fig. 5 in the format of the thesis Figure 18. Random-forest streams and
#    the two-layer etiology meta-learner scored against every label, with
#    paired patient-bootstrap CIs, plus source attribution.
#
#  INPUTS   (under Results/ unless stated)
#    ckpt_1h/L1_*_rf_final.npy, preds_etiology_1h.parquet, data/misc.json
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    fig5_rf_format18.png, fig5_rf_lifts.csv, fig5_rf_deltas.csv
#  RUN
#    python fig5_specialisation_rf.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import numpy as np, pandas as pd, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as AP
R='/PATH/TO/PROJECT/technical/Results/'; CK=R+'ultimate_full/ckpt_1h/'; V4='/PATH/TO/OUTPUT/'   # EDIT: point to your project / output folders
ids=pd.read_parquet(R+'paper_v3/_extract/test_1h_ids.parquet'); y=ids.is_sepsis_6h.values
labs={'Global':y,'Respiratory':y*ids.target_resp.values,'Urinary':y*ids.target_uri.values,'Other':y*ids.target_other.values}
est={'Global':np.load(CK+'L1_global_0_rf_final.npy'),'Respiratory':np.load(CK+'L1_etiology_0_rf_final.npy'),'Urinary':np.load(CK+'L1_etiology_1_rf_final.npy'),
     'Other':np.load(CK+'L1_etiology_2_rf_final.npy'),'Meta':pd.read_parquet(R+'ultimate_full/preds_etiology_1h.parquet')['CURRENT design (global+etiology NOSE-RF, XGB meta)'].values}
L={(e,l):AP(yl,p)/yl.mean() for e,p in est.items() for l,yl in labs.items()}
# paired patient bootstrap, stratified by septic status
pt=ids.groupby('subject_id').is_sepsis_stay.max(); code=ids.subject_id.map({s:i for i,s in enumerate(pt.index)}).values
pos,neg=np.where(pt.values==1)[0],np.where(pt.values==0)[0]; rng=np.random.default_rng(0)
comps=[('Respiratory','Respiratory','Global'),('Urinary','Urinary','Global'),('Other','Other','Global'),('Global','Meta','Global')]
D={c:[] for c in comps}
for b in range(200):
    w=np.bincount(np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]),minlength=len(pt))[code].astype(float)
    for lab,e1,e2 in comps:
        yl=labs[lab]; pw=(yl*w).sum()/w.sum()
        D[(lab,e1,e2)].append((AP(yl,est[e1],sample_weight=w)-AP(yl,est[e2],sample_weight=w))/pw)
out=[]
for k,v in D.items():
    v=np.array(v); out.append(dict(label=k[0],a=k[1],b=k[2],d_lift=L[(k[1],k[0])]-L[(k[2],k[0])],lo=np.quantile(v,.025),hi=np.quantile(v,.975)))
o=pd.DataFrame(out); print(o.round(3).to_string()); o.to_csv(V4+'fig5_rf_deltas.csv',index=False)
pd.DataFrame([dict(estimator=e,label=l,lift=v) for (e,l),v in L.items()]).to_csv(V4+'fig5_rf_lifts.csv',index=False)
# ---- figure in Fig 18 format
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#eeeeee','axes.axisbelow':True})
EC={'Global':'#d9604c','Respiratory':'#5fbf9f','Urinary':'#9b8fd6','Other':'#c7b84f','Meta':'#333333'}
NM={'Global':'Global stream','Respiratory':'Respiratory','Urinary':'Urinary','Other':'Other','Meta':'Etiology stack (meta)'}
fig=plt.figure(figsize=(12.5,4.6)); a=fig.add_axes([0.06,0.13,0.6,0.68]); b=fig.add_axes([0.74,0.13,0.22,0.68])
lab=list(labs); E=list(est); w_=0.16
for j,e in enumerate(E):
    for i,l in enumerate(lab):
        v=L[(e,l)]; own=(e==l) or (e=='Meta' and l=='Global')
        a.bar(i+(j-2)*w_,v,w_*0.92,color=EC[e] if own else 'white',edgecolor=EC[e],hatch=None if own else '///',lw=1.4,label=NM[e] if i==0 else None)
        a.text(i+(j-2)*w_,v+0.06,f'{v:.2f}',ha='center',fontsize=7.3)
a.axhline(1,ls='--',color='#aaaaaa',lw=1); a.set_xticks(range(4)); a.set_xticklabels(lab); a.set_xlabel('Evaluation label'); a.set_ylabel('Lift over chance (AUPRC / prevalence)')
a.set_ylim(0,5.9); a.legend(ncol=5,fontsize=8.3,frameon=False,loc='upper center',bbox_to_anchor=(0.5,1.11)); a.set_title('(a) Each estimator scored against each label',loc='left',fontsize=10,pad=30)
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('t',['#f3efe6','#c9dbd6','#6fa39f','#2f6b6b','#173f40'])
misc=json.load(open('/PATH/TO/OUTPUT/data/misc.json')); cmx=np.array(misc['attr_rf']['cm'])   # EDIT: point to your project / output folders
b.imshow(cmx,cmap=cm,vmin=0,vmax=0.6); b.grid(False)
for i in range(3):
    for j in range(3): b.text(j,i,f'{cmx[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if cmx[i,j]>0.42 else '#222')
b.set_xticks(range(3)); b.set_xticklabels(['Resp','Urin','Other']); b.set_yticks(range(3)); b.set_yticklabels(['Resp','Urin','Other'])
b.set_xlabel('Highest-ranked specialist'); b.set_ylabel('Coded infection source'); b.set_title('(b) Source attribution, septic hours',loc='left',fontsize=10)
fig.savefig(V4+'fig5_rf_format18.png',dpi=220)
print('attr rf balanced', np.diag(cmx).mean(), misc['attr_rf']['row_acc'])
