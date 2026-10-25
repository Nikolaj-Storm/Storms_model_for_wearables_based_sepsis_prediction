# Copyright (c) 2026 Nikolaj Storm Petersen. Licensed under CC BY-NC 4.0.
# Non-commercial use only. If you use or adapt this code, please cite the author.
# See LICENSE and CITATION.cff  |  https://creativecommons.org/licenses/by-nc/4.0/

# ============================================================================
#  fig1_architecture.py
#  Journal paper (2026), analysis step
#
#  PURPOSE
#    Draws the system diagram of both stack variants (paper Fig. 1).
#
#  INPUTS   (under Results/ unless stated)
#    none
#  OUTPUTS  (under /PATH/TO/OUTPUT)
#    figures/fig1_architecture.png
#  RUN
#    python fig1_architecture.py
#
#  USER-EDITABLE SETTINGS  (grep the body for the tag  EDIT:)
#    Project and output paths.
# ============================================================================

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['font.family']='Liberation Serif'
W=13.0
stages=[]  # (num,label,title,desc,boxes[(title,sub)], extra) ; boxes may be list of rows
S=[
 ("1","INPUT DATA","MIMIC-IV cohort","58,970 ICU stays · 45,833 patients · patient-level 80/20 split (47,218 train / 11,752 test stays)",
   [[("Six vitals per patient-hour","HR · RR · SpO$_2$ · Temp · SBP · DBP"),("Context","age · admission weight · hours since ICU admission")]]),
 ("2","FEATURE ENGINEERING","6 vitals + 3 context variables → 196 features",None,
   [[("Clinical composites","shock index, MAP, NEWS (vitals)"),("Temporal dynamics","expanding stats, CUSUM, slopes"),
     ("Biological interactions","fever-tachycardia, CRDI"),("24-h history and shock","6/12/24-h stats, low-MAP hours")]]),
 ("3","STREAM DEFINITION","Two ways to split the septic class into specialist targets","Same features, same learners, same layers. Only the basis for the specialist streams differs.",
   "SPLIT"),
 ("4","BASE LAYER (L1)","Two heterogeneous learners per stream",None,
   [[("NOSE random forest","5 balanced subsets × 150 trees, depth 8, 96 features"),("XGBoost","depth 8, learning rate 0.03, early stopping, 196 features")]],
   "5-fold patient-grouped out-of-fold predictions · refit on all training patients for the test set · A: 8 outputs · B: 24 outputs"),
 ("5","COMBINER LAYER (L2)","One pair of combiners per stream family",None,
   [[("Logistic regression","on global + family outputs + cross-stream variance"),("Small XGBoost","200 trees, depth 3, same inputs")]],
   "fitted out of fold · A: 1 family → 2 outputs · B: 2 families (k = 5, k = 6) → 4 outputs"),
 ("6","META-LEARNER (L3)","Logistic regression over the combiner outputs",None,
   [[("One probability per patient-hour","P(Sepsis-3 onset within 6 h) in [0, 1]")]]),
 ("7","THRESHOLD SELECTION","Policy parameter, not a fixed value",None,
   [[("Sensitivity 0.80","operating point of Barton et al. (2019)"),("≤ 10 false alerts per true alert","burden-capped operating point"),("≤ 20 false alerts per true alert","recall-priority operating point")]]),
 ("8","OUTPUT","Risk score per patient-hour",None,
   [[("All variants","{stay, hour, P$_{6h}$, alert flag} · per-stay first alert and alerts per shift"),("Variant A only","source-specific risks P$_{resp}$, P$_{uri}$, P$_{other}$ exposed for interpretation")]]),
]
H={ "1":2.05,"2":1.9,"3":3.55,"4":2.25,"5":2.25,"6":1.75,"7":1.75,"8":1.85}
gap=0.55
tot=sum(H.values())+gap*(len(S)-1)+0.2
fig=plt.figure(figsize=(W*0.62,tot*0.62)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,tot); ax.axis('off')
y=tot-0.1; L=0.25; R=W-0.25
def box(x0,y0,w,h,lw=1.1,fc='white',ec='black'):
    ax.add_patch(Rectangle((x0,y0),w,h,fill=True,fc=fc,ec=ec,lw=lw))
def inner_row(items,ytop,x0,x1,h=0.78,fs1=10.5,fs2=9.0):
    n=len(items); g=0.14; w=(x1-x0-g*(n-1))/n
    for i,(t,s) in enumerate(items):
        xx=x0+i*(w+g); box(xx,ytop-h,w,h,lw=0.9)
        ax.text(xx+w/2,ytop-0.27,t,ha='center',va='center',fontsize=fs1,fontweight='bold')
        ax.text(xx+w/2,ytop-0.56,s,ha='center',va='center',fontsize=fs2)
for st in S:
    num,lab,title,desc,boxes=st[:5]; extra=st[5] if len(st)>5 else None
    h=H[num]; box(L,y-h,R-L,h,lw=1.3)
    ax.text(L+0.25,y-0.28,f"{num} · "+" ".join(lab),fontsize=9.5,va='center',color='#222')
    ax.text(L+0.25,y-0.6,title,fontsize=12.5,fontweight='bold',va='center')
    cy=y-0.85
    if desc: ax.text(L+0.25,y-0.92,desc,fontsize=9.8,va='center'); cy=y-1.15
    if boxes=="SPLIT":
        mid=(L+R)/2; x0=L+0.25; x1=R-0.25
        for k,(name,kind,items,foot) in enumerate([
            ("Variant A · Etiology specialists","supervised · ICD-coded infection source",[("Global","all septic"),("Respiratory","lung source"),("Urinary","urinary source"),("Other","other sources")],"4 streams · each stream has a clinical meaning"),
            ("Variant B · Physiology clusters","unsupervised · k-means on PCA(10) pre-onset profiles",[("Global","all septic"),("k = 5","clusters C1 … C5"),("k = 6","clusters C1 … C6")],"12 streams · clusters carry no clinical label")]):
            xa = x0 if k==0 else mid+0.12; xb = mid-0.12 if k==0 else x1
            box(xa,cy-2.15,xb-xa,2.15,lw=1.0,fc='#f7f7f7')
            ax.text(xa+0.15,cy-0.27,name,fontsize=10.8,fontweight='bold',va='center')
            ax.text(xa+0.15,cy-0.55,kind,fontsize=9.2,style='italic',va='center')
            inner_row(items,cy-0.8,xa+0.15,xb-0.15,h=0.78,fs1=10,fs2=8.8)
            ax.text((xa+xb)/2,cy-1.9,foot,ha='center',fontsize=9.2,style='italic')
    else:
        for row in boxes:
            inner_row(row,cy,L+0.25,R-0.25); cy-=0.9
    if extra: ax.text((L+R)/2,cy-0.12,extra,ha='center',va='center',fontsize=9.2,style='italic')
    y-=h
    if num!="8": ax.text((L+R)/2,y-gap/2,"↓",ha='center',va='center',fontsize=13)
    y-=gap
fig.savefig('/PATH/TO/OUTPUT/figures/fig1_architecture.png',dpi=220)   # EDIT: point to your project / output folders
