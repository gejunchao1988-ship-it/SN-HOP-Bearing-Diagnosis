#!/usr/bin/env python3
"""SN-HOP background half-width sensitivity, source-only re-tuning.
Each fixed context halfwidth {0.8,1.2,1.6,2.0} uses the same raw-derived
8192-bin order cache. Target labels are used only for final evaluation.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')));RES=ROOT/'results';OUT=Path(__file__).resolve().parents[1]/'results';sys.path.insert(0,str(ROOT/'src'))
from core import BPFO,BPFI,ORDER_GRID,task_defs,masks,clf_model,classifier_candidates,metrics,select_candidates,norm_rows
meta=pd.read_csv(RES/'recordings.csv');y=meta.label.to_numpy();groups=meta.group.to_numpy();C=np.load(RES/'feature_cache.npz')['order'];Z=norm_rows(C,'snv')

def hop_context(x,w,maxo,context):
    p=[];cs=[]
    for base in (BPFO,BPFI):cs += list(np.arange(base,maxo+1e-10,base))
    for c in cs:
        dist=np.abs(ORDER_GRID-c);mid=dist<=w;back=(dist>w)&(dist<=context)
        if not mid.any() or not back.any():continue
        p.extend([x[:,mid].max(1),x[:,mid].mean(1)-x[:,back].mean(1)])
    for a,b in ((0,5),(5,10),(10,20),(20,30),(30,40),(40,50)):
        m=(ORDER_GRID>=a)&((ORDER_GRID<b) if b<50 else (ORDER_GRID<=b));p.extend([x[:,m].mean(1),x[:,m].std(1)])
    return np.stack(p,axis=1)

D=[];SEL=[];P=[]
clfs=list(classifier_candidates(True))
for context in (0.8,1.2,1.6,2.0):
    feats={(w,maxo):hop_context(Z,w,maxo,context) for w in (.2,.4,.6) for maxo in (15,25,35,50)}
    for direction,held in task_defs():
        tr,te,sg,*_=masks(meta,direction,held); candidates=[]
        for (w,maxo),X in feats.items():
            for c in clfs:
                cfg={**c,'halfwidth':w,'max_order':maxo};ss=[]
                for val in sg:
                    a=tr&(groups!=val);b=tr&(groups==val);model=clf_model(cfg);model.fit(X[a],y[a]);ss.append(balanced_accuracy_score(y[b],model.predict(X[b])))
                candidates.append({'cfg':cfg,'mean_cv':float(np.mean(ss)),'dim':X.shape[1]})
        chosen,ntie=select_candidates(candidates);cfg=chosen['cfg'];X=feats[(cfg['halfwidth'],cfg['max_order'])]
        model=clf_model(cfg);model.fit(X[tr],y[tr]);pred=model.predict(X[te]);ba,f1=metrics(y[te],pred)
        D.append({'background_halfwidth':context,'direction':direction,'test_groups':held,'target_ba':ba,'target_f1':f1,'feature_dim':X.shape[1]})
        SEL.append({'background_halfwidth':context,'direction':direction,'test_groups':held,**cfg,'source_cv_ba':chosen['mean_cv'],'n_tied':ntie})
        for idx,v in zip(np.flatnonzero(te),pred):P.append({'background_halfwidth':context,'direction':direction,'test_groups':held,'filename':meta.filename[idx],'bearing':meta.bearing[idx],'y_true':int(y[idx]),'y_pred':int(v)})
    print('context',context,'mean_ba',round(pd.DataFrame(D).query('background_halfwidth == @context').target_ba.mean()*100,4),flush=True)
    pd.DataFrame(D).to_csv(OUT/'background_width_tasks.csv',index=False);pd.DataFrame(SEL).to_csv(OUT/'background_width_selection.csv',index=False);pd.DataFrame(P).to_csv(OUT/'background_width_predictions.csv',index=False)
print('DONE',flush=True)
