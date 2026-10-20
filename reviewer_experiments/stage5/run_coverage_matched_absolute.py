#!/usr/bin/env python3
"""Control: absolute-frequency envelope features retaining same speed-dependent physical 0..50x-shaft-freq support as 0..50-order features; 256-bin Hz axis unchanged.
No interpolation to order coordinates; outside-band zero-padded in fixed Hz bins.
IMPORTANT: equal support, not equal effective spectral bin density at each rpm.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score,f1_score
ROOT=Path(__file__).resolve().parent; V1=ROOT/'SNHOP_canonical_reproducibility_v1'; OUT=ROOT/'results'
sys.path.insert(0,str(V1/'src'))
from core import norm_rows, clf_model, classifier_candidates, masks,task_defs,select_candidates
meta=pd.read_csv(V1/'results/recordings.csv'); y=meta.label.to_numpy(); groups=meta.group.to_numpy()
cache=np.load(V1/'results/feature_cache.npz')
fixed_abs=cache['absolute'].astype(np.float64)
f=np.linspace(0,1250,256)[None,:]
cutoff=(50.0*meta.measured_rpm.to_numpy()/60.0)[:,None]
X=np.where(f<=cutoff+1e-10,fixed_abs,0.0)
assert X.shape==(600,256)

def run():
    rows=[];pred=[];cv=[];start=time.time()
    for direction,held in task_defs():
        tr,te,sg,*_=masks(meta,direction,held)
        cand=[]
        for norm in ['raw','l2','snv']:
            rep=norm_rows(X,norm)
            for cc in classifier_candidates(False):
                cfg={'norm':norm,**cc};scores=[]
                for val in sg:
                    it=tr&(groups!=val);iv=tr&(groups==val)
                    model=clf_model(cfg);model.fit(rep[it],y[it]); yp=model.predict(rep[iv]);scores.append(float(balanced_accuracy_score(y[iv],yp)))
                avg=float(np.mean(scores));cand.append({'cfg':cfg,'mean_cv':avg,'Z':rep,'dim':256,'fold':scores})
                cv.append({'direction':direction,'test_groups':held,**cfg,'mean_source_cv':avg,'three_source_group_scores':json.dumps(scores)})
        # Source CV only; same selection policy as original experiment
        opt,ntie=select_candidates(cand); cfg=opt['cfg']; rep=opt['Z']; model=clf_model(cfg)
        model.fit(rep[tr],y[tr]); yp=model.predict(rep[te]); ba=float(balanced_accuracy_score(y[te],yp));mf=float(f1_score(y[te],yp,average='macro',zero_division=0))
        rows.append({'method':'absolute_50shaft_cutoff','direction':direction,'test_groups':held,'ba':ba,'macro_f1':mf,'source_cv_ba':opt['mean_cv'],'source_ties':ntie,**cfg})
        for idx,v in zip(np.flatnonzero(te),yp):pred.append({'method':'absolute_50shaft_cutoff','direction':direction,'test_groups':held,'filename':meta.filename[idx],'bearing':meta.bearing[idx],'y_true':int(y[idx]),'y_pred':int(v)})
        print(direction,held,round(ba*100,2),cfg,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'coverage_matched_absolute_tasks.csv',index=False)
    pd.DataFrame(pred).to_csv(OUT/'coverage_matched_absolute_predictions.csv',index=False)
    pd.DataFrame(cv).to_csv(OUT/'coverage_matched_absolute_source_cv.csv',index=False)
    s=df.groupby('direction').ba.mean();s['combined']=df.ba.mean()
    summary={'design':'Ablation restricts each 256-bin absolute-frequency envelope input to 0..min(1250,50*f_r) Hz by zero masking. In source and target this equals the physical coverage of the 0..50-order representation; frequencies remain in fixed hertz coordinates. Does NOT equalize spectral bin density or algorithmic inductive bias.', 'mean_ba_pct':{k:float(v*100) for k,v in s.items()},'execution_seconds':time.time()-start,'original_absolute_0_1250_ba_pct':float(100*pd.read_csv(V1/'results/absolute_tasks.csv').target_ba.mean()),'original_order_0_50_ba_pct':float(100*pd.read_csv(V1/'results/order_tasks.csv').target_ba.mean())}
    (OUT/'coverage_matched_absolute_design_summary.json').write_text(json.dumps(summary,indent=2))
    for r in rows:
        z=pd.DataFrame(pred);g=z[(z.direction==r['direction'])&(z.test_groups==r['test_groups'])]
        assert len(g)==120 and abs(balanced_accuracy_score(g.y_true,g.y_pred)-r['ba'])<1e-12
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':run()
