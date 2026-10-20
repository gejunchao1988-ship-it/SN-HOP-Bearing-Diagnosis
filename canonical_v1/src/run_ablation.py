import json
import numpy as np,pandas as pd
from core import *
from sklearn.metrics import balanced_accuracy_score,f1_score
cache=np.load(RES/'feature_cache.npz');X=cache['order'];meta=pd.read_csv(RES/'recordings.csv');y=meta.label.to_numpy();groups=meta.group.to_numpy();rpms=meta.rpm.to_numpy()
variants={'full':('physical','full'),'uniform_centers':('uniform','full'),'peak_context':('physical','peak_context'),'contrast_context':('physical','contrast_context'),'harmonics_only':('physical','harmonics_only'),'context_only':('physical','context_only')}
rows=[];pred=[];cvrows=[]
for variant,(center,comp) in variants.items():
 for direction,held in task_defs():
  tr,te,sg,src,tgt,hg=masks(meta,direction,held);cand=[]
  ws=[.2,.4,.6] if variant!='context_only' else [.2]; os=[15,25,35,50] if variant!='context_only' else [15]
  for w in ws:
   for o in os:
    Z=hop(X,w,o,centers=center,components=comp)
    for cc in classifier_candidates(True):
     cfg={'halfwidth':w,'max_order':o,**cc};fold=[]
     for val in sg:
      it=tr&(groups!=val);iv=tr&(groups==val);clf=clf_model(cfg);clf.fit(Z[it],y[it]);fold.append(balanced_accuracy_score(y[iv],clf.predict(Z[iv])))
     mean=float(np.mean(fold));cand.append({'cfg':cfg,'mean_cv':mean,'dim':Z.shape[1],'Z':Z});cvrows.append({'variant':variant,'direction':direction,'test_groups':held,**cfg,'mean_cv':mean,'fold_scores':json.dumps(fold),'dim':Z.shape[1]})
  sel,ntie=select_candidates(cand);cfg=sel['cfg'];Z=sel['Z'];clf=clf_model(cfg);clf.fit(Z[tr],y[tr]);yp=clf.predict(Z[te]);ba,f1=metrics(y[te],yp);rows.append({'variant':variant,'direction':direction,'test_groups':held,**cfg,'feature_dim':Z.shape[1],'inner_cv_ba':sel['mean_cv'],'n_mean_tied':ntie,'target_ba':ba,'target_f1':f1})
  for idx,v in zip(np.flatnonzero(te),yp):pred.append({'variant':variant,'direction':direction,'test_groups':held,'filename':meta.filename[idx],'y_true':int(y[idx]),'y_pred':int(v)})
  print(variant,direction,held,round(100*ba,2),flush=True)
pd.DataFrame(rows).to_csv(RES/'ablation_tasks.csv',index=False);pd.DataFrame(pred).to_csv(RES/'ablation_predictions.csv',index=False);pd.DataFrame(cvrows).to_csv(RES/'ablation_source_validation.csv',index=False)
s=pd.DataFrame(rows).groupby('variant')[['target_ba','target_f1']].mean().mul(100);s.to_csv(RES/'ablation_summary.csv');print(s)
