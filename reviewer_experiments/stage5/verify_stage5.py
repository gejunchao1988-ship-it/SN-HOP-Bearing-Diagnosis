"""Independent audit of stage-five saved model parameters and held-out predictions."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd,numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parent
V1=R/'SNHOP_canonical_reproducibility_v1'
OUT=R/'results'
meta=pd.read_csv(V1/'results/recordings.csv')
sys.path.insert(0,str(V1/'src'))
from core import norm_rows
X=norm_rows(np.load(V1/'results/feature_cache.npz')['order'],'snv').astype('float64')
pred=pd.read_csv(OUT/'dg_predictions.csv');task=pd.read_csv(OUT/'dg_tasks.csv')
models=np.load(OUT/'dg_models.npz')
assert len(pred)==4800 and len(task)==40
errors=[]
for r in task.itertuples(index=False):
  sub=pred[(pred.method==r.method)&(pred.direction==r.direction)&(pred.test_groups==r.test_groups)]
  assert len(sub)==120
  src,tgt=map(int,r.direction.split('_to_'))
  held=list(map(int,r.test_groups.split('-')))
  tr=(meta.rpm==src)&(~meta.group.isin(held));te=(meta.rpm==tgt)&meta.group.isin(held)
  assert tr.sum()==180 and te.sum()==120
  assert set(meta.bearing[tr]).isdisjoint(set(meta.bearing[te]))
  z=meta[te]
  assert sub.filename.tolist()==z.filename.tolist()
  key=f'{r.method}_{r.direction}_{r.test_groups}'
  means=models[key+'_standard_mean']; scales=models[key+'_standard_scale']
  assert np.max(np.abs(X[tr].mean(0)-means))<1e-6
  assert np.max(np.abs(np.where(X[tr].std(0)==0,1,X[tr].std(0))-scales))<1e-6
  W=models[key+'_coef']; b=models[key+'_intercept']; logits=((X[te]-means)/scales)@W+b
  logits-=logits.max(axis=1,keepdims=True);ps=np.exp(logits);ps/=ps.sum(axis=1,keepdims=True)
  yhat=ps.argmax(1)
  assert np.array_equal(yhat,sub.y_pred.to_numpy())
  assert np.max(np.abs(ps-sub[['p_healthy','p_outer','p_inner']].to_numpy()))<1e-10
  assert abs(balanced_accuracy_score(sub.y_true,sub.y_pred)-r.target_ba)<1e-12
  errors.append(float(np.max(np.abs(ps-sub[['p_healthy','p_outer','p_inner']].to_numpy()))))
cv=pd.read_csv(OUT/'dg_source_only_cv.csv');assert len(cv)==180
cvsel=pd.read_csv(OUT/'dg_source_selected.csv');assert len(cvsel)==40
ab=pd.read_csv(OUT/'coverage_matched_absolute_predictions.csv');ct=pd.read_csv(OUT/'coverage_matched_absolute_tasks.csv');assert len(ab)==2400 and len(ct)==20
for r in ct.itertuples(index=False):
  sub=ab[(ab.direction==r.direction)&(ab.test_groups==r.test_groups)]
  assert len(sub)==120 and abs(balanced_accuracy_score(sub.y_true,sub.y_pred)-r.ba)<1e-12
info={'dg_task_count':len(task),'dg_prediction_count':len(pred),'source_cv_config_count':len(cv),'physical_bearing_disjoint_all':True,'replayed_model_predictions_match':True,'max_probability_difference':max(errors),'coverage_tasks':len(ct),'coverage_predictions':len(ab),'metric_recalculation_all_match':True,'original_v1_snhop_mean_ba':float(pd.read_csv(V1/'results/snhop_tasks.csv').target_ba.mean()),'original_v1_full_order_mean_ba':float(pd.read_csv(V1/'results/order_tasks.csv').target_ba.mean())}
(OUT/'stage5_verification.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
