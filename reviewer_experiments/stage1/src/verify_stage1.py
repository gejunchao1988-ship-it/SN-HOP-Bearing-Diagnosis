import os
import json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score
R=Path(__file__).resolve().parents[1];O=R/'results'
checks={}
for stem,key in [('matched_dimension','rep'),('single_target_uda','variant'),('window_resolution','window'),('background_width','background_halfwidth'),('bandpass','band')]:
 tasks=pd.read_csv(O/(stem+'_tasks.csv'));pred=pd.read_csv(O/(stem+'_predictions.csv'))
 assert pred.duplicated(subset=[key,'direction','test_groups','filename']).sum()==0
 for vals,g in tasks.groupby([key,'direction','test_groups'],dropna=False):
  q=pred
  for col,val in zip([key,'direction','test_groups'],vals):q=q[q[col]==val]
  assert len(q)==(60 if stem=='single_target_uda' else 120),(stem,vals,len(q))
  ba=balanced_accuracy_score(q.y_true,q.y_pred)
  assert abs(ba-g.target_ba.iloc[0])<1e-10,(stem,vals,ba,g.target_ba.iloc[0])
 checks[stem]={'task_count':len(tasks),'predictions':len(pred),'balanced_accuracy_consistent':True}
assert len(pd.read_csv(O/'per_bearing_method_comparison.csv'))==60
assert len(pd.read_csv(O/'snhop_bearing_direction_recalls.csv'))==30
assert len(pd.read_csv(O/'end_to_end_timing_600.csv'))==600
j=json.loads((O/'end_to_end_timing_summary.json').read_text());assert j['all_cache_matches'] and j['max_abs_reprocessing_difference']<2e-6
raw=pd.read_csv(R/'evidence/raw15_verification_v2.csv');assert raw.sha256_ok.all() and raw.size_ok.all() and raw.found.sum()==600
checks['raw_archive']={'archive_count':15,'manifest_mat_files':600,'hash_verified':True}
checks['runtime']={'full_pipeline_recordings':600,'reprocessed_match_to_v1':True}
checks['v1_cache_sha256']=hashlib.sha256((Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')))/'results/feature_cache.npz').read_bytes()).hexdigest()
checks['all_checks_passed']=True
(R/'stage1_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
