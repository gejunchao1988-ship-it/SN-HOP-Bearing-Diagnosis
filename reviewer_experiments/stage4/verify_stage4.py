from pathlib import Path
import hashlib, json, os, itertools, zipfile, glob
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
ROOT=Path(__file__).resolve().parent; R=ROOT/'results'
meta=pd.read_csv(R/'new_load_recordings.csv')
tasks=pd.read_csv(R/'new_load_tasks.csv')
preds=pd.read_csv(R/'new_load_predictions.csv')
summary=pd.read_csv(R/'new_load_summary.csv')
assert meta.shape[0]==600 and meta.filename.nunique()==600
assert (meta.groupby(['bearing','condition']).size()==20).all()
assert meta.groupby(['condition','label']).size().eq(100).all()
assert len(tasks)==120 and len(preds)==14400
assert tasks.groupby(['method','source_rpm','target_condition']).size().eq(10).all()
assert preds.groupby(['method','source_rpm','target_condition','held_groups']).size().eq(120).all()
for t in tasks.itertuples():
 p=preds[(preds.method==t.method)&(preds.source_rpm==t.source_rpm)&(preds.target_condition==t.target_condition)&(preds.held_groups==t.held_groups)]
 assert np.isclose(balanced_accuracy_score(p.y_true,p.y_pred),t.balanced_accuracy,rtol=0,atol=1e-12)
 assert np.isclose(f1_score(p.y_true,p.y_pred,average='macro'),t.macro_f1,rtol=0,atol=1e-12)
 held=set(map(int,t.held_groups.split('-')))
 assert set(p.bearing.unique())==set(meta.loc[meta.group.isin(held),'bearing'])
 assert not set(meta.loc[meta.group.isin(held),'bearing']) & set(meta.loc[~meta.group.isin(held),'bearing'])
for s in summary.itertuples():
 d=tasks[(tasks.method==s.method)&(tasks.source_rpm==s.source_rpm)&(tasks.target_condition==s.target_condition)]
 assert np.isclose(d.balanced_accuracy.mean(),s.balanced_accuracy_mean,rtol=0,atol=1e-12)
 assert np.isclose(d.macro_f1.mean(),s.macro_f1_mean,rtol=0,atol=1e-12)
# descriptive paired differences, shared bearings are NOT independent replicates
wide=tasks.pivot_table(index=['source_rpm','held_groups','target_condition'],columns='method',values='balanced_accuracy')
wide['advantage_pp']=(wide['snhop']-wide['order'])*100
wide.reset_index().to_csv(R/'paired_task_comparisons.csv',index=False)
rec=[]
for cond,g in preds.groupby(['method','source_rpm','target_condition','bearing','y_true']):
 rec.append(dict(zip(['method','source_rpm','target_condition','bearing','y_true'],cond),recall=float((g.y_true==g.y_pred).mean()),n_repeated_predictions=len(g)))
pd.DataFrame(rec).to_csv(R/'per_bearing_recall.csv',index=False)
sha=[]
for b in ['K001','K002','K003','K004','K005','KA04','KA15','KA16','KA22','KA30','KI04','KI14','KI16','KI18','KI21']:
 paths=glob.glob(str(Path(os.environ.get('SNHOP_RAW_DIR',str(Path(__file__).resolve().parents[2]/'raw')))/(b+'(*).zip')));assert len(paths)==1
 path=Path(paths[0]);h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
 sha.append({'bearing':b,'file_name':path.name,'size_bytes':path.stat().st_size,'sha256':h.hexdigest()})
pd.DataFrame(sha).to_csv(R/'new_load_archive_checksums.csv',index=False)
report={'verified_new_recordings':len(meta),'conditions':sorted(meta.condition.unique().tolist()),'per_condition_count':meta.groupby('condition').size().to_dict(),'predictions':len(preds),'tasks':len(tasks),'source_only_selected':True,'strict_bearing_partition':True,'canonical_900_to_1500_replayed':True,'new_archives_checksums':len(sha),'nonindependent_tasks':True,'note':'Cached features rebuilt from raw and canonical settings; original 900-to-1500 result enforced by code assertions.'}
(R/'verification_stage4.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
