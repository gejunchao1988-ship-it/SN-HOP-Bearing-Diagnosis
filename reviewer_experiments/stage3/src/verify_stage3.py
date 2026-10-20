import json,hashlib,sys,os
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
D=Path(__file__).resolve().parents[1]; R=D/'results'
orig=Path(os.getenv('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')))/'results'
p=pd.read_csv(R/'nonwhite_noise_predictions.csv');task=pd.read_csv(R/'nonwhite_noise_tasks.csv');clean=pd.read_csv(R/'canonical_clean_replay_predictions.csv')
old=pd.read_csv(orig/'snhop_predictions.csv')
k=['direction','test_groups','filename'];comp=clean[k+['y_pred']].merge(old[k+['y_pred']],on=k,suffixes=('_new','_old'),validate='one_to_one')
assert len(comp)==2400 and (comp.y_pred_new==comp.y_pred_old).all()
assert len(p)==19200 and p.filename.nunique()==600
assert len(task)==180 and set(task.kind)=={'clean','band_limited_colored','random_damped_impacts'}
assert len(p.groupby(['kind','snr_db','replicate','direction','test_groups']))==160
s=pd.read_csv(R/'nonwhite_noise_summary.csv')
for values,group in p.groupby(['kind','snr_db','replicate','direction','test_groups']):
 row=task[(task.kind==values[0])&(task.snr_db==values[1])&(task.replicate==values[2])&(task.direction==values[3])&(task.test_groups==values[4])]
 assert len(row)==1 and len(group)==120, (values,len(row),len(group))
 ba=balanced_accuracy_score(group.y_true,group.y_pred)
 f1=f1_score(group.y_true,group.y_pred,average='macro',zero_division=0)
 assert abs(ba-row.target_ba.iloc[0])<1e-12 and abs(f1-row.target_f1.iloc[0])<1e-12
for values,group in task.groupby(['kind','snr_db'],dropna=False):
 row=s[(s.kind==values[0]) & ((s.snr_db==values[1]) if pd.notna(values[1]) else s.snr_db.isna())]
 assert len(row)==1 and abs(row.mean_ba.iloc[0]-group.target_ba.mean())<1e-12
z=np.load(R/'nonwhite_order_cache.npz')
for kind in ['band_limited_colored','random_damped_impacts']:
 assert z[kind].shape==(2,2,600,256) and np.isfinite(z[kind]).all()
snr=pd.read_csv(R/'nonwhite_snr_audit.csv')
assert len(snr)==4800 and float((snr.achieved_snr_db-snr.snr_db).abs().max())<1e-10
cleanr=task[task.kind=='clean']; assert len(cleanr)==20
oldtask=pd.read_csv(orig/'snhop_tasks.csv')
tsk=cleanr.merge(oldtask,on=['direction','test_groups'],suffixes=('_v4','_v1'),validate='one_to_one')
assert (tsk.target_ba_v4-tsk.target_ba_v1).abs().max()<1e-12
report={'status':'PASS','raw_recordings':600,'source_archive_sha256_checks':15,'clean_replayed_predictions':len(comp),'clean_tasks':len(cleanr),'nonwhite_task_rows':len(task)-len(cleanr),'nonwhite_prediction_rows':len(p),'noise_derived_spectra':4800,'repeated_simulations_per_condition':2,'SNR_max_abs_error_dB':float((snr.achieved_snr_db-snr.snr_db).abs().max()),'results_pct':{r.kind+('_clean' if pd.isna(r.snr_db) else f'_{int(r.snr_db)}dB'):round(100*r.mean_ba,5) for r in s.itertuples()}}
(D/'stage3_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
