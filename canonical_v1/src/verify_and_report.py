import json,time,hashlib,platform
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score,f1_score
from core import *
root=ROOT;res=RES
meta=pd.read_csv(res/'recordings.csv')
refs=json.loads((res/'reference_array_hashes.json').read_text())
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
cache=np.load(res/'feature_cache.npz')
array_checks={k:(list(cache[k].shape)==v['shape'] and str(cache[k].dtype)==v['dtype'] and ah(cache[k])==v['sha256_array_bytes']) for k,v in refs.items() if k in cache.files}
if (res/'noisy_order_cache.npz').exists():
    no=np.load(res/'noisy_order_cache.npz');v=refs['noisy_order'];array_checks['noisy_order']=(list(no['order'].shape)==v['shape'] and str(no['order'].dtype)==v['dtype'] and ah(no['order'])==v['sha256_array_bytes'] and no['snrs'].tolist()==v['snrs'])
checks={'reference_array_checks':array_checks,'prediction_checks':{},'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
for m in ['absolute','order','traditional23','pfn','snhop']:
    t=pd.read_csv(res/f'{m}_tasks.csv');p=pd.read_csv(res/f'{m}_predictions.csv');ok=(len(t)==20 and len(p)==2400 and p.filename.nunique()==600 and p.groupby('filename').size().eq(4).all())
    for (d,h),g in p.groupby(['direction','test_groups']):
        r=t[(t.direction==d)&(t.test_groups==h)].iloc[0];ok &= abs(balanced_accuracy_score(g.y_true,g.y_pred)-r.target_ba)<1e-12;ok &= abs(f1_score(g.y_true,g.y_pred,average='macro',zero_division=0)-r.target_f1)<1e-12
    checks['prediction_checks'][m]=bool(ok)
# check dependent outputs structurally
checks['dependent_checks']={
 'ablation_tasks_120': (res/'ablation_tasks.csv').exists() and len(pd.read_csv(res/'ablation_tasks.csv'))==120,
 'uda_tasks_present': (res/'uda_tasks.csv').exists() and len(pd.read_csv(res/'uda_tasks.csv'))>0,
 'noise_tasks_present': (res/'noise_tasks.csv').exists() and len(pd.read_csv(res/'noise_tasks.csv'))>0,
}
# timings for distinct selected SN-HOP configurations
X=cache['order'];sel=pd.read_csv(res/'snhop_selected.csv');pairs=sel[['halfwidth','max_order']].drop_duplicates().to_records(index=False);tim=[]
for w,o in pairs:
    vals=[]
    for _ in range(20):
        st=time.perf_counter();Z=hop(X,float(w),float(o));vals.append((time.perf_counter()-st)*1000)
    tim.append({'halfwidth':w,'max_order':o,'feature_dim':Z.shape[1],'ms_per_600_mean':float(np.mean(vals)),'ms_per_600_sd':float(np.std(vals))})
pd.DataFrame(tim).to_csv(res/'representation_timing.csv',index=False)
checks['all_passed']=all(array_checks.values()) and all(checks['prediction_checks'].values()) and all(checks['dependent_checks'].values())
(res/'verification.json').write_text(json.dumps(checks,indent=2))
# extra paired summaries saved for manuscript traceability
D=pd.read_csv(res/'ablation_tasks.csv');full=D[D.variant=='full'][['direction','test_groups','target_ba']].rename(columns={'target_ba':'full'});ars=[]
for v in ['uniform_centers','peak_context','contrast_context','harmonics_only','context_only']:
    z=full.merge(D[D.variant==v][['direction','test_groups','target_ba']].rename(columns={'target_ba':'x'}),on=['direction','test_groups']);diff=z.full-z.x
    ars.append({'variant':v,'full_gain_pp':100*diff.mean(),'wins':int((diff>1e-12).sum()),'ties':int((abs(diff)<=1e-12).sum()),'losses':int((diff<-1e-12).sum()),'raw_p':wilcoxon(diff,alternative='greater',zero_method='wilcox').pvalue})
pd.DataFrame(ars).to_csv(res/'ablation_paired_stats.csv',index=False)
U=pd.read_csv(res/'uda_tasks.csv');base=U[(U.method=='snhop')&(U.kind=='uda')&(U.variant=='NoDA')][['direction','test_groups','target_ba']].rename(columns={'target_ba':'base'});urs=[]
for m,v in [('snhop','CORAL'),('snhop','target_standardization'),('order','CORAL'),('pfn','CORAL'),('absolute','CORAL')]:
    g=U[(U.method==m)&(U.kind=='uda')&(U.variant==v)][['direction','test_groups','target_ba']].rename(columns={'target_ba':'x'});z=base.merge(g,on=['direction','test_groups']);diff=z.x-z.base
    urs.append({'method':m,'variant':v,'gain_vs_snhop_noda_pp':100*diff.mean(),'wins':int((diff>1e-12).sum()),'ties':int((abs(diff)<=1e-12).sum()),'losses':int((diff<-1e-12).sum()),'raw_p_greater':wilcoxon(diff,alternative='greater',zero_method='wilcox').pvalue})
pd.DataFrame(urs).to_csv(res/'uda_paired_stats.csv',index=False)
# report
s=pd.read_csv(res/'main_summary.csv');stats=pd.read_csv(res/'paired_stats.csv');ab=pd.read_csv(res/'ablation_summary.csv');uda=pd.read_csv(res/'uda_summary.csv');noise=pd.read_csv(res/'noise_summary.csv',header=[0,1],index_col=0)
cm=np.loadtxt(res/'snhop_mean_confusion.csv',delimiter=',');cmu=np.loadtxt(res/'snhop_uda_mean_confusion.csv',delimiter=',')
sel=pd.read_csv(res/'snhop_selected.csv');tim=pd.read_csv(res/'representation_timing.csv')
lines=['# SN-HOP canonical reproducible rerun','',f"Verification passed: **{checks['all_passed']}**.",'','## Main strict DG results','',s.to_markdown(index=False),'','## Paired comparisons vs SN-HOP','',stats.to_markdown(index=False),'','## Ablation','',ab.to_markdown(index=False),'','## UDA','',uda.to_markdown(index=False),'','## Noise robustness','',noise.to_markdown(),'','## Class recalls','',f"Strict DG recalls (%): Healthy {cm[0,0]*100:.2f}, Outer {cm[1,1]*100:.2f}, Inner {cm[2,2]*100:.2f}.",f"SN-HOP+CORAL recalls (%): Healthy {cmu[0,0]*100:.2f}, Outer {cmu[1,1]*100:.2f}, Inner {cmu[2,2]*100:.2f}.",'','## Selected SN-HOP configurations','',f"Half-width counts: {sel.halfwidth.value_counts().sort_index().to_dict()}",f"Maximum-order counts: {sel.max_order.value_counts().sort_index().to_dict()}",f"Feature dimension range: {int(sel.feature_dim.min())}-{int(sel.feature_dim.max())}; median {sel.feature_dim.median():.0f}.",'','## Timing','',tim.to_markdown(index=False),'','## Provenance','',"This directory is the canonical rerun implementation used for the updated manuscript. It was reconstructed after the earlier Stage G/H execution source was not preserved. Reported updated results are generated by the code and saved artifacts in this package; they are not back-filled from the historical manuscript values."]
(root/'FINAL_RERUN_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(checks,indent=2))
