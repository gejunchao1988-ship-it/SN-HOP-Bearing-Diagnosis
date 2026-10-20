"""Descriptive cross-bearing harmonic feature and frozen-prediction analysis.
Uses archived canonical data/results, never refits a classifier or changes original files.
The feature analysis is post hoc and uses true labels ONLY for grouped descriptions.
No inference from run-level or task-level pseudo-independence is made.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
BASE=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')))
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'src'))
from core import hop,norm_rows,BPFO,BPFI,ORDER_GRID
C=np.load(BASE/'results/feature_cache.npz');M=pd.read_csv(BASE/'results/recordings.csv');y=M.label.to_numpy();D=norm_rows(C['order'],'snv')
print('records:',len(M),', each bearing-speed count:', M.groupby(['bearing','rpm']).size().unique().tolist(),flush=True)
# Predictive failures from frozen stored out-of-bearing predictions
P=pd.read_csv(BASE/'results/snhop_predictions.csv')
conf=(P.groupby(['direction','bearing','y_true','y_pred']).size().rename('n').reset_index())
conf.to_csv(OUT/'frozen_bearing_confusions.csv',index=False)
per=(P.assign(correct=(P.y_true==P.y_pred).astype(int)).groupby(['direction','bearing','y_true']).agg(recall=('correct','mean'),n_predictions=('correct','size'),n_correct=('correct','sum')).reset_index())
per.to_csv(OUT/'frozen_bearing_recall.csv',index=False)
# Explicit high-speed failure rows
bad=conf[(conf.direction=='900_to_1500') & conf.bearing.isin(['KA15','KA22','KA30'])]
print('Outer-fault high-speed errors (80 repeated task predictions per bearing):\n',bad.to_string(index=False),flush=True)

# 4 fixed, untuned representations; Omax 35 and width .4 are originally specified candidate GRID values;
# no target data are used to select them and there are no claims of optimality.
reps={
 'full_order_256':D,
 'physical_harmonic_context':hop(C['order'],.4,35,centers='physical',components='full'),
 'uniform_harmonic_context':hop(C['order'],.4,35,centers='uniform',components='full'),
 'physical_harmonic_only':hop(C['order'],.4,35,centers='physical',components='harmonics_only'),
 'broad_context_only':hop(C['order'],.4,35,centers='physical',components='context_only'),
}
rows=[];pairrows=[]
for name,X in reps.items():
    # Descriptive global centering for similarity, not used in predictive pipeline
    mu=X.mean(0); X=X-mu
    means={(b,rpm):X[np.where((M.bearing.to_numpy()==b)&(M.rpm.to_numpy()==rpm))[0]].mean(0) for b in sorted(M.bearing.unique()) for rpm in [900,1500]}
    vecs=np.array([means[b,r] for b in sorted(M.bearing.unique()) for r in [900,1500]])
    lb={b:int(M[M.bearing==b].label.iloc[0]) for b in sorted(M.bearing.unique())}
    within=[];between=[];same_bearing=[]
    for b in lb:
        a,z=means[b,900],means[b,1500]
        same_bearing.append(np.dot(a,z)/(np.linalg.norm(a)*np.linalg.norm(z)+1e-12))
    for b1 in lb:
        for b2 in lb:
            if b1==b2: continue
            a,z=means[b1,900],means[b2,1500]
            cosine=np.dot(a,z)/(np.linalg.norm(a)*np.linalg.norm(z)+1e-12)
            key='same_fault_different_bearing' if lb[b1]==lb[b2] else 'different_fault_different_bearing'
            pairrows.append(dict(representation=name,bearing_900=b1,bearing_1500=b2,class_900=lb[b1],class_1500=lb[b2],pair_type=key,cosine=cosine))
            (within if lb[b1]==lb[b2] else between).append(cosine)
    rows.append({'representation':name,'dimensions':X.shape[1],
       'same_fault_cross_bearing_cross_speed_cosine_mean':np.mean(within),
       'different_fault_cross_bearing_cross_speed_cosine_mean':np.mean(between),
       'similarity_gap':np.mean(within)-np.mean(between),
       'same_bearing_cross_speed_cosine_mean':np.mean(same_bearing),
       'same_class_n_pairs':len(within),'different_class_n_pairs':len(between),
       'n_distinct_physical_bearings':15})
pd.DataFrame(rows).to_csv(OUT/'harmonic_cross_bearing_similarity.csv',index=False)
pd.DataFrame(pairrows).to_csv(OUT/'harmonic_pairwise_centroid_similarities.csv',index=False)
print('FEATURE SIMILARITY\n',pd.DataFrame(rows).to_string(index=False),flush=True)
# One-dimensional harmonic amplitude by physical sample and speed
Z=norm_rows(C['order'],'snv');harm=[]
for b in sorted(M.bearing.unique()):
    for speed in [900,1500]:
        ids=np.flatnonzero((M.bearing.to_numpy()==b)&(M.rpm.to_numpy()==speed))
        for typ,order in [('BPFO',BPFO),('BPFI',BPFI)]:
            for k in (1,2,3,4):
                c=k*order;cen=np.abs(ORDER_GRID-c)<=.4;bg=(np.abs(ORDER_GRID-c)>.4)&(np.abs(ORDER_GRID-c)<=1.2)
                if not cen.any() or not bg.any():continue
                p=Z[ids][:,cen].max(1); v=Z[ids][:,cen].mean(1)-Z[ids][:,bg].mean(1)
                harm.append(dict(bearing=b,label=int(y[ids[0]]),rpm=speed,band=typ,harmonic=k,peak_mean=float(np.mean(p)),contrast_mean=float(np.mean(v)),peak_sd=float(np.std(p,ddof=1)),contrast_sd=float(np.std(v,ddof=1)),n_recordings=len(ids)))
pd.DataFrame(harm).to_csv(OUT/'observed_harmonic_signatures.csv',index=False)
# failing bearings: compare true outer-race signatures across two rotational speeds
S=pd.DataFrame(harm);a=S[(S.label==1)&(S.band=='BPFO')&(S.harmonic<=3)].pivot_table(index=['bearing','harmonic'],columns='rpm',values='contrast_mean').reset_index()
a['delta_1500_minus_900']=a[1500]-a[900];a.to_csv(OUT/'outer_BPFO_contrast_shift.csv',index=False)
print('OUTER BPFO Contrast cross-speed:\n',a.to_string(index=False),flush=True)
(OUT/'feature_analysis_methods.json').write_text(json.dumps({'representation_width':0.4,'max_order':35,'background_width':1.2,'pair_comparison':'900 rpm bearing centroid vs 1500 rpm OTHER bearing centroid, matched class or mismatched class','global_centering':'description-only means over all 600 unlabeled feature vectors, NO model training','caveat':'bearing pair scores share bearings and are not independent replicates','code_version':'descriptive_v1'},indent=2),encoding='utf-8')
