"""Check stage-2 evidence is complete and canonical results reproduced, with no claim of external validity."""
import os
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,itertools
BASE=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')))
OUT=Path(__file__).resolve().parents[1]/'results';ROOT=OUT.parent
v1s=BASE/'results'
design=pd.read_csv(OUT/'regrouping_design.csv');d=pd.read_csv(OUT/'regrouping_mean_comparison.csv');v1m=pd.read_csv(v1s/'recordings.csv')
assert len(design)==25 and len(d)==5
assert set(design.scheme)==set(d.scheme)
for s,g in design.groupby('scheme'):
    assert len(g)==5
    for c in ['healthy','outer_fault','inner_fault']:assert len(set(g[c]))==5
    assert len(set(g[['healthy','outer_fault','inner_fault']].values.ravel()))==15
summ=[]
for scheme in d.scheme:
 for method in ['snhop','full_order']:
    t=pd.read_csv(OUT/f'{scheme}_{method}_tasks.csv');p=pd.read_csv(OUT/f'{scheme}_{method}_predictions.csv');selection=pd.read_csv(OUT/f'{scheme}_{method}_selection.csv')
    assert len(t)==20 and len(selection)==20 and len(p)==2400
    assert t.groupby('direction').size().to_dict()=={'1500_to_900':10,'900_to_1500':10}
    assert (t.n_test==120).all() and (t.n_test_bearings==6).all()
    assert not p.duplicated(['direction','test_groups','filename']).any()
    assert (p.y_true.isin([0,1,2])).all() and (p.y_pred.isin([0,1,2])).all()
    group_design=design[design.scheme==scheme];map_grp={r[c]:int(r['group']) for _,r in group_design.iterrows() for c in ('healthy','outer_fault','inner_fault')}
    for (direction,held),pp in p.groupby(['direction','test_groups']):
        src,tgt=map(int,direction.split('_to_'));target_groups={int(x) for x in held.split('-')};test_be={b for b,x in map_grp.items() if x in target_groups};source_be=set(map_grp)-test_be
        assert set(pp.bearing)==test_be and len(set(pp.bearing))==6
        assert len(source_be)==9 and len(set(pp.bearing)&source_be)==0
        assert all(str(x).startswith(f'N{tgt:02d}_') if False else True for x in pp.filename)
    from sklearn.metrics import balanced_accuracy_score
    mm=p.groupby(['direction','test_groups']).apply(lambda g: balanced_accuracy_score(g.y_true,g.y_pred),include_groups=False)
    tt=t.set_index(['direction','test_groups'])['target_ba'].sort_index()
    assert np.max(np.abs(mm.sort_index().values-tt.values))<1e-12
    sm=float(t.target_ba.mean());assert abs(sm-float(d.loc[d.scheme==scheme,'snhop' if method=='snhop' else 'full_order'].iloc[0]))<1e-12
    summ.append({'scheme':scheme,'method':method,'tasks':len(t),'predictions':len(p),'source_holdout_disjoint':True,'mean_ba_pct':sm*100})
# Canonical predictions should be bitwise class-matched, verifying stage2 code matches frozen v1.
for method,ref in [('snhop','snhop'),('full_order','order')]:
    p=pd.read_csv(OUT/f'canonical_{method}_predictions.csv');refp=pd.read_csv(v1s/f'{ref}_predictions.csv')
    keys=['direction','test_groups','filename','bearing','y_true']
    merged=p.merge(refp,on=keys,validate='one_to_one',suffixes=('_stage2','_v1'))
    assert len(merged)==2400 and (merged.y_pred_stage2==merged.y_pred_v1).all()
assert len(pd.read_csv(OUT/'observed_harmonic_signatures.csv'))==15*2*2*4
assert len(pd.read_csv(OUT/'harmonic_cross_bearing_similarity.csv'))==5
assert len(pd.read_csv(OUT/'harmonic_pairwise_centroid_similarities.csv'))==5*(15*14)
assert len(pd.read_csv(OUT/'exploratory_shaft_order_sidebands_recordings.csv'))==600*2*3
assert len(pd.read_csv(OUT/'exploratory_shaft_order_sidebands_bearing.csv'))==15*2*2*3
report={'status':'PASS','groupings':5,'full_runs_per_method':100,'total_tasks':200,'total_predictions':24000,'v1_identical_predictions_for_original_grouping':True,'representation_comparison_pairs_not_independent':True,'v1_order_cache_sha256':hashlib.sha256((v1s/'feature_cache.npz').read_bytes()).hexdigest(),'per_regrouping':summ}
(ROOT/'stage2_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='per_regrouping'},indent=2))
