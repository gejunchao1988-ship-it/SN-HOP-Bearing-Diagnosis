"""SN-HOP and full-order baseline under alternate, class-balanced physical-bearing groupings.
Each outer fold holds out 2 triplets / 6 complete physical bearings in both directions,
with nested source-only three-fold triplet CV; label-free group schedules fixed by seeds.
No target labels involved in parameter selection, no overwrites of canonical v1.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,itertools,hashlib
from pathlib import Path
import numpy as np,pandas as pd
BASE=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')))
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(BASE/'src'))
from core import GROUPS,hop,clf_model,classifier_candidates,norm_rows,select_candidates,metrics
meta=pd.read_csv(BASE/'results/recordings.csv');cache=np.load(BASE/'results/feature_cache.npz')
y=meta.label.to_numpy();bearing=meta.bearing.to_numpy();rpm=meta.rpm.to_numpy()
full=cache['order'];H=[hop(full,w,m) for w in [.2,.4,.6] for m in [15,25,35,50]]
REPS_H=[({'halfwidth':w,'max_order':m},H[i]) for i,(w,m) in enumerate(itertools.product([.2,.4,.6],[15,25,35,50]))]
REPS_O=[({'norm':n},norm_rows(full,n)) for n in ['raw','l2','snv']]

# Alter group composition, never class or bearing membership. Deterministic seeds chosen before looking at results.
healthy=[g[0] for g in GROUPS];outer=[g[1] for g in GROUPS];inner=[g[2] for g in GROUPS]
schemes=[('canonical', GROUPS)]
for seed in [20261020,20261021,20261022,20261023]:
    rng=np.random.default_rng(seed)
    # nontrivial permutations fixed by seed; healthy group IDs held to allow clear comparison
    op=rng.permutation(outer).tolist();ip=rng.permutation(inner).tolist()
    grp=[[healthy[i],op[i],ip[i]] for i in range(5)]
    assert grp!=GROUPS
    schemes.append((f'perm_seed_{seed}',grp))

pd.DataFrame([{'scheme':s,'group':i+1,'healthy':g[0],'outer_fault':g[1],'inner_fault':g[2]} for s,groups in schemes for i,g in enumerate(groups)]).to_csv(OUT/'regrouping_design.csv',index=False)

cand_h=list(classifier_candidates(True));cand_o=list(classifier_candidates(False))
selected_v1=pd.read_csv(BASE/'results/snhop_selected.csv')

def do_method(scheme, groups,method, reps,clfs):
    gidx=np.array([next(i+1 for i,g in enumerate(groups) if b in g) for b in bearing]); assert np.unique(gidx).size==5
    taskrows=[];predrows=[];selrows=[]
    for direction in ['1500_to_900','900_to_1500']:
        src,tgt=map(int,direction.split('_to_'))
        for held in itertools.combinations(range(1,6),2):
            name='-'.join(map(str,held));source=[i for i in range(1,6) if i not in held]
            tr=(rpm==src)&(~np.isin(gidx,held));te=(rpm==tgt)&(np.isin(gidx,held))
            assert int(tr.sum())==180 and int(te.sum())==120
            # source-fold splits shared by all feature settings and model candidates
            splits=[(tr&(gidx!=v),tr&(gidx==v)) for v in source]
            candidates=[]
            for rc,X in reps:
                for cc in clfs:
                    cfg={**rc,**cc};sc=[]
                    for it,iv in splits:
                        clf=clf_model(cfg);clf.fit(X[it],y[it]);sc.append(metrics(y[iv],clf.predict(X[iv]))[0])
                    candidates.append({'cfg':cfg,'mean_cv':float(np.mean(sc)),'dim':X.shape[1],'X':X,'fold_scores':sc})
            best,ntie=select_candidates(candidates)
            cfg=best['cfg'];X=best['X'];clf=clf_model(cfg);clf.fit(X[tr],y[tr]);yp=clf.predict(X[te]);ba,f1=metrics(y[te],yp)
            row={'scheme':scheme,'method':method,'direction':direction,'test_groups':name,'target_ba':ba,'target_f1':f1,'n_test':int(te.sum()),'n_test_bearings':6};taskrows.append(row)
            selrows.append({**row,**cfg,'feature_dim':X.shape[1],'source_cv_ba':best['mean_cv'],'n_ties':ntie,'source_fold_scores':json.dumps(best['fold_scores'])})
            for idx,v in zip(np.flatnonzero(te),yp):predrows.append({'scheme':scheme,'method':method,'direction':direction,'test_groups':name,'bearing':bearing[idx],'filename':meta.filename.iloc[idx],'y_true':int(y[idx]),'y_pred':int(v)})
            print(scheme,method,direction,name,f'BA={ba:.4f}',flush=True)
    # Append atomic-ish per-scheme output to uniquely named file; completed schemas only
    for suffix,df in [('tasks',pd.DataFrame(taskrows)),('predictions',pd.DataFrame(predrows)),('selection',pd.DataFrame(selrows))]:
        df.to_csv(OUT/f'{scheme}_{method}_{suffix}.csv',index=False)
    return np.mean([r['target_ba'] for r in taskrows])

if __name__=='__main__':
    start=time.monotonic();summary=[]
    print('FROZEN GROUPINGS',schemes,flush=True)
    for scheme,g in schemes:
        for method,reps,clfs in [('snhop',REPS_H,cand_h),('full_order',REPS_O,cand_o)]:
            existing=OUT/f'{scheme}_{method}_tasks.csv'
            mean=float(pd.read_csv(existing).target_ba.mean()) if existing.is_file() and len(pd.read_csv(existing))==20 else do_method(scheme,g,method,reps,clfs)
            print('COMPLETED',scheme,method,'mean',mean,'elapsed',round(time.monotonic()-start,1),flush=True)
            summary.append(dict(scheme=scheme,method=method,mean_target_ba=mean))
            pd.DataFrame(summary).to_csv(OUT/'regrouping_completed_summary.csv',index=False)
    d=pd.DataFrame(summary).pivot(index='scheme',columns='method',values='mean_target_ba')
    d['gain_pp']=(d.snhop-d.full_order)*100;d['snhop_pct']=d.snhop*100;d['full_order_pct']=d.full_order*100
    d.to_csv(OUT/'regrouping_mean_comparison.csv')
    v1=np.mean(pd.read_csv(BASE/'results/snhop_tasks.csv').target_ba)
    vfull=np.mean(pd.read_csv(BASE/'results/order_tasks.csv').target_ba)
    a=d.loc['canonical']
    assert abs(a.snhop-v1)<1e-12, f'canonical SNHOP mismatch {a.snhop} vs {v1}'
    assert abs(a.full_order-vfull)<1e-12, f'canonical order mismatch {a.full_order} vs {vfull}'
    print('CANONICAL MATCHED; GROUPING RESULTS\n',d.to_string(),flush=True)
    print('DONE elapsed',time.monotonic()-start,flush=True)
