#!/usr/bin/env python3
"""Prospective SN-HOP IEEE Access revision analyses. Does not modify v1.
Matched-dimension baselines use original source-selected SN-HOP dimension in each
outer task; all model selection/transform fitting is done solely on source bearings.
Single-target adaptation uses runs 1-10 of each target bearing separately and
predicts runs 11-20 of that same bearing without using target labels for adaptation.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import json,sys,hashlib,platform,time,itertools
from pathlib import Path
import numpy as np, pandas as pd, scipy, sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from scipy.linalg import eigh
ROOT=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1'))); DEST=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; OUT=DEST/'results'
sys.path.insert(0,str(ROOT/'src'))
from core import task_defs,masks,hop,clf_model,classifier_candidates,metrics,norm_rows,fisher_rank
cache=np.load(RES/'feature_cache.npz');meta=pd.read_csv(RES/'recordings.csv');y=meta.label.to_numpy();groups=meta.group.to_numpy();runs=meta.run.to_numpy()
X=norm_rows(cache['order'],'snv');selected=pd.read_csv(RES/'snhop_selected.csv');sel_index={(r.direction,r.test_groups):r for r in selected.itertuples(index=False)}


def covpow(a,p):
    evals,u=eigh(a,check_finite=False);return (u*np.maximum(evals,1e-12)**p)@u.T

def fit_transform_dim(xtrain,ytrain,xtest,k,method):
    if method=='PCA':
        p=PCA(n_components=int(k),svd_solver='randomized',random_state=20261019,n_oversamples=10)
        return p.fit_transform(xtrain),p.transform(xtest)
    if method=='Fisher':
        rank=fisher_rank(xtrain,ytrain)[:k]
        return xtrain[:,rank],xtest[:,rank]
    raise ValueError(method)


def run_dimension_matched():
    outputs=[];preds=[];selcv=[]
    clfs=list(classifier_candidates(True))
    for direction,held in task_defs():
        tr,te,sg,*_=masks(meta,direction,held); orig=sel_index[direction,held]; k=int(orig.feature_dim)
        print('MATCH',direction,held,'k',k,flush=True)
        for rep in ('PCA','Fisher'):
            folds=[]
            for val in sg:
                it=tr&(groups!=val);iv=tr&(groups==val)
                za,zb=fit_transform_dim(X[it],y[it],X[iv],k,rep)
                folds.append((za,y[it],zb,y[iv]))
            candidates=[]
            for cfg in clfs:
                scores=[]
                for za,ya,zb,yb in folds:
                    model=clf_model(cfg);model.fit(za,ya)
                    scores.append(metrics(yb,model.predict(zb))[0])
                candidates.append((float(np.mean(scores)),cfg,scores))
            best=max(s for s,_,_ in candidates)
            tied=[(s,c,fold) for s,c,fold in candidates if s>=best-1e-12]
            # same tie policy as v1: prefer logistic, then smaller C/gamma
            _,cfg,fold=min(tied,key=lambda t:(0 if t[1]['classifier']=='logreg' else 1,float(t[1]['C']),str(t[1]['gamma'])))
            selcv.append({'rep':rep,'direction':direction,'test_groups':held,'k':k,'classifier':cfg['classifier'],'C':cfg['C'],'gamma':cfg['gamma'],'source_cv_ba':best,'cv_folds':json.dumps(fold)})
            za,zb=fit_transform_dim(X[tr],y[tr],X[te],k,rep);model=clf_model(cfg);model.fit(za,y[tr]);yp=model.predict(zb);ba,f1=metrics(y[te],yp)
            outputs.append({'rep':rep,'direction':direction,'test_groups':held,'k':k,'target_ba':ba,'target_f1':f1,'source_cv_ba':best})
            for idx,v in zip(np.flatnonzero(te),yp):preds.append({'rep':rep,'direction':direction,'test_groups':held,'bearing':meta.bearing[idx],'filename':meta.filename[idx],'y_true':int(y[idx]),'y_pred':int(v)})
    pd.DataFrame(outputs).to_csv(OUT/'matched_dimension_tasks.csv',index=False)
    pd.DataFrame(preds).to_csv(OUT/'matched_dimension_predictions.csv',index=False)
    pd.DataFrame(selcv).to_csv(OUT/'matched_dimension_source_selection.csv',index=False)
    summary=pd.DataFrame(outputs).groupby('rep')[['target_ba','target_f1']].agg(['mean','std'])
    print('DIMENSION SUMMARY\n',summary.to_string(),flush=True)


def run_single_target_uda():
    outputs=[];preds=[];bearing_rows=[]
    for direction,held in task_defs():
        r=sel_index[direction,held]; tr,tt,*_=masks(meta,direction,held)
        F=hop(cache['order'],float(r.halfwidth),float(r.max_order))
        src_scale=StandardScaler().fit(F[tr]);S=src_scale.transform(F[tr]); T=src_scale.transform(F)
        base=clf_model({'classifier':r.classifier,'C':r.C,'gamma':r.gamma}).steps[-1][1];base.fit(S,y[tr])
        pvar={v:np.full(600,-1,int) for v in ('NoDA','CORAL_single_target','TargetZ_single_target')}
        for b in sorted(meta.bearing[tt].unique()):
            is_b=meta.bearing.to_numpy()==b
            ad=tt&is_b&(runs<=10);te=tt&is_b&(runs>=11)
            assert ad.sum()==10 and te.sum()==10
            pvar['NoDA'][te]=base.predict(T[te])
            # align the entire source domain to the unlabeled distribution of ONE bearing
            a=T[ad];Cs=np.cov(S,rowvar=False)+1e-3*np.eye(S.shape[1]);Ct=np.cov(a,rowvar=False)+1e-3*np.eye(S.shape[1]); aligned=(S-S.mean(0))@covpow(Cs,-0.5)@covpow(Ct,0.5)+a.mean(0)
            ca=clf_model({'classifier':r.classifier,'C':r.C,'gamma':r.gamma}).steps[-1][1];ca.fit(aligned,y[tr]);pvar['CORAL_single_target'][te]=ca.predict(T[te])
            # comparing with per-bearing target-only normalization, as in legacy adaptation study
            pvar['TargetZ_single_target'][te]=base.predict(StandardScaler().fit(F[ad]).transform(F[te]))
            for v in pvar:
                bearing_rows.append({'variant':v,'direction':direction,'test_groups':held,'bearing':b,'true_class':int(y[te][0]),'recall':float((pvar[v][te]==y[te]).mean())})
        test=tt&(runs>=11);assert test.sum()==60
        for v in pvar:
            yp=pvar[v][test]; assert np.all(yp>=0)
            ba,f1=metrics(y[test],yp)
            outputs.append({'variant':v,'direction':direction,'test_groups':held,'target_ba':ba,'target_f1':f1})
            for idx,vv in zip(np.flatnonzero(test),yp):preds.append({'variant':v,'direction':direction,'test_groups':held,'bearing':meta.bearing[idx],'filename':meta.filename[idx],'y_true':int(y[idx]),'y_pred':int(vv)})
        print('UDA single target',direction,held,[(v,round(metrics(y[test],pvar[v][test])[0],3)) for v in pvar],flush=True)
    pd.DataFrame(outputs).to_csv(OUT/'single_target_uda_tasks.csv',index=False)
    pd.DataFrame(preds).to_csv(OUT/'single_target_uda_predictions.csv',index=False)
    pd.DataFrame(bearing_rows).to_csv(OUT/'single_target_uda_bearing_recalls.csv',index=False)
    print('SINGLE TARGET SUMMARY\n',pd.DataFrame(outputs).groupby('variant')['target_ba'].mean().to_string(),flush=True)


def group_dependence_analysis():
    names=['absolute','order','traditional23','pfn']; out=[];bearing_rows=[]
    sn=pd.read_csv(RES/'snhop_predictions.csv')
    # 15 bearing-level scores pooled across 2 transfer directions & 4 held-out test configurations
    def bear_recall(df):
        return df.assign(correct=(df.y_pred==df.y_true).astype(float)).groupby('bearing')['correct'].mean()
    sn_r=bear_recall(sn)
    for mth in names:
        d=pd.read_csv(RES/f'{mth}_predictions.csv')
        keys=['direction','test_groups','filename','bearing','y_true']; joined=sn.merge(d,on=keys,suffixes=('_snhop','_base'),validate='one_to_one')
        assert len(joined)==len(sn)==2400
        br=bear_recall(d); deltas=(sn_r-br).sort_index()
        binfo=meta.drop_duplicates('bearing').set_index('bearing')[['group','label']]
        for b,v in deltas.items():bearing_rows.append({'baseline':mth,'bearing':b,'class_label':int(binfo.loc[b,'label']),'triplet':int(binfo.loc[b,'group']),'snhop_recall':sn_r.loc[b],'baseline_recall':br.loc[b],'difference':v})
        clusters=np.array([deltas[binfo.group==i].mean() for i in range(1,6)])
        effect=clusters.mean()
        rng=np.random.default_rng(20261019); reps=50000
        boot=clusters[rng.integers(0,5,size=(reps,5))].mean(1)
        ci=np.quantile(boot,[.025,.975]); exact_signs=np.asarray(list(itertools.product([-1,1],repeat=5)))
        nulls=(exact_signs*clusters).mean(1)
        p_two=float(np.mean(np.abs(nulls)>=abs(effect)-1e-12))
        out.append({'baseline':mth,'mean_difference_pp':effect*100,'triplet_boot_CI_low_pp':ci[0]*100,'triplet_boot_CI_high_pp':ci[1]*100,'exact_5cluster_sign_flip_p_two_sided':p_two,'n_physical_bearings':15,'n_arbitrary_balanced_triplets':5})
    pd.DataFrame(out).to_csv(OUT/'bearing_cluster_sensitivity.csv',index=False)
    pd.DataFrame(bearing_rows).to_csv(OUT/'per_bearing_method_comparison.csv',index=False)
    detail=sn.assign(correct=(sn.y_true==sn.y_pred).astype(float)).groupby(['bearing','direction']).agg(recall=('correct','mean'),n_appearances=('correct','size')).reset_index()
    detail['label']=detail.bearing.map(meta.drop_duplicates('bearing').set_index('bearing')['label'])
    detail.to_csv(OUT/'snhop_bearing_direction_recalls.csv',index=False)
    print('CLUSTER SENSITIVITY\n',pd.DataFrame(out).to_string(index=False),flush=True)


def main():
    st=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    print('Starting fixed prior-results analysis',flush=True)
    group_dependence_analysis(); print('cluster done',round(time.monotonic()-st,1),flush=True)
    run_dimension_matched();print('matched done',round(time.monotonic()-st,1),flush=True)
    run_single_target_uda();print('single-target done',round(time.monotonic()-st,1),flush=True)
    (DEST/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'elapsed_seconds':time.monotonic()-st,'main_cache_sha256':hashlib.sha256((RES/'feature_cache.npz').read_bytes()).hexdigest(),'v1_untouched':True},indent=2))
    print('DONE',round(time.monotonic()-st,1),flush=True)

if __name__=='__main__':main()
