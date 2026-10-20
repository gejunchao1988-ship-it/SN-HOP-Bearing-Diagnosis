#!/usr/bin/env python3
"""Pre-specified source-only bearing-group GroupDRO vs identical-model ERM.
All feature selection/model selection use SOURCE bearing groups only; no target data used.
GroupDRO is a small linear-classifier implementation of the group DRO objective,
not an implementation of any published bearing-specific neural DG architecture.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys,json,itertools,time,hashlib,platform,shutil
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
import sklearn,torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
V1=ROOT/'SNHOP_canonical_reproducibility_v1'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(V1/'src'))
from core import norm_rows, masks,task_defs

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
SEED=20261020
EPOCHS=160
LR=0.02

meta=pd.read_csv(V1/'results/recordings.csv'); yy=meta.label.to_numpy();grp=meta.group.to_numpy()
X=norm_rows(np.load(V1/'results/feature_cache.npz')['order'],'snv').astype('float64')
assert X.shape==(600,256)

# Reproducible GroupDRO (exponentially tilted group CE risk) with an L2 regularized linear classifier.
def fit_model(x,y,groups,kind,reg=0.01,eta=0.1):
    ids=np.unique(groups); scaler=StandardScaler().fit(x)
    tX=torch.tensor(scaler.transform(x),dtype=torch.float64)
    ty=torch.tensor(y.astype(int),dtype=torch.long)
    gi=[torch.tensor(np.flatnonzero(groups==g),dtype=torch.long) for g in ids]
    torch.manual_seed(SEED)
    # initialize identically (zeros) across ERM and DRO to isolate loss weighting
    W=torch.nn.Parameter(torch.zeros(x.shape[1],3,dtype=torch.float64))
    b=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64))
    opt=torch.optim.Adam([W,b],lr=LR)
    q=torch.ones(len(ids),dtype=torch.float64)/len(ids)
    for epoch in range(EPOCHS):
        opt.zero_grad()
        logits=tX@W+b
        lossvec=F.cross_entropy(logits,ty,reduction='none')
        losses=torch.stack([lossvec[z].mean() for z in gi])
        if kind=='GroupDRO':
            with torch.no_grad():
                # tilt based on current group risks; maintain normalized q
                q=torch.softmax(torch.log(q)+eta*losses.detach(),dim=0)
        loss=torch.sum(q*losses)+reg*torch.sum(W*W)
        loss.backward();opt.step()
    with torch.no_grad():
        logits=tX@W+b;lv=F.cross_entropy(logits,ty,reduction='none')
        gl=[float(lv[z].mean()) for z in gi]
    return dict(scaler=scaler,W=W.detach().numpy(),b=b.detach().numpy(),q=q.numpy(),group_ids=ids.tolist(),group_losses=gl)

def predict(model,x):
    logits=model['scaler'].transform(x)@model['W']+model['b']
    z=logits-logits.max(axis=1,keepdims=True)
    pp=np.exp(z);pp=pp/pp.sum(axis=1,keepdims=True)
    return np.argmax(pp,axis=1),pp

# Keeping candidates deliberately small mitigates tuning instability in 5 unique triplets.
candidates=[dict(method='ERM',reg=r,eta=0) for r in (0.001,0.01,0.1)]+[dict(method='GroupDRO',reg=r,eta=e) for r in (0.001,0.01,0.1) for e in (0.05,0.2)]


def run():
    t0=time.time();tasks=[];pred=[];cv=[];selected=[]; models={}
    for direction,held in task_defs():
        tr,te,sg,_,_,_=masks(meta,direction,held)
        assert tr.sum()==180 and te.sum()==120
        for config in candidates:
            scores=[]
            for v in sg:
                it=tr&(grp!=v);iv=tr&(grp==v)
                assert it.sum()==120 and iv.sum()==60 and set(meta.bearing[it]).isdisjoint(set(meta.bearing[iv]))
                model=fit_model(X[it],yy[it],grp[it],config['method'],config['reg'],config['eta'])
                yhat,_=predict(model,X[iv]);scores.append(float(balanced_accuracy_score(yy[iv],yhat)))
            cv.append({**config,'direction':direction,'test_groups':held,'source_cv_ba':float(np.mean(scores)), 'cv_scores':json.dumps(scores)})
        for method in ['ERM','GroupDRO']:
            eligible=[z for z in cv if z['direction']==direction and z['test_groups']==held and z['method']==method]
            best=max(z['source_cv_ba'] for z in eligible)
            choose=min([z for z in eligible if z['source_cv_ba']>=best-1e-12],key=lambda z:(z['reg'],z['eta']))
            model=fit_model(X[tr],yy[tr],grp[tr],method,choose['reg'],choose['eta'])
            yp,p=predict(model,X[te])
            ba=float(balanced_accuracy_score(yy[te],yp)); f1=float(f1_score(yy[te],yp,average='macro',zero_division=0))
            selkey=f'{method}_{direction}_{held}'
            models[selkey+'_coef']=model['W']; models[selkey+'_intercept']=model['b']
            models[selkey+'_standard_mean']=model['scaler'].mean_;models[selkey+'_standard_scale']=model['scaler'].scale_
            tasks.append(dict(method=method,direction=direction,test_groups=held,target_ba=ba,target_f1=f1,source_cv_ba=best,reg=choose['reg'],eta=choose['eta'],train_groups=json.dumps(model['group_ids']),last_group_weights=json.dumps(model['q'].tolist()),last_group_CE=json.dumps(model['group_losses'])))
            selected.append(dict(method=method,direction=direction,test_groups=held,source_cv_ba=best,reg=choose['reg'],eta=choose['eta']))
            for j,idx in enumerate(np.flatnonzero(te)):
                pred.append(dict(method=method,direction=direction,test_groups=held,bearing=meta.bearing.iloc[idx],filename=meta.filename.iloc[idx],y_true=int(yy[idx]),y_pred=int(yp[j]),p_healthy=float(p[j,0]),p_outer=float(p[j,1]),p_inner=float(p[j,2])))
            print(f'{direction} {held} {method:<8} {ba*100:.2f}%  best={choose["reg"]}/{choose["eta"]} sourceCV={best*100:.2f}%',flush=True)
    pd.DataFrame(tasks).to_csv(OUT/'dg_tasks.csv',index=False)
    pd.DataFrame(pred).to_csv(OUT/'dg_predictions.csv',index=False)
    pd.DataFrame(cv).to_csv(OUT/'dg_source_only_cv.csv',index=False)
    pd.DataFrame(selected).to_csv(OUT/'dg_source_selected.csv',index=False)
    np.savez_compressed(OUT/'dg_models.npz',**models)
    summary=pd.DataFrame(tasks).groupby(['method','direction']).agg(mean_ba=('target_ba','mean'),sd_ba=('target_ba','std'),mean_f1=('target_f1','mean')).reset_index()
    overall=pd.DataFrame(tasks).groupby('method').agg(mean_ba=('target_ba','mean'),mean_f1=('target_f1','mean')).reset_index(); overall['direction']='combined'
    summ=pd.concat([summary,overall],ignore_index=True);summ.to_csv(OUT/'dg_summary.csv',index=False)
    checks=[]
    for r in pd.DataFrame(tasks).itertuples():
        p=pd.DataFrame(pred); z=p[(p.method==r.method)&(p.direction==r.direction)&(p.test_groups==r.test_groups)]
        assert len(z)==120
        calc=balanced_accuracy_score(z.y_true,z.y_pred)
        assert abs(calc-r.target_ba)<1e-12
        checks.append({'method':r.method,'direction':r.direction,'test_groups':r.test_groups,'ba_recalc':float(calc),'matches':True})
    pd.DataFrame(checks).to_csv(OUT/'dg_independent_metric_recalc.csv',index=False)
    info=dict(seed=SEED, epochs=EPOCHS, learning_rate=LR, input='canonical v1 full order spectra, per-recording SNV', optimizer='Adam full batch', objective='linear ERM mean group CE, GroupDRO exponential-weighted group CE + L2 coefficient penalty', validation='3-fold SOURCE-bearing-triplet grouped CV; target excluded from train/scaler/selection', hyperparameters=candidates, n_fold_models=20*3*len(candidates), n_final_models=40,n_tasks=len(tasks),n_pred=len(pred), n_cv=len(cv),execution_seconds=time.time()-t0, libraries={'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},source_cache_sha256=hashlib.sha256((V1/'results/feature_cache.npz').read_bytes()).hexdigest())
    (OUT/'dg_design_and_environment.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('SUMMARY\n',summ.to_string(index=False),'\nTIME',round(time.time()-t0,1),flush=True)

if __name__=='__main__':run()
