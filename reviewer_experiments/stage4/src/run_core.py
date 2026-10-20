import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from core import *

cache=np.load(RES/'feature_cache.npz'); meta=pd.read_csv(RES/'recordings.csv'); y=meta.label.to_numpy(); groups=meta.group.to_numpy(); rpms=meta.rpm.to_numpy()


def evaluate_spectral(method,X,compact=False,snhop=False):
    tasks=[]; preds=[]; cvrows=[]; selected_rows=[]
    for direction,held in task_defs():
        tr,te,sg,src,tgt,hg=masks(meta,direction,held)
        if snhop:
            reps=[]
            for w in [.2,.4,.6]:
                for o in [15,25,35,50]:
                    Z=hop(X,w,o); reps.append(({'halfwidth':w,'max_order':o},Z))
            clfs=list(classifier_candidates(True))
        else:
            reps=[({'norm':n},norm_rows(X,n)) for n in ['raw','l2','snv']]
            clfs=list(classifier_candidates(False))
        cand=[]
        for rc,Z in reps:
            for cc in clfs:
                cfg={**rc,**cc}; fold=[]
                for val in sg:
                    it=tr&(groups!=val); iv=tr&(groups==val)
                    clf=clf_model(cfg); clf.fit(Z[it],y[it]); fold.append(balanced_accuracy_score(y[iv],clf.predict(Z[iv])))
                mean=float(np.mean(fold)); dim=Z.shape[1]
                row={'method':method,'direction':direction,'test_groups':held,**cfg,'mean_cv':mean,'fold_scores':json.dumps(fold),'dim':dim}
                cvrows.append(row); cand.append({'cfg':cfg,'mean_cv':mean,'fold':fold,'dim':dim,'Z':Z})
        sel,ntie=select_candidates(cand)
        cfg=sel['cfg'];Z=sel['Z'];clf=clf_model(cfg);clf.fit(Z[tr],y[tr]);yp=clf.predict(Z[te]);dec=clf.decision_function(Z[te]);ba,f1=metrics(y[te],yp)
        tasks.append({'method':method,'direction':direction,'test_groups':held,**cfg,'feature_dim':Z.shape[1],'inner_cv_ba':sel['mean_cv'],'n_mean_tied':ntie,'target_ba':ba,'target_f1':f1})
        selected_rows.append({'method':method,'direction':direction,'test_groups':held,**cfg,'feature_dim':Z.shape[1],'inner_cv_ba':sel['mean_cv'],'n_mean_tied':ntie})
        for j,idx in enumerate(np.flatnonzero(te)):
            preds.append({'method':method,'direction':direction,'test_groups':held,'filename':meta.filename[idx],'bearing':meta.bearing[idx],'run':meta.run[idx],'y_true':int(y[idx]),'y_pred':int(yp[j]),'score_0':float(dec[j,0]),'score_1':float(dec[j,1]),'score_2':float(dec[j,2])})
        print(method,direction,held,f'BA={ba*100:.2f}',cfg,'ties',ntie,flush=True)
    pd.DataFrame(tasks).to_csv(RES/f'{method}_tasks.csv',index=False);pd.DataFrame(preds).to_csv(RES/f'{method}_predictions.csv',index=False);pd.DataFrame(cvrows).to_csv(RES/f'{method}_source_validation.csv',index=False);pd.DataFrame(selected_rows).to_csv(RES/f'{method}_selected.csv',index=False)


def evaluate_traditional():
    X=cache['traditional']; tasks=[];preds=[];cvrows=[]
    for direction,held in task_defs():
        tr,te,sg,src,tgt,hg=masks(meta,direction,held);cand=[]
        for k in [5,10,15,23]:
            for cc in classifier_candidates(True):
                fold=[]
                for val in sg:
                    it=tr&(groups!=val);iv=tr&(groups==val);rank=fisher_rank(X[it],y[it])[:k]
                    clf=clf_model(cc);clf.fit(X[it][:,rank],y[it]);fold.append(balanced_accuracy_score(y[iv],clf.predict(X[iv][:,rank])))
                cfg={'k':k,**cc};mean=float(np.mean(fold));cvrows.append({'method':'traditional23','direction':direction,'test_groups':held,**cfg,'mean_cv':mean,'fold_scores':json.dumps(fold),'dim':k});cand.append({'cfg':cfg,'mean_cv':mean,'fold':fold,'dim':k})
        sel,ntie=select_candidates(cand);cfg=sel['cfg'];rank=fisher_rank(X[tr],y[tr])[:int(cfg['k'])];clf=clf_model(cfg);clf.fit(X[tr][:,rank],y[tr]);yp=clf.predict(X[te][:,rank]);dec=clf.decision_function(X[te][:,rank]);ba,f1=metrics(y[te],yp)
        tasks.append({'method':'traditional23','direction':direction,'test_groups':held,**cfg,'selected_features':json.dumps(rank.tolist()),'inner_cv_ba':sel['mean_cv'],'n_mean_tied':ntie,'target_ba':ba,'target_f1':f1})
        for j,idx in enumerate(np.flatnonzero(te)):preds.append({'method':'traditional23','direction':direction,'test_groups':held,'filename':meta.filename[idx],'bearing':meta.bearing[idx],'run':meta.run[idx],'y_true':int(y[idx]),'y_pred':int(yp[j]),'score_0':float(dec[j,0]),'score_1':float(dec[j,1]),'score_2':float(dec[j,2])})
        print('traditional23',direction,held,f'BA={ba*100:.2f}',cfg,'ties',ntie,flush=True)
    pd.DataFrame(tasks).to_csv(RES/'traditional23_tasks.csv',index=False);pd.DataFrame(preds).to_csv(RES/'traditional23_predictions.csv',index=False);pd.DataFrame(cvrows).to_csv(RES/'traditional23_source_validation.csv',index=False)

for method,X,compact,snhop in [('absolute',cache['absolute'],False,False),('order',cache['order'],False,False),('pfn',cache['pfn'],False,False),('snhop',cache['order'],True,True)]:
    evaluate_spectral(method,X,compact,snhop)
evaluate_traditional()

# summaries, confusion, paired stats
alltasks=[]
for mth in ['absolute','order','traditional23','pfn','snhop']:
    d=pd.read_csv(RES/f'{mth}_tasks.csv');alltasks.append(d)
summary=[]
for d in alltasks:
    mth=d.method.iloc[0]
    for direction,g in list(d.groupby('direction'))+[('combined',d)]:summary.append({'method':mth,'direction':direction,'balanced_accuracy_pct':100*g.target_ba.mean(),'macro_f1_pct':100*g.target_f1.mean(),'task_sd_pp':100*g.target_ba.std(ddof=1)})
pd.DataFrame(summary).to_csv(RES/'main_summary.csv',index=False)
base=pd.read_csv(RES/'snhop_tasks.csv')[['direction','test_groups','target_ba']].rename(columns={'target_ba':'snhop'})
stats=[]; rawps=[]
for mth in ['absolute','order','traditional23','pfn']:
    d=pd.read_csv(RES/f'{mth}_tasks.csv')[['direction','test_groups','target_ba']].rename(columns={'target_ba':'base'});z=base.merge(d,on=['direction','test_groups']);diff=z.snhop-z.base
    try:p=wilcoxon(diff,alternative='greater',zero_method='wilcox').pvalue
    except ValueError:p=1.0
    stats.append({'baseline':mth,'gain_pp':100*diff.mean(),'wins':int((diff>1e-12).sum()),'ties':int((abs(diff)<=1e-12).sum()),'losses':int((diff<-1e-12).sum()),'raw_p':p});rawps.append(p)
# Holm
orderidx=np.argsort(rawps);adj=[None]*len(rawps);running=0
for rank,i in enumerate(orderidx):
    val=min(1.0,(len(rawps)-rank)*rawps[i]);running=max(running,val);adj[i]=running
for s,a in zip(stats,adj):s['holm_p']=a
pd.DataFrame(stats).to_csv(RES/'paired_stats.csv',index=False)
# mean normalized confusion
for mth in ['snhop','pfn','order']:
    p=pd.read_csv(RES/f'{mth}_predictions.csv');cms=[]
    for _,g in p.groupby(['direction','test_groups']):cms.append(confusion_matrix(g.y_true,g.y_pred,labels=[0,1,2],normalize='true'))
    np.savetxt(RES/f'{mth}_mean_confusion.csv',np.mean(cms,axis=0),delimiter=',')
print(pd.read_csv(RES/'main_summary.csv').to_string(index=False));print(pd.read_csv(RES/'paired_stats.csv').to_string(index=False))
