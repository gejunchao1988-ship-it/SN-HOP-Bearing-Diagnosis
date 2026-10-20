"""Paderborn load-variation target-only evaluation; v1 model choices frozen.

No target bearing appears in fitting or model selection. Model and hyperparameter
choices are frozen from the canonical v1 SOURCE-ONLY CV result of each outer fold.
New target conditions are neither used to select nor to fit the model.
"""
from __future__ import annotations
import os, sys, glob, json, time, zipfile, io, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.metrics import balanced_accuracy_score, f1_score

HERE=Path(__file__).resolve().parent
RAW_DIR=Path(os.environ.get('SNHOP_RAW_DIR',str(Path(__file__).resolve().parents[2]/'raw')))
sys.path.insert(0,str(HERE/'src'))
from core import _load_record, GROUPS, GM, ORDER_GRID, norm_rows, hop, clf_model

BEARINGS=[b for triplet in GROUPS for b in triplet]
CONDITIONS=['N15_M01_F10','N15_M07_F04']
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
ARCHIVE=Path(os.environ.get('SNHOP_CANONICAL_ARCHIVE',str(HERE.parents[1]/'SNHOP_canonical_reproducibility_v1.zip')))
CODE_ZIP_PREFIX='SNHOP_canonical_reproducibility_v1/'
SRC_META={"canonical_code_file":"src/core.py","method":"500-10000 Hz 4th-order SOS, filtfilt; abs Hilbert envelope, FFT 8192, order interpolation on 0-50/256 grid, 31 nonoverlapping windows, per-recording mean", "source":"canonical cache and source-only frozen hyperparameters from September 2026 reproducibility v1", "target_conditions":CONDITIONS,"target_leakage":"target features/labels never used for fitting, parameter selection, normalization fitting", "notes":"The canonical original cross-speed 1500_to_900 model selection is reused for source 1500-to-new-load tests; 900_to_1500 for source 900-to-new-load tests. Independent across target tasks but physical specimens recur across folds."}
(OUT/'design.json').write_text(json.dumps(SRC_META,ensure_ascii=False,indent=2))


def extract():
    sos=butter(4,[500,10000],btype='bandpass',fs=64000,output='sos')
    f8192=np.fft.rfftfreq(8192,1/64000)
    X=[]; rows=[]
    start=time.time()
    for b in BEARINGS:
        paths=glob.glob(str(RAW_DIR / (b+'(*).zip')))
        if len(paths)!=1:raise RuntimeError('Ambiguous archive '+b)
        with zipfile.ZipFile(paths[0]) as z:
            filenames={Path(n).name:n for n in z.namelist() if n.endswith('.mat')}
            for condition in CONDITIONS:
                for run in range(1,21):
                    name=f'{condition}_{b}_{run}.mat'
                    x,rpm=_load_record(z,filenames[name])
                    seg=x[:8192*31].reshape(31,8192).copy()
                    seg-=seg.mean(axis=1,keepdims=True)
                    filt=sosfiltfilt(sos,seg,axis=1)
                    env=np.abs(hilbert(filt,axis=1))
                    env-=env.mean(axis=1,keepdims=True)
                    esp=np.log1p(np.abs(np.fft.rfft(env,axis=1)))
                    f_r=rpm/60.
                    ordspec=np.asarray([np.interp(ORDER_GRID,f8192/f_r,s) for s in esp],dtype=np.float32).mean(0)
                    X.append(ordspec)
                    label=0 if b.startswith('K0') else 1 if b.startswith('KA') else 2
                    rows.append({'filename':name,'condition':condition,'bearing':b,'label':label,'group':GM[b],'rpm':1500,'measured_rpm':rpm,'run':run,'samples':len(x)})
        print('extracted',b,'records',len(rows),'elapsed',round(time.time()-start,1),flush=True)
    np.savez_compressed(OUT/'new_load_order_cache.npz',order=np.asarray(X,np.float32))
    pd.DataFrame(rows).to_csv(OUT/'new_load_recordings.csv',index=False)
    print('saved extracted records',len(rows),flush=True)


def eval():
    import itertools
    base=HERE/'baseline'
    if (base/'feature_cache.npz').exists():
        old_cache=np.load(base/'feature_cache.npz')
        def get_csv(p):return pd.read_csv(base/Path(p).name)
        old_meta=get_csv('recordings.csv')
        conf_snhop=get_csv('snhop_selected.csv')
        conf_order=get_csv('order_selected.csv')
        original_snhop=get_csv('snhop_tasks.csv')
        original_order=get_csv('order_tasks.csv')
    else:
        with zipfile.ZipFile(ARCHIVE) as z:
            def get_csv(p):return pd.read_csv(io.BytesIO(z.read(CODE_ZIP_PREFIX+p)))
            old_cache=np.load(io.BytesIO(z.read(CODE_ZIP_PREFIX+'results/feature_cache.npz')))
            old_meta=get_csv('results/recordings.csv')
            conf_snhop=get_csv('results/snhop_selected.csv')
            conf_order=get_csv('results/order_selected.csv')
            original_snhop=get_csv('results/snhop_tasks.csv')
            original_order=get_csv('results/order_tasks.csv')
    new_cache=np.load(OUT/'new_load_order_cache.npz')
    new_meta=pd.read_csv(OUT/'new_load_recordings.csv')
    oldX=old_cache['order'];newX=new_cache['order']; yold=old_meta.label.to_numpy();ynew=new_meta.label.to_numpy()
    if len(newX)!=600:raise RuntimeError('Missing data')
    task=[]; preds=[]; conf=[]
    for method,configs,orig in [('snhop',conf_snhop,original_snhop),('order',conf_order,original_order)]:
        for source_rpm,orig_dir in [(900,'900_to_1500'),(1500,'1500_to_900')]:
            for a,b in itertools.combinations(range(1,6),2):
                held=f'{a}-{b}'
                cc=configs[(configs.direction==orig_dir)&(configs.test_groups==held)]
                assert len(cc)==1
                cfg=cc.iloc[0].to_dict()
                groups=old_meta.group.to_numpy()
                it=(old_meta.rpm.to_numpy()==source_rpm)&(~np.isin(groups,[a,b]))
                ictrl=(old_meta.rpm.to_numpy()==1500)&(np.isin(groups,[a,b]))
                new_test=np.isin(new_meta.group.to_numpy(),[a,b])
                if method=='snhop':
                    trX=hop(oldX[it],cfg['halfwidth'],int(cfg['max_order']))
                    tX_standard=hop(oldX[ictrl],cfg['halfwidth'],int(cfg['max_order']))
                    tX_new=hop(newX[new_test],cfg['halfwidth'],int(cfg['max_order']))
                else:
                    trX=norm_rows(oldX[it],cfg['norm']);tX_standard=norm_rows(oldX[ictrl],cfg['norm']);tX_new=norm_rows(newX[new_test],cfg['norm'])
                fcfg={'classifier':cfg['classifier'],'C':float(cfg['C']),'gamma':cfg['gamma'] if pd.notna(cfg['gamma']) else None}
                clf=clf_model(fcfg);clf.fit(trX,yold[it])
                controls=[('N15_M07_F10',old_meta[ictrl],yold[ictrl],tX_standard)]
                nf=new_meta[new_test].reset_index(drop=True)
                yn=ynew[new_test]
                for cond in CONDITIONS:
                    mc=(nf.condition.to_numpy()==cond)
                    controls.append((cond,nf[mc],yn[mc],tX_new[mc]))
                for cond,meta,ytrue,tx in controls:
                    pred=clf.predict(tx)
                    ba=balanced_accuracy_score(ytrue,pred)
                    f1=f1_score(ytrue,pred,average='macro',zero_division=0)
                    task.append({'method':method,'source_rpm':source_rpm,'target_condition':cond,'held_groups':held,'balanced_accuracy':ba,'macro_f1':f1,'source_bearings':9,'test_bearings':6,'target_recordings':len(ytrue),'configuration_origin':orig_dir})
                    for j,r in enumerate(meta.itertuples()):
                        preds.append({'method':method,'source_rpm':source_rpm,'target_condition':cond,'held_groups':held,'filename':r.filename,'bearing':r.bearing,'run':int(r.run),'y_true':int(ytrue[j]),'y_pred':int(pred[j])})
                # 900->1500 original tasks must reproduce exactly (locked protocol).
                if source_rpm==900:
                    canonical=orig[(orig.direction=='900_to_1500')&(orig.test_groups==held)].target_ba.iloc[0]
                    got=[t['balanced_accuracy'] for t in task if t['method']==method and t['source_rpm']==900 and t['target_condition']=='N15_M07_F10' and t['held_groups']==held][-1]
                    if abs(float(canonical)-got)>1e-12:raise RuntimeError(f'Canonical result mismatch: {method} {held}: {canonical} {got}')
                print('evaluated',method,source_rpm,held,flush=True)
    tasks=pd.DataFrame(task);preds=pd.DataFrame(preds)
    tasks.to_csv(OUT/'new_load_tasks.csv',index=False)
    preds.to_csv(OUT/'new_load_predictions.csv',index=False)
    summary=tasks.groupby(['method','source_rpm','target_condition'],sort=True).agg(balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),macro_f1_mean=('macro_f1','mean'),n_tasks=('balanced_accuracy','size')).reset_index()
    summary.to_csv(OUT/'new_load_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','eval']); args=p.parse_args()
    if args.stage=='extract':extract()
    else:eval()
