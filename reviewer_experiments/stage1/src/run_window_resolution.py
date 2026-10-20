#!/usr/bin/env python3
"""Controlled window-length sensitivity for reviewer R1-2.
The EXACT first 253952 raw samples are used in every setting, with an
8192-sample hop; longer windows overlap. Same filter/order interpolation;
training and hyperparameter choice are reselected using source-bearing CV.
No target bearing data is used for fitting, feature design or model selection.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,io,zipfile,time,json,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import butter,sosfiltfilt,hilbert
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')));DEST=Path(__file__).resolve().parents[1];RES=ROOT/'results';OUT=DEST/'results';RAWDIR=Path(os.environ.get('SNHOP_RAW_DIR',str(Path(__file__).resolve().parents[3]/'raw')))
sys.path.insert(0,str(ROOT/'src'))
from core import _load_record,ORDER_GRID,hop,task_defs,masks,clf_model,classifier_candidates,metrics,select_candidates
meta=pd.read_csv(RES/'recordings.csv');y=meta.label.to_numpy();groups=meta.group.to_numpy(); cache=np.load(RES/'feature_cache.npz')
source_manifest=pd.read_csv(ROOT/'evidence'/'paderborn_600_file_manifest.csv'); IDX={f:i for i,f in enumerate(meta.filename)}
for r in source_manifest.itertuples():assert r.mat_file in IDX
sos=butter(4,[500,10000],btype='bandpass',fs=64000,output='sos')
LENS=[16384,32768];N0=253952;HOP=8192


def make_spectrum(x,rpm,L):
    assert len(x)>=N0
    # No padding, and identical observation length for all 3 window settings.
    # Unlike the v1 nonoverlapping 8192-sample segments, larger windows overlap.
    view=np.lib.stride_tricks.sliding_window_view(x[:N0],L)[::HOP];seg=np.asarray(view,dtype=np.float64).copy()
    seg-=seg.mean(axis=1,keepdims=True)
    v=sosfiltfilt(sos,seg,axis=1)
    e=np.abs(hilbert(v,axis=1));e-=e.mean(axis=1,keepdims=True)
    f=np.fft.rfftfreq(L,d=1/64000)
    sp=np.log1p(np.abs(np.fft.rfft(e,axis=1)))
    coords=f/(rpm/60)
    return np.asarray([np.interp(ORDER_GRID,coords,s) for s in sp],np.float32).mean(0)


def raw_to_cache():
    N={L:np.empty((len(meta),256),np.float32) for L in LENS}
    timing=[];t0=time.monotonic()
    for arc,sub in source_manifest.groupby('archive',sort=False):
        with zipfile.ZipFile(RAWDIR/arc) as z:
            names={Path(n).name:n for n in z.namelist() if n.lower().endswith('.mat')}
            for r in sub.itertuples():
                t_read=time.perf_counter(); x,rpm=_load_record(z,names[r.mat_file]); read_s=time.perf_counter()-t_read
                idx=IDX[r.mat_file]
                assert abs(rpm-meta.measured_rpm[idx])<1e-6
                for L in LENS:
                    ts=time.perf_counter();N[L][idx]=make_spectrum(x,rpm,L)
                    timing.append({'filename':r.mat_file,'bearing':r.bearing,'window':L,'zip_read_seconds':read_s,'preprocess_seconds':time.perf_counter()-ts,'windows_per_recording':(N0-L)//HOP+1})
        print('raw processed',arc,'total',len(timing),'elapsed',round(time.monotonic()-t0,1),flush=True)
    assert len(timing)==1200
    np.savez_compressed(OUT/'window_order_cache.npz',order_16384=N[16384],order_32768=N[32768])
    pd.DataFrame(timing).to_csv(OUT/'window_preprocess_timing.csv',index=False)
    print('raw cache built in',round(time.monotonic()-t0,1),'s',flush=True)
    return N


def validate_reconstruction():
    for a in source_manifest.iloc[[0,79,290,599]].itertuples():
        with zipfile.ZipFile(RAWDIR/a.archive) as z:
            names={Path(n).name:n for n in z.namelist() if n.lower().endswith('.mat')}
            x,rpm=_load_record(z,names[a.mat_file])
        v=make_spectrum(x,rpm,8192);prior=cache['order'][IDX[a.mat_file]]
        diff=float(np.max(np.abs(v-prior)));print('check original preprocessing',a.mat_file,'difference',diff,flush=True)
        if diff>2e-6: raise RuntimeError(f'Preprocessing failed to reproduce old cache {diff}')


def select_and_evaluate(Z,L):
    tasks=[];pred=[];choices=[];clfs=list(classifier_candidates(True))
    for direction,held in task_defs():
        tr,te,sg,*_=masks(meta,direction,held)
        can=[]
        for width in [.2,.4,.6]:
            for maxo in [15,25,35,50]:
                P=hop(Z,width,maxo)
                for c in clfs:
                    config={**c,'halfwidth':width,'max_order':maxo};scores=[]
                    for val in sg:
                        it=tr&(groups!=val);iv=tr&(groups==val);model=clf_model(config);model.fit(P[it],y[it]);scores.append(balanced_accuracy_score(y[iv],model.predict(P[iv])))
                    can.append({'cfg':config,'mean_cv':float(np.mean(scores)),'dim':P.shape[1], 'fold':scores})
        sel,ntie=select_candidates(can);cfg=sel['cfg'];P=hop(Z,cfg['halfwidth'],cfg['max_order']);model=clf_model(cfg);model.fit(P[tr],y[tr]);yp=model.predict(P[te]);ba,f1=metrics(y[te],yp)
        tasks.append({'window':L,'direction':direction,'test_groups':held,'target_ba':ba,'target_f1':f1,'feature_dim':P.shape[1],'source_cv_ba':sel['mean_cv']})
        choices.append({'window':L,'direction':direction,'test_groups':held,**cfg,'source_cv_ba':sel['mean_cv'],'n_tied':ntie,'feature_dim':P.shape[1]})
        for idx,v in zip(np.flatnonzero(te),yp):pred.append({'window':L,'direction':direction,'test_groups':held,'bearing':meta.bearing[idx],'filename':meta.filename[idx],'y_true':int(y[idx]),'y_pred':int(v)})
        print('window',L,direction,held,'BA',round(ba*100,2),'CV',round(sel['mean_cv']*100,2),flush=True)
    return pd.DataFrame(tasks),pd.DataFrame(pred),pd.DataFrame(choices)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    validate_reconstruction()
    dest=OUT/'window_order_cache.npz'
    if dest.exists():
        loaded=np.load(dest);data={L:loaded[f'order_{L}'] for L in LENS}
    else:data=raw_to_cache()
    allr=[];allp=[];allc=[]
    for L in LENS:
        r,p,c=select_and_evaluate(data[L],L);allr.append(r);allp.append(p);allc.append(c)
        pd.concat(allr).to_csv(OUT/'window_resolution_tasks.csv',index=False)
        pd.concat(allp).to_csv(OUT/'window_resolution_predictions.csv',index=False)
        pd.concat(allc).to_csv(OUT/'window_resolution_source_selection.csv',index=False)
    print('WINDOW RESULTS\n',pd.concat(allr).groupby('window')[['target_ba','target_f1']].mean(),flush=True)
    print('DONE',flush=True)
if __name__=='__main__':main()
