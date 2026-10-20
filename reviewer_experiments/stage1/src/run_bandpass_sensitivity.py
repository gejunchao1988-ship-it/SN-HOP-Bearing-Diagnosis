#!/usr/bin/env python3
"""Reviewer R1-3: pre-specified band-pass sensitivity using full 600 raw recordings.
The bands are 200-12000 and 1000-8000 Hz, compared with original 500-10000 Hz.
Signals/segments/FFT/order grid/outer tasks are identical to v1; all models and
harmonic settings are re-selected ONLY on source-bearing validation folds.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import butter,sosfiltfilt,hilbert
ROOT=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')));RES=ROOT/'results';OUT=Path(__file__).resolve().parents[1]/'results';ROOTSRC=ROOT/'src';sys.path.insert(0,str(ROOTSRC));sys.path.insert(0,str(Path(__file__).parent))
from core import _load_record,ORDER_GRID
from run_window_resolution import select_and_evaluate
meta=pd.read_csv(RES/'recordings.csv');source=pd.read_csv(ROOT/'evidence/paderborn_600_file_manifest.csv');name_idx={f:i for i,f in enumerate(meta.filename)}
FS=64000;L=8192;N=31;ff=np.fft.rfftfreq(L,1/FS);BANDS={'200_12000':(200,12000),'1000_8000':(1000,8000)}

def spectrum(x,rpm,sos):
    seg=x[:L*N].reshape(N,L).copy();seg-=seg.mean(1,keepdims=True)
    f=sosfiltfilt(sos,seg,axis=1)
    e=np.abs(hilbert(f,axis=1));e-=e.mean(1,keepdims=True)
    sp=np.log1p(np.abs(np.fft.rfft(e,axis=1)))
    return np.asarray([np.interp(ORDER_GRID,ff/(rpm/60),s) for s in sp],np.float32).mean(0)

def build():
    feats={k:np.empty((600,256),np.float32) for k in BANDS};bands={k:butter(4,cut,btype='bandpass',fs=FS,output='sos') for k,cut in BANDS.items()}
    start=time.monotonic();rows=[]
    for arc,sub in source.groupby('archive',sort=False):
        with zipfile.ZipFile(Path(os.environ.get('SNHOP_RAW_DIR',str(Path(__file__).resolve().parents[3]/'raw')))/arc) as z:
            names={Path(n).name:n for n in z.namelist() if n.lower().endswith('.mat')}
            for r in sub.itertuples():
                x,rpm=_load_record(z,names[r.mat_file]);idx=name_idx[r.mat_file]
                for key in BANDS:
                    t=time.perf_counter();feats[key][idx]=spectrum(x,rpm,bands[key]);rows.append({'filename':r.mat_file,'band':key,'preprocess_seconds':time.perf_counter()-t})
        print('bandpass processed',arc,round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(OUT/'bandpass_order_cache.npz',**feats)
    pd.DataFrame(rows).to_csv(OUT/'bandpass_preprocess_timing.csv',index=False)
    return feats

def main():
    c=OUT/'bandpass_order_cache.npz';data={k:np.load(c)[k] for k in BANDS} if c.exists() else build()
    allr=[];allp=[];allsel=[]
    for key in BANDS:
        r,p,s=select_and_evaluate(data[key],key)
        allr.append(r.rename(columns={'window':'band'}));allp.append(p.rename(columns={'window':'band'}));allsel.append(s.rename(columns={'window':'band'}))
        pd.concat(allr).to_csv(OUT/'bandpass_tasks.csv',index=False);pd.concat(allp).to_csv(OUT/'bandpass_predictions.csv',index=False);pd.concat(allsel).to_csv(OUT/'bandpass_source_selection.csv',index=False)
    print('SUMMARY\n',pd.concat(allr).groupby('band')[['target_ba','target_f1']].mean().to_string(),flush=True)
    print('DONE',flush=True)
if __name__=='__main__':main()
