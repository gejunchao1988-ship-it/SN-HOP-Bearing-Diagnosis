#!/usr/bin/env python3
"""Full per-recording inference timing under fixed pre-fitted SN-HOP model.
Includes archive read+MAT decoding, 31-window 8192 FFT/filter/Hilbert,
SN-HOP feature transform and model predict. Hardware-specific; not training.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,zipfile,time,json,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import butter,sosfiltfilt,hilbert
ROOT=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')));RES=ROOT/'results';OUT=Path(__file__).resolve().parents[1]/'results';sys.path.insert(0,str(ROOT/'src'))
from core import _load_record,ORDER_GRID,hop,clf_model,masks
meta=pd.read_csv(RES/'recordings.csv');cache=np.load(RES/'feature_cache.npz');sel=pd.read_csv(RES/'snhop_selected.csv');source=pd.read_csv(ROOT/'evidence/paderborn_600_file_manifest.csv');rec_idx={f:i for i,f in enumerate(meta.filename)}
r=sel.iloc[0];tr,te,*_=masks(meta,r.direction,r.test_groups)
P=hop(cache['order'],r.halfwidth,r.max_order);model=clf_model(r);y=meta.label.to_numpy()
t0=time.perf_counter();model.fit(P[tr],y[tr]);fit_time=time.perf_counter()-t0
sos=butter(4,[500,10000],btype='bandpass',fs=64000,output='sos');ff=np.fft.rfftfreq(8192,d=1/64000)
def make_spectrum(x,rpm):
    s=x[:253952].reshape(31,8192).copy();s-=s.mean(1,keepdims=True)
    e=np.abs(hilbert(sosfiltfilt(sos,s,axis=1),axis=1));e-=e.mean(1,keepdims=True)
    a=np.log1p(np.abs(np.fft.rfft(e,axis=1)))
    return np.asarray([np.interp(ORDER_GRID,ff/(rpm/60),q) for q in a],np.float32).mean(0)

rows=[];errors=[]
for arc,sub in source.groupby('archive',sort=False):
    with zipfile.ZipFile(Path(os.environ.get('SNHOP_RAW_DIR',str(Path(__file__).resolve().parents[3]/'raw')))/arc) as z:
        names={Path(n).name:n for n in z.namelist() if n.lower().endswith('.mat')}
        for item in sub.itertuples():
            idx=rec_idx[item.mat_file]
            a=time.perf_counter();x,rpm=_load_record(z,names[item.mat_file]);b=time.perf_counter()
            order=make_spectrum(x,rpm);c=time.perf_counter()
            features=hop(order[None,:],r.halfwidth,r.max_order);d=time.perf_counter()
            pred=model.predict(features);e=time.perf_counter()
            err=float(np.max(np.abs(order-cache['order'][idx])));errors.append(err)
            rows.append({'filename':item.mat_file,'bearing':item.bearing,'read_decode_ms':(b-a)*1000,'filter_envelope_fft_ms':(c-b)*1000,'pooling_ms':(d-c)*1000,'classifier_ms':(e-d)*1000,'full_inference_ms':(e-a)*1000,'max_abs_cache_difference':err,'example_model_prediction':int(pred[0])})
    print('timed',arc,len(rows),flush=True)
res=pd.DataFrame(rows);res.to_csv(OUT/'end_to_end_timing_600.csv',index=False)
summary={'recordings':len(res),'trained_example_task':{'direction':r.direction,'held_out_groups':r.test_groups},'fit_seconds':fit_time,'timing_ms':{key:{'median':float(res[key].median()),'mean':float(res[key].mean()),'p95':float(res[key].quantile(.95))} for key in ['read_decode_ms','filter_envelope_fft_ms','pooling_ms','classifier_ms','full_inference_ms']},'max_abs_reprocessing_difference':max(errors),'all_cache_matches':max(errors)<2e-6,'timing_note':'Measured on hosted CPU with locally cached compressed ZIPs; excludes raw sensor acquisition, disk/network transfer, and model training. Single-thread BLAS. Not representative of field device latency.'}
(OUT/'end_to_end_timing_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2),flush=True)
