#!/usr/bin/env python3
"""SN-HOP revision Stage 3 — non-white, synthetic disturbance challenge.
Frozen clean-source classifier selections from canonical v1. No target label, noisy
sample or original test data is used for training/tuning. Synthetic disturbances
are NOT claimed to represent recorded industrial field measurements.

Conditions: band-limited colored noise (1200–6500 Hz, Butterworth),
randomly timed damped 3 kHz impacts; test input SNRs = 10 and 0 dB.
Two deterministic independent noise seeds per target recording, each
created from stable experimental constants + canonical index + type + replicate.
Physical shafts retain actual recorded measured rpm for order interpolation.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys, time, zipfile, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter,sosfiltfilt,hilbert
from sklearn.metrics import balanced_accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
V1=Path(os.getenv('SNHOP_V1_ROOT', str(Path(__file__).resolve().parents[3]/'canonical_v1')))
RAW=Path(os.getenv('SNHOP_RAW_DIR',str(Path(__file__).resolve().parents[3]/'raw')))
sys.path.insert(0,str(V1/'src'))
from core import _load_record, ORDER_GRID, hop, clf_model, task_defs, masks, sha256

RES=ROOT/'results'; RES.mkdir(parents=True,exist_ok=True)
meta=pd.read_csv(V1/'results'/'recordings.csv')
manifest=pd.read_csv(V1/'evidence'/'paderborn_600_file_manifest.csv')
y=meta.label.to_numpy(); names={s:i for i,s in enumerate(meta.filename)}
selected=pd.read_csv(V1/'results'/'snhop_selected.csv')
cache=np.load(V1/'results'/'feature_cache.npz')
ORDER=cache['order']
FS=64000;L=8192;N=31;TOTAL=L*N
FF=np.fft.rfftfreq(L,d=1/FS)
PRE=sosfiltfilt
SOS=butter(4,[500,10000],btype='bandpass',fs=FS,output='sos')
COLOR_SOS=butter(3,[1200,6500],btype='bandpass',fs=FS,output='sos')
KINDS=['band_limited_colored','random_damped_impacts']
SNRS=[10,0];REPS=[0,1]
BASE_SEED=20261019


def _normalise(q):
    q=np.asarray(q,dtype=np.float64); q-=q.mean()
    power=np.mean(q*q)
    if not np.isfinite(power) or power<1e-20: raise RuntimeError('Zero or invalid noise power')
    return q/np.sqrt(power)


def make_disturbance(kind,n,idx,rep):
    rng=np.random.default_rng(np.random.SeedSequence([BASE_SEED,int(idx),int(rep),KINDS.index(kind)]))
    if kind=='band_limited_colored':
        return _normalise(sosfiltfilt(COLOR_SOS,rng.standard_normal(n)))
    if kind=='random_damped_impacts':
        q=np.zeros(n,dtype=np.float64)
        # Approx. 13 events / s, nonuniform interarrival times and amplitudes.
        n_imp=rng.poisson(13*n/FS)
        tt=np.arange(512)/FS
        for k in range(n_imp):
            st=int(rng.integers(0,n))
            freq=float(rng.uniform(2600,4600))
            decay=float(rng.uniform(.00075,.0025))
            # Damped resonance excited by disturbances; not a simulated bearing label.
            pulse=np.exp(-tt/decay)*np.sin(2*np.pi*freq*tt)
            amp=float(rng.lognormal(mean=0,sigma=.45))*(1 if rng.random()<.5 else -1)
            ll=min(len(pulse),n-st)
            q[st:st+ll]+=amp*pulse[:ll]
        return _normalise(q)
    raise ValueError(kind)


def transform(x,rpm):
    z=x[:TOTAL].reshape(N,L).copy()
    z-=z.mean(axis=1,keepdims=True)
    filt=sosfiltfilt(SOS,z,axis=1)
    env=np.abs(hilbert(filt,axis=1)); env-=env.mean(axis=1,keepdims=True)
    sp=np.log1p(np.abs(np.fft.rfft(env,axis=1)))
    coord=FF/(rpm/60.)
    return np.asarray([np.interp(ORDER_GRID,coord,s) for s in sp],np.float32).mean(0)


def ensure_raw_integrity():
    checks=pd.read_csv(V1/'evidence'/'SNHOP_raw_archive_checksums.csv'); audit=[]
    for r in checks.itertuples():
        p=RAW/r.archive
        if not p.is_file():raise FileNotFoundError(p)
        h=sha256(p)
        if h!=r.sha256:raise RuntimeError('SHA-256 mismatch '+str(p))
        audit.append({'archive':r.archive,'actual_bytes':p.stat().st_size,'sha256':h,'verified':True})
    pd.DataFrame(audit).to_csv(ROOT/'evidence'/'raw15_integrity_stage3.csv',index=False)


def make_cache():
    assert len(meta)==600 and len(manifest)==600 and len(selected)==20
    B={kind:np.empty((len(REPS),len(SNRS),len(meta),256),dtype=np.float32) for kind in KINDS}
    energy=[]; t0=time.monotonic();clean_max=0
    for arc,group in manifest.groupby('archive',sort=False):
        with zipfile.ZipFile(RAW/arc) as z:
            mat_names={Path(m).name:m for m in z.namelist() if m.endswith('.mat')}
            for r in group.itertuples():
                idx=names[r.mat_file]
                x,rpm=_load_record(z,mat_names[r.mat_file]);x=x[:TOTAL]
                if not np.isclose(rpm,meta.measured_rpm[idx],atol=1e-6,rtol=0):
                    raise RuntimeError('RPM mismatch '+r.mat_file)
                original=transform(x,rpm)
                diff=np.max(np.abs(original-ORDER[idx]))
                clean_max=max(clean_max,float(diff))
                if diff>2e-6:raise RuntimeError('Clean preprocessing mismatch '+r.mat_file+' diff='+str(diff))
                pwr=np.mean((x-x.mean())**2)
                for ik,kind in enumerate(KINDS):
                    for ir,rep in enumerate(REPS):
                        disturbance=make_disturbance(kind,len(x),idx,rep)
                        for js,snr in enumerate(SNRS):
                            add=disturbance*np.sqrt(pwr/(10**(snr/10)))
                            achieved=10*np.log10(pwr/np.mean(add**2))
                            if abs(achieved-snr)>1e-9:raise RuntimeError('SNR not achieved')
                            B[kind][ir,js,idx]=transform(x+add,rpm)
                            energy.append({'filename':r.mat_file,'bearing':r.bearing,'kind':kind,'replicate':rep,'snr_db':snr,'achieved_snr_db':achieved})
        print('processed',arc,'elapsed_s',round(time.monotonic()-t0,2),flush=True)
    np.savez_compressed(RES/'nonwhite_order_cache.npz',**B,snrs_db=np.array(SNRS),replicates=np.array(REPS))
    pd.DataFrame(energy).to_csv(RES/'nonwhite_snr_audit.csv',index=False)
    (RES/'preprocess_reconstruction.json').write_text(json.dumps({'original_feature_max_abs_error':clean_max,'n_raw_records':len(meta),'noise_processes':KINDS,'snrs_db':SNRS,'replicates':REPS,'seed':BASE_SEED},indent=2),encoding='utf-8')
    return B


def evaluate(B):
    cleanp=pd.read_csv(V1/'results'/'snhop_predictions.csv')
    results=[]; predictions=[]; replay=[]
    for r in selected.to_dict('records'):
        direction=r['direction'];held=r['test_groups'];tr,te,*_=masks(meta,direction,held)
        cfg=dict(r); w=float(r['halfwidth']);mo=float(r['max_order'])
        base=hop(ORDER,w,mo);model=clf_model(cfg);model.fit(base[tr],y[tr])
        predicted=model.predict(base[te]);test_idx=np.flatnonzero(te)
        expected=cleanp[(cleanp.direction==direction)&(cleanp.test_groups==held)].set_index('filename')['y_pred'].to_dict()
        if len(expected)!=len(test_idx):raise RuntimeError('Canonical reference prediction count mismatch')
        if any(int(p)!=int(expected[meta.filename[i]]) for i,p in zip(test_idx,predicted)):
            raise RuntimeError('Clean canonical predictions could not be replayed: '+direction+' '+held)
        results.append({'kind':'clean','snr_db':None,'replicate':-1,'direction':direction,'test_groups':held,'target_ba':balanced_accuracy_score(y[te],predicted),'target_f1':f1_score(y[te],predicted,average='macro',zero_division=0)})
        replay.extend([{'direction':direction,'test_groups':held,'filename':meta.filename[i],'y_true':int(y[i]),'y_pred':int(predicted[j])} for j,i in enumerate(test_idx)])
        for kind in KINDS:
            for ir,rep in enumerate(REPS):
                for js,snr in enumerate(SNRS):
                    altered=hop(B[kind][ir,js],w,mo)
                    guess=model.predict(altered[te]);ba=balanced_accuracy_score(y[te],guess)
                    f1=f1_score(y[te],guess,average='macro',zero_division=0)
                    results.append({'kind':kind,'snr_db':snr,'replicate':rep,'direction':direction,'test_groups':held,'target_ba':ba,'target_f1':f1})
                    predictions.extend([{'kind':kind,'snr_db':snr,'replicate':rep,'direction':direction,'test_groups':held,'filename':meta.filename[i],'bearing':meta.bearing[i],'y_true':int(y[i]),'y_pred':int(guess[j])} for j,i in enumerate(test_idx)])
        print('evaluated',direction,held,flush=True)
    pd.DataFrame(results).to_csv(RES/'nonwhite_noise_tasks.csv',index=False)
    pd.DataFrame(predictions).to_csv(RES/'nonwhite_noise_predictions.csv',index=False)
    pd.DataFrame(replay).to_csv(RES/'canonical_clean_replay_predictions.csv',index=False)
    d=pd.DataFrame(results)
    summary=d.groupby(['kind','snr_db'],dropna=False).agg(mean_ba=('target_ba','mean'),std_task_ba=('target_ba','std'),mean_f1=('target_f1','mean'),n_task_appearances=('target_ba','size')).reset_index()
    summary.to_csv(RES/'nonwhite_noise_summary.csv',index=False)
    print('RESULT\n',summary.to_string(index=False),flush=True)
    # Distinct test files and 20 tasks × 2 noise realizations are NOT independent specimens.
    return summary


def main():
    ensure_raw_integrity()
    path=RES/'nonwhite_order_cache.npz'
    if path.exists():
        obj=np.load(path);B={k:obj[k] for k in KINDS}
        assert np.array_equal(obj['snrs_db'],SNRS)
    else:B=make_cache()
    evaluate(B)

if __name__=='__main__':main()
