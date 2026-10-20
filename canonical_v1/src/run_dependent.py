import json, shutil, argparse, io, zipfile
import numpy as np, pandas as pd
from pathlib import Path
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert
from core import *
cache=np.load(RES/'feature_cache.npz'); meta=pd.read_csv(RES/'recordings.csv');y=meta.label.to_numpy();groups=meta.group.to_numpy();rpms=meta.rpm.to_numpy();runs=meta.run.to_numpy()


def bare_estimator(cfg):
    return clf_model(cfg).steps[-1][1]

def covpow(C,p):
    v,u=np.linalg.eigh(C); return (u*np.maximum(v,1e-12)**p)@u.T

def coral_align(S,A,lam=1e-3):
    Cs=np.cov(S,rowvar=False)+lam*np.eye(S.shape[1]); Ct=np.cov(A,rowvar=False)+lam*np.eye(S.shape[1])
    return (S-S.mean(0))@covpow(Cs,-.5)@covpow(Ct,.5)+A.mean(0)

def representation(method,cfg,X):
    if method=='snhop':return hop(X,float(cfg['halfwidth']),float(cfg['max_order']))
    return norm_rows(X,str(cfg['norm']))

def uda_one(method,X,selected):
    rows=[];pred=[];cms=[]
    for r in selected.to_dict('records'):
        direction=r['direction'];held=r['test_groups'];tr,tt,sg,src,tgt,hg=masks(meta,direction,held);F=representation(method,r,X);sc=StandardScaler().fit(F[tr]);S=sc.transform(F[tr]);T=sc.transform(F);ad=tt&(runs<=10);te=tt&(runs>=11)
        base=bare_estimator(r);base.fit(S,y[tr]);yp0=base.predict(T[te])
        for label,yp in [('NoDA',yp0)]:
            ba,f1=metrics(y[te],yp);rows.append(dict(method=method,kind='uda',variant=label,k=10,direction=direction,test_groups=held,target_ba=ba,target_f1=f1))
            for idx,v in zip(np.flatnonzero(te),yp):pred.append(dict(method=method,kind='uda',variant=label,k=10,direction=direction,test_groups=held,filename=meta.filename[idx],bearing=meta.bearing[idx],y_true=int(y[idx]),y_pred=int(v)))
        alg=coral_align(S,T[ad]); clf=bare_estimator(r);clf.fit(alg,y[tr]);yp=clf.predict(T[te]);ba,f1=metrics(y[te],yp);rows.append(dict(method=method,kind='uda',variant='CORAL',k=10,direction=direction,test_groups=held,target_ba=ba,target_f1=f1));cms.append(confusion_matrix(y[te],yp,labels=[0,1,2],normalize='true'))
        for idx,v in zip(np.flatnonzero(te),yp):pred.append(dict(method=method,kind='uda',variant='CORAL',k=10,direction=direction,test_groups=held,filename=meta.filename[idx],bearing=meta.bearing[idx],y_true=int(y[idx]),y_pred=int(v)))
        # target standardization comparator
        ts=StandardScaler().fit(F[ad]); yp=base.predict(ts.transform(F[te]));ba,f1=metrics(y[te],yp);rows.append(dict(method=method,kind='uda',variant='target_standardization',k=10,direction=direction,test_groups=held,target_ba=ba,target_f1=f1))
        for idx,v in zip(np.flatnonzero(te),yp):pred.append(dict(method=method,kind='uda',variant='target_standardization',k=10,direction=direction,test_groups=held,filename=meta.filename[idx],bearing=meta.bearing[idx],y_true=int(y[idx]),y_pred=int(v)))
        # adaptation-size fixed test runs16-20
        te2=tt&(runs>=16)
        yp=base.predict(T[te2]);ba,f1=metrics(y[te2],yp);rows.append(dict(method=method,kind='adaptation_size',variant='NoDA',k=0,direction=direction,test_groups=held,target_ba=ba,target_f1=f1))
        for k in [1,3,5,10,15]:
            ad2=tt&(runs<=k);clf=bare_estimator(r);clf.fit(coral_align(S,T[ad2]),y[tr]);yp=clf.predict(T[te2]);ba,f1=metrics(y[te2],yp);rows.append(dict(method=method,kind='adaptation_size',variant='CORAL',k=k,direction=direction,test_groups=held,target_ba=ba,target_f1=f1))
    return pd.DataFrame(rows),pd.DataFrame(pred),np.mean(cms,axis=0) if cms else None

allr=[];allp=[]
for method in ['absolute','order','pfn','snhop']:
    X=cache['order'] if method=='snhop' else cache[method]
    selected=pd.read_csv(RES/f'{method}_selected.csv') if (RES/f'{method}_selected.csv').exists() else pd.read_csv(RES/f'{method}_tasks.csv')
    r,p,cm=uda_one(method,X,selected);allr.append(r);allp.append(p)
    if cm is not None:np.savetxt(RES/f'{method}_uda_mean_confusion.csv',cm,delimiter=',')
uda=pd.concat(allr,ignore_index=True);pred=pd.concat(allp,ignore_index=True);uda.to_csv(RES/'uda_tasks.csv',index=False);pred.to_csv(RES/'uda_predictions.csv',index=False)
summary=uda.groupby(['method','kind','variant','k'])[['target_ba','target_f1']].mean().mul(100).reset_index();summary.to_csv(RES/'uda_summary.csv',index=False)
print(summary.to_string(index=False))

# Raw-noise experiment. The cache can be regenerated from the 15 verified raw archives.
def build_noise_cache(raw_dir):
    raw_dir=Path(raw_dir)
    manifest=pd.read_csv(EVD/'paderborn_600_file_manifest.csv')
    checks=pd.read_csv(EVD/'SNHOP_raw_archive_checksums.csv')
    for r in checks.itertuples():
        p=raw_dir/r.archive
        if sha256(p)!=r.sha256: raise RuntimeError(f'SHA mismatch {p}')
    snrs=[20,10,6,3,0]
    N=np.zeros((3,5,len(meta),256),np.float32)
    sos=butter(4,[500,10000],btype='bandpass',fs=64000,output='sos')
    ff=np.fft.rfftfreq(8192,1/64000)
    idx_by_name={n:i for i,n in enumerate(meta.filename)}
    for arc,sub in manifest.groupby('archive',sort=False):
        with zipfile.ZipFile(raw_dir/arc) as z:
            names={Path(n).name:n for n in z.namelist() if n.lower().endswith('.mat')}
            for row in sub.itertuples():
                idx=idx_by_name[row.mat_file]
                obj=loadmat(io.BytesIO(z.read(names[row.mat_file])),struct_as_record=False,squeeze_me=False)
                key=[k for k in obj if not k.startswith('__')][0]
                rec=obj[key][0,0]
                ch={v.Name.item():np.asarray(v.Data).squeeze() for v in rec.Y.flat}
                x=ch['vibration_1'].astype(np.float64)[:31*8192]
                speed=float(np.mean(ch['speed'])); power=np.mean((x-x.mean())**2)
                def transform(v):
                    w=v.reshape(31,8192).copy(); w-=w.mean(1,keepdims=True)
                    e=np.abs(hilbert(sosfiltfilt(sos,w,axis=1),axis=1)); e-=e.mean(1,keepdims=True)
                    sp=np.log1p(np.abs(np.fft.rfft(e,axis=1)))
                    return np.asarray([np.interp(ORDER_GRID,ff/(speed/60.0),q) for q in sp],np.float32).mean(0)
                clean=transform(x)
                if not np.allclose(clean,cache['order'][idx],rtol=0,atol=1e-6):
                    raise RuntimeError(f'Clean preprocessing mismatch for {row.mat_file}')
                for rep in range(3):
                    rng=np.random.default_rng(np.random.SeedSequence([20260915,idx,rep]))
                    n=rng.standard_normal(x.shape); n-=n.mean(); n/=np.sqrt(np.mean(n*n))
                    for j,snr in enumerate(snrs):
                        N[rep,j,idx]=transform(x+n*np.sqrt(power/(10**(snr/10))))
        print('noise cache',arc,flush=True)
    np.savez_compressed(RES/'noisy_order_cache.npz',order=N,snrs=np.asarray(snrs))

noise_cache=RES/'noisy_order_cache.npz'
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',type=Path,default=None,help='Directory containing the 15 original Paderborn ZIP archives; needed only to regenerate the raw-noise cache.')
    ap.add_argument('--rebuild-noise-cache',action='store_true')
    args=ap.parse_args()
    if args.rebuild_noise_cache or not noise_cache.exists():
        if args.raw is None: raise SystemExit('Noise cache missing. Supply --raw /path/to/raw_archives.')
        build_noise_cache(args.raw)

if noise_cache.exists():
    nobj=np.load(noise_cache); N=nobj['order']; snrs=list(map(int,nobj['snrs']))
    sel=pd.read_csv(RES/'snhop_selected.csv');X=cache['order'];rows=[];pp=[]
    for r in sel.to_dict('records'):
        tr,tt,sg,src,tgt,hg=masks(meta,r['direction'],r['test_groups']);F=hop(X,float(r['halfwidth']),float(r['max_order']));clf=clf_model(r);clf.fit(F[tr],y[tr]);yp=clf.predict(F[tt]);ba,f1=metrics(y[tt],yp);rows.append(dict(snr='clean',replicate=-1,direction=r['direction'],test_groups=r['test_groups'],target_ba=ba,target_f1=f1))
        for rep in range(N.shape[0]):
            for j,snr in enumerate(snrs):
                Z=hop(N[rep,j],float(r['halfwidth']),float(r['max_order']));yp=clf.predict(Z[tt]);ba,f1=metrics(y[tt],yp);rows.append(dict(snr=str(snr),replicate=rep,direction=r['direction'],test_groups=r['test_groups'],target_ba=ba,target_f1=f1))
                for idx,v in zip(np.flatnonzero(tt),yp):pp.append(dict(snr=str(snr),replicate=rep,direction=r['direction'],test_groups=r['test_groups'],filename=meta.filename[idx],y_true=int(y[idx]),y_pred=int(v)))
    nd=pd.DataFrame(rows);nd.to_csv(RES/'noise_tasks.csv',index=False);pd.DataFrame(pp).to_csv(RES/'noise_predictions.csv',index=False)
    ns=nd.groupby('snr')[['target_ba','target_f1']].agg(['mean','std'])*100;ns.to_csv(RES/'noise_summary.csv');print(ns)
