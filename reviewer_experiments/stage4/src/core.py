import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import io, zipfile, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'; EVD=ROOT/'evidence'
GROUPS=[['K001','KA04','KI04'],['K002','KA15','KI14'],['K003','KA16','KI16'],['K004','KA22','KI18'],['K005','KA30','KI21']]
GM={b:i+1 for i,g in enumerate(GROUPS) for b in g}
BPFO=4*(1-6.75/28.55)
BPFI=4*(1+6.75/28.55)
ORDER_GRID=np.linspace(0,50,256)
ABS_GRID=np.linspace(0,1250,256)
PFN_GRID=np.linspace(0,5,128)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def _load_record(zipf, member):
    obj=loadmat(io.BytesIO(zipf.read(member)),struct_as_record=False,squeeze_me=False)
    key=[k for k in obj if not k.startswith('__')][0]
    rec=obj[key][0,0]
    ch={v.Name.item():np.asarray(v.Data).squeeze() for v in rec.Y.flat}
    return ch['vibration_1'].astype(np.float64), float(np.mean(ch['speed']))


def haar_packet_energy(x, levels=3):
    nodes=[x]
    inv=np.sqrt(0.5)
    for _ in range(levels):
        nxt=[]
        for a in nodes:
            if len(a)%2: a=a[:-1]
            ev=a[0::2]; od=a[1::2]
            nxt += [(ev+od)*inv, (ev-od)*inv]
        nodes=nxt
    e=np.array([np.mean(n*n) for n in nodes],float)
    return e/(e.sum()+1e-12)


def traditional23(x,fs=64000.0):
    x=x.astype(np.float64)
    mean=float(np.mean(x)); xc=x-mean
    ab=np.abs(xc); meanabs=float(ab.mean()); rms=float(np.sqrt(np.mean(xc*xc))); sd=float(xc.std())
    peak=float(ab.max()); p2p=float(xc.max()-xc.min())
    sk=float(skew(xc,bias=False)); ku=float(kurtosis(xc,fisher=False,bias=False))
    crest=peak/(rms+1e-12); impulse=peak/(meanabs+1e-12); shape=rms/(meanabs+1e-12)
    clearance=peak/((np.mean(np.sqrt(ab))+1e-12)**2); energy=float(np.mean(xc*xc))
    timef=[mean,sd,rms,peak,p2p,sk,ku,crest,impulse,shape,clearance,energy]
    sp=np.abs(np.fft.rfft(xc))**2
    f=np.fft.rfftfreq(len(xc),1/fs); den=sp.sum()+1e-12
    centroid=float((f*sp).sum()/den); rmsf=float(np.sqrt(((f*f)*sp).sum()/den)); spread=float(np.sqrt((((f-centroid)**2)*sp).sum()/den))
    wp=haar_packet_energy(xc,3).tolist()
    return np.asarray(timef+[centroid,rmsf,spread]+wp,dtype=np.float64)


def preprocess_all(raw_dir: Path):
    raw_dir=Path(raw_dir); RES.mkdir(exist_ok=True)
    manifest=pd.read_csv(EVD/'paderborn_600_file_manifest.csv')
    checks=pd.read_csv(EVD/'SNHOP_raw_archive_checksums.csv')
    audit=[]
    for r in checks.itertuples():
        p=raw_dir/r.archive; digest=sha256(p)
        if digest!=r.sha256: raise RuntimeError(f'SHA mismatch {p}')
        audit.append(dict(archive=r.archive,size=p.stat().st_size,sha256=digest,match=True))
    pd.DataFrame(audit).to_csv(RES/'archive_verification.csv',index=False)
    sos=butter(4,[500,10000],btype='bandpass',fs=64000,output='sos')
    f8192=np.fft.rfftfreq(8192,1/64000)
    f65536=np.fft.rfftfreq(65536,1/64000)
    hann=np.hanning(8192)
    order=[]; absolute=[]; pfn=[]; trad=[]; rows=[]
    for arc,sub in manifest.groupby('archive',sort=False):
        with zipfile.ZipFile(raw_dir/arc) as z:
            names={Path(n).name:n for n in z.namelist() if n.lower().endswith('.mat')}
            for r in sub.itertuples():
                x,rpm=_load_record(z,names[r.mat_file])
                seg=x[:8192*31].reshape(31,8192).copy(); seg-=seg.mean(axis=1,keepdims=True)
                filt=sosfiltfilt(sos,seg,axis=1)
                env=np.abs(hilbert(filt,axis=1)); env-=env.mean(axis=1,keepdims=True)
                esp=np.log1p(np.abs(np.fft.rfft(env,axis=1)))
                fr=rpm/60.0
                o=np.asarray([np.interp(ORDER_GRID,f8192/fr,s) for s in esp],dtype=np.float32).mean(0)
                a=np.asarray([np.interp(ABS_GRID,f8192,s) for s in esp],dtype=np.float32).mean(0)
                # PFN-core: high-resolution Hann spectrum normalized by BPFI frequency.
                hsp=np.log1p(np.abs(np.fft.rfft(seg*hann,n=65536,axis=1)))
                coord=f65536/(BPFI*fr)
                pp=np.asarray([np.interp(PFN_GRID,coord,s) for s in hsp],dtype=np.float32).mean(0)
                tr=traditional23(x[:8192*31])
                order.append(o); absolute.append(a); pfn.append(pp); trad.append(tr)
                rows.append(dict(filename=r.mat_file,bearing=r.bearing,label=int(r.label),group=GM[r.bearing],rpm=int(r.nominal_rpm),measured_rpm=rpm,run=int(r.run),samples=len(x)))
        print('processed',arc,'n=',len(rows),flush=True)
    np.savez_compressed(RES/'feature_cache.npz',order=np.asarray(order,np.float32),absolute=np.asarray(absolute,np.float32),pfn=np.asarray(pfn,np.float32),traditional=np.asarray(trad,np.float64))
    pd.DataFrame(rows).to_csv(RES/'recordings.csv',index=False)


def norm_rows(X,kind):
    X=np.asarray(X)
    if kind=='raw': return X
    if kind=='l2': return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
    if kind=='snv': return (X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-12)
    raise ValueError(kind)


def hop(X,width,max_order,centers='physical',components='full'):
    X=norm_rows(X,'snv'); g=ORDER_GRID; out=[]
    phys=[]
    for base in [BPFO,BPFI]:
        phys += list(np.arange(base,max_order+1e-10,base))
    if centers=='uniform':
        cs=np.linspace(max_order/(len(phys)+1),max_order*len(phys)/(len(phys)+1),len(phys)) if phys else []
    else: cs=phys
    include_peak=components in ('full','peak_context','harmonics_only')
    include_contrast=components in ('full','contrast_context','harmonics_only')
    include_context=components in ('full','peak_context','contrast_context','context_only')
    if components!='context_only':
        for c in cs:
            cen=np.abs(g-c)<=width; bg=(np.abs(g-c)>width)&(np.abs(g-c)<=1.2)
            if not cen.any() or not bg.any(): continue
            if include_peak: out.append(X[:,cen].max(1))
            if include_contrast: out.append(X[:,cen].mean(1)-X[:,bg].mean(1))
    if include_context:
        for a,b in [(0,5),(5,10),(10,20),(20,30),(30,40),(40,50)]:
            mask=(g>=a)&((g<b) if b<50 else (g<=b)); out += [X[:,mask].mean(1),X[:,mask].std(1)]
    return np.stack(out,axis=1)


def clf_model(cfg):
    if cfg['classifier']=='logreg':
        e=LogisticRegression(C=float(cfg['C']),max_iter=4000,solver='lbfgs',random_state=42)
    else:
        ga=cfg.get('gamma','scale'); ga=ga if ga=='scale' else float(ga)
        e=SVC(C=float(cfg['C']),gamma=ga,kernel='rbf',decision_function_shape='ovr')
    return make_pipeline(StandardScaler(),e)


def classifier_candidates(compact=False):
    if compact:
        lrc=[.01,.1,1,10]; svmc=[.1,1,10]; gam=['scale',.01]
    else:
        lrc=[.01,.1,1,10,100]; svmc=[.1,1,10,100]; gam=['scale',.01,.1,1.0]
    for C in lrc: yield dict(classifier='logreg',C=C,gamma=None)
    for C in svmc:
        for ga in gam: yield dict(classifier='svm',C=C,gamma=ga)


def feature_dim(cfg,rep):
    return rep.shape[1]


def tie_key(cfg,dim):
    # Explicit reproducibility rule: on mean-CV ties, prefer the lower-capacity
    # linear classifier, then fewer features, lower regularization parameter,
    # then deterministic gamma/geometry order. Uses source-side information only.
    ga=cfg.get('gamma')
    num_ga=(ga not in [None,'scale']) and pd.notna(ga)
    return (0 if cfg['classifier']=='logreg' else 1,
            int(dim),float(cfg['C']),0 if num_ga else 1,float(ga) if num_ga else 0,
            float(cfg.get('halfwidth',0)),float(cfg.get('max_order',0)),str(cfg.get('norm','')))


def select_candidates(candidates,tol=1e-12):
    best=max(c['mean_cv'] for c in candidates)
    tied=[c for c in candidates if c['mean_cv']>=best-tol]
    return min(tied,key=lambda c:tie_key(c['cfg'],c['dim'])), len(tied)


def task_defs():
    import itertools
    held=['-'.join(map(str,c)) for c in itertools.combinations(range(1,6),2)]
    return [(d,h) for d in ['1500_to_900','900_to_1500'] for h in held]


def masks(meta,direction,held):
    src,tgt=map(int,direction.split('_to_')); hg=list(map(int,held.split('-'))); sg=[g for g in range(1,6) if g not in hg]
    groups=meta.group.to_numpy(); rpms=meta.rpm.to_numpy()
    tr=np.isin(groups,sg)&(rpms==src); te=np.isin(groups,hg)&(rpms==tgt)
    return tr,te,sg,src,tgt,hg


def fisher_rank(X,y):
    classes=np.unique(y); mu=X.mean(0); num=np.zeros(X.shape[1]); den=np.zeros(X.shape[1])
    for c in classes:
        z=X[y==c]; num += len(z)*(z.mean(0)-mu)**2; den += ((z-z.mean(0))**2).sum(0)
    return np.argsort(-(num/(den+1e-12)))


def metrics(y,yp):
    return balanced_accuracy_score(y,yp), f1_score(y,yp,average='macro',zero_division=0)
