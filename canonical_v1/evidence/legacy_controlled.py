\
"""
Controlled PU speed-shift representation experiment.

Primary controlled pair:
    N15_M07_F10 (1500 rpm, M07, F10)
 -> N09_M07_F10 ( 900 rpm, M07, F10)

Data split:
    runs 1-3: source train / target adaptation pool
    runs 4-5: source validation / target test

The script compares:
    1) absolute-frequency envelope spectrum
    2) speed-normalized envelope order spectrum

No target labels are used for fitting or model selection.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

CODES = ["K001","KA01","KB23","KI04"]
MAP4 = {c:i for i,c in enumerate(CODES)}
MAP3 = {"K001":0, "KA01":1, "KB23":2, "KI04":2}

def read_pu(path: Path):
    d=loadmat(path,squeeze_me=False,struct_as_record=False)
    obj=d[path.stem][0,0]
    ch={y.Name.item():np.asarray(y.Data).squeeze() for y in obj.Y.flat}
    return ch["vibration_1"].astype(np.float64), float(np.mean(ch["speed"]))

def extract_pair(x,rpm,fs=64000.0,window=8192,stride=8192,bins=256,max_order=50.0,abs_max_hz=750.0):
    sos=butter(4,[500,10000],btype="bandpass",fs=fs,output="sos")
    f=np.fft.rfftfreq(window,1/fs)
    fg=np.linspace(0,abs_max_hz,bins)
    og=np.linspace(0,max_order,bins)
    A=[]; O=[]
    for s in np.arange(0,len(x)-window+1,stride):
        w=x[s:s+window].copy()
        w-=w.mean()
        env=np.abs(hilbert(sosfiltfilt(sos,w)))
        env-=env.mean()
        sp=np.log1p(np.abs(np.fft.rfft(env)))
        A.append(np.interp(fg,f,sp))
        O.append(np.interp(og,f/(rpm/60.0),sp))
    return np.asarray(A,np.float32), np.asarray(O,np.float32)

def build(root: Path, cond: str, taxonomy: str):
    lab=MAP4 if taxonomy=="4state" else MAP3
    out={}
    for split,runs in [("train",(1,2,3)),("eval",(4,5))]:
        aa=[]; oo=[]; yy=[]
        for c in CODES:
            for run in runs:
                p=root/f"{cond}_{c}_{run}.mat"
                x,rpm=read_pu(p)
                A,O=extract_pair(x,rpm)
                aa.append(A); oo.append(O); yy.extend([lab[c]]*len(A))
        out[split]=(np.concatenate(aa),np.concatenate(oo),np.asarray(yy,np.int64))
    return out

class MLP(nn.Module):
    def __init__(self,n_cls):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(.2),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),
            nn.Linear(64,n_cls))
    def forward(self,x): return self.net(x)

def run_one(source,target,rep,seed,n_cls):
    np.random.seed(seed); torch.manual_seed(seed)
    A,O,y=source["train"]; Av,Ov,yv=source["eval"]; At,Ot,yt=target["eval"]
    X=A if rep=="absolute" else O
    Xv=Av if rep=="absolute" else Ov
    Xt=At if rep=="absolute" else Ot
    sc=StandardScaler().fit(X)
    X=sc.transform(X).astype("float32")
    Xv=sc.transform(Xv).astype("float32")
    Xt=sc.transform(Xt).astype("float32")

    classes=np.arange(n_cls)
    cw=compute_class_weight(class_weight="balanced",classes=classes,y=y)
    ce=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float32))
    g=torch.Generator().manual_seed(seed)
    dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),
                  batch_size=64,shuffle=True,generator=g)
    m=MLP(n_cls); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    best=None; best_f1=-1; bad=0
    for _ in range(200):
        m.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True)
            loss=ce(m(xb),yb); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv=m(torch.from_numpy(Xv)).argmax(1).numpy()
        vf1=f1_score(yv,pv,average="macro")
        if vf1>best_f1+1e-6:
            best_f1=vf1; bad=0
            best={k:v.detach().clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=20: break
    m.load_state_dict(best); m.eval()
    with torch.no_grad(): pt=m(torch.from_numpy(Xt)).argmax(1).numpy()
    return {
        "source_val_macro_f1":float(best_f1),
        "target_accuracy":float(accuracy_score(yt,pt)),
        "target_balanced_accuracy":float(balanced_accuracy_score(yt,pt)),
        "target_macro_f1":float(f1_score(yt,pt,average="macro")),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n15-root",required=True)
    ap.add_argument("--n09-root",required=True)
    ap.add_argument("--taxonomy",choices=["4state","3class"],default="3class")
    ap.add_argument("--out",default="results/pu_controlled_representation.csv")
    args=ap.parse_args()
    D15=build(Path(args.n15_root),"N15_M07_F10",args.taxonomy)
    D09=build(Path(args.n09_root),"N09_M07_F10",args.taxonomy)
    n_cls=4 if args.taxonomy=="4state" else 3
    rows=[]
    for direction,S,T in [("N15F10_to_N09F10",D15,D09),("N09F10_to_N15F10",D09,D15)]:
        for rep in ["absolute","order"]:
            for seed in [42,43,44,45,46]:
                rows.append({"direction":direction,"representation":rep,"seed":seed,
                             **run_one(S,T,rep,seed,n_cls)})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).groupby(["direction","representation"])[
        ["target_accuracy","target_balanced_accuracy","target_macro_f1"]].agg(["mean","std"]))
if __name__=="__main__":
    main()
