"""Exploratory shaft-order sideband diagnostics; no revised classifier or accuracy claim."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
import numpy as np,pandas as pd
BASE=Path(os.environ.get('SNHOP_V1_ROOT',str(Path(__file__).resolve().parents[3]/'canonical_v1')))
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(BASE/'src'))
from core import ORDER_GRID,BPFO,BPFI,norm_rows
meta=pd.read_csv(BASE/'results/recordings.csv');X=norm_rows(np.load(BASE/'results/feature_cache.npz')['order'],'snv');print('grid spacing:', ORDER_GRID[1]-ORDER_GRID[0])
rows=[]
for i,r in meta.iterrows():
    for kind,first in [('BPFO',BPFO),('BPFI',BPFI)]:
        for h in [1,2,3]:
            c=h*first
            locs=[c-1,c,c+1]
            vals=[]
            for o in locs:
                mask=np.abs(ORDER_GRID-o)<=0.22
                assert np.any(mask)
                vals.append(float(X[i,mask].max()))
            rows.append(dict(bearing=r.bearing,label=r.label,rpm=r.rpm,filename=r.filename,type=kind,harmonic=h,minus1=vals[0],central=vals[1],plus1=vals[2],sideband_mean_minus_center=(vals[0]+vals[2])*0.5-vals[1]))
d=pd.DataFrame(rows);d.to_csv(OUT/'exploratory_shaft_order_sidebands_recordings.csv',index=False)
b=d.groupby(['bearing','label','rpm','type','harmonic']).agg(mean_sideband_minus_center=('sideband_mean_minus_center','mean'),sd_sideband_minus_center=('sideband_mean_minus_center','std')).reset_index()
b.to_csv(OUT/'exploratory_shaft_order_sidebands_bearing.csv',index=False)
print(b.groupby(['label','type','harmonic']).mean_sideband_minus_center.agg(['mean','std']).to_string())
print('CAVEAT: spectra bin width 0.196 order; nearby harmonics overlap; this is descriptive, NOT physical validated defect-sideband identification or prediction improvement.')
