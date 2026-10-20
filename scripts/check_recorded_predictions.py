#!/usr/bin/env python3
"""Check saved DG predictions against saved per-task scores (no raw data required).

This checks arithmetic consistency; it is not a fresh training/feature extraction run.
"""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'canonical_v1'/'results')
a=parser.parse_args()
results=a.results
methods=('absolute','order','traditional23','pfn','snhop')
all_ok=True
for method in methods:
    task_file=results/f'{method}_tasks.csv'
    pred_file=results/f'{method}_predictions.csv'
    if not task_file.exists() or not pred_file.exists():
        print(f'FAIL {method}: missing tasks or predictions',file=sys.stderr);all_ok=False;continue
    tasks=pd.read_csv(task_file)
    predictions=pd.read_csv(pred_file)
    ok=(len(tasks)==20 and len(predictions)==2400)
    n_verified=0
    for _,t in tasks.iterrows():
        task_preds=predictions[(predictions.direction==t.direction)&(predictions.test_groups==t.test_groups)]
        if len(task_preds)!=120:
            ok=False; continue
        ba=balanced_accuracy_score(task_preds.y_true,task_preds.y_pred)
        f1=f1_score(task_preds.y_true,task_preds.y_pred,average='macro',zero_division=0)
        ok &= bool(np.isclose(ba,t.target_ba,atol=1e-12))
        ok &= bool(np.isclose(f1,t.target_f1,atol=1e-12))
        n_verified+=1
    all_ok &= bool(ok)
    print(f'{method:14s}: {"PASS" if ok else "FAIL"}  {n_verified}/20 matched tasks; {len(predictions)} predictions')
print('PASS: saved prediction/task consistency' if all_ok else 'FAIL: see above')
sys.exit(0 if all_ok else 1)
