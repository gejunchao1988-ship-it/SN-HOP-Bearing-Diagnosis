#!/usr/bin/env python3
"""Validate saved DG predictions against per-task metrics, without re-running training."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path,
    default=Path(__file__).resolve().parents[1] / "canonical_v1" / "results")
args = parser.parse_args()
all_ok = True
for method in ("absolute", "order", "traditional23", "pfn", "snhop"):
    tasks_path = args.results / f"{method}_tasks.csv"
    predictions_path = args.results / f"{method}_predictions.csv"
    if not tasks_path.exists() or not predictions_path.exists():
        print(f"{method}: MISSING; upload the canonical_v1/results directory")
        all_ok = False
        continue
    tasks = pd.read_csv(tasks_path)
    predictions = pd.read_csv(predictions_path)
    ok = len(tasks) == 20 and len(predictions) == 2400
    n = 0
    for _, task in tasks.iterrows():
        subset = predictions.loc[
            (predictions.direction == task.direction) &
            (predictions.test_groups == task.test_groups)
        ]
        if len(subset) != 120:
            ok = False
            continue
        ba = balanced_accuracy_score(subset.y_true, subset.y_pred)
        f1 = f1_score(subset.y_true, subset.y_pred, average="macro", zero_division=0)
        ok = ok and bool(np.isclose(ba, task.target_ba, atol=1e-12))
        ok = ok and bool(np.isclose(f1, task.target_f1, atol=1e-12))
        n += 1
    all_ok &= ok
    print(f"{method}: {'PASS' if ok else 'FAIL'}; {n}/20 tasks")
sys.exit(0 if all_ok else 1)
