from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
ap=argparse.ArgumentParser()
ap.add_argument('--raw',type=Path,required=True,help='Directory containing the 15 Paderborn ZIP archives listed in evidence/SNHOP_raw_archive_checksums.csv')
a=ap.parse_args()
steps=[
    [sys.executable,str(SRC/'preprocess.py'),'--raw',str(a.raw)],
    [sys.executable,str(SRC/'run_core.py')],
    [sys.executable,str(SRC/'run_ablation.py')],
    [sys.executable,str(SRC/'run_dependent.py'),'--raw',str(a.raw),'--rebuild-noise-cache'],
    [sys.executable,str(SRC/'verify_and_report.py')],
    [sys.executable,str(SRC/'make_figures.py')],
]
for cmd in steps:
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
