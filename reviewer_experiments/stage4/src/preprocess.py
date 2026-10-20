from pathlib import Path
import argparse, json, platform
import numpy as np, scipy, sklearn, pandas as pd
from core import preprocess_all, ROOT, RES, sha256
ap=argparse.ArgumentParser(); ap.add_argument('--raw',type=Path,default=Path(str(Path(__file__).resolve().parents[3]/'raw'))); a=ap.parse_args()
preprocess_all(a.raw)
(RES/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'preprocess_sha256':sha256(__file__)},indent=2))
print('done')
