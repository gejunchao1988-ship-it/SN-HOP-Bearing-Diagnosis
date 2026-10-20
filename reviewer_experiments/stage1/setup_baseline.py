from pathlib import Path
from zipfile import ZipFile
root=Path(__file__).resolve().parent
z=root/'baseline_v1'/'SNHOP_canonical_reproducibility_v1.zip'
target=root/'baseline_v1'/'SNHOP_canonical_reproducibility_v1'
if not target.is_dir():
    with ZipFile(z) as pkg:pkg.extractall(root/'baseline_v1')
assert (target/'src/core.py').is_file() and (target/'results/feature_cache.npz').is_file()
print('Canonical v1 baseline restored:',target)
