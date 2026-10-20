# SN-HOP canonical reproducibility package

This package accompanies the updated IEEE Access manuscript **Physics-Guided Speed-Normalized Harmonic Order Pooling for Bearing Fault Diagnosis Across Speeds and Unseen Bearing Specimens**.

## What is included

- `src/`: complete executable preprocessing, strict-DG evaluation, ablation, UDA/noise evaluation, verification/reporting, and figure-generation code.
- `results/`: recording manifest produced by preprocessing, feature caches, selected configurations, inner source-validation records, task-level metrics, one-row-per-test-appearance predictions, confusion matrices, UDA/noise outputs, verification records, and timing results.
- `figures/`: regenerated publication figures corresponding to the updated experimental results.
- `evidence/`: the 600-recording file manifest, SHA-256 archive inventory, retained legacy preprocessing source, and source manuscript snapshot.
- `download/`: Paderborn archive download list and helper scripts.
- `logs/`: captured logs from the canonical rerun.
- `paper/`: updated submission manuscript (added in the final delivery package).

The 15 raw Paderborn ZIP archives are not duplicated inside this ZIP because they total about 2.12 GiB and are publicly downloadable. Their exact archive names and SHA-256 hashes are in `evidence/SNHOP_raw_archive_checksums.csv`; every required MAT filename is listed in `evidence/paderborn_600_file_manifest.csv`.

## Environment

The canonical rerun used Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, and Matplotlib 3.10.8. See `requirements.txt` and `results/environment.json`.

## Reproduce everything from the verified raw archives

From the package root:

```bash
python -m pip install -r requirements.txt
python src/run_all.py --raw /path/to/directory/containing/the/15/ZIPs
```

`run_all.py` performs archive verification, regenerates the 600-recording feature cache, runs strict DG baselines and SN-HOP, reruns representation ablations, rebuilds the raw-noise cache with deterministic seeds, runs UDA/adaptation-size experiments, verifies prediction/task consistency, and regenerates Figs. 5–10.

## Primary updated result

The canonical strict-DG bidirectional mean balanced accuracy is **56.67% for SN-HOP**, compared with **33.63%** for the absolute-frequency envelope representation, **41.33%** for the full order representation, **47.63%** for Traditional23+FS, and **55.42%** for the protocol-matched PFN-core comparator. See `FINAL_RERUN_REPORT.md` for the full results and caveats.

## Important provenance note

This is a reconstructed canonical implementation, not the missing original transient Stage G/H execution source. The updated manuscript is based on this rerun package. See `CODE_PROVENANCE.md`.
