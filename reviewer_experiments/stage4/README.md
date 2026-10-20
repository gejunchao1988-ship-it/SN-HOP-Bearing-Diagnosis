# SN-HOP revision stage 4 — Real load-condition validation

Date: 2026-10-20. Status: **completed complementary evaluation; NOT a final manuscript or external independent dataset validation**.

## Dataset and checks

Original 15 bearing identities reused; 15 new uploaded ZIP archives and 600 previously unexamined MAT acquisitions (300 per condition):

- `N15_M01_F10`: ~1500 rpm, torque 0.1 Nm, radial force 1000 N.
- `N15_M07_F04`: ~1500 rpm, torque 0.7 Nm, radial force 400 N.
- Original standard test control: `N15_M07_F10`: ~1500 rpm, torque 0.7 Nm, radial force 1000 N.

One extraneous old-condition MAT in K003.zip ignored. Each new ZIP was CRC-tested (15/15). Full SHA256 for ZIPs saved in `results/new_load_archive_checksums.csv` so downloaded files can be compared without embedding >2 GB of raw data.

A fixed 31x8192-sample prefix was used for every recording, identically to canonical original preprocessing. `src/core.py` is the canonical source version from the v1 reproducibility archive; `run_load_transfer_v1.py` reuses its `_load_record`, order-grid definitions, `hop`, normalizations, and classifier implementation. The two new conditions supplied measured speed values (~1500 rpm), not arbitrary nominal speed interpolation. See `results/new_load_recordings.csv`.

## Leakage-controlled evaluation

For each of ten outer held-triplet pairs, six physical target bearings are excluded from training and source-only parameter selection. Training uses nine distinct source bearings at either 900 rpm or 1500 rpm and *only* the standard load `N09_M07_F10` or `N15_M07_F10` respectively. Target testing uses only six held-out bearings at 1500 rpm under standard-load control, low torque, or low radial force.

SN-HOP and full-order-spectrum models use *frozen source-only configurations* from the corresponding canonical v1 fold. For 900 rpm sources, configurations from v1 `900_to_1500` are reused. For 1500 rpm sources, configurations selected in v1 `1500_to_900` are reused. No target class labels, recordings, or statistics influenced training, classifier choice, or parameter choice. This provides an informative source-frozen stress test across target loads; it is not newly optimized DG training for each target condition.

**Cross-check:** the `900_to_1500` standard-load control matches all ten original canonical transfer-task accuracies for both SN-HOP and full-order representations exactly (checked at 1e-12 tolerance).

## Mean balanced accuracy (10 held-out group pairs for each cell)

| Source rpm | Target condition | SN-HOP | Full order |
|---:|---|---:|---:|
| 900 | Standard N15_M07_F10 | 57.50% | 37.33% |
| 900 | Low torque N15_M01_F10 | 53.42% | 36.17% |
| 900 | Low radial force N15_M07_F04 | 54.50% | 35.67% |
| 1500 | Standard N15_M07_F10 | 53.50% | 51.08% |
| 1500 | Low torque N15_M01_F10 | 55.25% | 52.67% |
| 1500 | Low radial force N15_M07_F04 | 59.92% | 51.58% |

Each cell comprises 10 outer tasks and 120 target recordings per task. Total 120 method-specific tasks and 14,400 repeated test predictions, with **only 600 new distinct recordings**; some records are evaluated in several non-independent tasks.

## Interpretive limits

- This validates modest variations in **load at constant ~1500 rpm** for the same 15 bearing identities, not unseen bearing types, other datasets, continuously varying speed, or actual industrial field noise.
- The 10 outer held-group tasks share bearings and overlap in their training sets. Accuracy differences are descriptive and not evidence of 10 independent trials. Bearing-level uncertainty remains limited by 5 physical specimens per class.
- Low-radial-load average accuracy increases for source-1500; this is *not* causal evidence that lower load improves all diagnoses.
- PFN full framework and other published DG baselines are not included in this stage; new results must not be described as comprehensive SOTA superiority.
- Canonical historical code was reconstructed and audited; this stage reuses and freezes that canonical implementation rather than implying the original historical experimental code was preserved.

## Files

- `run_load_transfer_v1.py`: raw feature extraction and target-only evaluation; `extract` and `eval` modes.
- `src/core.py`: copied source implementation used for this stage.
- `baseline/`: canonical v1 feature cache, recordings, source-fold configurations and task summaries sufficient to reproduce the `eval` stage without another ZIP.
- `results/new_load_order_cache.npz`: newly extracted 600 x 256 order spectra.
- `results/new_load_recordings.csv`: actual new files, measured speeds, identities, labels.
- `results/new_load_tasks.csv`: per-task metrics.
- `results/new_load_predictions.csv`: each prediction (may repeat recording across held folds).
- `results/new_load_summary.csv`, `results/paired_task_comparisons.csv`, `results/per_bearing_recall.csv`: summary and descriptive comparisons.
- `results/new_load_archive_checksums.csv`: full SHA256 and size of 15 new raw ZIPs.
- `results/verification_stage4.json`: independent result recomputation/check summary.
- `extract.log`, `eval.log`: actual execution logs.

## Rerun

Use Python with `numpy`, `scipy`, `pandas`, and `scikit-learn`. `eval` uses the included cache and baseline, and can run without raw archives:

```bash
python run_load_transfer_v1.py eval
```

To reconstruct raw features, place all 15 new ZIPs in one directory with names following the uploaded `K001(3).zip`, `KA04(1).zip` pattern. Set `SNHOP_RAW_DIR` to that directory, and then:

```bash
python run_load_transfer_v1.py extract
python run_load_transfer_v1.py eval
python verify_stage4.py
```

Important: `verify_stage4.py` computes ZIP SHA256 by rereading all 15 archives and therefore needs `SNHOP_RAW_DIR` configured correctly.
