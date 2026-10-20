# SN-HOP IEEE Access — Stage-1 reviewer-response experiments (2026-10-19)

This is an evidence archive, **not** a revised manuscript or final resubmission. The original results and original reconstructed canonical v1 implementation were not overwritten. The copy of the v1 ZIP is provided in `baseline_v1/` as provenance and to enable repeat runs.

## Key numbers (mean balanced accuracy, 20 strict bearing-disjoint cross-speed tasks)

| Test | Balanced accuracy |
|---|---:|
| SN-HOP 8192 / 500–10,000 Hz / background=1.2 (canonical v1) | 56.67% |
| Full order + source-only PCA, matched taskwise dimension | 45.21% |
| Full order + source-only Fisher selection, matched dimension | 43.25% |
| 16384-point window, same 253952-sample support, re-tuned source-only | 56.88% |
| 32768-point window, same 253952-sample support, re-tuned source-only | 49.00% |
| 200–12,000 Hz bandpass, re-tuned source-only | 52.46% |
| 1,000–8,000 Hz bandpass, re-tuned source-only | 49.29% |
| Harmonic-background halfwidth 0.8, 1.2, 1.6, 2.0 | 55.17%, 56.67%, 51.50%, 50.83% |

Single target-bearing UDA on the *reserved runs 11–20 only*: NoDA **57.75%**, bearing-specific CORAL **42.50%**, bearing-specific target Z-standardization **39.17%**. This is an adaptation-test subset and MUST NOT be conflated with the v1 all-recording 56.67%.

Physical-bearing triplet sensitivity versus full order: +15.33pp; illustrative 5-cluster bootstrap interval [7.29, 27.17]pp; two-sided 5-triplet sign-flip p=0.0625. **Five artificial blocking triplets plus overlapping training sets do not establish independent inferential units.** Do not quote the old matched 20-task p-values as if 20 tasks were independent.

End-to-end timing from 600 actual archived MAT files: median **46.24 ms per four-second recording**, including reading/decompressing a local ZIP member, MAT parsing, filter, Hilbert, FFT, order interpolation, harmonic features, and a previously fitted classifier; this is hosted single-thread CPU timing, excluding sensor acquisition / network and training.

## Archive contents

- `SNHOP_IEEE_Access_返修阶段1实验报告.docx`: Chinese academic-style technical report.
- `src/run_stage1.py`: matched-dimension, 15-bearing / 5-triplet sensitivity, and single-target UDA.
- `src/run_window_resolution.py`: reprocessing both alternate FFT windows from raw vibration, followed by source-only nested classifier/pooling selection.
- `src/run_background_width.py`: re-tuned background neighborhood width study on original cached order spectra.
- `src/run_bandpass_sensitivity.py`: full raw-signal 2-band reprocessing and nested source-only model selection.
- `src/run_end_to_end_timing.py`: all 600 signal timing and cache consistency check.
- `src/verify_stage1.py`: validates each of the saved task-level and prediction-level records, all 600 recordings, and raw archive inventory.
- `results/*.csv, *.npz, *.json`: complete new task- and recording-level evidence; saved full reprocessing caches from raw data.
- `logs/`: original execution logs, including per-task outcomes.
- `evidence/`: raw archive SHA-256 manifest, the 600-file manifest, restored-archive audit.
- `baseline_v1/SNHOP_canonical_reproducibility_v1.zip`: original reconstructed v1 reproducibility archive.
- `stage1_verification.json`: consistency checks re-run before delivery.
- `SHA256SUMS.txt`: SHA-256 inventory for the stage-1 files.

Raw 15 ZIPs **are not re-bundled**: together they require ~2.12 GiB; they remain separately restored and checksummed in the conversation workspace/file library.

## Reproduction

Python dependencies can be installed from the extracted v1 package's `requirements.txt`. This revision package intentionally retains original choices and fixes no software/data inconsistencies silently.

```bash
python setup_baseline.py
python -m pip install -r baseline_v1/SNHOP_canonical_reproducibility_v1/requirements.txt
export SNHOP_RAW_DIR=/path/to/the/15/raw_paderborn_zip_archives
python src/verify_stage1.py
python src/run_stage1.py
python src/run_background_width.py
python src/run_window_resolution.py
python src/run_bandpass_sensitivity.py
python src/run_end_to_end_timing.py
python src/verify_stage1.py
```

Windows: in PowerShell, use `$env:SNHOP_RAW_DIR='C:\path\to\raw_ZIPs'` in place of `export`.

NOTE: Re-running overwrites **only stage-1** files in this extracted revision folder, never `baseline_v1/.../results`. The original v1 has its own full reproduction script that is unnecessary to rerun when generating the stage-1 evidence.

## Method and limits

- Predefined source/target separation: leave two of five 3-bearing triplets for test, train and model-select on the remaining 9 bearings. Both 1500→900 and 900→1500 rpm; 20 tasks.
- The PCA/Fisher representations use the per-task dimension selected by SN-HOP through source-only CV. Their PCA components/Fisher rankings are fit anew **inside each source training fold** and on all source bearings for final evaluation; classifiers are selected by 3-fold source-triplet CV.
- Window studies use the same first 253,952 samples for all window lengths with 8192 sample hop; longer windows overlap, so they alter the number of windows and averaging as well as physical resolution. No artificial zero-padding is treated as real resolution.
- CORAL-single adapts to 10 unlabeled recordings from **only one** held-out bearing and tests on its own disjoint runs 11–20; new target-test labels are not used for adaptation or model selection.
- Source triplets are human-imposed class-balanced blocking sets. A five-triplet bootstrap or sign-flip is sensitivity evidence, **not** an independent-sample significance claim because both source/training sets and test appearances overlap. Additional alternative groupings, more bearings, external datasets and robust statistics remain desirable.
- Fitted PFN-core is a protocol-matched representation, **not** a full implementation of the published PFN framework. No claim of statistical superiority to full PFN is justified.
- The canonical v1 was reconstructed after a prior transient implementation was lost; preserved historical provenance in baseline_v1/../CODE_PROVENANCE.md.

## Remaining reviewer requests

Alternative specimen groupings; actual cross-bearing harmonic/sideband analysis; per-class failure mechanisms; more realistic noise and possibly other speeds/loads; extended comparator discussion and manuscript edits; a point-by-point review response and highlighted PDF. These remain to be done before final IEEE Access resubmission.
