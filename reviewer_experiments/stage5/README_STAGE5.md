# SN-HOP IEEE Access Revision — Stage 5 (2026-10-20)

## Scope and epistemic limits
- Source-only **linear GroupDRO** vs same-architecture linear ERM on the 256-bin order spectrum; it is **not** a full published neural DG benchmark.
- Source-only nested three-fold bearing-triplet selection, 20 outer tests. Target recordings and target labels are never used in training, feature scaling, hyperparameter selection or optimizer tuning.
- Exploration of speed-specific 0–50-order physical bandwidth with fixed absolute-Hz spectral features. Zero masking introduces an rpm-specific cutoff artifact; this is **not a perfect frequency-resolution match**.
- The original PFN-core is **not** the full published PPAN (PFN + PGSA + conditional domain adaptation); no full PPAN reproduction is asserted.
- Original code base is the **audited reconstructed canonical v1**, not an untouched historic laboratory code snapshot. Original raw Paderborn ZIPs are not duplicated in this small archive.

## Core results (mean recording-level balanced accuracy, averaged across 20 overlapping held-bearing tasks)

| Method | 1500 to 900 | 900 to 1500 | Combined |
| --- | ---: | ---: | ---: |
| Linear ERM, full-order SNV | 48.17% | 53.00% | 50.58% |
| Linear GroupDRO, same features | 48.17% | 53.17% | 50.67% |
| SN-HOP canonical v1 (different representation / own source-selected model) | 55.83% | 57.50% | 56.67% |
| Absolute-frequency original 0–1250 Hz | — | — | 33.63% |
| Absolute frequency, each record capped at 50 times measured shaft frequency (exploratory) | 30.00% | 33.83% | 31.92% |

**DO NOT** interpret comparisons as independent tests on 20 bearing populations, as proof of dominance over all DG methods, or as full PPAN results. The independent physical specimens remain only 15, with 5 arbitrary balanced source groups.

## Execution

```bash
pip install numpy pandas scipy scikit-learn torch python-docx
python run_groupdro_dg.py
python run_coverage_matched_absolute.py
python verify_stage5.py
```

Expected relative inputs (included in this archive):
- `SNHOP_canonical_reproducibility_v1/src/core.py`
- `SNHOP_canonical_reproducibility_v1/results/feature_cache.npz`
- `SNHOP_canonical_reproducibility_v1/results/recordings.csv`
- `SNHOP_canonical_reproducibility_v1/results/{absolute,order,pfn,snhop}_tasks.csv`

`results/` preserves all fitted-model arrays, source-fold selection data, every held-out prediction and metric consistency audit. `dg_execution.log` and `coverage_execution.log` are actual job logs from this run. In `deliverables/`, the 16-comment English response is **still a working draft**: it requires final revised manuscript positions, a real public repository URL, and final editorial checks.

## Cited methodological background
- Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. "Distributionally Robust Neural Networks for Group Shifts." ICLR 2020. https://arxiv.org/abs/1911.08731
- Shi, Z. and Wang, K. "Physics-Guided Frequency Normalization Enables Cross-Speed Inner-Race Bearing Fault Transfer Diagnosis." *Electronics* 2026, 15(13), 2943. https://doi.org/10.3390/electronics15132943

## Important remaining tasks
1. Integrate stage 1–5 results into the 19-page submitted DOCX without damaging OMML equations, figures, or IEEE template layout.
2. Recheck overall inference text and page/line references; format figure/table numbering and bibliography order.
3. Consider whether a faithful full PPAN reproduction and additional strong DG networks are practicable under the *same* protocol. Do not invent missing scores.
4. Provide a real public code/configuration URL before claiming public access.
5. Generate clean DOCX/PDF, highlighted PDF, and final Author's Response file.
