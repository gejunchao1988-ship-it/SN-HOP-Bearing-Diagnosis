# SN-HOP IEEE Access reviewer revision — Stage 3 v4 (2026-10-19)

**Status:** Completed and prediction-verified additional raw-signal experiments. Not a final submission manuscript. The attached English response is a **working draft only** with open items clearly marked.

## Summary

- The 15 original ZIP files contain exactly two measured conditions: `N09_M07_F10` and `N15_M07_F10`; 20 genuine MAT recordings per bearing per condition, 600 recordings total.
- All 15 archives were rechecked against canonical SHA-256; the same canonical v1 clean-source model selections were frozen. All 2,400 canonical clean predictions were exactly replayed.
- Two **synthetic**, deliberately non-white test-time noise sources were imposed on raw vibration (clean-source training remained unchanged):
  - Band-limited colored noise obtained by filtering Gaussian innovations in the 1,200–6,500 Hz band.
  - Asynchronously occurring damped impact oscillations with randomized 2,600–4,600 Hz frequency, pulse decay and amplitudes.
- SNR per original raw sample (global RMS power) was 10 or 0 dB; two fixed random realizations per noise condition, **not independent experiments**.

| Raw test condition | Mean balanced accuracy | Mean macro-F1 |
| --- | ---: | ---: |
| Clean | 56.67% | 51.11% |
| Colored, 10 dB | 56.25% | 50.94% |
| Impulsive, 10 dB | 47.15% | 39.41% |
| Colored, 0 dB | 34.63% | 22.12% |
| Impulsive, 0 dB | 36.92% | 23.33% |

The 20 held-out configurations are **dependent** by physical bearing. The noisy results average over 2 seeds per configuration; this does **not** add independent target specimens. The simulated disturbances do not equal a real-world factory noise dataset.

## Original real load conditions: feasible but not yet acquired

Paderborn University's official dataset provides two further operating states: `N15_M01_F10` (1,500 rpm; 0.1 Nm; 1,000 N) and `N15_M07_F04` (1,500 rpm; 0.7 Nm; 400 N). We verified that **neither exists** in the currently recovered 15 ZIPs. The official landing page is
https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
and the conditions table is
https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/operating-conditions.

The original archive index is https://groups.uni-paderborn.de/kat/BearingDataCenter/ ; raw RAR archives and a compatible extractor are required to add *real* load tests. We have not downloaded or tested these additional source conditions.

## Files

- `SNHOP_IEEE_Access_返修阶段3实验报告.docx`: Chinese two-page experiment report.
- `Response_to_Reviewers_WORKING_DRAFT.docx`: English response working draft covering all 11 + 5 reviewer comments, with explicit missing-evidence list. **Not ready for submission; no manuscript page/line references yet.**
- `src/run_nonwhite_noise.py`: source-complete raw-signal preprocessing and noisy prediction evaluation.
- `src/verify_stage3.py`: checks raw-derived caches, 2,400 clean frozen predictions, 19,200 noisy predictions, task metrics and SNR power.
- `results/nonwhite_order_cache.npz`: raw-derived full 256-bin order spectra for every noise configuration. Does **not** contain synthetic raw vibration signal files.
- `results/nonwhite_noise_predictions.csv`, `nonwhite_noise_tasks.csv`, `nonwhite_noise_summary.csv`, `nonwhite_snr_audit.csv`, `canonical_clean_replay_predictions.csv`, `preprocess_reconstruction.json`.
- `evidence/`: file-level original data SHA-256 audit, official real-condition source references, and file archives required for the future load experiment.
- `figures/`: PNG and editable SVG chart.
- `logs/` and `stage3_verification.json`: execution logs and machine-read verification.

## Reproduce

Requires the canonical v1 package and the 15 real ZIP archives (the originals are **not** duplicated here).

```bash
export SNHOP_V1_ROOT=/path/to/SNHOP_canonical_reproducibility_v1
export SNHOP_RAW_DIR=/path/to/15_original_ZIPs
python -m pip install numpy pandas scipy scikit-learn python-docx matplotlib
python src/run_nonwhite_noise.py
python src/verify_stage3.py
python make_stage3_documents.py
```

`run_nonwhite_noise.py` reuses a previously generated cache if present; delete `results/nonwhite_order_cache.npz` to force complete preprocessing from real raw MAT files. The original ZIPs are SHA-256 checked before use, and clean test predictions are always compared with v1. The canonical v1 source code was itself reconstructed from prior archives (see canonical `CODE_PROVENANCE.md`), not recovered original transient code.

## Remaining resubmission blockers

1. The complete original published PFN method and a true representative DG method have not been reproduced; the PFN-core comparator is incomplete. Matched frequency coverage also needs formal review.
2. The real Paderborn torque/radial load conditions are published and available but have not been acquired here.
3. Complete revised manuscript, clean PDF, highlighted changes, verified manuscript page/line references, and actual public code repository remain pending.
4. The limited 15 specimens and below-deployment-grade recall cannot be resolved by repeated simulations or relabeling.
