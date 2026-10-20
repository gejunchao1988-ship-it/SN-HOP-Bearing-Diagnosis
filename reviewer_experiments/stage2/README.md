# IEEE Access SN-HOP — Stage 2 Reviewer Evidence (2026-10-19)

**Status: verified supplemental experiments, not the final submitted revision.**
The canonical v1 and Stage-1 v2 source, raw bearing archives, source DOCX, and reviewer letter were not overwritten.

## Main findings

All numbers below are calculated from saved prediction tables or actual fixed representation data; **no invented experiment outputs**.

### Alternative groupings: exact same nested source-only protocol for both representations

| Prespecified grouping | SN-HOP mean BA | Full order mean BA | Difference |
| --- | ---: | ---: | ---: |
| Original | 56.67% | 41.33% | +15.33 pp |
| Seed 20261020 | 59.38% | 37.00% | +22.38 pp |
| Seed 20261021 | 58.88% | 37.33% | +21.54 pp |
| Seed 20261022 | 51.17% | 38.33% | +12.83 pp |
| Seed 20261023 | 54.92% | 38.92% | +16.00 pp |

- Five sets of 5 physical triplets, with each triplet having one healthy, one outer-race damaged, one inner-race damaged physical specimen.
- Each grouping generates 10 leave-two-triplets-out tasks × 2 rotational-speed transfer directions; for each task, all model and feature parameters are selected through nested source-only three-fold bearing-triplet CV.
- SN-HOP grid uses 3 pooling halfwidths × 4 maximum harmonic orders × 10 lightweight classifiers; full order uses 3 row normalizations × 21 classifiers, exactly matching the canonical v1 protocol.
- Original grouping has been **verified to reproduce every one of the saved canonical SN-HOP and full-order predictions** (2,400 predictions per method).
- Same 15 physical bearings are reused in every grouping. This is a sensitivity analysis, **not** external validation or independent study replications. Grouping seeds are fixed, and no group is selected based on resulting accuracy.

### Cross-bearing physics-guided feature evidence

A fixed, pre-existing candidate grid configuration (`pooling halfwidth=0.4`, `maximum order=35`) defines the measured representation; no heldout labels are used to choose the configuration. Similarity is calculated over **30 bearing×speed centroids**, each averaged from the physical bearing's 20 recordings. The comparison is between 900-rpm centroids and 1500-rpm centroids **from different bearings**, divided by same/different true classes solely for *description*. Centroids are globally mean-centered using unlabeled features (never used in classification).

| Representation | Features | Same-class minus different-class mean cosine similarity |
| --- | ---: | ---: |
| Full order | 256 | 0.0869 |
| Physics-guided harmonics + broad-band context | 48 | 0.1585 |
| Uniformly spaced feature centers + same broad-band context | 48 | 0.0731 |
| Physics-guided harmonic features only | 36 | 0.1583 |
| Broad-band context only | 12 | 0.1452 |

The harmonic location provides plausible direct representational evidence but this **does not** prove a causal mechanism or statistical significance: centroid pair scores reuse the same physical bearings, and label categories are used to define the post hoc comparisons.

### Individual outer-race error patterns (900→1500 rpm)

| Physical bearing | Predicted healthy | Predicted outer | Predicted inner |
| --- | ---: | ---: | ---: |
| KA15 | 4 | 0 | 76 |
| KA22 | 45 | 0 | 35 |
| KA30 | 0 | 0 | 80 |

Each row contains 4 overlapping evaluation-task appearances of **the same 20 actual recordings**, not 80 independent recordings. In the high-speed target domain, KA15 and KA22 have negative BPFO first-harmonic local contrast; KA30 has low positive contrast but still fails, ruling out simple one-feature causal explanations. The frozen predictions and harmonic feature results are saved in CSV.

### Exploratory sideband observations

`results/exploratory_shaft_order_sidebands_*` stores ±1-shaft-order peak differences around BPFO/BPFI harmonics by recording and bearing; these are **descriptive, not verified sideband physics or evidence of higher diagnostic accuracy**. The order grid spacing is 0.19608 order; overlapping bins and finite FFT resolution limit interpretation.

## Contents

- `SNHOP_IEEE_Access_返修阶段2实验报告.docx` and PDF: Chinese results and limits, verified visually via DOCX rendering.
- `src/run_regrouping_sensitivity.py`: all five complete regrouping studies, classifier selection, test predictions.
- `src/run_feature_and_error_analysis.py`: cross-speed centroid similarities, per-bearing confusion, BPFO/BPFI contrast features.
- `src/run_sideband_analysis.py`: exploratory shaft-order sideband data.
- `src/verify_stage2.py`: checks integrity and exact predictions against canonical v1.
- `results/`: task-level / recording-level predictions, selected classifiers, protocol designs, direct feature signature CSVs.
- `figures/`: chart PNG/SVG figures for manuscript or response (review before inserting).
- `logs/`: original execution logs.
- `stage2_verification.json`: 5 groupings, 200 tasks total (both methods), 24,000 saved predictions, cross-checked.
- `SHA256SUMS.txt`: content hash inventory.

## Reproduce

Prepare the original canonical v1 package from `SNHOP_canonical_reproducibility_v1(1).zip` or the original v1 inside your Stage-1 v2 archive, and set its extracted root as the environment variable `SNHOP_V1_ROOT`.

```bash
export SNHOP_V1_ROOT=/path/to/SNHOP_canonical_reproducibility_v1
python -m pip install numpy pandas scipy scikit-learn matplotlib python-docx
python src/run_feature_and_error_analysis.py
python src/run_sideband_analysis.py
python src/run_regrouping_sensitivity.py    # CPU intensive (~3–4 min on reference environment)
python src/verify_stage2.py
python make_stage2_report.py
```

**Data provenance:** Feature cache and class-labeled frozen predictions are inherited from the canonical v1 reconstruction, itself explicitly documented as reconstructed because earlier transient code was not preserved. The original 15 raw archive ZIPs are **not included** in this Stage-2 zip; historical raw signal verification is saved with Stage-1 v2 evidence. This Stage-2 phase analyzes genuine canonical cached features and predictions, it does not claim to rerun fresh raw preprocessing.

## Reviewer response interpretation

- R1-4: new group sensitivity evidence; 15 physical specimens still an important limitation.
- R1-5: task dependence unresolved by regrouping; avoid independent-task p-values.
- R1-7: direct fixed-feature separation and BPFO feature examples; exploratory sidebands with stated limits.
- R1-8 / R2-5: explicit physical-bearing fault-class misclassification patterns.
- R2-1: no new external dataset or bearing geometry in this stage.

Further required before IEEE Access resubmission: non-white-noise/industrial interference test; evaluate practical additional operating conditions as feasible; update clean/highlighted manuscripts; submit English response to all 16 reviewer points.
