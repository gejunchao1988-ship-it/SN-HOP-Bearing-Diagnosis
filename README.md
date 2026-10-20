# SN-HOP — Physics-Guided Speed-Normalized Harmonic Order Pooling

**Private staging repository. Not a final public reproducibility release.**

Reproducibility materials for the manuscript *Physics-Guided Speed-Normalized Harmonic Order Pooling for Bearing Fault Diagnosis Across Speeds and Unseen Bearing Specimens* (IEEE Access resubmission in preparation; no acceptance or DOI is claimed).

## What this repository contains

- `canonical_v1/src/`: frozen, **reconstructed canonical** preprocessing, strict bearing-disjoint domain generalization (DG), baselines, ablations, UDA, noise analysis, plotting and verification code.
- `canonical_v1/results/`: retained task-level summaries, selection records, per-prediction CSV files and audit metadata; signal-derived NumPy caches are **not distributed** in this draft.
- `canonical_v1/evidence/`: public-dataset manifest and SHA-256 inventory when present.
- `reviewer_experiments/stage1` to `stage5`: scripts and tabular outputs retained from the five reviewer-response experiment stages, including matched-dimensional baselines, spectral/window sensitivity, alternate grouping, noise robustness, real load-condition evaluation, and a linear GroupDRO comparator.
- `provenance/`: the origin paths and SHA-256 digests of each copied file.

## Scope and caveats

**Code provenance:** The earlier transient execution source was not wholly preserved. The later frozen canonical implementation was reconstructed, audited, and rerun; it is **not** claimed to be an untouched historical source tree. See `canonical_v1/CODE_PROVENANCE.md`.

**Data:** Original Paderborn Bearing DataCenter raw recordings are *not* included. Obtain them from the official source and obey the data provider's terms. The main experiment uses 15 physical bearings and 600 four-second recordings at approximately 900 and 1500 rpm. The supplemental load evaluation used a further 600 measurements at ~1500 rpm and two additional load conditions.

**Interpretation:** Transfer tasks overlap in physical bearings and source subsets. The 20 tasks cannot be treated as 20 independent specimen populations. `PFN-core` is a physical-normalization comparator, not a full replication of PPAN. The Stage-5 GroupDRO is a **linear**, source-only comparator, not a deep-domain-generalization claim. No claim is made of broadly deployable industrial reliability.

**Reproducibility limitations of this curated draft:** Stored numerical results are supplied as outputs for inspection. This archive excludes substantial NumPy feature caches and the third-party raw measurements. As a consequence, numerical reruns require rebuilding caches from verified source data before running the staged scripts. Several original staged scripts assume their preserved development directory layouts or absolute `/mnt/data` paths; treat them as source provenance until their paths have been adapted and independently re-tested. **This archive should not be described as one-command, fully-tested end-to-end release.**

## Rebuild the original canonical analysis

1. Install compatible Python 3.13 and dependencies: `python -m pip install -r requirements.txt`.
2. Obtain the 15 Paderborn raw ZIP archives listed in `canonical_v1/evidence/SNHOP_raw_archive_checksums.csv` and verify them against the provided inventory.
3. From the `canonical_v1` directory, run: `python src/run_all.py --raw /path/to/folder/with/15/ZIPs`.
4. Inspect the generated `canonical_v1/results/` files and run the included verification script.

The original complete canonical v1 archive, with generated derived caches and logs, is retained privately with the manuscript evidence package. For independent peer review, validate that your own reconstructed results agree with its published task-level outputs before citing full reproduction.

## Additional conditions

`N15_M01_F10` (1500 rpm, 0.1 Nm, 1000 N) and `N15_M07_F04` (1500 rpm, 0.7 Nm, 400 N) were used in the load extension; scripts and outputs are under `reviewer_experiments/stage4`. This is a same-dataset, same-physical-specimen condition extension, not external-data validation.

## Licensing and citation

No software license is selected in this **draft**. Before publication, the authors should agree on an appropriate software license and confirm rights for all included material. Add the final paper citation and public repository/archival DOI **only after they exist**. Do not disclose any raw measurements without verifying their redistribution terms.


## Verified offline checks

A check requiring **only the committed tabular results** (and not raw vibration recordings or cached NumPy features) is:

```bash
python -m pip install -r requirements.txt
python scripts/check_recorded_predictions.py
```

This validates the five canonical representations' saved task metrics against all corresponding task-level predictions. It does **not** independently rerun the signal processing or classifier training. Full canonical reruns still require verified original raw archives (and significant CPU time); see the `canonical_v1` instructions.

## Environment and portable paths

The staging copy replaces embedded development-machine `/mnt/data` defaults in source paths with repository-relative defaults where possible. The source has been syntax-checked and recorded output arithmetic verified, **not** subjected to a complete independent cross-platform rerun. Pass the variables `SNHOP_V1_ROOT` and `SNHOP_RAW_DIR` when executing reviewer-specific analyses. The load-condition evaluator can also use `SNHOP_CANONICAL_ARCHIVE` to point to the private original canonical ZIP containing derived feature caches; that ZIP is intentionally excluded here. Review each script's docstring and input requirements before rerunning.

## Public release gate

This repository should remain **private** until the authors approve a code license, independently test the public installation/run instructions, and agree that no third-party data or credentials are included. Do not cite a code DOI or claim public availability until the actual URL and release exist.
