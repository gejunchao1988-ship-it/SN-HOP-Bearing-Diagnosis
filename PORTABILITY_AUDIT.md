# Portability and provenance audit

- No third-party raw `.mat` measurement files are included.
- Saved CSV results and historic experimental programs are included for inspection.
- Some historic source defaults were converted from temporary `/mnt/data` directories to project-relative data paths. See `provenance/portable_path_patches.json`.
- All packaged Python sources have been AST parsed, and the saved main predictions were rechecked against task-level accuracy/F1 metrics.
- The complete experimental workflows have not yet been rerun on a second machine; stage code may require archived NumPy features and the privately retained frozen canonical ZIP.
- `canonical_v1/CODE_PROVENANCE.md` discloses reconstructed-source provenance, which must not be removed.
- No open-source license has been selected, pending author agreement; public release requires license decision.
