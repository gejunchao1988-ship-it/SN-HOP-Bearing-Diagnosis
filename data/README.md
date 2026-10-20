# Third-party Paderborn measurements (not redistributed)

The University of Paderborn Bearing DataCenter distributes the measurements. The original recordings are not part of this repository. Obtain the 15 bearings identified in `../canonical_v1/evidence/paderborn_600_file_manifest.csv` through the [official Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download) and verify archive SHA-256 using `../canonical_v1/evidence/SNHOP_raw_archive_checksums.csv`.

The original experiment uses two conditions (`N09_M07_F10` and `N15_M07_F10`). Supplemental load-condition testing uses `N15_M01_F10` and `N15_M07_F04`, 20 measurements per bearing/condition. The supplementary data were supplied separately during the revision; they are also not redistributed here.

Place original ZIP files in an untracked `raw/` folder or outside the repository. See the root README for commands and the provenance limitations of this release.
