# Weekly progress record: Week 1

## Completed
- Downloaded HAR-PMD (7.3 GB) with a resumable command after repeated dropped connections. Verified size, extracted 2160 CSV files.
- Created the conda environment `fedhar`, exported environment.yml and tested that it rebuilds from scratch.
- Wrote and tested the parser (har_parser.py) on user 61, then ran it across all files.
- Ran the audit: coverage, sampling rates, timestamp resets, duplicate rows, gaps and window impact.
- Wrote the data dictionary, audit summary and scope page.

## Evidence produced
audit_per_file.csv, audit_missing.csv, audit_gaps.csv, audit_windows.csv, counts_*.csv, flagged_*.csv, user61_sanity.png, data_dictionary.md, audit_summary.md, scope.md.

## Hours
Reading: TODO h. Setup and download: TODO h. Parser and audit: TODO h. Writing: TODO h. Total: TODO h of 10.

## Blockers
- The Zenodo download was slow and kept dropping. Solved with a resume loop (many hours of wall-clock time, little active effort).
- The project overview names WISDM, but the dataset used is HAR-PMD. The supervisor confirmed at the first meeting that this is acceptable.
- Early audit scripts crashed or gave misleading numbers (the first window script failed on negative timestamps, and the 0.1 s gap flag fired on 94% of watch files). Both were fixed and the corrected scripts are in the repository.

## Next step
Week 2: build 10 s windows inside continuous segments, extract features, freeze participant splits and write the exclusion manifest.
