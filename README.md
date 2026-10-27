# Personalized federated learning for wearable HAR: Week 1

Week 1 covers data understanding: a parser, a data audit, a data dictionary and a project scope.

## Data

Dataset: HAR-PMD, Woo et al., Zenodo, doi:10.5281/zenodo.7939223 (CC BY 4.0).
File: data_publish.zip, 7.3 GB, md5 cdc7b79aaa0450d68d94f430f2a2de63.

Resumable download:

    curl -L -C - --retry 50 --retry-all-errors -o data_publish.zip "https://zenodo.org/records/7939223/files/data_publish.zip?download=1"

Check the hash with `md5 data_publish.zip` (macOS) or `md5sum data_publish.zip` (Linux), then unzip so the files sit at `data/data_publish/<user>/...`:

    mkdir -p data
    unzip -q data_publish.zip -d data

The raw data is not included in this repository.

## Environment

    conda env create -f environment.yml
    conda activate fedhar

## Run

    python audit.py          # per-file audit, missing files, count tables
    python audit_gaps.py     # gap and timestamp-reset analysis
    python audit_windows.py  # share of 10 s windows affected by gaps or low rate
    python explore_user61.py # parse one participant and plot a sanity check

Outputs are written to the project folder as CSV files and one PNG.

## Contents

- har_parser.py: parser. Reads one recording and keeps user, activity, device and environment as columns.
- audit.py, audit_gaps.py, audit_windows.py: data audit.
- data_dictionary.md: file layout, columns, units, collection facts.
- audit_summary.md: findings from the audit.
- scope.md: one-page project scope.
- progress_week1.md: weekly progress record.

## Notes

The user ID is an identifier only and must never be used as a model input.
