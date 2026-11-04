# Personalized federated learning for wearable activity recognition

This is my code for the project. It covers weeks 1 and 2 so far: reading the raw data, checking it, and turning it into windows and features a model can use. Training starts in week 3.

I'm using the HAR-PMD dataset (Woo et al., on Zenodo, doi:10.5281/zenodo.7939223, CC BY 4.0). It has phone and watch recordings from 120 people doing six activities. Only users 61 to 120 have a watch, and those 60 people are the ones I use for the main experiments.

## Get the data

The raw data is 7.3 GB, so it isn't in this repo. Download data_publish.zip from the Zenodo page. The download kept dropping on my connection, so I used this command. It picks up where it stopped, and you can run it again if it cuts out:

    curl -L -C - --retry 50 --retry-all-errors -o data_publish.zip "https://zenodo.org/records/7939223/files/data_publish.zip?download=1"

Check the file arrived intact. The md5 should be cdc7b79aaa0450d68d94f430f2a2de63:

    md5 data_publish.zip        # Mac
    md5sum data_publish.zip     # Linux

Unzip it into a folder called data, next to these scripts. The files should end up at data/data_publish/<user>/...

    mkdir -p data
    unzip -q data_publish.zip -d data

## Set up Python

    conda env create -f environment.yml
    conda activate fedhar

## Week 1: look at the data

    python audit.py            # checks every file, lists anything missing, counts rows
    python audit_gaps.py       # finds gaps and clock jumps
    python audit_windows.py    # how many 10 s windows the gaps would damage
    python explore_user61.py   # loads one user and plots a quick sanity check

The results land in the project folder as CSV files, plus one PNG.

## Week 2: windows, features and splits

Run these in order from the project folder. All five take a few minutes in total.

    python build_pipeline.py    # cuts files into 5 s windows and makes 30 features per window
    python make_splits.py       # picks the train, validation and test users (seed 42)
    python check_coverage.py    # tries other gap limits on the training users
    python audit_alignment.py   # checks whether phone and watch line up in time
    python checks.py            # sanity checks, every line should say OK

Once the split is saved I treat split_manifest.csv as final. Later weeks read that file, so please don't rerun make_splits.py unless you also mean to redo everything after it.

All the settings (window length, gap limit, seed and so on) are in preprocessing_config.json.

## What's in here

- har_parser.py reads one recording and keeps the user, activity, device and environment.
- pipeline.py has the shared code for the week 2 scripts.
- audit_*.py, build_pipeline.py, make_splits.py, check_coverage.py and checks.py are the scripts above.
- features_watch.csv.gz and features_phone.csv.gz hold the 30 features per window. They only have a window_id and the features. Everything else about a window (user, activity, environment, time) is in window_manifest.csv.gz.
- split_manifest.csv says which group each user is in.
- data_dictionary.md, audit_summary.md, preprocessing_notes.md, alignment_audit.md and scope.md explain what I found and decided.
- progress_week1.md and progress_week2.md are my weekly notes.

## Things to know

- The user ID is only for sorting people into groups. It is never an input to a model.
- The phone and watch clocks don't line up, so I'm not combining the two devices.
- The feature files are .csv.gz because the pyarrow package wouldn't load on my laptop. pandas reads them with pd.read_csv.
