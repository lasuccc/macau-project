# Initial data audit

Dataset: HAR-PMD (Woo et al., Zenodo doi:10.5281/zenodo.7939223, CC BY 4.0).
Method: 2160 raw CSV files parsed by har_parser.py and checked by audit.py, audit_gaps.py and audit_windows.py. Per-file tables are saved as CSV files in this folder.

## Coverage

- Files parsed: 2160 of 2160 expected. Missing: 0. Failed to parse: 0.
- 120 users with phone data; 60 users (61 to 120) with watch data.
- Every file has the expected column count (32 phone, 16 watch) and starts at Time = 0.
- Every file is exactly 600.0 s long. Folder numbers match filename user IDs in all files.
- No NaN cells. No duplicate timestamps.

## Sampling rate

| Device | Mean rows | Effective rate (rows / 600 s) | Min rows | Max rows |
|---|---|---|---|---|
| phone | 35,797 | about 59.7 Hz | 26,915 | 36,934 |
| watch | 31,191 | about 52.0 Hz | 17,950 | 38,044 |

These match the paper's reported averages (59.6 Hz phone, 51.7 Hz watch). The median step in audit_per_file.csv overstates the watch rate (66.7 Hz) because most steps are short and some are long, so rows divided by 600 is used as the effective rate.

Outliers: user 2 phone files (indoor 45 to 47 Hz across several activities; phone model Galaxy A8) and user 115 walking watch indoor (about 29.9 Hz, 17,950 rows). The device.txt file records the phone model only, so it cannot explain watch behaviour. With one user per model, a device effect cannot be separated from a user effect.

## Timestamp resets (11 files, 12 reset events, 6 users)

Affected users: 43, 88, 93, 95, 105, 114. Six phone files (users 43 and 93) and five watch files. User 93 has four affected phone files. Full list: flagged_resets.csv. Phone models of the affected users: 43 LG Q9, 88 Galaxy S21, 93 Galaxy S8+, 95 Galaxy S22+ 5G, 105 Galaxy S22, 114 Galaxy S22+ 5G. No common model.

Inspected examples (Time before and after the reset):

| File | Row | Time | Jump back |
|---|---|---|---|
| user 105 still watch indoor | 3674 | 72.870 to 69.405 | 3.465 s |
| user 114 walker watch indoor | 17738, 17739 | 347.670 to 346.981 to 342.795 | 0.689 s, then 4.186 s |
| user 93 walking phone indoor | 22547 | 411.194 to 411.024 | 0.170 s |
| user 43 walking phone indoor | 16224 | 297.895 to 297.269 | 0.626 s |

In each case time resumes counting forward from the earlier value, so part of the timeline is covered twice. In the two phone files inspected, the sensor values continue smoothly across the jump and no phone file contains duplicate rows, which points to a timestamp error and not repeated data. The watch files with resets differ (see below). These are interpretations from the audit, not confirmed causes.

One file has negative Time values: user 95 crutches watch outdoor (minimum Time -3.164 s, 49 rows below zero). Every file starts at Time = 0, so time dropped below zero after a reset. It is the only such file and needs splitting at the reset before windowing.

## Duplicate sensor rows (3 files)

User 105 electric watch indoor (51 rows), user 105 still watch indoor (49 rows) and user 114 walker watch indoor (5 rows). All three also contain timestamp resets, so some data in them appears repeated. See flagged_duplicates.csv.

## Gaps

Gaps over 0.1 s occur in 298 of 1440 phone files and 675 of 720 watch files. For the watch this threshold mostly reflects normal irregular sampling, so 0.5 s and 1 s are used as dropout thresholds.

| Threshold | Phone files | Watch files |
|---|---|---|
| gap over 0.5 s | 90 of 1440 | 11 of 720 |
| gap over 1 s | 20 of 1440 | 8 of 720 |

Three patterns appear:

1. User 2 phone files show slow, irregular sampling throughout (400 to 520 gaps over 0.1 s per indoor file, 84 to 147 s spent in such gaps, but a maximum gap of only about 1.2 s). This is consistent with a device or user effect, not dropouts.
2. User 115 walking watch indoor runs at about 30 Hz with 551 gaps over 0.1 s and a maximum gap of 0.27 s.
3. Real dropouts: user 88 walking watch outdoor has a 30.0 s gap and six gaps over 1 s (this file also has a reset). Users 38, 89, 21 and 90 have phone files with several gaps of 1 to 2.8 s.

Full counts: audit_gaps.csv.

## Window impact

Method (audit_windows.py): each file is cut into non-overlapping 10 s bins from t = 0 (60 per file). A bin is "bad" if it contains a gap over 1 s or fewer than 80% of the samples expected at the nominal rate (60 Hz phone, 52 Hz watch).

| | Phone | Watch |
|---|---|---|
| Windows | 86,400 | 43,200 |
| Windows with a gap over 0.5 s | 794 (0.92%) | 18 (0.04%) |
| Windows with a gap over 1 s | 36 (0.04%) | 15 (0.03%) |
| Short windows | 319 (0.37%) | 93 (0.22%) |
| Bad windows | 340 (0.39%) | 99 (0.23%) |

The problem is concentrated in a few users. User 2 phone accounts for 305 of the 340 bad phone windows (90%). User 115 watch accounts for 60 of the 99 bad watch windows, and user 88 watch for 15. Without those, about 35 phone and 24 watch windows remain bad. Users 90, 89, 38 and 21 (phone) and 67, 114 and 91 (watch) have only a few each.

Caveats: the counts are approximate. Windows are fixed bins from t = 0, windows that contain a timestamp reset are not flagged here, and rows with negative Time are left out of the bin counts. Week 2 will recompute these within continuous segments. Per-file results: audit_windows.csv.

## Implications for Week 2

1. Windows will be defined by time (10 s), not sample count, because rates differ by device and file.
2. Windows will be built inside continuous segments. A reset, or a gap above a chosen threshold, starts a new segment.
3. A minimum sample count per window will be set per device.
4. Files with resets or duplicates are candidates for splitting or exclusion. Each decision goes in the exclusions log.
5. Users with persistently low sampling rates (user 2 phone, user 115 watch) are candidates for a flag or exclusion, to be decided in Week 2.
6. Time restarts at 0 in every file, so phone and watch recordings cannot be paired by timestamp. Fusion is conditional on an alignment audit, and single-device results are the fallback.
7. Watch experiments cover 60 users, not 120.

## Dataset choice

The project overview describes the WISDM dataset (51 users, 18 activities, 31/10/10 split). This audit covers HAR-PMD (120 users, 6 activities). The supervisor confirmed at the first supervision meeting that HAR-PMD is acceptable for the project. Split sizes and window parameters will be set from HAR-PMD's population in Week 2.
