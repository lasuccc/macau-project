Weekly progress record, week 2

What I did
- Cut every recording into segments at clock jumps and gaps over 0.2 seconds.
- Resampled to 60 readings per second, cut 5 second windows and worked out 30 features per window.
- Chose the watch users (61 to 120) as the study group and split them 36/12/12 with seed 42.
- Tested other gap limits on the training users and kept 0.2 seconds.
- Checked whether phone and watch line up in time. They do not, so I will not combine them.
- Ran checks.py. Every check passed.

What I produced
features_watch.csv.gz, features_phone.csv.gz, window_manifest.csv.gz, split_manifest.csv, exclusions.csv, counts_by_user_class.csv, filter_counts.csv, coverage_by_gap_rule.csv, alignment_audit.md, preprocessing_config.json, feature_columns.txt, preprocessing_notes.md.

Hours
Reading: 2 h. Segments and features: 2 h. Split and checks: 2 h. Timing check: 2 h. Writing: 2 h. Total: 10 h of 10.

Problems
- The week 2 document mentions a study group and split on page 2, and my copy has no page 2. I used the watch users and kept the 31/10/10 ratio from the project overview. I still need my supervisor to confirm this.
- The pyarrow package would not load, so I saved the features as compressed CSV.
