# Week 5 interpretation (about 300 words)

Write short plain sentences in your own words. Use main_results.csv, natural_vs_mixed.csv, client_stats.csv and the plots.
Quote numbers with their seed. Delete these prompts when you are done.

1. Central vs FedAvg: which scored higher on the 12 test users, by how much, and how many of the 12 users went each way (test_per_user_comparison.csv)?
2. Which FedAvg candidate won on validation (1 or 3 local epochs), and by how much? Say the choice was made before test users were scored.
3. Work done: central trained for X epochs and Y optimizer steps, FedAvg for Z rounds and W optimizer steps. These are not the same kind of computation, so say what the comparison does and does not show.
4. How different are the natural clients? Say what the label distributions look like (near uniform or not) and what the feature-mean heatmap shows. Remember that similar label mixes do not mean identical motion.
5. Mixing control: did mixing the windows change the score? Give the primary-seed difference. If you ran the 5-seed check, say whether the difference is larger than the seed-to-seed spread. If it is not, report a null effect.
6. Limits: one main seed, 12 test users, one learning rate, mixing changes several things at once so it cannot isolate one cause.
