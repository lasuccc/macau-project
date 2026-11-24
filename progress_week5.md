Week 5 progress

Frozen before scoring test users (from frozen_config.json)
- Federated candidate chosen on validation: 3 local epochs, learning rate 0.03, 50-round cap, selected round 46
- Central: learning rate 0.03, selected epoch 49
- Freeze time: see frozen_config.json

Main comparison on the 12 test users (from main_results.csv)
- Central MLP, pooled macro F1 0.9205, mean per-user macro F1 0.9192
- FedAvg MLP, pooled macro F1 0.8739, mean per-user macro F1 0.8725
- Test users where FedAvg beats central: 1 of 12
- Work done: central 39,641 optimizer steps, FedAvg 31,740 optimizer steps

Heterogeneity
- Windows per client: min 1410, max 1440
- Mean class-proportion distance from the pooled mix: 0.0013
- Average spread of client means across features: 0.212

Mixing control (from natural_vs_mixed.csv, seed 42)
- Natural test macro F1 0.8739, mixed 0.8868, difference +0.0129
- 5-seed validation check: mixed beat natural in all 5 seeds, mean last-round difference about 0.012, natural seed-to-seed spread about 0.003

Checks
- reassignment_checks.txt all PASS: yes
- Same test windows for both models (eval_manifest_hash): yes
- Test accessed only after the freeze (test_access_log.txt): yes, three accesses (main comparison, mixing run, and the mixing run repeated by the seed check)