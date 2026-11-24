# Week 5 instructions (HAR-PMD, main comparison)

## What "two FedAvg candidates" means
In Week 4 you ran one FedAvg setting. Week 5 asks for two, so you can choose between them:
- Candidate 1: one local epoch per round
- Candidate 2: three local epochs per round

Everything else stays the same for both: the learning rate, the 36 clients, 10 clients per round, the 50-round cap and the seed.
You score both on the validation users only, keep the better one, and write that choice down in a frozen file.
Only after that do you score the test users, once, with the frozen central model and the frozen FedAvg model.

## Two things to know before you start
1. **Rounds.** Your document says a 50-round cap. If your teammate's thread says 100 rounds, that is for their WISDM brief. Your Week 4 timing was about 0.1 s per round, so 50 rounds need no reduction.
2. **Learning rate.** Your Week 4 page said to use the selected centralized learning rate, which was 0.03, and that is what you ran. The Week 4 notebook has 0.01 typed in as a fixed prototype value, and your teammate quoted 0.01. Keep 0.03, and ask your supervisor to confirm. If the answer is 0.01, open `week5_config.json`, change `"fed_lr": null` to `"fed_lr": 0.01`, and run the steps below. The central model keeps its Week 3 learning rate either way.

## Reading (1 h)
- McMahan et al. 2017, 30 min. Read the experimental discussion of non-IID data and local computation.
- Matplotlib quick start, 20 min.
- PyTorch reproducibility notes, 10 min.

## The scripts
| Script | What it does | Touches test users? |
|---|---|---|
| `week5_candidates.py` | Runs the two FedAvg candidates for 50 rounds, scores validation only | No |
| `week5_freeze.py` | Picks the winner by validation score and writes `frozen_config.json` | No |
| `week5_main.py` | Scores the central and FedAvg models on validation and test | Yes, after the freeze |
| `week5_heterogeneity.py` | Plots client sizes, class mixes and feature means | No |
| `week5_mixing.py` | Mixing control: reshuffles the training windows into pseudo-clients and retrains | Yes, after the freeze |
| `week4_addons.py` | Optional Week 4 tidy-up (see Step 9) | No |

Test scoring is blocked until `frozen_config.json` exists. Every test access is written to `test_access_log.txt`, so you can show the order of events.

## Steps

### Step 0: finish Week 4 first
Week 4 should be committed and tagged (`week4-fedavg-prototype`). Check with `git status -sb`.

### Step 1: unzip
```
cd "/Users/lucas/Macau Project"

conda activate fedhar

unzip -o ~/Downloads/week5_scripts.zip

ls week5_common.py week5_candidates.py week5_freeze.py week5_main.py week5_heterogeneity.py week5_mixing.py week5_config.json
```
All seven names should print.

### Step 2: run the two candidates
```
python week5_candidates.py
```
You get one line per candidate with its best validation round and macro F1. It should finish in a few minutes.
You should also see "Week 4 prototype repeated exactly ... YES". That means your E=1 candidate reproduces your Week 4 run for the first 10 rounds, which proves the runs are repeatable. If it says NO, the learning rate differs from the one you used in Week 4.

It writes `candidate_E1_history.csv`, `candidate_E3_history.csv`, `candidates_validation_summary.csv`, the best and last checkpoint for each candidate, and `fedavg_candidates_validation.png`.

### Step 3: look at the candidates, then freeze
```
open fedavg_candidates_validation.png

cat candidates_validation_summary.csv
```
Then freeze:
```
python week5_freeze.py
```
The rule was written before you saw any test score: highest best-round validation macro F1, with a tie going to the smaller number of local epochs. The script prints the winner and writes `frozen_config.json`.
Run it only once. After the freeze, do not rerun `week5_candidates.py`, because `week5_main.py` checks that no frozen file has changed.
If you need to redo this before any test scoring, delete `frozen_config.json`. The script refuses to freeze again once `test_access_log.txt` exists.

### Step 4: the main comparison
```
python week5_main.py
```
You should see three OK lines:
- the frozen files are unchanged
- both checkpoints reproduce their frozen validation scores
- both models were scored on the identical test windows

Then a table with validation and test scores for each model, the work each did, and how many test users FedAvg beat central on.
It writes `main_results.csv`, `per_user_metrics.csv`, `test_per_user_comparison.csv`, `confusion_test_central_vs_fedavg.png`, `selected_central.pt`, `selected_fedavg.pt` and `central_history.csv`.

Central epochs and federated rounds are not equivalent computation. The table lists epochs, rounds, optimizer steps and processed examples side by side so you can say that honestly.

### Step 5: heterogeneity
```
python week5_heterogeneity.py

open natural_client_distributions.png
```
It prints the numbers and writes `client_stats.csv`, `feature_mean_spread.csv` and the figure.
On your data, expect nearly every client to have about 1440 windows and nearly the same class mix, since each person recorded about the same amount of each activity. The page says not to manufacture label imbalance, so say the labels are close to uniform. Any difference between people shows up in the feature-mean heatmap instead. Similar label mixes do not mean identical motion.

### Step 6: the mixing control
```
python week5_mixing.py
```
It shuffles the sorted training window IDs once with seed 42, and deals them into 36 pseudo-clients that keep the original client sizes. Then it retrains FedAvg with the same settings and the same sampler.
First it prints the reassignment checks. All must say PASS: every window appears once, the training set is unchanged, client sizes are kept, validation and test windows are untouched. If any says FAIL, stop and send me the output.
It writes `reassignment_manifest.csv`, `reassignment_checks.txt`, `natural_vs_mixed.csv`, `mixed_history.csv`, `mixed_best.pt`, `mixing_heterogeneity_summary.csv` and `natural_vs_mixed_curve.png`.

One seed can mislead when the effect is small. For a noise estimate, run:
```
python week5_mixing.py --sweep
```
This repeats natural vs mixed for 5 seeds on validation users only, so it does not add any test looks. It writes `natural_vs_mixed_seeds.csv` and `natural_vs_mixed_seed_summary.csv`. If the mixed-minus-natural difference is smaller than the seed-to-seed spread, report a null effect. The page says to report it if that is what you see.
The sweep repeats the whole primary run too, so it also writes test scores for the primary seed again. That is harmless and shows in `test_access_log.txt`.

### Step 7: write the interpretation (about 300 words)
```
cp interpretation_template.md interpretation.md

open -e interpretation.md
```
Use the prompts. Delete them when you are done. Send me the numbers and I will help you word it.

### Step 8: progress note
```
open -e progress_week5.md
```

### Step 9: optional Week 4 tidy-up
The Week 4 notebook also lists `run_config.json` and a numerical weighting example in its submission. This writes both from your existing files:
```
python week4_addons.py
```
It creates `week4_run_config.json` and `weighting_example.md`.

### Step 10: commit and tag
```
git add *.py *.json *.csv *.md *.png *.txt *.pt

git status
```
Check that `data/` and zip files are not listed. Then:
```
git commit -m "Week 5: main comparison, heterogeneity and mixing control"

git tag week5-main-comparison

git push

git push --tags
```
If the push fails, the commit is safe on your laptop. Retry later.

### Step 11: message your supervisor
> Hi, Week 5 is done. I compared two FedAvg candidates (1 and 3 local epochs, learning rate [x], 50-round cap) on validation users, froze the better one, then scored the central MLP and the frozen FedAvg model on the 12 test users. Test pooled macro F1 was [central] for central and [fedavg] for FedAvg. For the mixing control I reshuffled the training windows into pseudo-clients with the original sizes and retrained; the test score changed by [difference]. [One line on the 5-seed check.] Is the learning rate [x] correct for the federated runs?

## What goes where
| Week 5 requirement | File |
|---|---|
| main_results.csv | `main_results.csv` |
| natural_client_distributions.png | `natural_client_distributions.png` |
| reassignment_manifest.csv | `reassignment_manifest.csv` |
| natural_vs_mixed.csv | `natural_vs_mixed.csv` |
| Selected checkpoints | `selected_central.pt`, `selected_fedavg.pt`, `mixed_best.pt` |
| Histories | `central_history.csv`, `candidate_E1_history.csv`, `candidate_E3_history.csv`, `mixed_history.csv` |
| 300-word interpretation | `interpretation.md` |
| Counts and seeds in each table | columns inside each table |
| Preserves the training multiset and client sizes | `reassignment_checks.txt` |
| Identical evaluation manifest | `eval_manifest_hash` column in `main_results.csv` |
| Every score has a saved run | histories and checkpoints above |

## Common problems
- `Blocked: freeze the configurations first`: run `python week5_freeze.py` after `week5_candidates.py`.
- `Stop: test scoring has already happened or the freeze already exists`: that is on purpose. Do not refreeze after seeing test scores.
- `... changed after the freeze`: a frozen file was rewritten. Delete `frozen_config.json` and `test_access_log.txt` only if no test score has been looked at. Otherwise tell me.
- `training windows changed since the scaler was fitted`: the split changed after Week 3. Rerun `week3_prepare.py` and the Week 3 training.
- `FileNotFoundError ... mlp_selection.json`: Week 3 did not finish.
- Anything else: send me the last 15 lines.
