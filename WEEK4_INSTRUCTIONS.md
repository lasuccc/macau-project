# Week 4 instructions (HAR-PMD, federated averaging)

## What Week 4 is for
You build federated averaging (FedAvg) and prove it is correct, then run a short 10-round trial.
Each of your 36 training people is one "client". Every round, 10 clients are picked. Each one trains the current global model
on its own windows for one epoch. The server then averages their weights, giving bigger clients more say.
The goal this week is a correct prototype. A low score is not a sign of a bug, so the checks matter more than the score.
No test user is scored.

## Before you start
You need the Week 3 files in `/Users/lucas/Macau Project`:
`scaler.json`, `mlp_initial_state.pt`, `mlp_selection.json`, `label_map.json`, `split_manifest.csv`, `window_manifest.csv.gz`, `features_watch.csv.gz`.

## Reading (1 h total)
- McMahan et al. 2017 Federated Averaging, 40 min. Study Algorithm 1. Match three things to the code: local epochs (E), batch size (B), and the fraction of clients picked each round (C).
- PyTorch saving and loading models, 10 min. Understand what a state_dict is, and why a saved copy is independent of the live model.
- PyTorch reproducibility notes, 10 min. Find the three places randomness comes from: initial weights, which clients get picked, and batch order.

## What the code does
- `fedavg_common.py` holds the pieces: building clients, local training, the averaging function, scoring the global model.
- `fedavg_checks.py` runs the correctness checks and writes the two check files.
- `run_fedavg.py` runs the rounds, saves a checkpoint each round, and writes the history, the curve and the runtime file.

## Where the document's rules live in the code
| Rule from the Week 4 page | Where it is checked |
|---|---|
| One client per training person, both environments | client table at the bottom of client_independence_checks.txt |
| 10 clients per round, no replacement | selection checks |
| Each client starts from an independent copy of the global weights, fresh optimizer | independence checks |
| One local epoch | run_fedavg.py |
| Average = sum(n_k x weights_k) / sum(n_k) | aggregation checks |
| n_k not multiplied by local epochs | aggregation checks |
| Scalar test: 2 and 8 with counts 1 and 3 gives 6.5 | first line of aggregation_checks.txt |
| Changing client A leaves client B and the global state alone | independence checks |
| Single-client round returns that client's trained state | independence checks |
| Zero learning rate keeps the starting weights | aggregation checks |
| Seed 42, the Week 3 learning rate | run_fedavg.py |
| Per-round clients, counts, loss, metrics, time, checkpoint ID | fedavg_10round_history.csv |
| 50-round estimate, hardware, thread count | runtime_budget.md |

## Steps

### Step 0: finish Week 3 if you have not
Commit and tag Week 3 first (progress note, `git add`, commit, tag `week3-baselines`, push). Week 4 builds on it.

### Step 1: unzip the Week 4 files
```
cd "/Users/lucas/Macau Project"

conda activate fedhar

unzip -o ~/Downloads/week4_scripts.zip

ls fedavg_common.py fedavg_checks.py run_fedavg.py
```
All three names should print.

### Step 2: run the checks
```
python fedavg_checks.py
```
Every line should start with PASS and the last line should say ALL CHECKS PASSED.
This writes `aggregation_checks.txt` and `client_independence_checks.txt`.
If any line says FAIL, stop and send me the output. Do not run the rounds yet.

Also read the bottom of `client_independence_checks.txt`. It lists your 36 clients with their window counts per environment.
If a client has no windows in one environment, the line "every client has windows from both environments" says FAIL.
That would be a data fact to report to your supervisor, not a code bug.

### Step 3: quick trial (2 rounds)
```
python run_fedavg.py --rounds 2
```
Files get a `quick_` prefix, so nothing real is overwritten. You should see two round lines with a client list, a loss and a validation score.
Delete the trial files afterwards:
```
rm -r quick_*
```

### Step 4: the real 10-round run
```
python run_fedavg.py
```
It prints one line per round. Expect the first round to be a bit slower than the rest.
It writes:
- `fedavg_10round_history.csv`, one row per round
- `fedavg_validation_curve.png`
- `fedavg_prototype.pt`, the global model after round 10
- `fedavg_checkpoints/`, one checkpoint per round with an ID in its name
- `fedavg_round0_start.json`, the untrained starting score, used for the curve
- `runtime_budget.md`

### Step 5: look at the results
```
open fedavg_validation_curve.png

cat runtime_budget.md

head -4 fedavg_10round_history.csv
```
Things to look for:
- Validation loss should go down over the rounds. The score may wobble in the early rounds, because different clients get picked each time.
- After only 10 rounds the score will probably sit below the central MLP (0.926, trained for 50 full epochs). That is normal. The page says a low score alone is not evidence of wrong averaging.
- Rerun `python run_fedavg.py` once more and compare the first line of `fedavg_10round_history.csv`. The same seed should give the same clients and the same checkpoint IDs.

### Step 6: finish runtime_budget.md
```
open -e runtime_budget.md
```
Fill in the Week 5 plan section. The page's rule is that the planned Week 5 runs must fit the time you have. If you do not know the Week 5 plan yet, say so there and ask your supervisor.

### Step 7: fill in the progress note
```
open -e progress_week4.md
```

### Step 8: commit and tag
```
git add *.py *.json *.csv *.md *.png *.txt fedavg_checkpoints fedavg_prototype.pt

git status
```
Check that `data/`, zip files and `quick_` files are not listed. Then:
```
git commit -m "Week 4: FedAvg prototype with correctness checks"

git tag week4-fedavg-prototype

git push

git push --tags
```
If the push fails, the commit is safe on your laptop. Retry later. The README commit from Week 2 also still needs pushing.

### Step 9: message your supervisor
> Hi, Week 4 is done. I built FedAvg with one client per training user (36 clients, both environments), 10 clients per round without replacement, one local epoch, weights proportional to each client's number of training windows. All aggregation and client-independence checks pass, including the 2 and 8 with counts 1 and 3 equals 6.5 test, the zero learning rate test and the single-client test. The 10-round run uses seed 42 and the Week 3 learning rate 0.03. Validation macro F1 went from [start] to [round 10]. A 50-round run should take about [estimate] minutes. What are the Week 5 runs, so I can check they fit the time?

## Common problems
- `FileNotFoundError ... mlp_selection.json`: Week 3 did not finish. Rerun `python train_mlp.py`.
- `training windows changed since the scaler was fitted`: the split changed after Week 3. Rerun `python week3_prepare.py` and Week 3 training.
- A FAIL line in the checks: send me the whole output.
- `Do not use --smoke in Week 4`: use `--rounds 2` for a quick trial instead.
