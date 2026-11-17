# Week 3 instructions (HAR-PMD, central baselines)

Read this top to bottom once. Then do the steps in order. Every command goes in the Mac Terminal.

## What Week 3 is for

You train three simple models on all the training users together (no federated learning yet):

1. A model that always guesses the most common class. This is the floor. Anything real must beat it.
2. Logistic regression.
3. A small neural network (MLP).

You score all three on the 12 validation users only. You do not touch the 12 test users this week.
Week 4 reuses your scaler, your split and your starting network weights, so they have to be saved exactly.

## Before you start

You need these files from Week 2 in `/Users/lucas/Macau Project`:
`features_watch.csv.gz`, `window_manifest.csv.gz`, `split_manifest.csv`, `feature_columns.txt`, `preprocessing_config.json`.
Check with:
```
cd "/Users/lucas/Macau Project"
ls features_watch.csv.gz window_manifest.csv.gz split_manifest.csv feature_columns.txt preprocessing_config.json
```
All five names should print with no "No such file" error.

## Step 0: set up the tools

```
cd "/Users/lucas/Macau Project"
conda activate fedhar
pip install torch
unzip -o ~/Downloads/week3_scripts.zip
python -c "import torch, sklearn; print(torch.__version__, sklearn.__version__)"
```
The last line should print two version numbers. If `pip install torch` fails, send me the error text.
Do not install pyarrow. The scripts read the `.csv.gz` files on purpose.

## Step 1: decide the split (do this before any training)

Your Week 2 split used scikit-learn's GroupShuffleSplit. The Week 3 document says to sort the user IDs, shuffle with
`numpy default_rng(42).permutation`, then take the first 36 as train, the next 12 as validation and the last 12 as test.
Both give 36/12/12 but they pick different people.

Compare them (this changes nothing):
```
python resplit_like_doc.py
```
You will see a table. The "same users in both" column is small. That confirms the two methods differ.

Switch to the document's method:
```
python resplit_like_doc.py --apply
```
This does three things:
- copies your old file to `split_manifest_week2_groupshuffle.csv`
- writes the new `split_manifest.csv`
- rewrites `split_class_counts.csv`

Why switch now: nothing has been trained or scored on test yet, so switching leaks nothing. Later it would force you to redo every number.
Tell your supervisor at the check-in. If they say keep the old split, restore it with:
```
cp split_manifest_week2_groupshuffle.csv split_manifest.csv
```
Only do this restore before step 4.

Run `python resplit_like_doc.py --apply` only once. Running it a second time just rewrites the same split, so it is harmless, but it overwrites nothing important.

## Step 2: smoke test (tiny and fast)

The Week 3 document says to run one file, then two users, then everyone. `--smoke` is the two-user version.
```
python week3_prepare.py --smoke
python train_logistic.py --smoke
python train_mlp.py --smoke
python diagnostics.py --smoke
python report_validation.py --smoke
```
Smoke scores will look bad, and the overfit check will print LOW. That is expected with 3 epochs.
What matters: no red error text. Files starting with `smoke_` appear. You can delete them after.

## Step 3: the real run

Run these five in this exact order.

```
python week3_prepare.py
```
Checks that no user or window sits in two splits. Fits the scaler on training windows only. Prints a class count table.
Look at the table: every class should have thousands of windows in train and a similar share in val.

```
python train_logistic.py
```
Prints the majority-class score, then the logistic regression result as JSON.
Check `"converged": true`. If it says false, the features are probably not scaled. Do not just raise max_iter. Tell me.

```
python train_mlp.py
```
Prints one line per epoch for each learning rate (0.01 then 0.03), then a JSON summary with the chosen learning rate.
Takes a few minutes on a laptop. The best epoch is the one with the highest validation Macro F1. Ties go to the earlier epoch.

```
python diagnostics.py
```
Two checks:
- A: the network trained on 120 windows should reach about 100% training accuracy. It prints OK or LOW.
- B: with shuffled training labels, validation Macro F1 should land near 0.17. It prints OK or TOO HIGH.

If A says LOW: labels, feature order or scaling is wrong. If B says TOO HIGH: something leaks. Stop and send me the output.
Do not carry on to report numbers while either check fails.

```
python report_validation.py
```
Prints the final table: majority, logistic, MLP. Pooled Macro F1 and mean per-user Macro F1 are separate columns because they are different numbers.

## Step 4: what you now have

| File | What it is |
|---|---|
| `scaler.json`, `scaler.joblib` | Scaler fitted on training windows only. Week 4 reuses it. |
| `label_map.json` | still 0, walking 1, crutches 2, walker 3, manual 4, electric 5 |
| `evaluation_contract.json` | Hashes of the train, validation and test window lists, manifests and config, plus library versions |
| `logistic_model.joblib`, `logistic_info.json` | Logistic regression and its convergence info |
| `mlp_initial_state.pt` | The network's starting weights. Federated runs start from these. |
| `mlp_lr0.01_best.pt`, `mlp_lr0.03_best.pt`, `mlp_selected.pt` | Best epoch for each learning rate, and the chosen one |
| `mlp_history_lr*.csv`, `mlp_curves.png` | Loss and validation score per epoch |
| `central_validation_metrics.csv` | The main results table |
| `central_validation_per_user.csv` | Macro F1 per validation user, with classes missing for that user |
| `confusion_*.csv`, `confusion_mlp.png` | Confusion matrices |
| `diagnostics.json` | Results of the two sanity checks |
| `error_analysis_facts.json` | Numbers to use in your error analysis |
| `class_counts_train_val.csv` | Windows per class in train and validation |

## Step 5: error analysis

Copy the template and fill it in:
```
cp error_analysis_template.md error_analysis.md
open -e error_analysis.md
```
Use `error_analysis_facts.json` and `confusion_mlp.png`. Write short plain sentences:
- the three biggest confusions and how many windows each one is
- the weakest users and what they have in common (for example a missing class)
- one line saying validation has only 12 users, so one odd user can move the scores

The paper reports overlap between walking aids and between still and wheelchair classes. You can say whether you see the same,
but call it a comparison with the paper, not a result you proved.

## Step 6: fill in the progress note

```
open -e progress_week3.md
```
Copy the numbers from `central_validation_metrics.csv`. Write the diagnostics results. Note the split switch under open questions.

## Step 7: commit

```
git add *.py *.json *.csv *.md *.png *.pt *.joblib
git status
```
Check that `data/` and zip files are not listed. They are ignored on purpose.
```
git commit -m "Week 3: central baselines on validation"
git tag week3-baselines
git push
git push --tags
```
If GitHub gives a 500 error again, the commit is safe locally. Try again later. Your earlier README commit still needs a push too.

## What Week 3 asks for, and where it is

| Requirement | Where |
|---|---|
| Scaler fit on training only, parameters saved | `week3_prepare.py`, `scaler.json` |
| Fixed six-label map | `label_map.json` |
| Majority baseline | `train_logistic.py`, `pred_majority.csv` |
| Logistic regression C=1, max_iter=1000 | `train_logistic.py` |
| MLP 30-64-6, SGD, batch 64, 50 epochs, lr 0.01 and 0.03 | `train_mlp.py` |
| Initial weights saved for FedAvg | `mlp_initial_state.pt` |
| One shared evaluate() function | `baseline_common.py` |
| Pooled Macro F1, accuracy, per-class recall, confusion matrix | `central_validation_metrics.csv`, `confusion_*.csv` |
| Mean per-user Macro F1 as its own column, absent classes per user | `central_validation_per_user.csv` |
| Overfit and shuffled-label checks | `diagnostics.py`, `diagnostics.json` |
| Evaluated window hash saved | `evaluation_contract.json` |
| No test scoring | No script reads test features |

## Common problems

- `ModuleNotFoundError: torch`: you forgot `conda activate fedhar` or `pip install torch`.
- `AssertionError: training windows changed since the scaler was fitted`: you changed the split after step 3. Rerun `week3_prepare.py` and everything after it.
- `windows from a user missing in split_manifest.csv`: the split file and feature file disagree. Send me `head split_manifest.csv`.
- Logistic regression not converged: send me `logistic_info.json`.
- Anything else: copy the last 15 lines of Terminal output to me.

## Check-in message for your supervisor

Hi, Week 3 is done. I trained a majority baseline, logistic regression and a 30-64-6 MLP on the 36 training users and scored them on the 12 validation users only. No test user was scored. I switched the participant split to the seed-42 permutation method from the Week 3 document, and kept my Week 2 split as a backup. Results are in central_validation_metrics.csv. Is it fine to keep the new split for the federated weeks?
