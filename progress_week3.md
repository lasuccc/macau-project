Week 3 progress

What I ran
1. resplit_like_doc.py (compared the two split methods, then switched to the document's method before any training)
2. week3_prepare.py (leakage checks, scaler fitted on training windows only, hashes saved)
3. train_logistic.py (majority reference and logistic regression)
4. train_mlp.py (30-64-6 network, learning rates 0.01 and 0.03)
5. diagnostics.py (overfit a small subset, shuffled labels)
6. report_validation.py (the validation table, confusion matrix, per-user scores)

Numbers (copy from central_validation_metrics.csv)
- Majority baseline, pooled macro F1:
- Logistic regression, pooled macro F1:
- MLP, pooled macro F1 and chosen learning rate:
- Mean per-user macro F1 for each (separate column, not the same as pooled):

Checks
- Overfit test:
- Shuffled-label test:
- Logistic regression converged:
- No test user scored: yes


- Switched the split to the document's method. Old split kept as split_manifest_week2_groupshuffle.csv.

Majority 0.048, logistic 0.886, MLP 0.926 (lr 0.03, epoch 49). Mean per-user macro F1 is 0.048, 0.884 and 0.925.
Both diagnostics okay. First overfit run used 1500 epochs and reached 0.95, and raised it to 5000 epochs, which reached 1.0.
The MLP was still improving at epoch 49 of the 50 epoch limit, so the limit capped training.
The split switched