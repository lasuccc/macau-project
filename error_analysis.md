Week 3 error analysis (validation users only)

All numbers come from the MLP (learning rate 0.03, epoch 49) scored on the 12 validation users. No test user was scored.

Which classes get mixed up

The model gets walking right almost every time (recall 0.98). Crutches is the hardest class at 0.88.

The three biggest mistakes are:
- crutches predicted as walker: 202 windows (7% of crutches windows)
- crutches predicted as manual wheelchair: 131 windows (5%)
- still predicted as electric wheelchair: 130 windows (5%)

Crutches and walker also get mixed up the other way, with 4% of walker windows predicted as crutches. This is the pair I expected. Both are walking aids, and the paper reports overlap between them. The still and electric wheelchair mix-up also matches the paper's overlap between still and the wheelchair classes. My guess is that a person sitting in an electric wheelchair moves the wrist very little, so it looks like standing still. I have not tested that guess.

Logistic regression shows the same weak spots, only worse. Its crutches recall is 0.770 against 0.876 for the MLP.

Which users are hard

The weakest validation users are 71 (macro F1 0.883) and 83 (0.889). Users 72, 102, 107, 80 and 67 sit between 0.913 and 0.919. The best user is 96 at 0.960, so the gap from best to worst is about 0.08. The middle of the group is close to the overall mean of 0.925, so the model works about equally well for most people and no single user pulls the average down. Every validation user has all six classes, so a missing class does not explain any low score.


What this does not tell us is

Validation has only 12 users, so one odd user moves the numbers a lot.

I picked the learning rate and the best epoch using these same validation users, so the validation scores are a little optimistic. The test users are the honest check, and I have not scored them.

The model was still improving slowly at epoch 49 out of the 50-epoch limit, so the limit capped training. A longer run might score a bit higher.
