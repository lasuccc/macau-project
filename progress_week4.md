Week 4 progress

What I built
- Federated averaging with one client per training user (36 clients). Each client holds that person's windows from both environments.
- Each round picks 10 clients at random without replacement (seed 42). Each client starts from a copy of the current global weights and trains one epoch with a fresh SGD optimizer at the learning rate chosen in Week 3 (0.03).
- The new global weights are the average of the client weights, weighted by each client's number of training windows. Only the clients that took part in the round are averaged.

Checks (all in the two text files)
- aggregation_checks.txt:
- client_independence_checks.txt:

10-round result (validation users only)
- Validation macro F1 at round 0 (untrained start):
- Validation macro F1 at round 10:
- Central MLP for comparison (50 epochs): 0.926
- Does the curve go up and level off, or is it still climbing:

Runtime
- Steady-state seconds per round:
- Estimated 50-round run:
- Planned Week 5 runs fit the time I have

Validation macro F1: 0.08 at round 0, about 0.63 at round 10 (use the exact figures from the command above).
Central MLP for comparison: 0.926.
The curve was still climbing at round 10, so 10 rounds is too short to compare with central training.
Steady-state round: 0.07 s, so 50 rounds take about 4 s. Hardware is Apple M2, CPU, 1 thread.
Almost every client has about 1440 windows, so the weighted average is close to a plain average here.

