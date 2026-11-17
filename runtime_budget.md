# Runtime budget

## What I measured
- Rounds run: 10 (seed 42, learning rate 0.03, 10 clients per round, 1 local epoch)
- First round: 0.82 s (includes warm-up, so I left it out of the steady-state average)
- Steady-state round, mean of rounds 2 to 10: 0.07 s
  (training 0.04 s, validation scoring 0.03 s)
- Slowest round: 0.82 s

## Estimate for a 50-round run
0.07 s x 50 = 3 s, about 0.1 minutes

| Plan | Runs | Estimated time |
|---|---|---|
| One 50-round run | 1 | 0.1 min |
| Three seeds | 3 | 0.2 min |
| Five seeds | 5 | 0.3 min |
| Ten seeds | 10 | 0.6 min |

Fill in the Week 5 plan here and check it against the numbers above:
- Planned Week 5 runs: dunno yet
- Total time: one 50-round run takes about 4 seconds, so even ten seeds take well under a minute
- Fits the time I have: as long as the Week 5 runs are of similar siz

## Hardware and settings
- Machine: Apple M2, 8 logical cores
- OS: macOS-26.5.1-arm64-arm-64bit
- Python 3.11.18, PyTorch 2.15.0
- PyTorch threads: 1 (set to 1 on purpose)
- Device: CPU

## Caveat
The estimate comes from 9 rounds. Rounds with different clients can take a little more or less time, because clients have different numbers of windows.
