"""Step 2: the 10-round FedAvg prototype. Seed 42, the learning rate chosen in Week 3, one local epoch.
  python run_fedavg.py              the real 10-round run
  python run_fedavg.py --rounds 2   quick trial (files get a quick_ prefix, nothing is overwritten)"""
import argparse
import os
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fedavg_common import (CLIENTS_PER_ROUND, HERE, SEED, build_clients, evaluate_global, initial_state,
                           run_round, select_clients, selected_lr, state_id)

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=10)
args = ap.parse_args()
P = "" if args.rounds == 10 else "quick_"
ckpt_dir = HERE / f"{P}fedavg_checkpoints"
ckpt_dir.mkdir(exist_ok=True)

clients, va = build_clients()
lr = selected_lr()
state = initial_state()
rng = np.random.default_rng(SEED)
print(f"{len(clients)} clients, {sum(c['n'] for c in clients.values())} training windows, lr {lr}, "
      f"{min(CLIENTS_PER_ROUND, len(clients))} clients per round, {torch.get_num_threads()} thread(s)")

res0, _, _ = evaluate_global(state, va)
pd.Series({"checkpoint_id": state_id(state), **res0}).to_json(HERE / f"{P}fedavg_round0_start.json", indent=1)
print(f"round  0 (start, untrained)  val macro F1 {res0['macro_f1_pooled']:.4f}")

rows = []
for r in range(1, args.rounds + 1):
    t0 = time.perf_counter()
    sel = select_clients(rng, list(clients))
    state, counts, train_loss = run_round(state, clients, sel, lr, r)
    t1 = time.perf_counter()
    res, per_user, cm = evaluate_global(state, va)
    t2 = time.perf_counter()
    cid = state_id(state)
    torch.save({"state_dict": state, "round": r, "checkpoint_id": cid, "selected_clients": sel, "lr": lr},
               ckpt_dir / f"round_{r:02d}_{cid}.pt")
    rows.append({"round": r, "selected_clients": ";".join(map(str, sel)), "client_counts": ";".join(map(str, counts)),
                 "total_windows": int(sum(counts)), "train_loss": train_loss, "val_loss": res["val_loss"],
                 "val_accuracy": res["accuracy"], "val_macro_f1": res["macro_f1_pooled"],
                 "val_macro_f1_user_mean": res["macro_f1_user_mean"],
                 "train_time_s": t1 - t0, "eval_time_s": t2 - t1, "round_time_s": t2 - t0, "checkpoint_id": cid})
    print(f"round {r:2d}  clients {sel}  train loss {train_loss:.4f}  val loss {res['val_loss']:.4f}  "
          f"val macro F1 {res['macro_f1_pooled']:.4f}  {t2 - t0:.2f}s  ckpt {cid}")

H = pd.DataFrame(rows)
H.to_csv(HERE / f"{P}fedavg_10round_history.csv", index=False)
torch.save({"state_dict": state, "round": args.rounds, "checkpoint_id": rows[-1]["checkpoint_id"], "lr": lr,
            "seed": SEED, "note": "FedAvg prototype after the last round. Not the final model."},
           HERE / f"{P}fedavg_prototype.pt")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
x = [0] + list(H["round"])
axs[0].plot(x, [res0["macro_f1_pooled"]] + list(H.val_macro_f1), marker="o")
axs[0].set_xlabel("round"); axs[0].set_ylabel("validation macro F1"); axs[0].set_title("FedAvg validation score")
axs[1].plot(x, [res0["val_loss"]] + list(H.val_loss), marker="o", label="validation loss")
axs[1].plot(H["round"], H.train_loss, marker="s", label="client training loss (weighted)")
axs[1].set_xlabel("round"); axs[1].set_ylabel("loss"); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / f"{P}fedavg_validation_curve.png", dpi=150); plt.close(fig)

# ---- runtime budget ----
try:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
except Exception:
    cpu = ""
steady = H[H["round"] >= 2] if len(H) >= 3 else H
per_round = float(steady.round_time_s.mean())
est50 = per_round * 50
first = float(H.round_time_s.iloc[0])
(HERE / f"{P}runtime_budget.md").write_text(f"""# Runtime budget

## What I measured
- Rounds run: {args.rounds} (seed {SEED}, learning rate {lr}, {min(CLIENTS_PER_ROUND, len(clients))} clients per round, 1 local epoch)
- First round: {first:.2f} s (includes warm-up, so I left it out of the steady-state average)
- Steady-state round, mean of rounds {int(steady['round'].min())} to {int(steady['round'].max())}: {per_round:.2f} s
  (training {steady.train_time_s.mean():.2f} s, validation scoring {steady.eval_time_s.mean():.2f} s)
- Slowest round: {H.round_time_s.max():.2f} s

## Estimate for a 50-round run
{per_round:.2f} s x 50 = {est50:.0f} s, about {est50 / 60:.1f} minutes

| Plan | Runs | Estimated time |
|---|---|---|
| One 50-round run | 1 | {est50 / 60:.1f} min |
| Three seeds | 3 | {3 * est50 / 60:.1f} min |
| Five seeds | 5 | {5 * est50 / 60:.1f} min |
| Ten seeds | 10 | {10 * est50 / 60:.1f} min |

Fill in the real Week 5 plan here and check it against the numbers above:
- Planned Week 5 runs:
- Total estimated time:
- Fits the time I have: yes / no

## Hardware and settings
- Machine: {cpu or platform.processor() or 'unknown'}, {os.cpu_count()} logical cores
- OS: {platform.platform()}
- Python {platform.python_version()}, PyTorch {torch.__version__}
- PyTorch threads: {torch.get_num_threads()} (set to 1 on purpose)
- Device: CPU

## Caveat
The estimate comes from {len(steady)} rounds. Rounds with different clients can take a little more or less time, because clients have different numbers of windows.
""")
print(f"\nSteady-state round {per_round:.2f}s -> 50 rounds about {est50 / 60:.1f} min. Files written with prefix '{P}'.")
