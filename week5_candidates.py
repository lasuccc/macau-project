"""Step 1: the two FedAvg candidates (1 or 3 local epochs), same learning rate and clients, 50-round cap.
Scored on validation users only. Nothing here touches test users."""
import json
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from baseline_common import HERE
from week5_common import CFG, build_clients, fed_lr, out, run_fedavg, save_ckpt

clients, va = build_clients()
lr, rounds, seed = fed_lr(), CFG["rounds"], CFG["seed"]
print(f"{len(clients)} clients, lr {lr} (fixed), {CFG['clients_per_round']} clients per round, {rounds} rounds, seed {seed}")
summary, hists = [], {}
for E in CFG["candidate_local_epochs"]:
    r = run_fedavg(clients, va, lr, E, rounds, seed, label=f"E{E}")
    H = r["history"]
    H.to_csv(HERE / f"candidate_E{E}_history.csv", index=False)
    meta = dict(local_epochs=E, lr=lr, seed=seed)
    save_ckpt(HERE / f"candidate_E{E}_best.pt", r["best_state"], round=r["best_round"], **meta)
    save_ckpt(HERE / f"candidate_E{E}_last.pt", r["last_state"], round=rounds, **meta)
    b = H[H["round"] == r["best_round"]].iloc[0]
    summary.append({"candidate_local_epochs": E, "lr": lr, "clients_per_round": CFG["clients_per_round"],
                    "rounds_cap": rounds, "seed": seed, "best_round": r["best_round"],
                    "best_val_macro_f1": b.val_macro_f1, "best_val_accuracy": b.val_accuracy,
                    "last_val_macro_f1": H.val_macro_f1.iloc[-1],
                    "optimizer_steps_to_best": int(b.optimizer_steps_cum), "optimizer_steps_total": int(H.optimizer_steps_cum.iloc[-1]),
                    "examples_total": int(H.examples_cum.iloc[-1]), "runtime_total_s": H.round_time_s.sum(),
                    "best_checkpoint_id": b.checkpoint_id})
    hists[E] = H
    print(f"E={E}: best validation round {r['best_round']}, macro F1 {b.val_macro_f1:.4f}, last round {H.val_macro_f1.iloc[-1]:.4f}, "
          f"{H.round_time_s.sum():.1f}s")
S = pd.DataFrame(summary)
S.to_csv(HERE / "candidates_validation_summary.csv", index=False)

central = json.loads(out("mlp_selection.json").read_text())
cf1 = central["runs"][str(central["selected_lr"])]["val_macro_f1"]
fig, ax = plt.subplots(figsize=(6.5, 4))
for E, H in hists.items():
    ax.plot(H["round"], H.val_macro_f1, label=f"FedAvg, {E} local epoch(s)")
ax.axhline(cf1, color="grey", linestyle="--", label=f"central MLP (validation {cf1:.3f})")
ax.set_xlabel("round"); ax.set_ylabel("validation macro F1"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "fedavg_candidates_validation.png", dpi=150); plt.close(fig)

# Reproducibility check: the E=1 candidate should repeat the Week 4 prototype for its first 10 rounds.
p = HERE / "fedavg_10round_history.csv"
if p.exists() and lr == pd.read_csv(p).get("lr", pd.Series([lr])).iloc[0] and 1 in hists:
    W4 = pd.read_csv(p)
    n = min(len(W4), 10)
    same = list(W4.checkpoint_id[:n]) == list(hists[1].checkpoint_id[:n])
    print("Week 4 prototype repeated exactly (same checkpoint IDs for rounds 1 to", n, "):", "YES" if same else "NO, check the learning rate")
print("\nNext: python week5_freeze.py")
