"""Optional tidy-up for Week 4. The Week 4 notebook also lists run_config.json and a numerical weighting example.
Run once, after Week 4 is finished. Writes week4_run_config.json and weighting_example.md. Reads only Week 4 and Week 3 files."""
import json
import platform
import numpy as np
import pandas as pd
import torch
from baseline_common import HERE, out, sha256_file

H = pd.read_csv(HERE / "fedavg_10round_history.csv")
best = H.loc[H.val_macro_f1.idxmax()]                      # first maximum, so a tie keeps the earlier round
steady = float(H[H["round"] >= 2].round_time_s.mean())
sel = json.loads(out("mlp_selection.json").read_text())
cfg = {"seed": 42, "rounds": int(H["round"].max()), "local_epochs": 1, "lr": sel["selected_lr"], "clients_per_round": 10,
       "batch_size": 64, "best_round": int(best["round"]), "best_validation_macro_f1": float(best.val_macro_f1),
       "seconds_per_round_steady": steady, "estimated_50_round_seconds": 50 * steady,
       "feature_file_sha256": sha256_file(HERE / "features_watch.csv.gz"),
       "split_file_sha256": sha256_file(HERE / "split_manifest.csv"),
       "initial_state_sha256": sha256_file(HERE / "mlp_initial_state.pt"),
       "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "threads": torch.get_num_threads()}
(HERE / "week4_run_config.json").write_text(json.dumps(cfg, indent=2))

counts = [int(c) for c in H.client_counts.iloc[0].split(";")]
w = np.array(counts) / sum(counts)
(HERE / "weighting_example.md").write_text(f"""# How the sample weights work (numerical example)

Each client's weights count in proportion to its number of training windows, n_k.
New weight = sum(n_k x weight_k) / sum(n_k).

## Toy example
Two models have the value 2 and 8. Model A has 1 window, model B has 3 windows.
(1 x 2 + 3 x 8) / (1 + 3) = 26 / 4 = 6.5.
A plain mean would give 5. The bigger client pulls the result toward its own value.
The check in aggregation_checks.txt confirms the code returns 6.5.

## Real example, round 1 of my 10-round run
Clients picked: {H.selected_clients.iloc[0].replace(';', ', ')}
Their window counts: {', '.join(map(str, counts))}
Total windows: {sum(counts)}
Each client's weight is its count divided by the total. They range from {w.min():.4f} to {w.max():.4f}.
A plain mean would give every client {1 / len(counts):.4f}.
The weights are almost equal because every client has between about {min(counts)} and {max(counts)} windows.
So on this data the weighted average is close to the plain average. The 6.5 test is what shows the weighting is coded correctly.

The weights use the number of unique windows, not the number of batches and not the number of local epochs.
Only the clients picked in that round are in the sum and in the denominator.
""")
print("Wrote week4_run_config.json and weighting_example.md")
print(f"Best round {int(best['round'])}, validation macro F1 {float(best.val_macro_f1):.4f}, steady {steady:.3f}s per round")
