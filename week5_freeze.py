"""Step 2: choose the FedAvg candidate on VALIDATION results and freeze both configurations.
After this file exists, test scoring is allowed. Before it exists, it is blocked."""
import datetime
import json
import sys
import pandas as pd
from baseline_common import BASE, HERE, out, sha256_file
from week5_common import CFG, fed_lr

if (HERE / "test_access_log.txt").exists() or (HERE / "frozen_config.json").exists():
    sys.exit("Stop: test scoring has already happened or the freeze already exists. Do not freeze again after seeing test results.")

S = pd.read_csv(HERE / "candidates_validation_summary.csv")
# Predeclared rule: highest best-round validation macro F1. A tie goes to the smaller number of local epochs.
S = S.sort_values(["best_val_macro_f1", "candidate_local_epochs"], ascending=[False, True]).reset_index(drop=True)
win = S.iloc[0]
E = int(win.candidate_local_epochs)
sel = json.loads(out("mlp_selection.json").read_text())
clr = str(sel["selected_lr"])
crun = sel["runs"][clr]
contract = json.loads(out("evaluation_contract.json").read_text())
n_train = contract["n_train_windows"]
epochs_cap = BASE["mlp"]["max_epochs"]
files = ["mlp_selected.pt", f"candidate_E{E}_best.pt", "candidates_validation_summary.csv", "mlp_selection.json",
         "scaler.json", "evaluation_contract.json"] + [f"candidate_E{e}_history.csv" for e in CFG["candidate_local_epochs"]]
frozen = {
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "note": "Frozen from validation results only. Test users had not been scored at this point.",
    "selection_rule": "Federated: highest best-round validation macro F1; tie goes to fewer local epochs. Best round = highest validation macro F1, tie keeps the earlier round. Central: Week 3 rule.",
    "seed": CFG["seed"],
    "central": {"checkpoint": "mlp_selected.pt", "lr": sel["selected_lr"], "best_epoch": crun["best_epoch"],
                "epochs_trained": epochs_cap, "val_macro_f1": crun["val_macro_f1"],
                "optimizer_steps_to_best": crun["updates_to_best"], "optimizer_steps_total": crun["total_updates"],
                "examples_to_best": crun["best_epoch"] * n_train, "examples_total": epochs_cap * n_train,
                "runtime_s": crun["runtime_s"]},
    "federated": {"candidate_local_epochs": E, "checkpoint": f"candidate_E{E}_best.pt", "lr": fed_lr(),
                  "clients_per_round": CFG["clients_per_round"], "rounds_cap": int(win.rounds_cap),
                  "selected_round": int(win.best_round), "val_macro_f1": float(win.best_val_macro_f1)},
    "all_candidates": S.to_dict("records"),
    "data": {k: contract[k] for k in ["train_window_hash", "val_window_hash", "test_window_id_hash", "split_manifest_hash",
                                       "window_manifest_file_hash"]},
    "file_sha256": {f: sha256_file(HERE / f) for f in files}}
(HERE / "frozen_config.json").write_text(json.dumps(frozen, indent=2, default=float))
print(S[["candidate_local_epochs", "best_round", "best_val_macro_f1", "last_val_macro_f1", "runtime_total_s"]].round(4).to_string(index=False))
print(f"\nFrozen. Federated: {E} local epoch(s), lr {fed_lr()}, selected round {int(win.best_round)} of {int(win.rounds_cap)}.")
print(f"Frozen. Central: lr {sel['selected_lr']}, epoch {crun['best_epoch']} of {epochs_cap}.")
print("Test users have not been scored. Next: python week5_main.py")
