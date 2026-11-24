"""Step 3: the main comparison. Central MLP and the frozen FedAvg model, scored on the validation and test users."""
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from baseline_common import HERE, LABELS, SEED, load_scaled, out, sha256_file
from week5_common import CFG, evaluate_global, load_state, load_test

frozen = json.loads((HERE / "frozen_config.json").read_text())
for f, h in frozen["file_sha256"].items():
    assert sha256_file(HERE / f) == h, f"{f} changed after the freeze"
print("OK   frozen files unchanged since the freeze")
tr, va = load_scaled()
te = load_test("week5_main.py")
contract = json.loads(out("evaluation_contract.json").read_text())
assert contract["val_window_hash"] == frozen["data"]["val_window_hash"]
FC, FF = frozen["central"], frozen["federated"]
models = {"Central MLP": load_state(HERE / FC["checkpoint"]), f"FedAvg MLP (E={FF['candidate_local_epochs']})": load_state(HERE / FF["checkpoint"])}
hist = pd.read_csv(HERE / f"candidate_E{FF['candidate_local_epochs']}_history.csv")
fb = hist[hist["round"] == FF["selected_round"]].iloc[0]
n_train = contract["n_train_windows"]
info = {"Central MLP": dict(unit="epochs", selected=FC["best_epoch"], trained=FC["epochs_trained"], local_epochs="", clients_per_round="",
                            lr=FC["lr"], steps_sel=FC["optimizer_steps_to_best"], steps_tot=FC["optimizer_steps_total"],
                            ex_sel=FC["examples_to_best"], ex_tot=FC["examples_total"], runtime=FC["runtime_s"]),
        list(models)[1]: dict(unit="rounds", selected=FF["selected_round"], trained=FF["rounds_cap"], local_epochs=FF["candidate_local_epochs"],
                              clients_per_round=FF["clients_per_round"], lr=FF["lr"], steps_sel=int(fb.optimizer_steps_cum),
                              steps_tot=int(hist.optimizer_steps_cum.iloc[-1]), ex_sel=int(fb.examples_cum),
                              ex_tot=int(hist.examples_cum.iloc[-1]), runtime=float(hist.round_time_s.sum()))}
rows, per_user, cms = [], [], {}
for name, state in models.items():
    for split, pack, mh in (("validation", va, contract["val_window_hash"]), ("test", te, contract["test_window_id_hash"])):
        res, pu, cm = evaluate_global(state, pack)
        i = info[name]
        rows.append({"model": name, "split": split, "n_eval_windows": res["n_windows"], "n_eval_users": res["n_users"],
                     "eval_manifest_hash": mh[:16], "accuracy": res["accuracy"], "macro_f1_pooled": res["macro_f1_pooled"],
                     "macro_f1_user_mean": res["macro_f1_user_mean"],
                     **{f"recall_{l}": res[f"recall_{l}"] for l in LABELS},
                     "training_unit": i["unit"], "units_selected": i["selected"], "units_trained": i["trained"],
                     "local_epochs": i["local_epochs"], "clients_per_round": i["clients_per_round"], "lr": i["lr"],
                     "optimizer_steps_to_selected": i["steps_sel"], "optimizer_steps_total": i["steps_tot"],
                     "examples_to_selected": i["ex_sel"], "examples_total": i["ex_tot"], "runtime_s_total": round(i["runtime"], 2),
                     "train_users": contract["n_train_users"], "train_windows": n_train, "seed": CFG["seed"]})
        pu.insert(0, "split", split); pu.insert(0, "model", name)
        per_user.append(pu)
        cms[(name, split)] = cm
M = pd.DataFrame(rows)
# sanity: the saved checkpoints must reproduce their frozen validation scores
cv = M[(M.model == "Central MLP") & (M.split == "validation")].macro_f1_pooled.iloc[0]
fv = M[(M.model != "Central MLP") & (M.split == "validation")].macro_f1_pooled.iloc[0]
assert abs(cv - FC["val_macro_f1"]) < 1e-9, "central checkpoint does not reproduce its Week 3 validation score"
assert abs(fv - FF["val_macro_f1"]) < 1e-9, "federated checkpoint does not reproduce its frozen validation score"
print("OK   both checkpoints reproduce their frozen validation scores")
assert M[M.split == "test"].eval_manifest_hash.nunique() == 1 and M[M.split == "test"].n_eval_windows.nunique() == 1
print("OK   both models were scored on the identical test windows")
M.to_csv(HERE / "main_results.csv", index=False)
P = pd.concat(per_user)
P.to_csv(HERE / "per_user_metrics.csv", index=False)
names = list(models)
t = P[P.split == "test"].pivot(index="user", columns="model", values="macro_f1")
cmp_ = pd.DataFrame({"user": t.index, "central_macro_f1": t[names[0]].values, "fedavg_macro_f1": t[names[1]].values})
cmp_["fedavg_minus_central"] = cmp_.fedavg_macro_f1 - cmp_.central_macro_f1
cmp_.to_csv(HERE / "test_per_user_comparison.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, name in zip(axs, names):
    cm = cms[(name, "test")]; rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(LABELS, rotation=45, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels(LABELS)
    for a in range(6):
        for b in range(6):
            ax.text(b, a, f"{rn[a, b]:.2f}", ha="center", va="center", color="white" if rn[a, b] > 0.5 else "black", fontsize=7)
    ax.set_title(f"{name}, test users"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
fig.tight_layout(); fig.savefig(HERE / "confusion_test_central_vs_fedavg.png", dpi=150); plt.close(fig)
shutil.copy(HERE / FC["checkpoint"], HERE / "selected_central.pt")
shutil.copy(HERE / FF["checkpoint"], HERE / "selected_fedavg.pt")
shutil.copy(HERE / "mlp_history_lr{}.csv".format(FC["lr"]), HERE / "central_history.csv")

show = ["model", "split", "n_eval_windows", "n_eval_users", "accuracy", "macro_f1_pooled", "macro_f1_user_mean"]
print("\n" + M[show].round(4).to_string(index=False))
print("\nWork done (to the selected checkpoint / in total):")
print(M[M.split == "test"][["model", "units_selected", "units_trained", "optimizer_steps_to_selected", "examples_to_selected", "runtime_s_total"]].to_string(index=False))
print(f"\nTest users where FedAvg beats central: {(cmp_.fedavg_minus_central > 0).sum()} of {len(cmp_)}")
