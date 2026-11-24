"""Step 5: the mixing control. Deal the training windows into pseudo-clients of the original sizes, then retrain FedAvg.
  python week5_mixing.py            the saved-seed control (this is the one that goes in natural_vs_mixed.csv)
  python week5_mixing.py --sweep    also repeat natural vs mixed for 5 seeds on VALIDATION ONLY (extra evidence for noise)"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from baseline_common import HERE, out, sha256_file, sha256_ids
from week5_common import (CFG, build_clients, client_summary, evaluate_global, fed_lr, load_state, load_test, run_fedavg,
                          save_ckpt, state_id)

frozen = json.loads((HERE / "frozen_config.json").read_text())
for f, h in frozen["file_sha256"].items():
    assert sha256_file(HERE / f) == h, f"{f} changed after the freeze"
FF = frozen["federated"]
E, lr, rounds, seed, mix_seed = FF["candidate_local_epochs"], FF["lr"], FF["rounds_cap"], CFG["seed"], CFG["mix_seed"]
contract = json.loads(out("evaluation_contract.json").read_text())
clients, va = build_clients()
old_ids = sorted(clients)


def make_mixed(clients, s):
    """Sort window IDs, shuffle once with seed s, deal into pseudo-clients with the original client sizes."""
    ids = np.concatenate([clients[c]["ids"] for c in old_ids]).astype(str)
    owner = np.concatenate([[c] * clients[c]["n"] for c in old_ids])
    X = torch.cat([clients[c]["X"] for c in old_ids]); y = torch.cat([clients[c]["y"] for c in old_ids])
    order = np.argsort(ids, kind="stable")                      # sorted window IDs
    shuffled = order[np.random.default_rng(s).permutation(len(order))]
    new, pos, new_owner = {}, 0, np.empty(len(ids), dtype=int)
    for c in old_ids:
        take = shuffled[pos:pos + clients[c]["n"]]; pos += clients[c]["n"]
        new[c] = {"X": X[take], "y": y[take], "n": int(len(take)), "ids": ids[take], "env_counts": {}}
        new_owner[take] = c
    return new, ids, owner, new_owner, X, y


mixed, wid, old_owner, new_owner, Xall, yall = make_mixed(clients, mix_seed)

# ---- the checks the page asks for ----
checks, failed = [], []
def check(ok, text, detail=""):
    checks.append(f"{'PASS' if ok else 'FAIL'}  {text}" + (f"  ({detail})" if detail else ""))
    if not ok: failed.append(text)
mids = np.concatenate([mixed[c]["ids"] for c in old_ids])
check(len(set(mids)) == len(mids) == len(wid), "every training window appears exactly once after the reassignment", f"{len(mids)} windows")
check(set(mids) == set(wid) and sha256_ids(mids) == contract["train_window_hash"], "the set of training windows is unchanged (matches the Week 3 training hash)")
check(all(mixed[c]["n"] == clients[c]["n"] for c in old_ids), "every pseudo-client has the same size as the original client it replaces")
check(np.array_equal(np.bincount(torch.cat([mixed[c]["y"] for c in old_ids]).numpy()), np.bincount(yall.numpy())) and
      abs(float(torch.cat([mixed[c]["X"] for c in old_ids]).double().sum() - Xall.double().sum())) < 1e-3,
      "the pooled training data is the same multiset (class counts and feature sums match)")
check(sha256_ids(va["ids"]) == contract["val_window_hash"], "validation windows are identical to Week 3")
te = load_test("week5_mixing.py")
check(sha256_ids(te["ids"]) == contract["test_window_id_hash"] == frozen["data"]["test_window_id_hash"], "test windows are identical to Week 3")
check(not (set(mids) & set(va["ids"])) and not (set(mids) & set(te["ids"])), "no validation or test window entered a pseudo-client")
stay = float((old_owner == new_owner).mean())
check(True, "windows that landed back in their own client by chance", f"{stay:.3f}, about 1/{len(old_ids)} = {1 / len(old_ids):.3f} expected")
M = pd.DataFrame({"window_id": wid, "old_client": old_owner, "new_client": new_owner, "mix_seed": mix_seed}).sort_values("window_id")
M.to_csv(HERE / "reassignment_manifest.csv", index=False)
(HERE / "reassignment_checks.txt").write_text("\n".join(checks) + "\n")
print("\n".join(checks))
if failed: sys.exit("A reassignment check failed. Stop.")

# ---- heterogeneity of natural vs mixed ----
sn, _, _ = client_summary(clients); sm, _, _ = client_summary(mixed)
pd.DataFrame([{"condition": "natural", **sn}, {"condition": "mixed", **sm}]).to_csv(HERE / "mixing_heterogeneity_summary.csv", index=False)

# ---- primary run: natural and mixed, same schedule, same sampler seed, same settings ----
nat = run_fedavg(clients, va, lr, E, rounds, seed, "natural")
assert state_id(nat["best_state"]) == json.loads(json.dumps(torch.load(HERE / FF["checkpoint"], weights_only=True)["checkpoint_id"])), \
    "the natural run did not reproduce the frozen FedAvg checkpoint"
print("OK   natural run reproduces the frozen FedAvg checkpoint exactly")
mix = run_fedavg(mixed, va, lr, E, rounds, seed, "mixed")
mix["history"].to_csv(HERE / "mixed_history.csv", index=False)
save_ckpt(HERE / "mixed_best.pt", mix["best_state"], round=mix["best_round"], local_epochs=E, lr=lr, seed=seed, mix_seed=mix_seed)
rows = []
for cond, r, cl in (("natural", nat, clients), ("mixed", mix, mixed)):
    H = r["history"]; b = H[H["round"] == r["best_round"]].iloc[0]
    rt, _, _ = evaluate_global(r["best_state"], te)
    rows.append({"condition": cond, "training_seed": seed, "mix_seed": mix_seed if cond == "mixed" else "", "local_epochs": E, "lr": lr,
                 "rounds": rounds, "clients": len(cl), "clients_per_round": CFG["clients_per_round"],
                 "train_windows": sum(c["n"] for c in cl.values()), "best_round": r["best_round"],
                 "val_macro_f1_best": b.val_macro_f1, "val_macro_f1_last": H.val_macro_f1.iloc[-1],
                 "test_macro_f1_pooled": rt["macro_f1_pooled"], "test_macro_f1_user_mean": rt["macro_f1_user_mean"],
                 "test_accuracy": rt["accuracy"], "test_windows": rt["n_windows"], "test_users": rt["n_users"],
                 "optimizer_steps_total": int(H.optimizer_steps_cum.iloc[-1]), "examples_total": int(H.examples_cum.iloc[-1]),
                 "runtime_s": round(H.round_time_s.sum(), 2), "checkpoint_id": b.checkpoint_id})
R = pd.DataFrame(rows)
diff = {"condition": "mixed minus natural"}
for k in ["val_macro_f1_best", "val_macro_f1_last", "test_macro_f1_pooled", "test_macro_f1_user_mean", "test_accuracy"]:
    diff[k] = R[k].iloc[1] - R[k].iloc[0]
R = pd.concat([R, pd.DataFrame([diff])], ignore_index=True)
R.to_csv(HERE / "natural_vs_mixed.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(nat["history"]["round"], nat["history"].val_macro_f1, label="natural clients (one person each)", color="#0072B2")
ax.plot(mix["history"]["round"], mix["history"].val_macro_f1, label="mixed pseudo-clients", color="#D55E00")
ax.set_xlabel("round"); ax.set_ylabel("validation macro F1"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "natural_vs_mixed_curve.png", dpi=150); plt.close(fig)
print("\n" + R[["condition", "best_round", "val_macro_f1_best", "val_macro_f1_last", "test_macro_f1_pooled", "test_macro_f1_user_mean"]].round(4).to_string(index=False))
print("\nHeterogeneity (natural vs mixed):")
print(pd.read_csv(HERE / "mixing_heterogeneity_summary.csv").round(4).to_string(index=False))

# ---- optional: noise estimate, validation only ----
if "--sweep" in sys.argv:
    srows = []
    for s in CFG["seed_sweep"]:
        m_s, *_ = make_mixed(clients, s)
        for cond, cl in (("natural", clients), ("mixed", m_s)):
            r = run_fedavg(cl, va, lr, E, rounds, s, cond)
            H = r["history"]
            srows.append({"condition": cond, "seed": s, "local_epochs": E, "lr": lr, "rounds": rounds,
                          "best_round": r["best_round"], "val_macro_f1_best": H.val_macro_f1.max(), "val_macro_f1_last": H.val_macro_f1.iloc[-1]})
        print(f"seed {s} done")
    S = pd.DataFrame(srows)
    S.to_csv(HERE / "natural_vs_mixed_seeds.csv", index=False)
    agg = S.groupby("condition")[["val_macro_f1_best", "val_macro_f1_last"]].agg(["mean", "std"]).round(4)
    agg.to_csv(HERE / "natural_vs_mixed_seed_summary.csv")
    print("\nValidation only, 5 seeds (mean and standard deviation):\n", agg.to_string())
    p = S.pivot(index="seed", columns="condition", values="val_macro_f1_last")
    print("\nMixed minus natural at the last round, per seed:", (p["mixed"] - p["natural"]).round(4).to_dict())
print("\nTest access log:\n" + (HERE / "test_access_log.txt").read_text())
