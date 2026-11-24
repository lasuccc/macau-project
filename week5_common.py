"""Shared Week 5 pieces: config, a FedAvg runner that counts steps and examples, a test loader with a freeze guard."""
import copy
import datetime
import json
import math
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from baseline_common import (DEVICE, FEATURES, HERE, LABEL_ID, LABELS, N_CLASSES, evaluate, load_scaled, out,
                             read_features, sha256_file, sha256_ids)
from fedavg_common import (BATCH, aggregate, build_clients, evaluate_global, initial_state, make_model,
                           select_clients, selected_lr, state_id)

CFG = json.loads((HERE / "week5_config.json").read_text())


def fed_lr():
    """The federated learning rate stays fixed for every candidate. Default: the central learning rate chosen in Week 3."""
    return float(CFG["fed_lr"]) if CFG["fed_lr"] is not None else selected_lr()


def local_train5(global_state, client, lr, round_idx, client_id, epochs, seed):
    """Same as Week 4 local_train, with a seed argument and a step counter. For seed 42 and epochs 1 it gives the same result."""
    model = make_model()
    model.load_state_dict(copy.deepcopy(global_state))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    gen = torch.Generator().manual_seed(seed * 1_000_003 + round_idx * 1009 + client_id)
    loss_fn = nn.CrossEntropyLoss()
    n, total, steps = client["n"], 0.0, 0
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(client["X"][idx]), client["y"][idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            steps += 1
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, total / (n * epochs), steps


def run_fedavg(clients, va, lr, epochs, rounds, seed, label=""):
    """FedAvg from the shared initial state. Every round is scored on the validation users only.
    The best round is the one with the highest validation macro F1; a tie keeps the earlier round."""
    state = initial_state()
    rng = np.random.default_rng(seed)
    ids = sorted(clients)
    rows, best_f1, best_round, best_state = [], -1.0, None, None
    steps_cum = ex_cum = 0
    time_cum = 0.0
    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        sel = select_clients(rng, ids, CFG["clients_per_round"])
        states, counts, losses, steps = [], [], [], 0
        for cid in sel:
            s, l, st = local_train5(state, clients[cid], lr, r, cid, epochs, seed)
            states.append(s); counts.append(clients[cid]["n"]); losses.append(l); steps += st
        state = aggregate(states, counts)
        res, _, _ = evaluate_global(state, va)
        dt = time.perf_counter() - t0
        steps_cum += steps
        ex_cum += epochs * sum(counts)
        time_cum += dt
        rows.append({"round": r, "seed": seed, "local_epochs": epochs, "lr": lr,
                     "selected_clients": ";".join(map(str, sel)), "client_counts": ";".join(map(str, counts)),
                     "optimizer_steps_round": steps, "optimizer_steps_cum": steps_cum,
                     "examples_round": epochs * sum(counts), "examples_cum": ex_cum,
                     "train_loss": float(np.average(losses, weights=counts)), "val_loss": res["val_loss"],
                     "val_accuracy": res["accuracy"], "val_macro_f1": res["macro_f1_pooled"],
                     "val_macro_f1_user_mean": res["macro_f1_user_mean"],
                     "round_time_s": dt, "runtime_cum_s": time_cum, "checkpoint_id": state_id(state)})
        if res["macro_f1_pooled"] > best_f1:
            best_f1, best_round, best_state = res["macro_f1_pooled"], r, copy.deepcopy(state)
    return {"history": pd.DataFrame(rows), "best_round": best_round, "best_state": best_state,
            "last_state": state, "label": label}


def save_ckpt(path, state, **meta):
    torch.save({"state_dict": state, "checkpoint_id": state_id(state), **meta}, path)


def load_state(path):
    return torch.load(path, weights_only=True)["state_dict"]


def load_test(caller):
    """Test windows, scaled with the saved Week 3 scaler. Blocked until the configurations are frozen.
    Every access is written to test_access_log.txt."""
    if not (HERE / "frozen_config.json").exists():
        raise SystemExit("Blocked: freeze the configurations first (python week5_freeze.py). Test scoring comes after the freeze.")
    F = read_features(DEVICE)
    M = pd.read_csv(HERE / "window_manifest.csv.gz", usecols=["window_id", "user", "activity", "device", "status"])
    M = M[(M.device == DEVICE) & (M.status == "kept")]
    S = pd.read_csv(HERE / "split_manifest.csv")[["user", "split"]]
    D = F.merge(M[["window_id", "user", "activity"]], on="window_id", validate="one_to_one")
    D = D.merge(S, on="user", how="left", validate="many_to_one")
    T = D[D.split == "test"].reset_index(drop=True)
    contract = json.loads(out("evaluation_contract.json").read_text())
    assert sha256_ids(T.window_id) == contract["test_window_id_hash"], "test windows differ from the Week 3 contract"
    assert np.isfinite(T[FEATURES].to_numpy()).all()
    sc = json.loads(out("scaler.json").read_text())
    assert sc["features"] == FEATURES
    X = ((T[FEATURES].to_numpy() - np.array(sc["mean"])) / np.array(sc["scale"])).astype(np.float32)
    with open(HERE / "test_access_log.txt", "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}  {caller}  test windows {len(T)}  "
                f"hash {contract['test_window_id_hash'][:16]}\n")
    return {"X": X, "y": T.activity.map(LABEL_ID).to_numpy().astype(int), "users": T.user.to_numpy(), "ids": T.window_id.to_numpy()}


def client_summary(clients):
    """Heterogeneity numbers for a set of clients, in training-standardized feature space."""
    ids = sorted(clients)
    all_y = torch.cat([clients[c]["y"] for c in ids]).numpy()
    glob = np.bincount(all_y, minlength=N_CLASSES) / len(all_y)
    tv, means = [], []
    for c in ids:
        y = clients[c]["y"].numpy()
        tv.append(0.5 * np.abs(np.bincount(y, minlength=N_CLASSES) / len(y) - glob).sum())
        means.append(clients[c]["X"].numpy().mean(0))
    means = np.array(means)
    return {"clients": len(ids), "mean_class_tv_distance": float(np.mean(tv)),
            "mean_feature_std_across_clients": float(means.std(0).mean()),
            "mean_client_mean_vector_norm": float(np.linalg.norm(means, axis=1).mean())}, np.array(tv), means
