"""Shared FedAvg pieces for Week 4: clients, local training, aggregation, evaluation, hashing."""
import copy
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from baseline_common import BASE, HERE, LABELS, N_CLASSES, SEED, SMOKE, evaluate, load_scaled, out

if SMOKE:
    sys.exit("Do not use --smoke in Week 4. Use --rounds 2 for a quick trial.")
torch.set_num_threads(1)                      # fixed thread count, recorded in runtime_budget.md
HIDDEN = BASE["mlp"]["hidden"]
BATCH = BASE["mlp"]["batch_size"]
CLIENTS_PER_ROUND = 10


def make_model():
    return nn.Sequential(nn.Linear(30, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, N_CLASSES))


def selected_lr():
    return float(json.loads(out("mlp_selection.json").read_text())["selected_lr"])


def initial_state():
    return torch.load(out("mlp_initial_state.pt"), weights_only=True)


def state_id(state):
    """Short fingerprint of a model state. Used as the checkpoint ID."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def build_clients():
    """One client per TRAINING user. Each client holds all that user's windows (indoor and outdoor).
    Returns (clients, validation pack)."""
    tr, va = load_scaled()
    M = pd.read_csv(HERE / "window_manifest.csv.gz", usecols=["window_id", "environment"])
    env_of = dict(zip(M.window_id, M.environment))
    env = np.array([env_of[w] for w in tr["ids"]])
    clients = {}
    for u in sorted(set(tr["users"])):
        m = tr["users"] == u
        clients[int(u)] = {"X": torch.from_numpy(tr["X"][m].copy()), "y": torch.from_numpy(tr["y"][m].copy()).long(),
                           "n": int(m.sum()), "ids": tr["ids"][m],
                           "env_counts": {e: int((env[m] == e).sum()) for e in sorted(set(env))}}
    return clients, va


def local_train(global_state, client, lr, round_idx, client_id, epochs=1):
    """Start from an independent copy of the global weights, fresh SGD (no momentum). One epoch by default."""
    model = make_model()
    model.load_state_dict(copy.deepcopy(global_state))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    gen = torch.Generator().manual_seed(SEED * 1_000_003 + round_idx * 1009 + client_id)
    loss_fn = nn.CrossEntropyLoss()
    n, total = client["n"], 0.0
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
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, total / (n * epochs)


def aggregate(states, counts):
    """FedAvg: each tensor = sum(n_k * w_k) / sum(n_k). Only the states passed in take part.
    n_k is the client's number of training windows. It is NOT multiplied by local epochs."""
    total = float(sum(counts))
    new = {}
    for k in states[0]:
        acc = sum(s[k].double() * (c / total) for s, c in zip(states, counts))
        new[k] = acc.to(states[0][k].dtype)
    return new


def select_clients(rng, client_ids, k=CLIENTS_PER_ROUND):
    """min(k, number of clients) clients, no replacement."""
    ids = sorted(client_ids)
    return sorted(int(x) for x in rng.choice(ids, size=min(k, len(ids)), replace=False))


def run_round(global_state, clients, selected, lr, round_idx, epochs=1):
    states, counts, losses = [], [], []
    for cid in selected:
        s, l = local_train(global_state, clients[cid], lr, round_idx, cid, epochs)
        states.append(s)
        counts.append(clients[cid]["n"])
        losses.append(l)
    new = aggregate(states, counts)
    train_loss = float(np.average(losses, weights=counts))
    return new, counts, train_loss


def evaluate_global(state, va):
    model = make_model()
    model.load_state_dict(state)
    model.eval()
    X, y = torch.from_numpy(va["X"].copy()), torch.from_numpy(va["y"].copy()).long()
    with torch.no_grad():
        logits = model(X)
        loss = nn.CrossEntropyLoss()(logits, y).item()
        pred = logits.argmax(1).numpy()
    res, per_user, cm = evaluate(va["y"], pred, va["users"])
    res["val_loss"] = loss
    return res, per_user, cm


def max_abs_diff(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)
