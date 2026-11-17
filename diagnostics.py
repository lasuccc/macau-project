"""Step 4 checks: can the MLP overfit a tiny training subset, and does it fall to chance on shuffled labels?
Neither check touches test users. These are diagnostics, not tuning."""
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from baseline_common import BASE, SEED, SMOKE, load_scaled, macro_f1, out

cfg, dg = BASE["mlp"], BASE["diagnostics"]
tr, va = load_scaled()
init_state = torch.load(out("mlp_initial_state.pt"))
loss_fn = nn.CrossEntropyLoss()


def fresh():
    m = nn.Sequential(nn.Linear(30, cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], 6))
    m.load_state_dict(copy.deepcopy(init_state))
    return m


def train(model, X, y, lr, epochs, seed, log_at=()):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    gen = torch.Generator().manual_seed(seed)
    log = {}
    for e in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
        if e in log_at:
            model.eval()
            with torch.no_grad():
                out_ = model(X)
                log[e] = {"train_loss": loss_fn(out_, y).item(), "train_accuracy": float((out_.argmax(1) == y).float().mean())}
    return log


# A) Overfit a small subset: 20 windows per class.
rng = np.random.default_rng(SEED)
per = 3 if SMOKE else dg["overfit_windows_per_class"]
pick = np.concatenate([rng.choice(np.flatnonzero(tr["y"] == k), size=per, replace=False) for k in range(6)])
Xs, ys = torch.from_numpy(tr["X"][pick]), torch.from_numpy(tr["y"][pick]).long()
E = 30 if SMOKE else dg["overfit_max_epochs"]
log = train(fresh(), Xs, ys, dg["overfit_lr"], E, SEED, log_at=sorted({E // 6, E // 2, E}))
final_acc = log[E]["train_accuracy"]
print("A) overfit a", len(pick), "window subset:", json.dumps(log))
print("   final training accuracy:", round(final_acc, 3), "->", "OK, it can learn" if final_acc >= 0.95 - 1e-6 else "LOW: check labels, feature order and scaling")

# B) Shuffled training labels: validation Macro F1 should drop to roughly chance (about 1/6).
Xt, yt = torch.from_numpy(tr["X"]), torch.from_numpy(rng.permutation(tr["y"])).long()
m = fresh()
train(m, Xt, yt, dg["shuffle_lr"], 2 if SMOKE else dg["shuffle_epochs"], SEED)
m.eval()
with torch.no_grad():
    f1_shuffled = macro_f1(va["y"], m(torch.from_numpy(va["X"])).argmax(1).numpy())
print("B) validation macro F1 with shuffled training labels:", round(f1_shuffled, 4), "(chance is about 0.167)",
      "-> OK" if f1_shuffled < 0.30 else "-> TOO HIGH: look for leakage")
out("diagnostics.json").write_text(json.dumps({
    "overfit_subset": {"n_windows": len(pick), "epochs": E, "lr": dg["overfit_lr"], "log": log, "passed": final_acc >= 0.95 - 1e-6},
    "shuffled_labels": {"epochs": 2 if SMOKE else dg["shuffle_epochs"], "val_macro_f1": f1_shuffled, "passed": f1_shuffled < 0.30}}, indent=1))
