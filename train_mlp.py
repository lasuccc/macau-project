"""Step 3: 30-64-6 MLP, cross-entropy, plain SGD, batch size 64, up to 50 epochs, learning rates 0.01 and 0.03."""
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from baseline_common import BASE, SEED, SMOKE, evaluate, load_scaled, macro_f1, out, save_predictions

cfg = BASE["mlp"]
EPOCHS = 3 if SMOKE else cfg["max_epochs"]
BS = cfg["batch_size"]
tr, va = load_scaled()
Xtr, ytr = torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"]).long()
Xva, yva = torch.from_numpy(va["X"]), torch.from_numpy(va["y"]).long()
n = len(ytr)
updates_per_epoch = int(np.ceil(n / BS))


def make_model():
    return nn.Sequential(nn.Linear(30, cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], 6))


torch.manual_seed(SEED)
init_state = copy.deepcopy(make_model().state_dict())
torch.save(init_state, out("mlp_initial_state.pt"))          # reused by the federated runs in week 4
loss_fn = nn.CrossEntropyLoss()
summary = {}
for lr in cfg["learning_rates"]:
    model = make_model()
    model.load_state_dict(init_state)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    gen = torch.Generator().manual_seed(SEED)
    hist, best, t0 = [], {"f1": -1.0}, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(Xva)
            val_loss = loss_fn(logits, yva).item()
            pred = logits.argmax(1).numpy()
        f1 = macro_f1(va["y"], pred)
        hist.append({"epoch": epoch, "train_loss": running / n, "val_loss": val_loss, "val_macro_f1": f1,
                     "val_accuracy": float((pred == va["y"]).mean()), "updates_so_far": epoch * updates_per_epoch})
        if f1 > best["f1"]:                                   # strict ">" keeps the earlier epoch on ties
            with torch.no_grad():
                train_pred = model(Xtr).argmax(1).numpy()
            best = {"f1": f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict()), "pred": pred,
                    "train_macro_f1": macro_f1(tr["y"], train_pred)}
        print(f"lr {lr}  epoch {epoch:2d}  train loss {hist[-1]['train_loss']:.4f}  "
              f"val loss {val_loss:.4f}  val macro F1 {f1:.4f}")
    runtime = time.time() - t0
    pd.DataFrame(hist).to_csv(out(f"mlp_history_lr{lr}.csv"), index=False)
    torch.save({"state_dict": best["state"], "epoch": best["epoch"], "lr": lr, "val_macro_f1": best["f1"],
                "train_macro_f1": best["train_macro_f1"], "updates_to_best": best["epoch"] * updates_per_epoch},
               out(f"mlp_lr{lr}_best.pt"))
    save_predictions(f"mlp_lr{lr}", va, best["pred"])
    summary[lr] = {"best_epoch": best["epoch"], "val_macro_f1": best["f1"], "train_macro_f1": best["train_macro_f1"],
                   "updates_per_epoch": updates_per_epoch, "updates_to_best": best["epoch"] * updates_per_epoch,
                   "total_updates": EPOCHS * updates_per_epoch, "runtime_s": round(runtime, 1)}

# Choose the learning rate by validation Macro F1; a tie goes to the smaller learning rate.
chosen = max(cfg["learning_rates"], key=lambda r: (summary[r]["val_macro_f1"], -r))
ckpt = torch.load(out(f"mlp_lr{chosen}_best.pt"))
torch.save(ckpt, out("mlp_selected.pt"))
out("mlp_selection.json").write_text(json.dumps({"selected_lr": chosen, "selected_model": f"mlp_lr{chosen}",
                                                 "runs": {str(k): v for k, v in summary.items()}}, indent=1))
print(json.dumps({"selected_lr": chosen, "runs": summary}, indent=1))
