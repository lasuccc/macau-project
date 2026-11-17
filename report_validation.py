"""Step 5: the one validation table, the confusion matrix, per-user scores, loss curves and error-analysis facts.
Validation only. The test users are never scored here."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from baseline_common import LABELS, evaluate, out

sel = json.loads(out("mlp_selection.json").read_text())
models = {"majority": "majority", "logistic": "logistic", "mlp": sel["selected_model"]}
rows, per_user_all, cms = [], [], {}
for name, file_key in models.items():
    P = pd.read_csv(out(f"pred_{file_key}.csv"))
    res, pu, cm = evaluate(P.y_true, P.y_pred, P.user)
    rows.append({"model": name + (f" ({file_key})" if name == "mlp" else ""), **res})
    pu.insert(0, "model", name)
    per_user_all.append(pu)
    cms[name] = cm
    P.to_csv(out(f"central_validation_{name}.csv"), index=False)
T = pd.DataFrame(rows)
T.to_csv(out("central_validation_metrics.csv"), index=False)
pd.concat(per_user_all).to_csv(out("central_validation_per_user.csv"), index=False)
show = ["model", "n_windows", "n_users", "accuracy", "macro_f1_pooled", "macro_f1_user_mean"] + [f"recall_{l}" for l in LABELS]
print(T[show].round(3).to_string(index=False))

for name, cm in cms.items():
    pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]).to_csv(out(f"confusion_{name}.csv"))
cm = cms["mlp"]
rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(rn, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(6)); ax.set_xticklabels(LABELS, rotation=45, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels(LABELS)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center", color="white" if rn[i, j] > 0.5 else "black", fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("MLP validation confusion (rows sum to 1)")
fig.colorbar(im); fig.tight_layout(); fig.savefig(out("confusion_mlp.png"), dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for lr in (0.01, 0.03):
    H = pd.read_csv(out(f"mlp_history_lr{lr}.csv"))
    axs[0].plot(H.epoch, H.train_loss, label=f"train lr {lr}")
    axs[0].plot(H.epoch, H.val_loss, "--", label=f"val lr {lr}")
    axs[1].plot(H.epoch, H.val_macro_f1, label=f"lr {lr}")
axs[0].set_xlabel("epoch"); axs[0].set_ylabel("loss"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("epoch"); axs[1].set_ylabel("validation macro F1"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out("mlp_curves.png"), dpi=150); plt.close(fig)

# Facts for the error analysis (you write the words; these are the numbers).
pairs = [(LABELS[i], LABELS[j], int(cm[i, j])) for i in range(6) for j in range(6) if i != j]
pairs.sort(key=lambda p: -p[2])
pu = pd.concat(per_user_all)
pu = pu[pu.model == "mlp"].sort_values("macro_f1")
facts = {"top_confusions_mlp": [{"true": a, "pred": b, "count": c} for a, b, c in pairs[:5]],
         "weakest_users_mlp": pu.head(5)[["user", "macro_f1", "n_windows", "absent_classes"]].to_dict("records"),
         "users_with_absent_classes": int((pu.absent_classes != "").sum()),
         "per_user_macro_f1_min_median_max": [float(pu.macro_f1.min()), float(pu.macro_f1.median()), float(pu.macro_f1.max())]}
out("error_analysis_facts.json").write_text(json.dumps(facts, indent=1, default=str))
print("\nTop confusions:", facts["top_confusions_mlp"][:3])
print("Users with at least one class missing:", facts["users_with_absent_classes"])
