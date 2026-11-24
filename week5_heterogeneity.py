"""Step 4: how different are the natural clients? Training users only. No test users, no model."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from baseline_common import FEATURES, HERE, LABELS
from week5_common import build_clients, client_summary

clients, _ = build_clients()
ids = sorted(clients)
summ, tv, means = client_summary(clients)
N = np.array([clients[c]["n"] for c in ids])
props = np.array([np.bincount(clients[c]["y"].numpy(), minlength=6) / clients[c]["n"] for c in ids])

rows = []
for i, c in enumerate(ids):
    env = clients[c]["env_counts"]
    rows.append({"client": c, "n_windows": int(N[i]), **{f"n_{k}": v for k, v in env.items()},
                 **{f"prop_{l}": props[i, j] for j, l in enumerate(LABELS)},
                 "class_tv_distance_to_global": tv[i], "mean_abs_feature_mean": float(np.abs(means[i]).mean()),
                 "max_abs_feature_mean": float(np.abs(means[i]).max()), "feature_mean_vector_norm": float(np.linalg.norm(means[i]))})
pd.DataFrame(rows).to_csv(HERE / "client_stats.csv", index=False)
within = np.mean([clients[c]["X"].numpy().std(0) for c in ids], axis=0)
pd.DataFrame({"feature": FEATURES, "std_of_client_means": means.std(0), "mean_within_client_std": within,
              "between_over_within": means.std(0) / within}).to_csv(HERE / "feature_mean_spread.csv", index=False)

okabe = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
fig = plt.figure(figsize=(14, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.45, wspace=0.18)
x = np.arange(len(ids))
ax = fig.add_subplot(gs[0, 0])
ax.bar(x, N, color="#0072B2")
ax.axhline(N.mean(), color="black", linestyle="--", linewidth=1)
ax.set_xticks(x[::2]); ax.set_xticklabels([ids[i] for i in range(0, len(ids), 2)], rotation=90, fontsize=7)
ax.set_ylim(0, N.max() * 1.1); ax.set_xlabel("client (participant ID)"); ax.set_ylabel("training windows")
ax.set_title(f"Windows per client (min {N.min()}, max {N.max()}, mean {N.mean():.0f})", fontsize=10)
ax = fig.add_subplot(gs[0, 1])
bottom = np.zeros(len(ids))
for j, l in enumerate(LABELS):
    ax.bar(x, props[:, j], bottom=bottom, color=okabe[j], label=l, width=0.9)
    bottom += props[:, j]
ax.set_xticks(x[::2]); ax.set_xticklabels([ids[i] for i in range(0, len(ids), 2)], rotation=90, fontsize=7)
ax.set_ylim(0, 1); ax.set_xlabel("client (participant ID)"); ax.set_ylabel("share of the client's windows")
ax.set_title("Class proportions per client", fontsize=10)
ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
ax = fig.add_subplot(gs[1, :])
lim = float(np.ceil(np.percentile(np.abs(means), 99) * 2) / 2)
im = ax.imshow(means, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
ax.set_yticks(x); ax.set_yticklabels(ids, fontsize=6); ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=7)
ax.set_ylabel("client"); ax.set_title("Each client's mean for each feature, in training-standardized units (0 = the pooled training mean)", fontsize=10)
fig.colorbar(im, ax=ax, pad=0.01).set_label("client mean (standard deviations)")
fig.savefig(HERE / "natural_client_distributions.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print(f"Clients: {len(ids)}   windows per client: min {N.min()}, max {N.max()}, mean {N.mean():.1f}")
print(f"Mean class-proportion distance from the pooled mix (0 = identical, 1 = disjoint): {summ['mean_class_tv_distance']:.4f}")
print(f"Largest class proportion gap for any client and class: {np.abs(props - props.mean(0)).max():.3f}")
print(f"Average spread of client means across features: {summ['mean_feature_std_across_clients']:.3f} standard deviations")
print(f"Typical client mean vector length: {summ['mean_client_mean_vector_norm']:.3f}")
print("Features with the largest spread between clients:")
print(pd.read_csv(HERE / "feature_mean_spread.csv").sort_values("std_of_client_means", ascending=False).head(5).round(3).to_string(index=False))
