"""Step 1: the correctness checks. Writes aggregation_checks.txt and client_independence_checks.txt."""
import copy
import numpy as np
import torch
from fedavg_common import (CLIENTS_PER_ROUND, HERE, aggregate, build_clients, initial_state, local_train,
                           max_abs_diff, run_round, select_clients, selected_lr, state_id)

lines_a, lines_b, failed = [], [], []


def check(buf, ok, text, detail=""):
    buf.append(f"{'PASS' if ok else 'FAIL'}  {text}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failed.append(text)


clients, va = build_clients()
ids = sorted(clients)
g0 = initial_state()
lr = selected_lr()
A, B, C = ids[0], ids[1], ids[2]

# ---------- aggregation checks ----------
r = aggregate([{"w": torch.tensor([2.0])}, {"w": torch.tensor([8.0])}], [1, 3])["w"].item()
check(lines_a, abs(r - 6.5) < 1e-9, "scalar test: models 2 and 8 with counts 1 and 3 give 6.5", f"got {r}")

r = aggregate([{"w": torch.tensor([2.0])}, {"w": torch.tensor([8.0])}], [5, 5])["w"].item()
check(lines_a, abs(r - 5.0) < 1e-9, "equal counts give the plain mean 5.0", f"got {r}")

r1 = aggregate([{"w": torch.tensor([2.0])}, {"w": torch.tensor([8.0])}], [1, 3])["w"].item()
r2 = aggregate([{"w": torch.tensor([2.0])}, {"w": torch.tensor([8.0])}], [5, 15])["w"].item()
check(lines_a, abs(r1 - r2) < 1e-9, "scaling every count by 5 changes nothing (so counts are not multiplied by epochs)", f"{r1} vs {r2}")

sel = select_clients(np.random.default_rng(1), ids)
new, counts, _ = run_round(g0, clients, sel, lr, 1)
check(lines_a, counts == [clients[c]["n"] for c in sel], "weights are each client's own number of training windows, nothing else")
new2, counts2, _ = run_round(g0, clients, sel, lr, 1, epochs=2)
check(lines_a, counts2 == counts, "with 2 local epochs the weights stay the same (n_k is not multiplied by epochs)")

# independent numpy reference for a real round with the participating clients only
locals_ = [local_train(g0, clients[c], lr, 1, c)[0] for c in sel]
w = np.array([clients[c]["n"] for c in sel], dtype=np.float64)
ref_diff = max(float(np.abs(np.average(np.stack([s[k].double().numpy() for s in locals_]), axis=0, weights=w)
                            - new[k].double().numpy()).max()) for k in new)
check(lines_a, ref_diff < 1e-6, "real round matches an independent numpy weighted average", f"max difference {ref_diff:.2e}")

only = sel[:3]
l3 = [local_train(g0, clients[c], lr, 1, c)[0] for c in only]
n3, _, _ = run_round(g0, clients, only, lr, 1)
check(lines_a, max_abs_diff(n3, aggregate(l3, [clients[c]["n"] for c in only])) == 0.0,
      "only participating clients are averaged (a 3-client round uses just those 3)")

perm = list(reversed(sel))
lp = [local_train(g0, clients[c], lr, 1, c)[0] for c in perm]
dp = max_abs_diff(aggregate(lp, [clients[c]["n"] for c in perm]), new)
check(lines_a, dp < 1e-6, "client order does not matter", f"max difference {dp:.2e}")

z, _, _ = run_round(g0, clients, sel, 0.0, 1)
dz = max_abs_diff(z, g0)
check(lines_a, dz < 1e-7, "zero learning rate: aggregation keeps the starting weights", f"max difference {dz:.2e}")
check(lines_a, all(new[k].dtype == g0[k].dtype for k in g0), "tensor types are preserved (float32 stays float32)")

# ---------- client independence checks ----------
h0 = state_id(g0)
sA, _ = local_train(g0, clients[A], lr, 1, A)
check(lines_b, state_id(g0) == h0, "training client A does not change the global state", f"id {h0}")

sB_alone, _ = local_train(g0, clients[B], lr, 1, B)
_ = local_train(g0, clients[A], lr, 1, A)
sB_after, _ = local_train(g0, clients[B], lr, 1, B)
check(lines_b, max_abs_diff(sB_alone, sB_after) == 0.0, "training client A first does not change client B's result")

fake = copy.deepcopy(clients[A])
fake["X"] = fake["X"] + 5.0
_ = local_train(g0, fake, lr, 1, A)
sB_third, _ = local_train(g0, clients[B], lr, 1, B)
check(lines_b, max_abs_diff(sB_alone, sB_third) == 0.0, "changing client A's data does not change client B's result")

check(lines_b, max_abs_diff(sA, g0) > 0, "a trained client really differs from the global weights (training happened)")
ptr_shared = any(sA[k].data_ptr() == g0[k].data_ptr() for k in g0)
snap = state_id(g0)
for k in sA:
    sA[k].zero_()
check(lines_b, (not ptr_shared) and state_id(g0) == snap, "a client's returned weights are a separate copy; editing them leaves the global state alone")

one, _, _ = run_round(g0, clients, [C], lr, 1)
sC, _ = local_train(g0, clients[C], lr, 1, C)
check(lines_b, max_abs_diff(one, sC) == 0.0, "a single-client round returns exactly that client's trained state")

rng = np.random.default_rng(42)
ok_sel, sizes = True, []
for _ in range(10):
    s = select_clients(rng, ids)
    sizes.append(len(s))
    ok_sel &= len(s) == len(set(s)) == min(CLIENTS_PER_ROUND, len(ids)) and set(s) <= set(ids)
check(lines_b, ok_sel, f"each round picks min(10, {len(ids)}) = {min(CLIENTS_PER_ROUND, len(ids))} distinct clients", f"sizes {sizes}")
check(lines_b, sorted(select_clients(np.random.default_rng(1), ids[:4])) == ids[:4], "with only 4 clients all 4 are picked")
first = [select_clients(rng2, ids) for rng2 in [np.random.default_rng(42)] for _ in range(3)]
again = [select_clients(rng2, ids) for rng2 in [np.random.default_rng(42)] for _ in range(3)]
check(lines_b, first == again, "the same seed gives the same client picks")

n_total = sum(c["n"] for c in clients.values())
all_ids = np.concatenate([c["ids"] for c in clients.values()])
check(lines_b, len(clients) == len(ids) and len(set(all_ids)) == len(all_ids) == n_total,
      f"{len(clients)} clients, one per training user; no window is in two clients; {n_total} windows in total")
missing = [u for u, c in clients.items() if len(c["env_counts"]) < 2 or min(c["env_counts"].values()) == 0]
check(lines_b, not missing, "every client has windows from both environments",
      "all clients have both" if not missing else f"clients missing an environment: {missing}")

hdr = "Client table (user, windows, windows per environment)"
lines_b.append("")
lines_b.append(hdr)
for u in ids:
    lines_b.append(f"  user {u}: {clients[u]['n']} windows, {clients[u]['env_counts']}")

(HERE / "aggregation_checks.txt").write_text("\n".join(lines_a) + "\n")
(HERE / "client_independence_checks.txt").write_text("\n".join(lines_b) + "\n")
print("\n".join(lines_a)); print(); print("\n".join(lines_b[:-len(ids) - 2]))
print("\nALL CHECKS PASSED" if not failed else f"\n{len(failed)} CHECK(S) FAILED: {failed}")
raise SystemExit(1 if failed else 0)
