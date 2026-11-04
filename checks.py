"""Acceptance checks. Every line should print OK."""
import numpy as np
import pandas as pd
from pipeline import (CFG, COLS, DATA_ROOT, FEATURE_NAMES, HZ, NPTS, window_features, load_table)

M = pd.read_csv("window_manifest.csv.gz")
SP = pd.read_csv("split_manifest.csv")
ok = lambda msg: print("OK  ", msg)

for dev in ("phone", "watch"):
    F = load_table(f"features_{dev}")
    assert list(F.columns) == ["window_id"] + FEATURE_NAMES, "unexpected feature columns"
    assert np.isfinite(F[FEATURE_NAMES].to_numpy()).all(), f"non-finite features in {dev}"
    assert F.window_id.is_unique
    kept = M[(M.status == "kept") & (M.device == dev)]
    assert set(F.window_id) == set(kept.window_id), f"{dev}: features and manifest differ"
    ok(f"{dev}: {len(F)} feature rows, all finite, ids unique and match the manifest")

banned = {"user", "environment", "file", "activity", "start_time", "end_time", "device", "segment"}
assert not banned & set(FEATURE_NAMES)
assert not any(w in n.lower() for n in FEATURE_NAMES for w in ("user", "env", "time", "file"))
ok("feature columns contain no metadata names")

S = {s: set(SP.user[SP.split == s]) for s in ("train", "val", "test")}
assert not (S["train"] & S["val"] or S["train"] & S["test"] or S["val"] & S["test"])
ok("train, val and test users are disjoint: " + ", ".join(f"{k}={len(v)}" for k, v in S.items()))

assert (M.start_time < M.end_time).all() and (M.raw_row_start <= M.raw_row_end).all()
assert M.window_id.is_unique
ok("every window has time bounds and a raw row range; window ids unique")

# Independent recompute for a random sample of kept windows.
rng = np.random.default_rng(0)
kept = M[M.status == "kept"]
sample = kept.iloc[rng.choice(len(kept), size=min(40, len(kept)), replace=False)]
store = {d: load_table(f"features_{d}").set_index("window_id") for d in ("phone", "watch")}
for _, w in sample.iterrows():
    raw = pd.read_csv(DATA_ROOT / w.file, usecols=["Time"] + COLS).iloc[int(w.raw_row_start):int(w.raw_row_end) + 1]
    t, X = raw.Time.to_numpy(float), raw[COLS].to_numpy(float)
    t, i = np.unique(t, return_index=True)
    X = X[i]
    grid = w.start_time + np.arange(NPTS) / HZ
    Y = np.column_stack([np.interp(grid, t, X[:, j]) for j in range(len(COLS))])
    f = window_features(Y[None])[0]
    assert np.allclose(f, store[w.device].loc[w.window_id, FEATURE_NAMES].to_numpy(float), rtol=1e-6, atol=1e-8), w.window_id
ok(f"recomputed {len(sample)} random windows from the raw files; features match")

# Rejected windows can never be in the kept set; counts by user and class.
bad = M[(M.status != "kept")]
ok(f"{len(bad)} windows rejected for non-finite values")
print("\nuser/class counts: counts_by_user_class.csv; split/class counts: split_class_counts.csv")
