"""Step 1: leakage checks, fit and save the scaler on TRAINING windows only, save the label map and hashes."""
import json
import platform
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from baseline_common import (BASE, FEATURES, HERE, LABELS, PRE, SEED, SMOKE, load_data, out,
                             sha256_file, sha256_ids)

D, test_ids = load_data()
tr, va = D[D.split == "train"], D[D.split == "val"]
tr_ids, va_ids = set(tr.window_id), set(va.window_id)

# Leakage checks: nothing from validation or test may reach the scaler.
assert tr_ids and va_ids
assert tr_ids.isdisjoint(va_ids) and tr_ids.isdisjoint(test_ids) and va_ids.isdisjoint(test_ids), "window ids overlap"
assert set(tr.user).isdisjoint(set(va.user)), "a user is in train and validation"
sm = pd.read_csv(HERE / "split_manifest.csv")
test_users = set(sm.user[sm.split == "test"])
assert set(tr.user).isdisjoint(test_users) and set(va.user).isdisjoint(test_users), "a user is also a test user"
print("OK   window ids and users are disjoint across train, validation and test")

scaler = StandardScaler().fit(tr[FEATURES].to_numpy())     # training windows only
assert scaler.n_samples_seen_ == len(tr_ids)
fit_hash = sha256_ids(tr.window_id)
out("scaler.json").write_text(json.dumps({
    "features": FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
    "n_windows": int(scaler.n_samples_seen_), "fit_window_hash": fit_hash,
    "note": "fitted on training windows only; reused unchanged for validation, test and later weeks"}, indent=1))
joblib.dump(scaler, out("scaler.joblib"))
out("label_map.json").write_text(json.dumps({n: i for i, n in enumerate(LABELS)}, indent=1))
print("OK   scaler fitted on", len(tr_ids), "training windows and saved")

counts = pd.DataFrame({"train": tr.groupby("y").size(), "val": va.groupby("y").size()}).reindex(range(len(LABELS)), fill_value=0)
counts.index = LABELS
counts.to_csv(out("class_counts_train_val.csv"))
print(counts.to_string())

try:
    import torch
    torch_version = torch.__version__
except ImportError:
    torch_version = None
contract = {
    "seed": SEED, "label_map": {n: i for i, n in enumerate(LABELS)}, "features": FEATURES,
    "device": PRE["cohort"]["device"], "smoke_run": SMOKE,
    "n_train_windows": len(tr_ids), "n_val_windows": len(va_ids), "n_test_windows_ids_only": len(test_ids),
    "n_train_users": int(tr.user.nunique()), "n_val_users": int(va.user.nunique()),
    "train_users": sorted(map(int, tr.user.unique())), "val_users": sorted(map(int, va.user.unique())),
    "train_window_hash": fit_hash, "val_window_hash": sha256_ids(va_ids), "test_window_id_hash": sha256_ids(test_ids),
    "window_manifest_file_hash": sha256_file(HERE / "window_manifest.csv.gz"),
    "split_manifest_hash": sha256_file(HERE / "split_manifest.csv"),
    "preprocessing_config_hash": sha256_file(HERE / "preprocessing_config.json"),
    "baseline_config_hash": sha256_file(HERE / "baseline_config.json"),
    "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                 "sklearn": sklearn.__version__, "torch": torch_version, "platform": platform.platform()}}
out("evaluation_contract.json").write_text(json.dumps(contract, indent=1))
print("OK   evaluation_contract.json saved. Validation window hash:", contract["val_window_hash"][:16], "...")
