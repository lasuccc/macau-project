"""Shared loading, label map, hashing and the ONE evaluation function used by every model."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

HERE = Path(__file__).resolve().parent
PRE = json.loads((HERE / "preprocessing_config.json").read_text())
BASE = json.loads((HERE / "baseline_config.json").read_text())
SMOKE = "--smoke" in sys.argv                      # tiny run: 2 train users, 2 validation users, 3 epochs
PREFIX = "smoke_" if SMOKE else ""
LABELS = BASE["labels"]
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
N_CLASSES = len(LABELS)
SEED = BASE["seed"]
DEVICE = PRE["cohort"]["device"]
FEATURES = (HERE / "feature_columns.txt").read_text().split()   # explicit, ordered feature list
assert len(FEATURES) == 30 and PRE["activities"] == LABELS


def out(name):
    return HERE / f"{PREFIX}{name}"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def sha256_ids(ids):
    return sha256_bytes("\n".join(sorted(map(str, ids))).encode())


def read_features(device):
    p = HERE / f"features_{device}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(HERE / f"features_{device}.csv.gz")


def load_data():
    """Returns (train_val_frame, test_window_ids). Test FEATURES are never kept in memory here."""
    F = read_features(DEVICE)
    assert list(F.columns) == ["window_id"] + FEATURES, "feature columns differ from feature_columns.txt"
    M = pd.read_csv(HERE / "window_manifest.csv.gz", usecols=["window_id", "user", "activity", "device", "status"])
    M = M[(M.device == DEVICE) & (M.status == "kept")]
    S = pd.read_csv(HERE / "split_manifest.csv")[["user", "split"]]
    D = F.merge(M[["window_id", "user", "activity"]], on="window_id", validate="one_to_one")
    D = D.merge(S, on="user", how="left", validate="many_to_one")
    assert D.split.notna().all(), "windows from a user missing in split_manifest.csv"
    assert np.isfinite(D[FEATURES].to_numpy()).all(), "non-finite features"
    D["y"] = D.activity.map(LABEL_ID)
    assert D.y.notna().all(), "unknown activity name"
    D["y"] = D.y.astype(int)
    D = D[D.split != "excluded"]
    test_ids = set(D.window_id[D.split == "test"])
    D = D[D.split != "test"].reset_index(drop=True)
    if SMOKE:
        keep = (sorted(D.user[D.split == "train"].unique())[:2] + sorted(D.user[D.split == "val"].unique())[:2])
        D = D[D.user.isin(keep)].reset_index(drop=True)
    return D, test_ids


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=list(range(N_CLASSES)), average="macro", zero_division=0)


def evaluate(y_true, y_pred, users):
    """The single evaluation contract: six fixed class IDs, zero_division=0.
    Pooled and per-user Macro F1 are separate numbers."""
    y_true, y_pred, users = np.asarray(y_true), np.asarray(y_pred), np.asarray(users)
    L = list(range(N_CLASSES))
    res = {"n_windows": len(y_true), "n_users": len(set(users)),
           "accuracy": accuracy_score(y_true, y_pred),
           "macro_f1_pooled": macro_f1(y_true, y_pred)}
    for name, r in zip(LABELS, recall_score(y_true, y_pred, labels=L, average=None, zero_division=0)):
        res[f"recall_{name}"] = r
    rows = []
    for u in sorted(set(users)):
        m = users == u
        absent = [LABELS[k] for k in L if not (y_true[m] == k).any()]
        rows.append({"user": u, "n_windows": int(m.sum()), "macro_f1": macro_f1(y_true[m], y_pred[m]),
                     "absent_classes": ";".join(absent)})
    per_user = pd.DataFrame(rows)
    res["macro_f1_user_mean"] = float(per_user.macro_f1.mean())
    return res, per_user, confusion_matrix(y_true, y_pred, labels=L)


def load_scaled():
    """Loads the SAVED scaler (never refits). Returns train and validation sets as dicts."""
    D, _ = load_data()
    sc = json.loads(out("scaler.json").read_text())
    assert sc["features"] == FEATURES
    tr, va = D[D.split == "train"], D[D.split == "val"]
    assert sha256_ids(tr.window_id) == sc["fit_window_hash"], \
        "training windows changed since the scaler was fitted; rerun week3_prepare.py"
    mean, scale = np.array(sc["mean"]), np.array(sc["scale"])
    def pack(df):
        return {"X": ((df[FEATURES].to_numpy() - mean) / scale).astype(np.float32),
                "y": df.y.to_numpy(), "users": df.user.to_numpy(), "ids": df.window_id.to_numpy()}
    return pack(tr), pack(va)


def save_predictions(model_name, va, y_pred):
    pd.DataFrame({"model": model_name, "window_id": va["ids"], "user": va["users"],
                  "y_true": va["y"], "y_pred": np.asarray(y_pred)}).to_csv(out(f"pred_{model_name}.csv"), index=False)
