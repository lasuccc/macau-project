"""Eligible cohort and frozen participant split. Run once, commit splits, never regenerate."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pipeline import CFG

M = pd.read_csv("window_manifest.csv.gz")
K = M[M.status == "kept"]
ACTS, MIN_W = CFG["activities"], CFG["cohort"]["min_windows_per_class"]

def eligible(dev):
    c = (K[K.device == dev].groupby(["user", "activity"]).size().unstack(fill_value=0)
         .reindex(columns=ACTS, fill_value=0))
    return (c >= MIN_W).all(axis=1)

users = pd.DataFrame({"user": sorted(M.user.unique())})
users["has_watch"] = users.user.isin(M[M.device == "watch"].user.unique())
users["phone_eligible"] = users.user.map(eligible("phone")).fillna(False).astype(bool)
users["watch_eligible"] = users.user.map(eligible("watch")).fillna(False).astype(bool)
dev = CFG["cohort"]["device"]
users["in_cohort"] = users[f"{dev}_eligible"]

def reason(r):
    if r.in_cohort:
        return ""
    if dev == "watch" and not r.has_watch:
        return "no watch data (phone only user)"
    return f"fewer than {MIN_W} kept windows in at least one class ({dev})"

users["reason"] = users.apply(reason, axis=1)
users["split"] = "excluded"
c = users.index[users.in_cohort].to_numpy()
sp = CFG["split"]
g1 = GroupShuffleSplit(n_splits=1, test_size=sp["test_frac"], random_state=sp["seed"])
rest, test = next(g1.split(c, groups=users.loc[c, "user"]))
g2 = GroupShuffleSplit(n_splits=1, test_size=sp["val_frac_of_remainder"], random_state=sp["seed"])
tr, va = next(g2.split(c[rest], groups=users.loc[c[rest], "user"]))
users.loc[c[test], "split"] = "test"
users.loc[c[rest][va], "split"] = "val"
users.loc[c[rest][tr], "split"] = "train"

S = {s: set(users.user[users.split == s]) for s in ("train", "val", "test")}
assert not (S["train"] & S["val"] or S["train"] & S["test"] or S["val"] & S["test"]), "users overlap"
users.to_csv("split_manifest.csv", index=False)

KK = K[K.device == dev].merge(users[["user", "split"]], on="user")
cc = KK.groupby(["split", "activity"]).size().unstack(fill_value=0).reindex(columns=ACTS)
cc.to_csv("split_class_counts.csv")
print(users.split.value_counts().to_string())
print("\nkept", dev, "windows per split and class\n", cc.to_string())
print("\nnot in cohort:", users[~users.in_cohort][["user", "reason"]].shape[0], "users")
print(users[~users.in_cohort & users.has_watch][["user", "reason"]].to_string(index=False))
print("cohort users without phone eligibility (matter for Week 7):",
      users[users.in_cohort & ~users.phone_eligible].user.tolist())
