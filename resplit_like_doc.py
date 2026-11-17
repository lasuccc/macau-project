"""Compare the Week 2 split with the Week 3 document's method, and optionally switch.
Document method: sorted watch user IDs -> numpy default_rng(42).permutation -> first 36 train, next 12 validation, last 12 test.
  python resplit_like_doc.py            only compares, changes nothing
  python resplit_like_doc.py --apply    backs up the old file, then writes the new split (run BEFORE any training)"""
import shutil
import sys
import numpy as np
import pandas as pd

sm = pd.read_csv("split_manifest.csv")
cohort = sorted(sm.user[sm.in_cohort.astype(bool)].astype(int))
assert len(cohort) == 60, f"expected 60 cohort users, found {len(cohort)}"
order = np.random.default_rng(42).permutation(np.array(cohort))
new = {int(u): s for ids, s in ((order[:36], "train"), (order[36:48], "val"), (order[48:], "test")) for u in ids}
old = dict(zip(sm.user.astype(int), sm.split))
print("Users in each split, old method vs document method:")
for s in ("train", "val", "test"):
    o = {u for u in cohort if old[u] == s}
    n = {u for u in cohort if new[u] == s}
    print(f"  {s:5s} old {len(o)}  new {len(n)}  same users in both: {len(o & n)}")
if "--apply" not in sys.argv:
    print("\nNothing changed. Add --apply to switch.")
    sys.exit()
shutil.copy("split_manifest.csv", "split_manifest_week2_groupshuffle.csv")
sm["split"] = [new[int(u)] if int(u) in new else "excluded" for u in sm.user]
sm.to_csv("split_manifest.csv", index=False)
cls = pd.read_csv("window_manifest.csv.gz", usecols=["user", "activity", "device", "status"])
cls = cls[(cls.device == "watch") & (cls.status == "kept")].merge(sm[["user", "split"]], on="user")
cls.groupby(["split", "activity"]).size().unstack(fill_value=0).to_csv("split_class_counts.csv")
print("\nSwitched. Old file saved as split_manifest_week2_groupshuffle.csv. split_class_counts.csv rewritten.")
print(cls.groupby(["split", "activity"]).size().unstack(fill_value=0).to_string())
