"""Segment, interpolate, window and extract features for every recording.
Usage: python build_pipeline.py            (all users)
       python build_pipeline.py 61 62      (test run; outputs get a test_ prefix)
"""
import sys
import numpy as np
import pandas as pd
from pipeline import CFG, FEATURE_NAMES, recordings, process_file, save_table

only = {int(a) for a in sys.argv[1:]}
tag = "test_" if only else ""
files, segs, wins, feats = [], [], [], []
for r in recordings():
    if only and r["user"] not in only:
        continue
    f, s, w, F = process_file(r)
    files.append(f); segs.extend(s); wins.extend(w); feats.append((r["device"], F))

E, S, M = pd.DataFrame(files), pd.DataFrame(segs), pd.DataFrame(wins)
F = pd.DataFrame(np.vstack([f for _, f in feats]), columns=FEATURE_NAMES)
F.insert(0, "window_id", M["window_id"].to_numpy())
K = F[M["status"].to_numpy() == "kept"]
dev = M.loc[M["status"] == "kept", "device"].to_numpy()

E.to_csv(f"{tag}exclusions.csv", index=False)
S.to_csv(f"{tag}segment_log.csv", index=False)
M.to_csv(f"{tag}window_manifest.csv.gz", index=False)
for d in ("phone", "watch"):
    print("saved", save_table(K[dev == d].reset_index(drop=True), f"{tag}features_{d}"))
open(f"{tag}feature_columns.txt", "w").write("\n".join(FEATURE_NAMES) + "\n")

(E.groupby(["user", "activity", "device"])
   [["nominal_windows", "windows_produced", "windows_nonfinite", "windows_kept"]].sum()
   .assign(windows_rejected=lambda d: d.nominal_windows - d.windows_kept)
   .to_csv(f"{tag}counts_by_user_class.csv"))

g = E.groupby("device")
t = pd.DataFrame({
    "1 nominal (files x 120)": g.nominal_windows.sum(),
    "2 after segmentation": g.windows_produced.sum(),
    "3 after finite check (kept)": g.windows_kept.sum()})
t["kept %"] = (100 * t["3 after finite check (kept)"] / t["1 nominal (files x 120)"]).round(2)
t.T.to_csv(f"{tag}filter_counts.csv")
print("\nwindow counts before and after each filter\n", t.T.to_string())
print("\nevents:", E[["reset_breaks", "gap_breaks", "dup_timestamps_dropped", "dup_conflicts"]].sum().to_dict())
print("files with at least one gap break:", int((E.gap_breaks > 0).sum()), "of", len(E))
print("\nkept windows by device and activity")
print(M[M.status == "kept"].groupby(["device", "activity"]).size().unstack(0).to_string())
