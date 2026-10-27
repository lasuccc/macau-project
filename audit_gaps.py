import numpy as np
import pandas as pd
from har_parser import list_recordings

rows = []
for r in list_recordings():
    t = pd.read_csv(r["path"], usecols=["Time"])["Time"].to_numpy(dtype=float)
    dt = np.diff(t)
    rows.append({
        "user": r["user"], "activity": r["activity"],
        "device": r["device"], "environment": r["environment"],
        "n": len(t),
        "median_dt": float(np.median(dt[dt > 0])),
        "p99_dt": float(np.quantile(dt, 0.99)),
        "gaps_gt_0.1s": int((dt > 0.1).sum()),
        "gaps_gt_0.5s": int((dt > 0.5).sum()),
        "gaps_gt_1s": int((dt > 1.0).sum()),
        "time_in_gaps_gt_0.1s": float(dt[dt > 0.1].sum()),
        "max_gap_s": float(dt.max()),
        "n_resets": int((dt < 0).sum()),
        "first_reset_at_s": float(t[np.argmax(dt < 0)]) if (dt < 0).any() else np.nan,
    })
g = pd.DataFrame(rows)
g.to_csv("audit_gaps.csv", index=False)
for c in ["gaps_gt_0.1s", "gaps_gt_0.5s", "gaps_gt_1s", "n_resets"]:
    print(c, g.groupby("device")[c].apply(lambda s: int((s > 0).sum())).to_dict())
print("\nworst 15 files by time spent in gaps > 0.1 s:")
print(g.sort_values("time_in_gaps_gt_0.1s", ascending=False).head(15)[
    ["user","activity","device","environment","gaps_gt_0.1s","gaps_gt_1s","time_in_gaps_gt_0.1s","max_gap_s"]].to_string())
