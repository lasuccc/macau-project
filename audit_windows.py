import numpy as np
import pandas as pd
from har_parser import list_recordings

NOMINAL = {"phone": 60, "watch": 52}   # Hz, from the paper
rows = []
for r in list_recordings():
    t = pd.read_csv(r["path"], usecols=["Time"])["Time"].to_numpy(dtype=float)
    dt = np.diff(t)
    n_win = int(t.max() // 10)
    b = np.floor(t / 10).astype(int)
    ok = (b >= 0) & (b < n_win)
    counts = np.bincount(b[ok], minlength=n_win)
    gb = b[1:]
    m = (gb >= 0) & (gb < n_win) & (dt > 0)
    maxgap = np.zeros(n_win)
    np.maximum.at(maxgap, gb[m], dt[m])
    short = counts < 0.8 * NOMINAL[r["device"]] * 10
    rows.append({
        "user": r["user"], "activity": r["activity"],
        "device": r["device"], "environment": r["environment"],
        "windows": n_win,
        "gap_gt_0.5s": int((maxgap > 0.5).sum()),
        "gap_gt_1s": int((maxgap > 1.0).sum()),
        "short_windows": int(short.sum()),
        "bad_windows": int(((maxgap > 1.0) | short).sum()),
        "has_reset": bool((dt < 0).any()),
        "min_time": float(t.min()),
        "negative_time_rows": int((t < 0).sum()),
    })
w = pd.DataFrame(rows)
w.to_csv("audit_windows.csv", index=False)
s = w.groupby("device")[["windows", "gap_gt_0.5s", "gap_gt_1s", "short_windows", "bad_windows"]].sum()
for c in ["gap_gt_0.5s", "gap_gt_1s", "short_windows", "bad_windows"]:
    s[c + "_pct"] = (100 * s[c] / s["windows"]).round(2)
print(s.T.to_string())
print("\nfiles with reset:", int(w.has_reset.sum()))
print("files with negative Time values:", int((w.negative_time_rows > 0).sum()))
print(w[w.negative_time_rows > 0][["user", "activity", "device", "environment", "min_time", "negative_time_rows"]].to_string())
print("\nusers with most bad windows:")
print(w.groupby(["user", "device"]).bad_windows.sum().sort_values(ascending=False).head(10).to_string())
