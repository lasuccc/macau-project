import numpy as np
import pandas as pd
from har_parser import (list_recordings, load_recording, ACTIVITIES,
                        ENVIRONMENTS, EXPECTED_COLUMNS, META_COLS)

recs = list_recordings()
rows, failed = [], []
for r in recs:
    try:
        df = load_recording(r["path"])
    except Exception as e:
        failed.append({"path": str(r["path"]), "error": str(e)})
        continue
    sensors = df.drop(columns=META_COLS)
    t = df["Time"].to_numpy(dtype=float)
    dt = np.diff(t)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if len(pos) else np.nan
    rows.append({
        "user": r["user"], "activity": r["activity"],
        "device": r["device"], "environment": r["environment"],
        "folder_matches_user": r["path"].parent.parent.name == str(r["user"]),
        "rows": len(df),
        "duration_s": float(t[-1] - t[0]),
        "starts_at_zero": bool(t[0] == 0),
        "median_dt_s": med,
        "approx_hz": 1 / med if med == med else np.nan,
        "resets": int((dt < 0).sum()),
        "duplicate_timestamps": int((dt == 0).sum()),
        "long_gaps": int((dt > 5 * med).sum()) if med == med else 0,
        "max_gap_s": float(dt.max()) if len(dt) else np.nan,
        "nan_cells": int(sensors.isna().sum().sum()),
        "duplicate_rows": int(sensors.duplicated().sum()),
        "columns_ok": sensors.shape[1] == EXPECTED_COLUMNS[r["device"]],
    })

audit = pd.DataFrame(rows)
audit.to_csv("audit_per_file.csv", index=False)
pd.DataFrame(failed).to_csv("audit_failed.csv", index=False)

found = set(zip(audit.user, audit.activity, audit.device, audit.environment))
expected = {(u, a, "phone", e) for u in range(1, 121) for a in ACTIVITIES for e in ENVIRONMENTS}
expected |= {(u, a, "watch", e) for u in range(61, 121) for a in ACTIVITIES for e in ENVIRONMENTS}
missing = sorted(expected - found)
pd.DataFrame(missing, columns=["user", "activity", "device", "environment"]).to_csv("audit_missing.csv", index=False)

audit.groupby(["device", "activity", "environment"]).agg(files=("rows", "size"), rows=("rows", "sum")).to_csv("counts_device_activity_env.csv")
audit.groupby(["user", "device"]).agg(files=("rows", "size"), rows=("rows", "sum")).to_csv("counts_user_device.csv")
audit.groupby(["user", "activity", "device"]).agg(rows=("rows", "sum")).to_csv("counts_user_activity_device.csv")

print(f"parsed: {len(audit)} | expected: {len(expected)} | missing: {len(missing)} | failed: {len(failed)}")
flags = ["resets", "duplicate_timestamps", "long_gaps", "nan_cells", "duplicate_rows"]
print("\nfiles with at least one issue, by check:")
print((audit[flags] > 0).sum())
print("\nfolder/user mismatches:", int((~audit.folder_matches_user).sum()))
print("column-count mismatches:", int((~audit.columns_ok).sum()))
print("files not starting at t=0:", int((~audit.starts_at_zero).sum()))
print("\nper-device summary:")
print(audit.groupby("device")[["rows", "duration_s", "approx_hz"]].describe().T.to_string())
