"""Effect of the gap rule on TRAINING users only (reads split_manifest.csv)."""
import pandas as pd
from pipeline import recordings, process_file, NOMINAL_WINDOWS

sp = pd.read_csv("split_manifest.csv")
train = set(sp.user[sp.split == "train"])
recs = [r for r in recordings() if r["user"] in train]
rows = []
for gap in (0.2, 0.3, 0.5, 1.0):
    kept = {"phone": 0, "watch": 0}
    nom = {"phone": 0, "watch": 0}
    for r in recs:
        f = process_file(r, max_gap=gap)[0]
        kept[r["device"]] += f["windows_kept"]
        nom[r["device"]] += NOMINAL_WINDOWS
    for d in kept:
        if nom[d]:
            rows.append({"max_gap_s": gap, "device": d, "kept": kept[d], "nominal": nom[d],
                         "kept_pct": round(100 * kept[d] / nom[d], 2)})
out = pd.DataFrame(rows)
out.to_csv("coverage_by_gap_rule.csv", index=False)
print(out.pivot(index="max_gap_s", columns="device", values="kept_pct").to_string())
print("\ntraining users:", len(train))
