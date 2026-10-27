import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from har_parser import list_recordings, load_recording

recs = [r for r in list_recordings() if r["user"] == 61]
print("files for user 61:", len(recs))   # expect 24

frames = {(r["device"], r["activity"], r["environment"]): load_recording(r["path"]) for r in recs}
for key, df in frames.items():
    print(key, df.shape, "duration_s=%.1f" % df["Time"].iloc[-1])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, act in zip(axes, ["still", "walking"]):
    d = frames[("phone", act, "indoor")]
    d = d[d["Time"] < 10]
    ax.plot(d["Time"], d["LAccX"], label="LAccX")
    ax.plot(d["Time"], d["LAccY"], label="LAccY")
    ax.plot(d["Time"], d["LAccZ"], label="LAccZ")
    ax.set_title(f"user 61 phone {act} indoor, first 10 s")
    ax.legend()
plt.tight_layout()
plt.savefig("user61_sanity.png")
print("saved user61_sanity.png")
