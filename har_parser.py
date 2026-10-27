"""Parser for HAR-PMD (Zenodo doi:10.5281/zenodo.7939223)."""
import re
from pathlib import Path
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent / "data" / "data_publish"
ACTIVITIES = ["still", "walking", "crutches", "walker", "manual", "electric"]
ENVIRONMENTS = ["indoor", "outdoor"]
EXPECTED_COLUMNS = {"phone": 32, "watch": 16}
META_COLS = ["user", "activity", "device", "environment"]
NAME_PATTERN = re.compile(
    r"^(?P<user>\d+)_(?P<activity>still|walking|crutches|walker|manual|electric)"
    r"_(?P<device>phone|watch)_(?P<environment>indoor|outdoor)\.csv$"
)


def parse_name(path):
    m = NAME_PATTERN.match(Path(path).name)
    if m is None:
        return None
    meta = m.groupdict()
    meta["user"] = int(meta["user"])
    return meta


def list_recordings(root=DATA_ROOT):
    recs = []
    for p in sorted(Path(root).rglob("*.csv")):
        meta = parse_name(p)
        if meta is None:
            print("skipped (unrecognised name):", p)
            continue
        meta["path"] = p
        recs.append(meta)
    return recs


def load_recording(path):
    meta = parse_name(path)
    if meta is None:
        raise ValueError(f"unrecognised file name: {path}")
    df = pd.read_csv(path)
    for i, col in enumerate(META_COLS):
        df.insert(i, col, meta[col])
    return df
