"""Shared functions for the Week 2 pipeline (segment, interpolate, window, features)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from har_parser import list_recordings

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data" / "data_publish"
CFG = json.loads((HERE / "preprocessing_config.json").read_text())
COLS = CFG["streams"]["accel"] + CFG["streams"]["gyro"]
STATS = CFG["feature_stats"]
FEATURE_NAMES = [f"{c}_{s}" for c in COLS for s in STATS]
HZ, WIN_S = CFG["grid_hz"], CFG["window_s"]
NPTS = HZ * WIN_S
NOMINAL_WINDOWS = int(CFG["time"]["recording_s"] // WIN_S)

STAT_FUNCS = {
    "mean": lambda W: W.mean(1),
    "std": lambda W: W.std(1, ddof=CFG["std_ddof"]),
    "min": lambda W: W.min(1),
    "max": lambda W: W.max(1),
    "rms": lambda W: np.sqrt((W ** 2).mean(1)),
}


def recordings():
    return list_recordings(DATA_ROOT)


def window_features(W):
    """W has shape (n_windows, NPTS, 6). Returns (n_windows, 30), channel-major then stat."""
    f = np.stack([STAT_FUNCS[s](W) for s in STATS], axis=2)
    return f.reshape(len(W), -1)


def segment_bounds(t, max_gap):
    dt = np.diff(t)
    cuts = np.where((dt < 0) | (dt > max_gap))[0] + 1
    return np.r_[0, cuts], np.r_[cuts, len(t)], int((dt < 0).sum()), int((dt > max_gap).sum())


def dedupe(ts, Xs):
    keep = np.r_[True, np.diff(ts) > 0]
    n_dup = int((~keep).sum())
    if n_dup == 0:
        return keep, 0, 0
    last = np.maximum.accumulate(np.where(keep, np.arange(len(ts)), 0))
    d = ~keep
    return keep, n_dup, int((Xs[d] != Xs[last[d]]).any(axis=1).sum())


def process_file(r, max_gap=None):
    """Returns (file_row, segment_rows, window_rows, feature_array_for_all_windows)."""
    max_gap = CFG["max_gap_s"] if max_gap is None else max_gap
    raw = pd.read_csv(r["path"], usecols=["Time"] + COLS)
    t = raw["Time"].to_numpy(float)
    X = raw[COLS].to_numpy(float)
    starts, ends, n_reset, n_gap = segment_bounds(t, max_gap)
    rel = str(Path(r["path"]).relative_to(DATA_ROOT))
    key = {"user": r["user"], "activity": r["activity"], "device": r["device"], "environment": r["environment"]}
    seg_rows, win_rows, feats = [], [], []
    n_dup = n_conf = 0
    for seg, (s, e) in enumerate(zip(starts, ends)):
        ts, Xs, idx = t[s:e], X[s:e], np.arange(s, e)
        keep, nd, nc = dedupe(ts, Xs)
        n_dup += nd
        n_conf += nc
        ts, Xs, idx = ts[keep], Xs[keep], idx[keep]
        n_win = 0
        if len(ts) >= 2:
            n_win = (int(np.floor((ts[-1] - ts[0]) * HZ + 1e-9)) + 1) // NPTS
        seg_rows.append({**key, "file": rel, "segment": seg, "t_start": ts[0], "t_end": ts[-1],
                         "n_rows": len(ts), "n_windows": n_win,
                         "raw_row_start": int(idx[0]), "raw_row_end": int(idx[-1])})
        if n_win == 0:
            continue
        grid = ts[0] + np.arange(n_win * NPTS) / HZ
        Y = np.column_stack([np.interp(grid, ts, Xs[:, j]) for j in range(len(COLS))])
        F = window_features(Y.reshape(n_win, NPTS, len(COLS)))
        finite = np.isfinite(F).all(axis=1)
        for w in range(n_win):
            a, b = grid[w * NPTS], grid[w * NPTS + NPTS - 1]
            i0 = max(int(np.searchsorted(ts, a, "right")) - 1, 0)
            i1 = min(int(np.searchsorted(ts, b, "left")), len(ts) - 1)
            win_rows.append({
                "window_id": f"{r['user']}_{r['activity']}_{r['device']}_{r['environment']}_s{seg}_w{w}",
                **key, "file": rel, "segment": seg, "window": w,
                "start_time": a, "end_time": b,
                "raw_row_start": int(idx[i0]), "raw_row_end": int(idx[i1]), "n_raw_samples": i1 - i0 + 1,
                "max_step_s": float(np.diff(ts[i0:i1 + 1]).max()) if i1 > i0 else np.nan,
                "status": "kept" if finite[w] else "rejected_nonfinite",
            })
        feats.append(F)
    produced = len(win_rows)
    kept = sum(1 for w in win_rows if w["status"] == "kept")
    file_row = {**key, "file": rel, "rows": len(t), "segments": len(starts), "reset_breaks": n_reset,
                "gap_breaks": n_gap, "dup_timestamps_dropped": n_dup, "dup_conflicts": n_conf,
                "nominal_windows": NOMINAL_WINDOWS, "windows_produced": produced,
                "windows_lost_to_segmentation": NOMINAL_WINDOWS - produced,
                "windows_nonfinite": produced - kept, "windows_kept": kept}
    return file_row, seg_rows, win_rows, (np.vstack(feats) if feats else np.empty((0, len(FEATURE_NAMES))))


def save_table(df, stem):
    try:
        df.to_parquet(f"{stem}.parquet", index=False)
        return f"{stem}.parquet"
    except ImportError:
        df.to_csv(f"{stem}.csv.gz", index=False, float_format="%.7g")
        return f"{stem}.csv.gz"


def load_table(stem):
    p = Path(f"{stem}.parquet")
    return pd.read_parquet(p) if p.exists() else pd.read_csv(f"{stem}.csv.gz")
