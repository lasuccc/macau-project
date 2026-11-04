"""Phone/watch clock audit. Rule is frozen on TRAINING users only, before looking at val/test.

offset_s: the same motion appears offset_s later in the watch file than in the phone file
(watch_time = phone_time + offset_s). A real shared clock gives an offset near 0 with a peak far
above what mismatched (control) pairs reach.
"""
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from pipeline import CFG, recordings

A = CFG["alignment"]
BIN, MAXLAG = A["bin_s"], A["max_lag_s"]
NB = int(round(CFG["time"]["recording_s"] / BIN))
ACTS = [a for a in CFG["activities"] if a != "still"]      # still has too little motion to correlate

sp = pd.read_csv("split_manifest.csv")
train = sorted(sp.user[(sp.split == "train") & sp.has_watch])
recs = {(r["user"], r["activity"], r["device"], r["environment"]): r["path"] for r in recordings()}
segs = pd.read_csv("segment_log.csv")
wins = pd.read_csv("window_manifest.csv.gz")
wins = wins[(wins.status == "kept") & (wins.device == "phone")]

def envelope(path):
    d = pd.read_csv(path, usecols=["Time", "GyrX", "GyrY", "GyrZ"])
    t = d.Time.to_numpy(float)
    if (np.diff(t) < 0).any():
        return None
    g = np.sqrt((d[["GyrX", "GyrY", "GyrZ"]].to_numpy(float) ** 2).sum(axis=1))
    b = np.clip(np.floor(t / BIN).astype(int), 0, NB - 1)
    c = np.bincount(b, minlength=NB)
    s = np.bincount(b, weights=g, minlength=NB)
    e = np.full(NB, np.nan)
    e[c > 0] = s[c > 0] / c[c > 0]
    good = ~np.isnan(e)
    e = np.interp(np.arange(NB), np.flatnonzero(good), e[good])
    return (e - e.mean()) / (e.std() + 1e-12)

def best_lag(phone, watch):
    c = correlate(watch, phone, mode="full", method="fft") / NB
    lags = correlation_lags(len(watch), len(phone))
    m = np.abs(lags) <= MAXLAG / BIN
    k = np.argmax(c[m])
    return lags[m][k] * BIN, float(c[m][k])

def covered_phone_windows(u, act, env, off):
    pw = wins[(wins.user == u) & (wins.activity == act) & (wins.environment == env)]
    ws = segs[(segs.user == u) & (segs.activity == act) & (segs.environment == env) & (segs.device == "watch")]
    n = 0
    for a, b in zip(pw.start_time, pw.end_time):
        n += bool(((ws.t_start <= a + off) & (ws.t_end >= b + off)).any())
    return n, len(pw)

rows = []
for i, u in enumerate(train):
    for kind, v in (("true", u), ("control", train[(i + 1) % len(train)])):
        for act in ACTS:
            for env in ("indoor", "outdoor"):
                kp, kw = (u, act, "phone", env), (v, act, "watch", env)
                row = {"pair": kind, "user": u, "watch_user": v, "activity": act, "environment": env}
                if kp not in recs or kw not in recs:
                    rows.append({**row, "skipped": "missing file"}); continue
                p, w = envelope(recs[kp]), envelope(recs[kw])
                if p is None or w is None:
                    rows.append({**row, "skipped": "timestamp reset"}); continue
                off, peak = best_lag(p, w)
                row.update(skipped="", offset_s=off, peak_corr=peak, overlap_s=CFG["time"]["recording_s"] - abs(off))
                if kind == "true":
                    cw, tw = covered_phone_windows(u, act, env, off)
                    row.update(phone_windows=tw, candidate_pairs=cw)
                rows.append(row)

R = pd.DataFrame(rows)
R.to_csv("alignment_audit.csv", index=False)
T, C = (R[(R.pair == k) & (R.skipped == "")] for k in ("true", "control"))
thr = float(C.peak_corr.quantile(0.95))                     # correlation a mismatched pair reaches 5% of the time
tol = A["max_timing_error_s"]
ok = (T.peak_corr >= thr) & (T.offset_s.abs() <= tol)
rate = float(ok.mean()) if len(T) else 0.0
ctrl_rate = float(((C.peak_corr >= thr) & (C.offset_s.abs() <= tol)).mean()) if len(C) else float("nan")
good = T[ok]
p95 = float(good.offset_s.abs().quantile(0.95)) if len(good) else float("nan")
accepted = rate >= A["required_pass_rate"]
frozen = round(min(np.ceil(p95 * 10) / 10, tol), 1) if accepted else None

md = f"""# Phone and watch alignment audit (training users only)

Files carry `Time` in seconds since each recording started, so no shared clock is stored. Resetting both
first readings to zero is not evidence of alignment. This audit estimates the offset between each phone
file and the watch file for the same user, activity and environment from the 0.1 s gyroscope-magnitude
envelope (cross-correlation, lags up to +/-{MAXLAG:.0f} s), and compares it with control pairs
(phone of one user against the watch of a different training user, same activity and environment).

Rule, fixed before looking at validation or test users:
- control threshold = 95th percentile of control peak correlation = {thr:.3f}
- a true pair passes if peak correlation >= that threshold and |offset| <= {tol} s
- pairing is accepted if at least {A['required_pass_rate']:.0%} of true training pairs pass

Evidence ({len(train)} training users, activities {', '.join(ACTS)}):
- true pairs used: {len(T)}; control pairs used: {len(C)}; skipped for timestamp resets or missing files: {int((R.skipped != '').sum())}
- median peak correlation: true {T.peak_corr.median():.3f}, control {C.peak_corr.median():.3f}
- true pairs passing: {rate:.1%}; control pairs passing: {ctrl_rate:.1%}
- |offset| among passing pairs: median {good.offset_s.abs().median() if len(good) else float('nan'):.2f} s, 95th percentile {p95:.2f} s
- overlap of the two recordings after the offset: median {T.overlap_s.median():.1f} s
- phone windows with full watch coverage after the offset: {int(T.candidate_pairs.sum())} of {int(T.phone_windows.sum())}

Decision: {'pairing ACCEPTED at about ' + str(frozen) + ' s timing error. Window pairs must be rebuilt on a common grid in Week 7.' if accepted else 'fusion UNAVAILABLE. Keep single-device results and an unpaired phone-versus-watch comparison.'}

Limits: the envelope resolution is 0.1 s; the audit uses training users only; the phone position
(hand, pocket, bag) varies by user and weakens correlation with the wrist-worn watch.
"""
open("alignment_audit.md", "w").write(md)
print(md)
