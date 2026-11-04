# Phone and watch alignment audit (training users only)

Files carry `Time` in seconds since each recording started, so no shared clock is stored. Resetting both
first readings to zero is not evidence of alignment. This audit estimates the offset between each phone
file and the watch file for the same user, activity and environment from the 0.1 s gyroscope-magnitude
envelope (cross-correlation, lags up to +/-30 s), and compares it with control pairs
(phone of one user against the watch of a different training user, same activity and environment).

Rule, fixed before looking at validation or test users:
- control threshold = 95th percentile of control peak correlation = 0.119
- a true pair passes if peak correlation >= that threshold and |offset| <= 1.0 s
- pairing is accepted if at least 80% of true training pairs pass

Evidence (36 training users, activities walking, crutches, walker, manual, electric):
- true pairs used: 353; control pairs used: 353; skipped for timestamp resets or missing files: 14
- median peak correlation: true 0.340, control 0.056
- true pairs passing: 29.2%; control pairs passing: 0.6%
- |offset| among passing pairs: median 0.30 s, 95th percentile 0.90 s
- overlap of the two recordings after the offset: median 597.5 s
- phone windows with full watch coverage after the offset: 41684 of 42255

Decision: fusion UNAVAILABLE. Keep single-device results and an unpaired phone-versus-watch comparison.

Limits: the envelope resolution is 0.1 s; the audit uses training users only; the phone position
(hand, pocket, bag) varies by user and weakens correlation with the wrist-worn watch.

## Interpretation
Real pairs match far better than control pairs (median peak correlation 0.340 against 0.056), so the phone and watch recordings of the same session do contain the same movement: 291 of 353 real pairs (82%) exceed the control threshold. The pairing still fails the frozen rule because only 103 of those 291 have an offset within 1 s. The others are spread over 1 to 3 s (71), 3 to 10 s (113) and 10 to 30 s (4). The two devices therefore have no shared clock, and the offset varies from file to file.

Median peak correlation by activity: electric 0.62, walking 0.52, crutches 0.35, walker 0.26, manual 0.07. Wheelchair pushing gives almost no match, which is plausible if the phone is carried away from the wrists.

Offsets estimated from a 0.1 s gyroscope envelope are approximate, especially for repeating movement such as walking, so they are not used to pair windows.

Consequence: cross-device fusion is marked unavailable. Week 7 compares phone and watch inputs as separate models on the same users and the same split, without pairing windows. An estimate-then-pair method would be a different procedure and would need its own validation.

## What this means

The phone and the watch from the same session record the same movement. Real pairs match far better than mismatched pairs, with a median correlation of 0.340 against 0.056. In total, 291 of 353 real pairs beat the control threshold.

The timing does not match. Only 103 of those 291 pairs have a time offset within 1 second. The rest are off by 1 to 3 seconds (71 pairs), 3 to 10 seconds (113 pairs) or more than 10 seconds (4 pairs). The two devices started recording at different moments, and the gap changes from file to file.

Electric wheelchair and walking match best, with medians of 0.62 and 0.52. Crutches score 0.35, walker 0.26 and manual wheelchair 0.07. Pushing a wheelchair by hand probably moves the wrists in a way that a phone in a pocket or bag does not copy.

The offsets come from a rough method, and they can be wrong when the movement repeats, as in walking. So I do not use them to pair windows.

Decision: I will not combine phone and watch. In week 7 I will train one phone model and one watch model on the same users and the same split, and compare them.
