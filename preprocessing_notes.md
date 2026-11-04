# Preprocessing notes, week 2

All the settings are in preprocessing_config.json. The code is in pipeline.py and build_pipeline.py.

## What the timestamps mean

The Time column counts seconds from the start of each recording. The dataset authors already cut 30 seconds off the start and end of every 11 minute recording. Every file is exactly 600 seconds long, so I did not trim anything.

## Cutting files into pieces

I cut a file into separate pieces, called segments, wherever the clock jumps backwards or the sensor goes quiet for more than 0.2 seconds. I never sorted the rows. If a timestamp repeats, I keep the first row and note whether the repeated row had different values.

Across all 2,160 files I found 12 clock jumps and 13,557 gaps over 0.2 seconds. The gaps sit in 199 files, and most of those are phone files. No timestamp repeated.

I picked the 0.2 second limit myself, so I tested other limits on the 36 training users. The table shows the share of windows each limit keeps.

| Gap limit | Phone | Watch |
|---|---|---|
| 0.2 s | 99.63% | 99.90% |
| 0.3 s | 99.65% | 99.92% |
| 0.5 s | 99.93% | 99.94% |
| 1.0 s | 99.99% | 99.94% |

The strictest limit loses less than half a percent of phone windows and almost nothing for the watch. I kept 0.2 seconds.

## Windows

The sensors record at uneven times. To get exactly 60 readings per second, I drew straight lines between neighbouring readings and read the line at even times. I only do this inside one segment, and I never fill a gap.

Each window is 5 seconds long, which is 300 readings per channel. Windows do not overlap. If a window contains a broken value (not a number), I drop it. No window had one.

## Features

Each window becomes 30 numbers. For each of six channels (three for acceleration, three for rotation) I take the mean, standard deviation, minimum, maximum and RMS. RMS is the typical size of the values. The file feature_columns.txt lists all 30 names.

The features are not scaled yet. In week 3 I will scale them using the training users only.

The feature files hold the 30 numbers and a window_id, and nothing else. User, activity, environment, file name and time are in window_manifest.csv, so the model never sees them.

I saved the features as compressed CSV because the pyarrow package on my laptop would not load.

## How many windows I kept

| | Phone | Watch |
|---|---|---|
| Possible (files x 120) | 172,800 | 86,400 |
| After cutting into segments | 169,432 (98.05%) | 86,315 (99.90%) |
| After the broken value check | 169,432 | 86,315 |

Most of the phone loss comes from a few users with no watch. User 17 lost 1,204 of 1,440 windows and user 2 lost 785. Users 21, 90, 89, 38, 93 and 43 lost between 127 and 227 each. Users 89, 90 and 93 have a watch, and they keep about 86 to 88% of their phone windows.

## Who is in the study

Users 61 to 120 have watch data. All 60 had enough windows in all six activities, so all 60 are in. Users 1 to 60 only have a phone, so they are out of the watch study.

I split the 60 users into 36 train, 12 validation and 12 test, using seed 42. This keeps the 31/10/10 ratio from the project overview. Nobody is in two groups. The classes are balanced: about 8,630 windows per class in train, 2,880 in validation and 2,870 to 2,880 in test. The counts are in split_class_counts.csv.

## Phone and watch timing

The phone and watch clocks do not line up. The details are in alignment_audit.md. I will not combine the two devices.

## Checks

checks.py passes. All features are numbers, every window_id is unique and matches the manifest, no user or environment column is in the features, and no user appears in two groups. The script also picked 40 windows at random, rebuilt them from the raw files, and got the same features.
