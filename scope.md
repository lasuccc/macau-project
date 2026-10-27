# Project scope: personalized federated learning for wearable activity recognition

## Task
Classify the activity performed in a window of phone or watch motion data. The task is activity classification, not participant identification.

## Dataset
HAR-PMD: Woo et al., Human Activity Recognition Dataset for Pedestrians with Mobility Disabilities. Zenodo, doi:10.5281/zenodo.7939223 (CC BY 4.0). Described in Scientific Data 13:211 (2026), doi:10.1038/s41597-025-06527-y.

120 users, 6 activities (still, walking, crutches, walker, manual wheelchair, electric wheelchair), indoor and outdoor, 600 s per recording. Phone data for all 120 users, watch data for users 61 to 120.

The project overview names the WISDM dataset (51 users, 18 activities). The supervisor confirmed at the first supervision meeting that HAR-PMD is acceptable. Split sizes and window length will be set from HAR-PMD's population in Week 2 and checked with the supervisor.

## Research questions
- RQ1 (priority): centralized training versus federated averaging on the same held-out users.
- RQ3 (priority): does fine-tuning on a small labelled support set improve a global model for an unseen user?
- RQ2 (exploratory): heterogeneity across training users, plus one controlled reassignment of training windows.
- RQ4 (exploratory, conditional): phone versus watch input, and fusion only if pairing is reliable.

## Planned protocol
- Participant-disjoint training, validation and test users with split seed 42. The 31/10/10 split assumed 51 users, so sizes will be set in Week 2 from the eligible population.
- Non-overlapping 10 s windows inside each user, activity and continuous recording segment.
- Features: mean, standard deviation, minimum, maximum and RMS per axis. For linear acceleration and gyroscope that is 30 features per device.
- Preprocessing fitted on training users only.
- Baselines: logistic regression and a small MLP, evaluated with one shared evaluator.

## What the Week 1 audit changes
- All 2160 files parse, with exactly 600 s each. Sampling is about 59.7 Hz (phone) and 52.0 Hz (watch), irregular for the watch.
- 11 files have timestamp resets (one with negative time values) and 3 have duplicate rows. Windows will be built inside continuous segments, with each exclusion logged.
- A few users have poor sampling: user 2 phone (about 45 to 47 Hz) and user 115 watch (about 30 Hz). User 2 accounts for 90% of the bad phone windows. These are candidates for a flag or exclusion in Week 2.
- Time restarts at 0 in every file, so phone and watch recordings cannot be paired by timestamp. RQ4 fusion is conditional on an alignment check, with single-device results as the fallback.
- Watch experiments cover 60 users, not 120.

## Why the participant ID is not a feature
The ID identifies the person, not the activity. A model that sees it could score well by memorizing users, which says nothing about activity recognition, and it would fail on test users, who are unseen by design. It would also break the participant-disjoint evaluation. In this dataset the ID is the folder name and the filename prefix, so it must be dropped when feature tables are built.

## Out of scope
New data collection, clinical claims, real-device networking, complex deep architectures, formal privacy mechanisms, full meta-learning.

## Limits to state in the report
Participants had no mobility disabilities and simulated the aid conditions. Recordings are 10 minutes per activity under supervision. Participants were young (mean age 23.8). Phone position varies by user and is not recorded in the files. The study is a single-computer simulation, so it does not show deployment feasibility or privacy protection.
