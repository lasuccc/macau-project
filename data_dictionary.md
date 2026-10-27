# Data dictionary: HAR-PMD

Dataset: Woo et al., Human Activity Recognition Dataset for Pedestrians with Mobility Disabilities.
Zenodo, doi:10.5281/zenodo.7939223 (CC BY 4.0). Paper: Scientific Data 13:211 (2026), doi:10.1038/s41597-025-06527-y.

## File layout

    data_publish/<user>/device.txt        smartphone model
    data_publish/<user>/os_version.txt    Android version
    data_publish/<user>/phone/<user>_<activity>_phone_<environment>.csv
    data_publish/<user>/watch/<user>_<activity>_watch_<environment>.csv   (users 61-120 only)

Total: 2160 CSV files (120 users x 12 phone files, plus 60 users x 12 watch files). Verified by audit.py.

## Identifiers

| Field | Values | Source |
|---|---|---|
| user | 1 to 120. Users 1-60 have phone only; 61-120 have phone and watch | folder name and filename prefix |
| activity | still, walking, crutches, walker, manual (manual wheelchair), electric (electric wheelchair) | filename |
| device | phone, watch | subfolder and filename |
| environment | indoor, outdoor | filename |

The user ID is an identifier only. It must never be a model input, and test users must be unseen.

## Columns

Every file has a Time column. Units follow the Android Sensors API.

| Column(s) | Description | Unit | Phone | Watch |
|---|---|---|---|---|
| Time | Seconds since the start of this recording (restarts at 0 in every file) | s | yes | yes |
| LAccX, LAccY, LAccZ | Linear acceleration (gravity excluded) | m/s^2 | yes | yes |
| GyrX, GyrY, GyrZ | Rate of rotation | rad/s | yes | yes |
| MagX, MagY, MagZ | Geomagnetic field strength | uT | yes | yes |
| GraX, GraY, GraZ | Gravity vector | m/s^2 | yes | yes |
| AccX, AccY, AccZ | Acceleration including gravity | m/s^2 | yes | yes |
| Ori_Azimuth, Ori_Pitch, Ori_Roll | Orientation angles | rad | yes | no |
| RotVec_0 to RotVec_3 | Rotation vector | unitless | yes | no |
| Game_RotVec_0 to Game_RotVec_3 | Rotation vector without magnetometer | unitless | yes | no |
| Pressure | Air pressure | hPa | yes | no |
| Height | Altitude | m | yes | no |
| Light | Illuminance | lx | yes | no |
| Step | Step count | steps | yes | no |
| Proxi | Proximity | cm | yes | no |

Phone files: 32 columns. Watch files: 16 columns. The parser adds user, activity, device and environment.

## Collection facts (from the paper)

- Target rate 60 Hz. Reported average rates: phone 59.6 Hz, watch 51.7 Hz. Released data are not interpolated.
- Each activity was recorded for 11 minutes per environment; the first and last 30 s were removed, leaving 600 s per file.
- The watch was worn on the non-dominant wrist. Phone position was chosen by the participant (hand, pocket, bag, or other) and stayed fixed within an activity. The position is not recorded in the files.
- Participants were 127 recruited (mean age 23.8, SD 2.7, 44 female); 7 were excluded, leaving 120. They did not have mobility disabilities and simulated the aid conditions.
- Indoor and outdoor recordings come from two separate visits.

## Known limits

- Time restarts at 0 in every file, so phone and watch recordings of the same activity cannot be aligned by timestamp.
- Smartwatch data exists for only 60 users.
- Sampling is irregular, especially for the watch.
