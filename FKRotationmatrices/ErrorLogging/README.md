# UR5e, VICON, and forward-kinematics comparison

## Run

Extract the ZIP and run from its folder:

```sh
python -m pip install -r requirements.txt
python compare_poses.py
```

This produces static figures, four MP4s, a paired pose/error CSV, the FK
transformation matrices, and a JSON report. It also opens interactive plots
and animations. MP4 export requires FFmpeg on PATH. `--no-video` skips video
export while retaining interactive animations. `--no-show` exports without
opening windows. Defaults: 20 fps, 2x playback, about 25.9 seconds per video.
`--speed 1` gives real-time playback.

## Trajectories and the FK model

Three trajectories are overlaid in the UR-base coordinate system:

1. Logged actual UR TCP position and orientation.
2. Measured VICON rigid-body centroid and orientation, mapped using the
   previously estimated effective alignment.
3. UR5e forward kinematics calculated from `actual_q1_rad` through
   `actual_q6_rad`, using the DH formulas and dimensions in the supplied
   `forwardkinematics1.py`.

No `target_*` data are used. The supplied FK function is reproduced with NumPy
array arithmetic so every native joint sample can be evaluated efficiently.
The original uploaded script is included unchanged for reference, but its
example prints and plotting side effects are not imported at runtime. The
comparison script itself does not need SymPy.

The DH parameters are unchanged:

```text
 a     = [0, -0.425, -0.3922, 0, 0, 0] metres
 d     = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996] metres
 alpha = [pi/2, 0, 0, pi/2, -pi/2, 0] radians
 T     = A1 @ A2 @ A3 @ A4 @ A5 @ A6
```

The user confirmed zero flange-to-TCP offset. The mobile-base transforms in
that script are not applied because comparison is in the UR-base frame. FK
receives no fitted positional or rotational correction. These nominal DH
parameters are compared as supplied; the script does not assume they exactly
reproduce the physical robot's calibrated model.

## Error definitions

Each static error PNG and running error animation has four panels: principal
rotation angle in radians, then signed X, Y, and Z translation errors in mm.
Translation errors are expressed along the UR-base axes. Positive values mean
the compared position is greater than the reference coordinate.

| Comparison | Translation error | Orientation error |
|---|---|---|
| UR versus VICON | p_UR - p_VICON_aligned | angle(R_VICON_aligned.T @ R_UR) |
| FK versus VICON | p_FK - p_VICON_aligned | angle(R_VICON_aligned.T @ R_FK) |
| FK versus UR | p_FK - p_UR | angle(R_UR.T @ R_FK) |

VICON is the reference (treated as truth) for both VICON comparisons. The
requested direct FK-to-UR comparison necessarily uses UR as its reference.
Treating VICON as truth defines how errors are calculated; the fitted
alignment itself remains the previous UR-based calibration, as requested.
The angle is the unsigned principal angle in [0, pi], calculated from relative
rotation matrices, not from differences of rotation-vector components.

## Fixed calibration and timing

`calibration.json` preserves the previous matrices, loop period, and start
times exactly. It is loaded by default; FK is not used to refit the alignment.
For other recordings explicitly use `--refit`; optional start/period overrides
also require `--refit`. `--calibration PATH` selects another saved calibration.
The default saved calibration is intended only for the original supplied logs.

The common nominal event is the bottom of a loop. UR starts at 7.464375 s
and VICON starts at 12.690000 s in their respective files. The period is
12.909570 s. Four whole loops are retained, ending at 51.638280 s of common
elapsed time. The final 5.241720-second partial loop is omitted from every
comparison, graph, animation, and exported pose/matrix data. Input CSVs remain
unchanged. The exact loop endpoint is included in the paired interpolation.

The shaded first loop is the interval originally used to estimate VICON
alignment; the other three loops were not used to fit it. Repeated-loop index
is ambiguous without external synchronization. There is no speed scaling,
time warping, or independent time fit for FK; FK uses the same UR timestamps.

## Units and interpolation

UR joint angles and both sensors' rotation vectors are radians. UR positions
are metres. VICON declares 100 Hz and millimetres; its positions are converted
to metres and timestamps reconstructed from frame numbers. Tracker RX/RY/RZ
are assumed to be helical rotation vectors, as in default Tracker CSV export,
not Euler angles. If the export was deliberately configured to use Euler
angles, change `read_vicon` before using the results. Official reference:
https://vicon-help.atlassian.net/wiki/spaces/Tracker310/pages/13926860/Quaternion%2Boutput%2Bfor%2BCSV%2Bexport

Static path/component figures show valid native samples, including FK computed
at each native joint timestamp. Paired errors and animations use a 100 Hz
common elapsed-time grid, with linear position interpolation and SLERP
rotation interpolation. FK is evaluated at native joint samples before pose
interpolation. No outliers are discarded, no orientation is forced constant,
and no measurements are extrapolated. Tracking gaps over 0.1 s cause an error.

## Effective transform and centroid/TCP limitation

The physical relationship would be:

```text
T_UR_TCP = T_UR_VICON @ T_VICON_marker @ T_marker_TCP
```

These nearly constant-orientation data do not constrain the physical
VICON-to-UR frame translation and centroid-to-TCP lever arm separately with
useful confidence. The retained model is:

```text
p_UR = R_UR_VICON @ p_VICON + t_effective
R_TCP = R_UR_VICON @ R_marker @ C_marker_TCP
```

`effective_vicon_to_ur_matrix.csv` contains R and t_effective (metres):

```text
[-0.9996090164  -0.0042868537   0.0276303667  -0.2119165205]
[ 0.0039318526  -0.9999091931  -0.0128897703  -0.6559788745]
[ 0.0276831143  -0.0127760921   0.9995351003   0.1419786474]
[ 0             0             0             1           ]
```

`marker_to_tcp_orientation_matrix.csv` contains the right-hand orientation
alignment C. Its zero translation is an arbitrary modeling convention that
places all nominal centroid displacement in t_effective. It is not a measured
zero centroid-to-TCP offset. The effective translation must not be used as a
separately measured frame-origin translation. Orientation-dependent lever-arm
residuals remain possible. VICON errors quantify disagreement after this
calibration, not independent absolute metrology accuracy.

## Output files

- `trajectory_3d.png`: three 3D position paths with local body axes, plus three
  orientation trajectories colored by time. Orientation plot axes are rotation
  vectors, not Euler angles.
- `pose_components_over_time.png`: X/Y/Z and rotation-vector components for
  all three trajectories over time.
- `pose_animation.mp4`: three overlaid moving poses.
- `pose_errors_over_time.png` and `principal_rotation_error_animation.mp4`:
  UR minus VICON, now showing principal angle and all three signed translations.
- `fk_vs_vicon_errors_over_time.png` and `fk_vs_vicon_error_animation.mp4`.
- `fk_vs_ur_errors_over_time.png` and `fk_vs_ur_error_animation.mp4`.
- `paired_actual_poses_and_errors.csv`: all three synchronized poses and the
  principal angle, signed translations, and positional norm for all three pairs.
  Legacy unprefixed error columns still refer to UR versus VICON.
- `fk_transformations.npz`: full 4x4 UR-base-to-TCP FK matrices at retained
  native joint sample times, with actual joint angles and elapsed timestamps.
- `alignment_report.json`: timing, limitations, and full-loop/calibration/holdout
  error summaries, including axis-wise signed bias, RMSE, and maximum absolute
  errors. Matrix CSVs and `calibration.json` retain the existing calibration.

Load FK matrices in Python:

```python
import numpy as np
fk = np.load('results/fk_transformations.npz')
T = fk['T_ur_base_tcp']       # shape (N, 4, 4)
t = fk['elapsed_s']          # shape (N,)
q = fk['actual_joints_rad']  # shape (N, 6)
```

## Results for all four retained loops

| Comparison | Position RMSE | Principal-angle RMS |
|---|---:|---:|
| UR versus VICON | 2.158 mm | 0.014595 rad |
| FK versus VICON | 2.191 mm | 0.015556 rad |
| FK versus UR | 0.673 mm | 0.006366 rad |

All comparisons use identical start times, sample times, and loop cutoff.

## Validation

The vectorized FK function was checked against the original scalar function's
expressions at 20 actual joint configurations, using NumPy matrix arithmetic
for the original function: all matrix entries agreed to machine precision.
The fixed calibration was checked for exact preservation. The paired CSV
contains only complete loops and the signed errors were checked directly
against its pose columns. PNGs and video frames were inspected; every MP4
uses the same frame count and duration.
