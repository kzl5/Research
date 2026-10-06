#!/usr/bin/env python3
"""Compare UR actual TCP, VICON, and actual-joint forward kinematics.

Install: python -m pip install numpy scipy pandas matplotlib
Run: python compare_poses.py --ur ur5e_log_2026-10-02_20-54-19.csv --vicon 5spire3.csv
Use --no-show for batch export; --no-video to skip MP4 (otherwise ffmpeg required).

Conventions: column vectors, metres, radians; ^U R_V maps VICON vectors into
UR-base vectors. Tracker RX/RY/RZ are assumed helical rotation vectors, NOT
Euler angles. UR uses rotation vectors. Only actual_tcp_* and actual_q* UR
fields are read. FK uses the DH model from forwardkinematics1.py, zero TCP
offset, and no mobile-base transform. calibration.json preserves prior alignment.

Full physical model: ^U T_TCP = ^U T_V @ ^V T_marker @ ^marker T_TCP.
Nearly constant orientations make the two translations non-identifiable.
This script estimates p_U = R_UV @ p_V + t_effective and
R_TCP = R_UV @ R_marker @ C_marker_TCP. t_effective includes the unknown
centroid/TCP displacement at the nominal orientation. It is NOT a separately
measured frame-origin translation. A zero lever-arm gauge is used, so changes
in marker orientation can produce uncorrected lever-arm position effects.
Calibration and timing are estimated on the first complete common loop;
subsequent samples are evaluated without refitting or time warping.
The comparison is truncated to whole loops; the final partial loop is omitted.
"""
from pathlib import Path
import argparse
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
from matplotlib.lines import Line2D


def read_ur(path):
    """Read actual measured TCP and actual joints; preserve elapsed timestamps."""
    data = pd.read_csv(path)
    cols = ['time_s'] + [f'actual_tcp_{a}_m' for a in 'xyz'] + [
        f'actual_tcp_{a}_rad' for a in ('rx', 'ry', 'rz')] + [
        f'actual_q{i}_rad' for i in range(1, 7)]
    raw = data[cols].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(raw).all(axis=1)
    d = raw.loc[valid].drop_duplicates('time_s').sort_values('time_s')
    a = np.array(d.to_numpy(), dtype=float, copy=True)    
    if len(a) < 100:
        raise ValueError('Too few valid UR samples.')
    return {'t': a[:, 0], 'p': a[:, 1:4], 'R': Rotation.from_rotvec(a[:, 4:7]),
            'q': a[:, 7:13], 'rows': len(data), 'removed': int(len(data)-len(a))}


def dh_transformation_matrix(q, a, d, alpha):
    """NumPy form of the standard DH matrix in forwardkinematics1.py.

    q may be an array of angles. Exactly the same DH convention and dimensions
    as the supplied SymPy script; vectorized for all logged joint samples.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    cq, sq, ca, sa = np.cos(q), np.sin(q), np.cos(alpha), np.sin(alpha)
    A = np.zeros((len(q), 4, 4))
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = cq, -sq*ca, sq*sa, a*cq
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = sq, cq*ca, -cq*sa, a*sq
    A[:, 2, 1], A[:, 2, 2], A[:, 2, 3], A[:, 3, 3] = sa, ca, d, 1
    return A


def ur5e_fk(joint_angles):
    """Return N x 4 x 4 UR-base-to-flange transforms for actual joints (radians).

    Exact parameter values from the supplied ur5e_fk; zero TCP offset as
    confirmed by the user. No mobile-base transform or fitted FK correction.
    """
    q = np.atleast_2d(np.asarray(joint_angles, dtype=float))
    if q.shape[1] != 6:
        raise ValueError('FK requires six joint angles per sample.')
    a = [0, -0.425, -0.3922, 0, 0, 0]
    d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
    alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    for j in range(6):
        T = T @ dh_transformation_matrix(q[:, j], a[j], d[j], alpha[j])
    return T


def fk_pose_from_joints(ur):
    matrices = ur5e_fk(ur['q'])
    return {'t': ur['t'], 'p': matrices[:, :3, 3],
            'R': Rotation.from_matrix(matrices[:, :3, :3]), 'T': matrices}


def read_vicon(path):
    """Read the supplied Tracker Objects export and its declared frame rate."""
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('Frame,Sub Frame,'))
    rate = float(lines[header-2].split(',')[0])
    units = lines[header+1].split(',')
    if units[2:5] != ['rad']*3 or units[5:8] != ['mm']*3:
        raise ValueError('Expected VICON radians and millimetres.')
    d = pd.read_csv(path, skiprows=header)
    d = d[['Frame', 'Sub Frame', 'RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']].apply(
        pd.to_numeric, errors='coerce')
    numeric = d.loc[d['Frame'].notna()]
    valid = np.isfinite(numeric).all(axis=1)
    good = numeric.loc[valid].sort_values(['Frame', 'Sub Frame'])
    if not (good['Sub Frame'] == 0).all():
        raise ValueError('Nonzero VICON subframes require an explicit subframe time model.')
    good = good.drop_duplicates('Frame')
    a = np.array(good.to_numpy(), dtype=float, copy=True)
    return {'t': (a[:, 0]-1)/rate, 'p': a[:, 5:8]/1000,
            'R': Rotation.from_rotvec(a[:, 2:5]), 'rate': rate,
            'rows': len(numeric), 'removed': int(len(numeric)-len(a))}


def loop_bottoms(pose):
    """Detect repeated Z minima for these loops (both recording Z axes point up).

    A short filter is used ONLY to locate events, never to alter displayed data.
    This dataset-specific detector rejects partial endpoint minima. Override
    starts and period on the command line for another trajectory geometry.
    """
    t, z = pose['t'], pose['p'][:, 2]
    dt = np.median(np.diff(t))
    uniform = np.arange(t[0], t[-1], dt)
    n = max(5, int(round(0.4/dt)) | 1)
    smooth = savgol_filter(np.interp(uniform, t, z), n, 2)
    peaks, _ = find_peaks(-smooth, distance=int(8/dt), prominence=0.1)
    if len(peaks) < 2:
        raise ValueError('Cannot detect loops; provide --ur-start --vicon-start --period.')
    times = uniform[peaks]
    return times, float(np.median(np.diff(times)))


def sample_pose(pose, times):
    """Linear position interpolation and geodesic (SLERP) rotation interpolation.

    Never extrapolate or interpolate across tracking gaps exceeding 0.1 seconds.
    This only pairs measured samples; no commanded poses are generated.
    """
    t = pose['t']
    times = np.asarray(times)
    if times.min() < t[0] or times.max() > t[-1]:
        raise ValueError('Requested times extend beyond measured data.')
    j = np.clip(np.searchsorted(t, times, side='right'), 1, len(t)-1)
    if np.any(t[j]-t[j-1] > 0.1):
        raise ValueError('Tracking gap >0.1 s in comparison; select a continuous interval.')
    p = np.column_stack([np.interp(times, t, pose['p'][:, k]) for k in range(3)])
    return p, Slerp(t, pose['R'])(times)


def rigid_position_fit(source, destination):
    """Least-squares Kabsch rigid registration; no scale fit or reflection."""
    x = source-source.mean(axis=0)
    y = destination-destination.mean(axis=0)
    u, s, vt = np.linalg.svd(x.T @ y)
    if s[-1] < s[0]*1e-5:
        raise ValueError('Calibration path does not constrain a full 3D rotation.')
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ correction @ u.T
    t = destination.mean(axis=0)-R @ source.mean(axis=0)
    return R, t, s


def homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def transform_pose(p, rotations, T_effective, C_marker_tcp):
    """Map positions with the effective offset; align body axes on the right."""
    R = Rotation.from_matrix(T_effective[:3, :3])
    return R.apply(p)+T_effective[:3, 3], R*rotations*C_marker_tcp


def estimate_alignment(ur, vicon, ur_start, vicon_start, period, refine=True):
    """Estimate timing, spatial registration and mounting orientation on ONE loop.

    There is a whole-loop synchronization ambiguity. Choose the first detected
    complete loop in each recording; optimize only a local +/-0.5 s phase shift.
    No time scaling, dynamic time warping or holdout calibration is performed.
    """
    elapsed = np.arange(0, period, 0.02)
    pv, rv = sample_pose(vicon, vicon_start+elapsed)

    def fit(delta):
        pu, _ = sample_pose(ur, ur_start+delta+elapsed)
        R, t, s = rigid_position_fit(pv, pu)
        residual = pv @ R.T+t-pu
        return float(np.mean(np.sum(residual**2, axis=1))), R, t, s

    delta = 0.0
    if refine:
        opt = minimize_scalar(lambda d: fit(d)[0], bounds=(-0.5, 0.5),
                              method='bounded', options={'xatol': 1e-7})
        if not opt.success:
            raise RuntimeError('Timing optimization failed.')
        delta = float(opt.x)
        if abs(delta) > 0.49:
            warnings.warn('Timing optimum is near search boundary; check start events.')
    _, R, t, s = fit(delta)
    _, ru = sample_pose(ur, ur_start+delta+elapsed)
    # C is a fixed marker-axis to TCP-axis rotation, estimated on calibration data.
    C = ((Rotation.from_matrix(R)*rv).inv()*ru).mean()
    return homogeneous(R, t), C, ur_start+delta, delta, s


def principal_angle(reference, measured):
    """Angle of R_reference.T @ R_measured, in [0, pi] radians."""
    return (reference.inv()*measured).magnitude()


def equal_axes(ax, points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo+hi)/2
    radius = max((hi-lo).max()/2, 0.02)*1.25
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel='UR-base X (m)', ylabel='UR-base Y (m)', zlabel='UR-base Z (m)')
    ax.view_init(elev=25, azim=-55)


def draw_frame(ax, p, R, length=0.035, alpha=1, linestyle='-'):
    """Draw local body axes as columns of the orientation matrix."""
    for k, color in enumerate(('crimson', 'green', 'royalblue')):
        q = p+length*R[:, k]
        ax.plot(*np.vstack((p, q)).T, color=color, alpha=alpha, linestyle=linestyle)


def make_plots(ur, vicon, fk, T, C, starts, times, out):
    """Overlay raw measured UR/VICON and FK derived from native actual joints."""
    u0, v0 = starts
    duration = times[-1]
    umask = (ur['t'] >= u0) & (ur['t'] <= u0+duration)
    vmask = (vicon['t'] >= v0) & (vicon['t'] <= v0+duration)
    vp, vr = transform_pose(vicon['p'][vmask], vicon['R'][vmask], T, C)
    series = [('UR actual TCP', ur['p'][umask], ur['R'][umask], ur['t'][umask]-u0, 'darkorange', '-'),
              ('VICON reference', vp, vr, vicon['t'][vmask]-v0, 'purple', '--'),
              ('FK from actual joints', fk['p'][umask], fk['R'][umask], fk['t'][umask]-u0, 'deepskyblue', ':')]
    fig = plt.figure(figsize=(14, 7))
    a = fig.add_subplot(121, projection='3d')
    b = fig.add_subplot(122, projection='3d')
    for label, p, r, t, color, style in series:
        a.plot(*p.T, color=color, linestyle=style, linewidth=1.3, label=label)
        a.scatter(*p[0], color=color, s=30)
        for i in np.linspace(0, len(p)-1, 8, dtype=int):
            draw_frame(a, p[i], r[i].as_matrix(), alpha=.4, linestyle=style)
        vectors = r.as_rotvec()
        b.plot(*vectors.T, color=color, alpha=.45, linewidth=.7, label=label)
        b.scatter(*vectors[::10].T, c=t[::10], cmap='viridis', s=2)
    equal_axes(a, np.vstack([s[1] for s in series]))
    a.set_title('Three overlaid trajectories and local pose axes')
    a.legend(loc='upper left', fontsize=8)
    b.set(xlabel='rotvec X (rad)', ylabel='rotvec Y (rad)', zlabel='rotvec Z (rad)',
          title='Orientation over time (color = elapsed seconds)')
    b.legend(fontsize=8)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, duration), cmap='viridis'),
                      ax=b, shrink=.65, pad=.1)
    cb.set_label('Elapsed time (s)')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out/'trajectory_3d.png', dpi=180)
    fig2, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for label, p, r, t, color, style in series:
        vectors = r.as_rotvec()
        for k, axis in enumerate('XYZ'):
            axs[0, k].plot(t, p[:, k], color=color, linestyle=style, linewidth=.9, label=label)
            axs[1, k].plot(t, vectors[:, k], color=color, linestyle=style, linewidth=.9)
            axs[0, k].set_ylabel(f'{axis} (m)')
            axs[1, k].set(ylabel=f'rotvec {axis} (rad)', xlabel='Common elapsed time (s)')
    for ax in axs.flat:
        ax.grid(alpha=.25)
    axs[0, 0].legend(fontsize=8)
    fig2.suptitle('UR actual, VICON reference, and forward kinematics in the UR-base frame')
    fig2.tight_layout()
    fig2.savefig(out/'pose_components_over_time.png', dpi=180)


def pose_errors(compared_p, compared_r, reference_p, reference_r):
    """Signed compared-minus-reference translation and unsigned principal angle."""
    return compared_p-reference_p, principal_angle(reference_r, compared_r)


def error_figure(times, xyz_error, angles, period, label):
    """Shared four-panel layout used by every static and animated error plot."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    values = [angles, *xyz_error.T*1000]
    colors = ['purple', 'crimson', 'green', 'royalblue']
    ylabels = ['Principal angle (rad)', 'X error (mm)', 'Y error (mm)', 'Z error (mm)']
    for k, (ax, y, color, ylabel) in enumerate(zip(axs, values, colors, ylabels)):
        if k == 0:
            ax.set_ylim(0, max(float(y.max())*1.15, .0001))
        else:
            lo, hi = float(y.min()), float(y.max())
            margin = max((hi-lo)*.12, .01)
            ax.set_ylim(min(lo, 0)-margin, max(hi, 0)+margin)
            ax.axhline(0, color='black', linewidth=.7, alpha=.6)
        ax.set(xlim=(0, times[-1]), ylabel=ylabel)
        ax.axvspan(0, period, color='grey', alpha=.12)
        ax.axvline(period, color='grey', linestyle='--', linewidth=.8)
        ax.grid(alpha=.25)
    axs[-1].set_xlabel('Common elapsed time (s)')
    fig.suptitle(label+' | signed translation = compared − reference', fontsize=12, y=.985)
    fig.tight_layout(rect=(0, 0, 1, .90))
    return fig, axs, values, colors


def plot_errors(times, xyz_error, angles, period, label, path):
    fig, axs, values, colors = error_figure(times, xyz_error, angles, period, label)
    for ax, y, color in zip(axs, values, colors):
        ax.plot(times, y, color=color, linewidth=.9)
    fig.text(.5, .935, 'Complete loops only; shaded first loop used for VICON alignment', ha='center')
    fig.savefig(path, dpi=180)
    return fig


def animation_indices(times, fps, speed):
    """Use the same measured-time schedule for every animation, including endpoint."""
    return np.unique(np.r_[np.searchsorted(times, np.arange(0, times[-1], speed/fps)), len(times)-1])


def animate_pose(times, pu, ru, pv, rv, pf, rf, out, fps=20, speed=2, export=True):
    """Animate UR actual, aligned VICON, and actual-joint FK together."""
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    positions = (pu, pv, pf)
    rotations = (ru, rv, rf)
    colors = ('darkorange', 'purple', 'deepskyblue')
    styles = ('-', '--', ':')
    for p, color in zip(positions, colors):
        ax.plot(*p.T, color=color, alpha=.12, linewidth=.8)
    equal_axes(ax, np.vstack(positions))
    trails = [ax.plot([], [], [], color=c, linestyle=s, linewidth=1.6)[0] for c, s in zip(colors, styles)]
    points = [ax.plot([], [], [], marker='o', color=c, linestyle='')[0] for c in colors]
    axes = [[ax.plot([], [], [], color=c, linewidth=2, linestyle=style)[0]
             for c in ('crimson', 'green', 'royalblue')] for style in styles]
    ax.legend(handles=[Line2D([], [], color=c, linestyle=s, label=l) for c, s, l in
              zip(colors, styles, ('UR actual TCP', 'VICON reference', 'FK from actual joints'))],
              loc='upper left', fontsize=9)
    title = ax.set_title('')
    mats = [r.as_matrix() for r in rotations]
    def update(i):
        for sensor, p in enumerate(positions):
            trails[sensor].set_data_3d(*p[:i+1].T)
            points[sensor].set_data_3d(*p[i:i+1].T)
            for k in range(3):
                endpoint = p[i]+.04*mats[sensor][i, :, k]
                axes[sensor][k].set_data_3d(*np.vstack((p[i], endpoint)).T)
        title.set_text(f'Three poses | t = {times[i]:.2f} s | playback {speed:g}x\n'
                       f'Position versus VICON: UR {1000*np.linalg.norm(pu[i]-pv[i]):.2f} mm, '
                       f'FK {1000*np.linalg.norm(pf[i]-pv[i]):.2f} mm')
        return trails+points+[a for group in axes for a in group]+[title]
    anim = FuncAnimation(fig, update, frames=animation_indices(times, fps, speed),
                         interval=1000/fps, blit=False)
    if export:
        if not writers.is_available('ffmpeg'):
            raise RuntimeError('Install ffmpeg or use --no-video.')
        anim.save(out/'pose_animation.mp4', writer=FFMpegWriter(fps=fps, bitrate=2200), dpi=100)
    return anim


def animate_errors(times, xyz_error, angles, period, label, path, fps=20, speed=2, export=True):
    """Reveal principal angle and signed X/Y/Z errors with synchronized cursors."""
    fig, axs, values, colors = error_figure(times, xyz_error, angles, period, label)
    lines, points, cursors = [], [], []
    for ax, color in zip(axs, colors):
        lines.append(ax.plot([], [], color=color, linewidth=1.3)[0])
        points.append(ax.plot([], [], marker='o', color=color, markersize=4, linestyle='')[0])
        cursors.append(ax.axvline(0, color=color, alpha=.3, linewidth=.8))
    status = fig.text(.5, .925, '', ha='center', fontsize=10)
    def update(i):
        for line, point, cursor, y in zip(lines, points, cursors, values):
            line.set_data(times[:i+1], y[:i+1])
            point.set_data([times[i]], [y[i]])
            cursor.set_xdata([times[i], times[i]])
        xyz = xyz_error[i]*1000
        status.set_text(f't = {times[i]:.2f} s | angle = {angles[i]:.5f} rad '
                        f'({np.rad2deg(angles[i]):.3f}°) | playback {speed:g}x\n'
                        f'X = {xyz[0]:+.3f} mm | Y = {xyz[1]:+.3f} mm | Z = {xyz[2]:+.3f} mm')
        return lines+points+cursors+[status]
    anim = FuncAnimation(fig, update, frames=animation_indices(times, fps, speed),
                         interval=1000/fps, blit=False)
    if export:
        if not writers.is_available('ffmpeg'):
            raise RuntimeError('Install ffmpeg or use --no-video.')
        anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2200), dpi=100)
    return anim


def error_stats(position, angle, mask):
    if not np.any(mask):
        return None
    p, a = position[mask]*1000, angle[mask]
    return {'samples': int(mask.sum()), 'position_rmse_mm': float(np.sqrt(np.mean(p*p))),
            'position_median_mm': float(np.median(p)), 'position_p95_mm': float(np.percentile(p, 95)),
            'position_max_mm': float(p.max()), 'principal_angle_rms_rad': float(np.sqrt(np.mean(a*a))),
            'principal_angle_p95_rad': float(np.percentile(a, 95)),
            'principal_angle_max_rad': float(a.max())}


def comparison_stats(xyz_error, angle, times, period):
    result = {}
    for label, mask in [('all', np.ones(len(times), dtype=bool)),
                        ('calibration', times < period), ('holdout', times >= period)]:
        stats = error_stats(np.linalg.norm(xyz_error, axis=1), angle, mask)
        if stats is not None:
            e = xyz_error[mask]*1000
            stats['xyz_bias_mm'] = e.mean(axis=0).tolist()
            stats['xyz_rmse_mm'] = np.sqrt(np.mean(e*e, axis=0)).tolist()
            stats['xyz_max_abs_mm'] = np.abs(e).max(axis=0).tolist()
        result[label] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ur', default='C:\\Users\\kevin\\Documents\\GitHub\\Research\\FKRotationmatrices\\ErrorLogging\\ur5e_log_2026-10-02_20-54-19.csv')
    parser.add_argument('--vicon', default='C:\\Users\\kevin\\Documents\\GitHub\\Research\\FKRotationmatrices\\ErrorLogging\\5spire3.csv')
    parser.add_argument('--out', default='C:\\Users\\kevin\\Documents\\GitHub\\Research\\FKRotationmatrices\\ErrorLogging\\results')
    parser.add_argument('--ur-start', type=float, help='UR absolute time at common loop bottom, seconds')
    parser.add_argument('--vicon-start', type=float, help='VICON absolute time at common loop bottom, seconds')
    parser.add_argument('--period', type=float, help='Calibration loop duration, seconds')
    parser.add_argument('--no-refine', action='store_true', help='Use supplied/event starts exactly')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--speed', type=float, default=2, help='Animation playback speed multiplier')
    parser.add_argument('--calibration', default=str(Path(__file__).with_name('calibration.json')),
                        help='Saved fixed calibration and timing for these recordings')
    parser.add_argument('--refit', action='store_true', help='Explicitly refit alignment for new recordings')
    args = parser.parse_args()
    if args.fps <= 0 or args.speed <= 0:
        parser.error('fps and speed must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ur, vicon = read_ur(args.ur), read_vicon(args.vicon)
    fk = fk_pose_from_joints(ur)
    if args.refit:
        ue, up = loop_bottoms(ur) if args.ur_start is None or args.period is None else ([], args.period)
        ve, vp = loop_bottoms(vicon) if args.vicon_start is None or args.period is None else ([], args.period)
        u0 = float(ue[0]) if args.ur_start is None else args.ur_start
        v0 = float(ve[0]) if args.vicon_start is None else args.vicon_start
        period = float(np.median([up, vp])) if args.period is None else args.period
        T, C, u0, delta, singular = estimate_alignment(ur, vicon, u0, v0, period, not args.no_refine)
    else:
        if any(x is not None for x in (args.ur_start, args.vicon_start, args.period)):
            parser.error('Start/period overrides require --refit; fixed calibration includes timing.')
        saved = json.loads(Path(args.calibration).read_text())
        if Path(args.ur).name != saved['ur_file'] or Path(args.vicon).name != saved['vicon_file']:
            parser.error('Fixed calibration belongs to the supplied recordings; use --refit for others.')
        for path in (Path(args.ur), Path(args.vicon)):
            digest = saved.get('input_sha256', {}).get(path.name)
            if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                parser.error('Input contents differ from fixed calibration; use --refit for new recordings.')
        T = np.array(saved['effective_vicon_to_ur_matrix'])
        C = Rotation.from_matrix(np.array(saved['marker_to_tcp_orientation_zero_translation_gauge'])[:3, :3])
        u0, v0, period = (saved[k] for k in ('ur_common_start_s', 'vicon_common_start_s', 'calibration_loop_s'))
        delta, up, vp = (saved[k] for k in ('ur_phase_refinement_s', 'detected_ur_period_s', 'detected_vicon_period_s'))
        singular = np.array(saved['kabsch_singular_values'])
    available_duration = min(ur['t'][-1]-u0, vicon['t'][-1]-v0)
    complete_loops = int(np.floor(available_duration/period))
    if complete_loops < 1:
        raise ValueError('The synchronized overlap does not contain a complete loop.')
    duration = complete_loops*period
    # Coarser sensor's nominal rate. Do not invent data outside the overlap.
    dt = max(np.median(np.diff(ur['t'])), 1/vicon['rate'])
    times = np.arange(0, duration, dt)
    if not np.isclose(times[-1], duration, rtol=0, atol=1e-9):
        times = np.r_[times, duration]
    pu, ru = sample_pose(ur, u0+times)
    pv, rv = sample_pose(vicon, v0+times)
    pv, rv = transform_pose(pv, rv, T, C)
    pf, rf = sample_pose(fk, u0+times)
    make_plots(ur, vicon, fk, T, C, (u0, v0), times, out)
    comparisons = [
        ('ur_vs_vicon', 'UR actual − VICON reference', *pose_errors(pu, ru, pv, rv),
         'pose_errors_over_time.png', 'principal_rotation_error_animation.mp4'),
        ('fk_vs_vicon', 'FK − VICON reference', *pose_errors(pf, rf, pv, rv),
         'fk_vs_vicon_errors_over_time.png', 'fk_vs_vicon_error_animation.mp4'),
        ('fk_vs_ur', 'FK − UR actual reference', *pose_errors(pf, rf, pu, ru),
         'fk_vs_ur_errors_over_time.png', 'fk_vs_ur_error_animation.mp4')]
    for key, label, xyz, angle, png, video in comparisons:
        plot_errors(times, xyz, angle, period, label, out/png)
    epos = np.linalg.norm(comparisons[0][2], axis=1)
    eangle = comparisons[0][3]
    native_mask = (ur['t'] >= u0) & (ur['t'] <= u0+duration)
    np.savez_compressed(out/'fk_transformations.npz', elapsed_s=ur['t'][native_mask]-u0,
                        ur_absolute_s=ur['t'][native_mask], actual_joints_rad=ur['q'][native_mask],
                        T_ur_base_tcp=fk['T'][native_mask])
    C4 = homogeneous(C.as_matrix(), np.zeros(3))
    np.savetxt(out/'effective_vicon_to_ur_matrix.csv', T, delimiter=',', fmt='%.10f')
    np.savetxt(out/'marker_to_tcp_orientation_matrix.csv', C4, delimiter=',', fmt='%.10f')
    rows = {'elapsed_s': times, 'ur_absolute_s': u0+times, 'vicon_absolute_s': v0+times,
            'position_error_mm': epos*1000, 'principal_angle_rad': eangle,
            'principal_angle_deg': np.rad2deg(eangle), 'is_calibration': times < period}
    for prefix, p, r in [('ur_actual', pu, ru), ('vicon_aligned', pv, rv), ('fk', pf, rf)]:
        for k, a in enumerate('xyz'):
            rows[f'{prefix}_{a}_m'] = p[:, k]
        for k, a in enumerate(('rx', 'ry', 'rz')):
            rows[f'{prefix}_{a}_rad'] = r.as_rotvec()[:, k]
    for key, label, xyz, angle, png, video in comparisons:
        for k, a in enumerate('xyz'):
            rows[f'{key}_{a}_error_mm'] = xyz[:, k]*1000
        rows[f'{key}_position_error_mm'] = np.linalg.norm(xyz, axis=1)*1000
        rows[f'{key}_principal_angle_rad'] = angle
    pd.DataFrame(rows).to_csv(out/'paired_actual_poses_and_errors.csv', index=False)
    report = {'model': 'p_U = R_UV p_V + t_effective; R_TCP = R_UV R_marker C_marker_TCP',
              'limitations': ['t_effective combines frame-origin offset and nominal centroid/TCP offset.',
                              'Physical centroid/TCP lever arm and frame translation cannot be separated reliably with constant orientation.',
                              'Orientation mounting bias is fitted on one loop; angle errors measure residual disagreement, not absolute calibration accuracy.',
                              'Repeated-loop index is ambiguous without external synchronization.',
                              'Time offset is fitted on one loop; no speed scaling or time warping.',
                              'VICON rotations assumed helical rotation vectors as in default Tracker CSV export.'],
              'effective_vicon_to_ur_matrix': T.tolist(), 'marker_to_tcp_orientation_zero_translation_gauge': C4.tolist(),
              'ur_common_start_s': u0, 'vicon_common_start_s': v0, 'ur_phase_refinement_s': delta,
              'calibration_loop_s': period, 'detected_ur_period_s': up, 'detected_vicon_period_s': vp,
              'overlap_s': float(times[-1]), 'comparison_dt_s': dt,
              'available_overlap_s': float(available_duration),
              'excluded_final_partial_loop_s': float(available_duration-duration),
              'rotation_det': float(np.linalg.det(T[:3, :3])), 'kabsch_singular_values': singular.tolist(),
              'start_position_disagreement_mm': float(epos[0]*1000),
              'ur_valid_samples': len(ur['t']), 'vicon_valid_samples': len(vicon['t']),
              'ur_removed_rows': ur['removed'], 'vicon_removed_rows': vicon['removed'],
              'all': error_stats(epos, eangle, np.ones(len(times), dtype=bool)),
              'calibration': error_stats(epos, eangle, times < period),
              'holdout': error_stats(epos, eangle, times >= period),
              'complete_loop_count': complete_loops,
              'holdout_complete_loops': error_stats(epos, eangle, times >= period),
              'alignment_refitted': args.refit,
              'signed_error_convention': 'compared minus reference, expressed in UR-base axes',
              'fk_model': 'Standard DH dimensions and order from forwardkinematics1.py; actual_q1_rad..actual_q6_rad; zero flange-to-TCP offset; no mobile-base transform',
              'comparisons': {key: comparison_stats(xyz, angle, times, period)
                              for key, label, xyz, angle, png, video in comparisons},
              'ur_orientation_from_start_max_rad': float(principal_angle(ru[0], ru).max()),
              'vicon_orientation_from_start_max_rad': float(principal_angle(rv[0], rv).max())}
    (out/'alignment_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    animations = []
    if not (args.no_show and args.no_video):
        animations.append(animate_pose(times, pu, ru, pv, rv, pf, rf, out, args.fps, args.speed, not args.no_video))
        for key, label, xyz, angle, png, video in comparisons:
            animations.append(animate_errors(times, xyz, angle, period, label, out/video,
                                             args.fps, args.speed, not args.no_video))
    if not args.no_show:
        plt.show()
    else:
        # Retain animation through save, then release figures for batch runs.
        plt.close('all')
    return report, animations


if __name__ == '__main__':
    main()
