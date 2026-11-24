import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


# ========================= USER SETTINGS =========================

# State order: world x, world y, chassis yaw, q1, ..., q6.
BASE_START = [0.0, 0.0, 0.0]  # meters, meters, radians
ARM_START_DEG = [40, -60, -60, 80, 40, 40]

# Upright mount. Translation is expressed in chassis coordinates.
MOUNT_OFFSET_M = [0.0, 0.0, 0.515]
MOUNT_PITCH_DEG = 0.0

# TCP translation from DH frame 6, expressed in frame 6.
TCP_OFFSET_M = [0.0, 0.0, 0.0]

# Desired WORLD-frame TCP pose:
# [x, y, z, rx, ry, rz], meters and rotation-vector radians.
# Leave as None to run the example.
TARGET_POSE = None
DEMO_TRANSLATION_M = [1.20, 0.35, 0.0]

TAU_CASES_DEG = [5.0, 20.0]
NUMBER_FINAL = 5

# Collect extra candidates before refinement.
POOL_SIZE = 20
MAX_ATTEMPTS = 160
RANDOM_SEED = 42

POSITION_TOL_M = 1e-6
ANGLE_TOL_RAD = 1e-6

# Example arm limits. Replace with your permitted joint ranges.
JOINT_LIMITS_DEG = [(-360.0, 360.0)] * 6

# Explicit dimensionless movement weights.
# Reference scales are 1 meter for x/y and 1 radian for angles.
REFERENCE_SCALES = np.ones(9)
COST_WEIGHTS = np.r_[1.0, 1.0, np.full(7, 0.75)]

# Numerical variable scaling for the first-stage solver.
SEARCH_SCALES = np.ones(9)

# Nominal UR5e standard DH parameters: a, d, alpha.
DH = [
    (0.0,     0.1625,  np.pi / 2),
    (-0.4250, 0.0,     0.0),
    (-0.3922, 0.0,     0.0),
    (0.0,     0.1333,  np.pi / 2),
    (0.0,     0.0997, -np.pi / 2),
    (0.0,     0.0996,  0.0),
]


def transform(rotation, position):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


def kinematics(state):
    """Return world TCP transform and link-frame origins for plotting."""
    x, y, yaw = state[:3]

    T = transform(
        Rotation.from_euler("z", yaw).as_matrix(),
        [x, y, 0.0],
    )
    points = [T[:3, 3].copy()]

    mount_rotation = Rotation.from_euler(
        "y", MOUNT_PITCH_DEG, degrees=True
    ).as_matrix()
    T = T @ transform(mount_rotation, MOUNT_OFFSET_M)
    points.append(T[:3, 3].copy())

    for theta, (a, d, alpha) in zip(state[3:], DH):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        T = T @ np.array([
            [c, -s * ca,  s * sa, a * c],
            [s,  c * ca, -c * sa, a * s],
            [0,      sa,      ca,     d],
            [0,       0,       0,     1],
        ])
        points.append(T[:3, 3].copy())

    T = T @ transform(np.eye(3), TCP_OFFSET_M)
    points.append(T[:3, 3].copy())

    return T, np.asarray(points)


def pose_error(state, target):
    current, _ = kinematics(state)

    e_p = target[:3, 3] - current[:3, 3]
    R_err = target[:3, :3] @ current[:3, :3].T
    e_R = Rotation.from_matrix(R_err).as_rotvec()

    return np.r_[e_p, e_R]


def pose_is_valid(state, target, lower, upper):
    error = pose_error(state, target)

    return (
        np.all(np.isfinite(state))
        and np.all(state >= lower)
        and np.all(state <= upper)
        and np.linalg.norm(error[:3]) <= POSITION_TOL_M
        and np.linalg.norm(error[3:]) <= ANGLE_TOL_RAD
    )


def angular_difference(a, b):
    """Shortest signed differences for yaw and all six joints."""
    difference = a[2:] - b[2:]
    return np.arctan2(np.sin(difference), np.cos(difference))


def separated(state, accepted, tau):
    """Require at least one angular difference >= tau against each state."""
    return all(
        np.max(np.abs(angular_difference(state, old))) >= tau
        for old in accepted
    )


def movement_cost(state, start):
    change = (state - start) / REFERENCE_SCALES
    return np.sum(COST_WEIGHTS * change**2)


def stage_one(start, target, lower, upper, tau):
    """Collect distinct configurations that reach the same TCP target."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Center random guesses near the target horizontally.
    center = start.copy()
    center[:2] += pose_error(start, target)[:2]

    # Numerical seed exploration ranges around the center.
    # x/y remain unbounded during optimization.
    spread = np.r_[
        0.75, 0.75,             # +/- 0.75 meters
        np.pi,                  # +/- 180 degrees of chassis yaw
        np.full(6, np.pi),       # +/- 180 degrees per arm joint
    ]

    pool = []

    for attempt in range(MAX_ATTEMPTS):
        seed = center + rng.uniform(-1.0, 1.0, 9) * spread

        if attempt == 0:
            seed = start.copy()
        elif attempt == 1:
            seed = center.copy()

        seed = np.clip(seed, lower, upper)

        fit = least_squares(
            pose_error,
            seed,
            args=(target,),
            bounds=(lower, upper),
            x_scale=SEARCH_SCALES,
            max_nfev=300,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

        if (
            pose_is_valid(fit.x, target, lower, upper)
            and separated(fit.x, pool, tau)
        ):
            pool.append(fit.x.copy())

        if len(pool) >= POOL_SIZE:
            break

    if len(pool) < NUMBER_FINAL:
        raise RuntimeError(
            f"Stage 1 found only {len(pool)} distinct feasible candidates "
            f"for tau={np.rad2deg(tau):g} deg "
            f"in {MAX_ATTEMPTS} attempts."
        )

    print(
        f"Stage 1: {len(pool)} candidates "
        f"after {attempt + 1} attempts."
    )

    return sorted(
        pool,
        key=lambda state: movement_cost(state, start),
    )


def stage_two(pool, start, target, lower, upper, tau):
    """Refine candidates while preserving separation from accepted results."""
    accepted = []
    statuses = []
    bounds = list(zip(lower, upper))

    for seed in pool:
        # Earlier refinements may have moved close to this seed.
        if not separated(seed, accepted, tau):
            continue

        constraints = [
            {
                "type": "eq",
                "fun": lambda state: pose_error(state, target),
            }
        ]

        if accepted:
            old_states = np.asarray(accepted)

            # For each earlier result, choose an angular coordinate
            # that already distinguishes this seed from that result.
            witnesses = np.asarray([
                np.argmax(np.abs(angular_difference(seed, old))) + 2
                for old in old_states
            ])
            reference_angles = old_states[
                np.arange(len(old_states)), witnesses
            ]

            def separation_constraints(state):
                # Nonnegative when the selected angular differences
                # have shortest magnitudes of at least tau.
                differences = state[witnesses] - reference_angles
                return np.cos(tau) - np.cos(differences)

            constraints.append({
                "type": "ineq",
                "fun": separation_constraints,
            })

        result = minimize(
            movement_cost,
            seed,
            args=(start,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 300},
        )

        use_refined = (
            pose_is_valid(result.x, target, lower, upper)
            and separated(result.x, accepted, tau)
            and movement_cost(result.x, start)
            <= movement_cost(seed, start)
        )

        if use_refined:
            accepted.append(result.x.copy())
            statuses.append(f"refined ({result.message})")
        else:
            accepted.append(seed.copy())
            statuses.append(
                f"retained feasible seed ({result.message})"
            )

    if len(accepted) < NUMBER_FINAL:
        raise RuntimeError(
            f"Stage 2 retained only {len(accepted)} separated results. "
            "Increase POOL_SIZE and MAX_ATTEMPTS."
        )

    order = sorted(
        range(len(accepted)),
        key=lambda i: movement_cost(accepted[i], start),
    )

    return [
        (accepted[i], statuses[i])
        for i in order[:NUMBER_FINAL]
    ]


def plot_configurations(start, target, results):
    """Plot the common start, final skeletons, and target TCP pose."""
    fig = plt.figure(figsize=(14, 7))
    colors = plt.get_cmap("tab10").colors[:NUMBER_FINAL]

    all_states = [start] + [
        state
        for group in results.values()
        for state, _ in group
    ]
    points = [kinematics(state)[1] for state in all_states]

    # Target coordinate axes, each 0.15 m long.
    desired_axes = (
        target[:3, 3] + 0.15 * target[:3, :3].T
    )

    extent = np.vstack(
        points + [target[:3, 3][None, :], desired_axes]
    )
    midpoint = (extent.max(axis=0) + extent.min(axis=0)) / 2
    radius = max(np.ptp(extent, axis=0).max() * 0.55, 0.3)

    for panel, (tau_deg, group) in enumerate(
        results.items(), start=1
    ):
        ax = fig.add_subplot(
            1, len(results), panel, projection="3d"
        )

        _, initial_points = kinematics(start)
        ax.plot(
            *initial_points.T,
            "--o", color="0.25", lw=2, ms=3, label="Start",
        )

        for index, (state, _) in enumerate(group, start=1):
            _, geometry = kinematics(state)
            color = colors[index - 1]

            ax.plot(
                *geometry.T,
                "-o", color=color, lw=1.5, ms=3,
                label=f"End {index}",
            )

            # Chassis origin and forward heading.
            base = geometry[0]
            ax.scatter(*base, marker="s", color=color, s=35)
            heading = base + 0.18 * np.array([
                np.cos(state[2]), np.sin(state[2]), 0.0
            ])
            ax.plot(
                *np.vstack([base, heading]).T,
                color=color, lw=2,
            )

        ax.scatter(
            *target[:3, 3],
            color="black", marker="*", s=160,
            label="Target TCP",
        )

        for axis, color in enumerate(("red", "green", "blue")):
            segment = np.vstack([
                target[:3, 3], desired_axes[axis]
            ])
            ax.plot(*segment.T, color=color, lw=2)
            ax.text(
                *desired_axes[axis],
                f"TCP {'XYZ'[axis]}", color=color,
            )

        ax.set(
            xlabel="World X (m)",
            ylabel="World Y (m)",
            zlabel="World Z (m)",
            title=f"Minimum angular separation {tau_deg:g} degrees",
            xlim=(midpoint[0] - radius, midpoint[0] + radius),
            ylim=(midpoint[1] - radius, midpoint[1] + radius),
            zlim=(midpoint[2] - radius, midpoint[2] + radius),
        )
        ax.set_box_aspect((1, 1, 1))
        ax.legend(loc="upper left", fontsize=8)

    fig.subplots_adjust(
        left=0.02, right=0.98, bottom=0.10, top=0.90,
        wspace=0.05,
    )
    plt.show()
    return fig


def main():
    start = np.r_[BASE_START, np.deg2rad(ARM_START_DEG)]
    joint_bounds = np.deg2rad(JOINT_LIMITS_DEG)

    # One complete yaw interval centered on the starting heading.
    lower = np.r_[
        -np.inf, -np.inf, start[2] - np.pi,
        joint_bounds[:, 0],
    ]
    upper = np.r_[
        np.inf, np.inf, start[2] + np.pi,
        joint_bounds[:, 1],
    ]

    if np.any(start < lower) or np.any(start > upper):
        raise ValueError(
            "Starting joints must lie within the configured bounds."
        )

    if TARGET_POSE is None:
        target, _ = kinematics(start)
        target[:3, 3] += DEMO_TRANSLATION_M
    else:
        rotation = Rotation.from_rotvec(
            TARGET_POSE[3:]
        ).as_matrix()
        target = transform(rotation, TARGET_POSE[:3])

    print(
        "Start base [m, m, deg]:",
        np.r_[start[:2], np.rad2deg(start[2])],
    )
    print("Start joints (deg):", ARM_START_DEG)
    print("Target world xyz (m):", target[:3, 3])
    print(
        "Target rotation vector (rad):",
        Rotation.from_matrix(target[:3, :3]).as_rotvec(),
    )

    results = {}

    for tau_deg in TAU_CASES_DEG:
        tau = np.deg2rad(tau_deg)
        print(f"\\nTau = {tau_deg:g} degrees")

        pool = stage_one(start, target, lower, upper, tau)
        group = stage_two(
            pool, start, target, lower, upper, tau
        )
        results[tau_deg] = group

        for index, (state, status) in enumerate(group, start=1):
            error = pose_error(state, target)

            print(
                f"End {index}: "
                f"cost={movement_cost(state, start):.6f}"
            )
            print(
                "  Base [m, m, deg]:",
                np.round(
                    np.r_[state[:2], np.rad2deg(state[2])], 4
                ),
            )
            print(
                "  Joints (deg):",
                np.round(np.rad2deg(state[3:]), 3),
            )
            print(
                f"  Errors: {np.linalg.norm(error[:3]):.3e} m, "
                f"{np.linalg.norm(error[3:]):.3e} rad"
            )
            print(" ", status)

        pairwise = [
            np.rad2deg(np.max(np.abs(
                angular_difference(a[0], b[0])
            )))
            for i, a in enumerate(group)
            for b in group[:i]
        ]

        print(
            "Smallest pairwise separation: "
            f"{min(pairwise):.6f} degrees"
        )
        print("End 1 has the lowest cost in this returned group.")

    plot_configurations(start, target, results)
    return start, target, results


if __name__ == "__main__":
    main()