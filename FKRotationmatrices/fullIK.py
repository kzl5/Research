import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# Standard UR5e DH parameters: (a, d, alpha), in meters and radians.
DH = [
    (0.0,     0.1625,  np.pi / 2),
    (-0.4250, 0.0,     0.0),
    (-0.3922, 0.0,     0.0),
    (0.0,     0.1333,  np.pi / 2),
    (0.0,     0.0997, -np.pi / 2),
    (0.0,     0.0996,  0.0),
]

# Fixed arm mount, expressed in the chassis frame.
# +90 deg about Y puts the arm's local +Z along the chassis's +X.
T_MOUNT = np.eye(4)
T_MOUNT[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
T_MOUNT[:3, 3] = [0.0, 0.0, 0.515]  # Replace with your measured offset.


def arm_fk(q):
    """Return the transform from the UR5e base to DH frame 6."""
    T = np.eye(4)
    for theta, (a, d, alpha) in zip(q, DH):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = T @ np.array([
            [c, -s * ca,  s * sa, a * c],
            [s,  c * ca, -c * sa, a * s],
            [0,      sa,      ca,     d],
            [0,       0,       0,     1],
        ])
    return T


def world_fk(state):
    """Return the world-frame TCP pose for [x, y, yaw, q1, ..., q6]."""
    x, y, yaw = state[:3]
    c, s = np.cos(yaw), np.sin(yaw)
    T_base = np.array([
        [c, -s, 0, x],
        [s,  c, 0, y],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])
    # This example puts the TCP at DH frame 6.
    return T_base @ T_MOUNT @ arm_fk(state[3:])


def solve_ik(target, start):
    # Smaller scales make changes in that coordinate more costly.
    # Order: base x, base y, base yaw, then the six arm joints.
    scales = np.r_[1.0, 1.0, 0.75, np.full(6, 0.75)]

    # Example search bounds. Replace with your workspace and joint limits.
    bounds = [(-3.0, 3.0)] * 2 + [(-np.pi, np.pi)]
    bounds += [(-2 * np.pi, 2 * np.pi)] * 6

    def cost(state):
        return np.sum(((state - start) / scales) ** 2)

    def pose_error(state):
        current = world_fk(state)
        e_p = target[:3, 3] - current[:3, 3]
        R_err = target[:3, :3] @ current[:3, :3].T
        e_R = Rotation.from_matrix(R_err).as_rotvec()
        return np.r_[e_p, e_R]

    result = minimize(
        cost,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": pose_error},
        options={"ftol": 1e-10, "maxiter": 200},
    )

    error = pose_error(result.x)
    position_error = np.linalg.norm(error[:3])
    orientation_error = np.linalg.norm(error[3:])

    if not result.success or position_error > 1e-6 or orientation_error > 1e-6:
        raise RuntimeError(f"IK failed from this starting guess. {result.message}")

    return result.x, position_error, orientation_error


# ----- Starting configuration -----
# Base x and y are in meters. Yaw and arm joint angles are in radians.
start = np.r_[0.0, 0.0, 0.0, np.deg2rad([40, -60, -60, 80, 40, 40])]

# ----- Target TCP pose in the world frame -----
# For the demo, keep the starting tool orientation and move the goal
# 1.20 m along world X and 0.35 m along world Y.
target = world_fk(start)
target[:3, 3] += [1.20, 0.35, 0.0]
# For your own target, assign its 4x4 world-frame transform to `target`.

solution, position_error, orientation_error = solve_ik(target, start)

x, y, yaw = solution[:3]
print(f"Parking pose: x={x:.3f} m, y={y:.3f} m, "
      f"yaw={np.rad2deg(yaw):.2f} deg")
print("Arm joints (deg):", np.round(np.rad2deg(solution[3:]), 2))
print(f"TCP position error: {position_error:.3e} m")
print(f"TCP orientation error: {orientation_error:.3e} rad")