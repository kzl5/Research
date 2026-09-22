from sympy import *
import math

#### MOBILE BASE ######

theta = symbols('theta')
x_base = symbols('x_base')
y_base = symbols('y_base')
z_base = symbols('z_base')

#Mobile Base Rotation Matrix
Mobile_Base_RM = Matrix([[cos(theta), -sin(theta), 0, x_base],
                         [sin(theta), cos(theta), 0, y_base],
                         [0, 0, 1, z_base],
                         [0, 0, 0, 1]])

x_mounting = symbols('x_mounting')
y_mounting = symbols('y_mounting')
z_mounting = symbols('z_mounting')

#Mobile Base to UR5e Robotic Arm Transformation Matrix
Mobile_Base_to_UR5e_RM = Matrix([[0, 0, 1, x_mounting],
                                 [0, 1, 0, y_mounting],
                                 [-1, 0, 0, z_mounting],
                                 [0, 0, 0, 1]])

#### UR5e Robotic Arm ######

#DH Parameters symbols
q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')
a1, a2, a3, a4, a5, a6 = symbols('a1 a2 a3 a4 a5 a6')
d1, d2, d3, d4, d5, d6 = symbols('d1 d2 d3 d4 d5 d6')
alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha1 alpha2 alpha3 alpha4 alpha5 alpha6')

#DH Parameters for UR5e Robotic Arm
q1 = math.radians(40);   a1 = 0;       d1 = 0.1625; alpha1 = pi/2
q2 = math.radians(-60);  a2 = -0.425;  d2 = 0;      alpha2 = 0
q3 = math.radians(-60);  a3 = -0.3922; d3 = 0;      alpha3 = 0
q4 = math.radians(80);   a4 = 0;       d4 = 0.1333; alpha4 = pi/2
q5 = math.radians(40);   a5 = 0;       d5 = 0.0997; alpha5 = -pi/2
q6 = math.radians(40);   a6 = 0;       d6 = 0.0996; alpha6 = 0

#DH Parameter list
dh_params = [
    [q1, a1, d1, alpha1],
    [q2, a2, d2, alpha2],
    [q3, a3, d3, alpha3],
    [q4, a4, d4, alpha4],
    [q5, a5, d5, alpha5],
    [q6, a6, d6, alpha6]
]

#Denavit Hartenberg Transformation Matrix
def dh_transformation_matrix(q, a, d, alpha):
    return Matrix([[cos(q), -sin(q)*cos(alpha), sin(q)*sin(alpha), a*cos(q)],
                   [sin(q), cos(q)*cos(alpha), -cos(q)*sin(alpha), a*sin(q)],
                   [0, sin(alpha), cos(alpha), d],
                   [0, 0, 0, 1]])

#Forward Kinematics Transformation Matrix for UR5e Robotic Arm
T01 = dh_transformation_matrix(*dh_params[0])
T12 = dh_transformation_matrix(*dh_params[1])
T23 = dh_transformation_matrix(*dh_params[2])
T34 = dh_transformation_matrix(*dh_params[3])
T45 = dh_transformation_matrix(*dh_params[4])
T56 = dh_transformation_matrix(*dh_params[5])

Tbaseto6 = Mobile_Base_RM * Mobile_Base_to_UR5e_RM * T01 * T12 * T23 * T34 * T45 * T56

T0to6 = T01 * T12 * T23 * T34 * T45 * T56


def ur5e_fk(q1, q2, q3, q4, q5, q6):
    a = [0, -0.425, -0.3922, 0, 0, 0]
    d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
    alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
    q = [q1, q2, q3, q4, q5, q6]

    T = Matrix([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

    for qi, ai, di, alphai in zip(q, a, d, alpha):
        T = T * dh_transformation_matrix(qi, ai, di, alphai)

    return T

def rotation_vector_from_transform(T):

    R = T[:3, :3]

    r00 = float(R[0, 0])
    r01 = float(R[0, 1])
    r02 = float(R[0, 2])

    r10 = float(R[1, 0])
    r11 = float(R[1, 1])
    r12 = float(R[1, 2])

    r20 = float(R[2, 0])
    r21 = float(R[2, 1])
    r22 = float(R[2, 2])

    # Total rotation angle
    cos_theta = (r00 + r11 + r22 - 1.0) / 2.0

    # Clamp for floating-point safety
    cos_theta = max(-1.0, min(1.0, cos_theta))

    theta = math.acos(cos_theta)

    # No rotation
    if abs(theta) < 1e-10:
        return 0.0, 0.0, 0.0

    # Special case near 180 degrees
    if abs(math.pi - theta) < 1e-6:
        kx = math.sqrt(max(0.0, (r00 + 1.0) / 2.0))
        ky = math.sqrt(max(0.0, (r11 + 1.0) / 2.0))
        kz = math.sqrt(max(0.0, (r22 + 1.0) / 2.0))

        if r21 - r12 < 0:
            kx = -kx
        if r02 - r20 < 0:
            ky = -ky
        if r10 - r01 < 0:
            kz = -kz

        return theta * kx, theta * ky, theta * kz

    # General case
    scale = theta / (2.0 * math.sin(theta))

    Rx = scale * (r21 - r12)
    Ry = scale * (r02 - r20)
    Rz = scale * (r10 - r01)

    return Rx, Ry, Rz


# Example usage:
print("Forward Kinematics and Rotation Vector Calculation for UR5e Robotic Arm\n")
T_1 = ur5e_fk(math.radians(40), math.radians(-60), math.radians(-60), math.radians(80), math.radians(40), math.radians(40))
print(T_1)
R_1 = rotation_vector_from_transform(T_1)
print("Rotation vector components (Rx, Ry, Rz):", R_1)

print("\n")
T_2 = ur5e_fk(math.radians(-40), math.radians(-40), math.radians(-80), math.radians(-80), math.radians(40), math.radians(40))
print(T_2)
R_2 = rotation_vector_from_transform(T_2)
print("Rotation vector components (Rx, Ry, Rz):", R_2)

print("\n")
T_3 = ur5e_fk(math.radians(0), math.radians(-90), math.radians(0), math.radians(-90), math.radians(0), math.radians(0))
print(T_3)
R_3 = rotation_vector_from_transform(T_3)
print("Rotation vector components (Rx, Ry, Rz):", R_3)

### UR5e PolyScope values:
poses = [
    {
        "name": "Run 4060",
        "joints": tuple(math.radians(angle) for angle in (40, -60, -60, 80, 40, 40)),
        "measured": [35.37, -243.99, 836.31, 1.145, -0.004, 0.225],
    },
    {
        "name": "Run 4040",
        "joints": tuple(math.radians(angle) for angle in (-40, -40, -80, -80, 40, 40)),
        "measured": [-162.11, -137.34, 848.84, 0.516, 2.217, -1.793],
    },
    {
        "name": "Run 0090",
        "joints": tuple(math.radians(angle) for angle in (0, -90, 0, -90, 0, 0)),
        "measured": [-1.06, -233.66, 1080.30, 0.005, 2.224, -2.224],
    }
]


def calculate_pose_errors(poses):
    errors = []

    for pose in poses:
        transform = ur5e_fk(*pose["joints"])
        predicted_position = [float(value) * 1000.0 for value in transform[:3, 3]]
        predicted_rotation = rotation_vector_from_transform(transform)
        measured = pose["measured"]
        errors.append([
            predicted - expected
            for predicted, expected in zip(
                predicted_position + list(predicted_rotation), measured
            )
        ])

    return errors


pose_errors = calculate_pose_errors(poses)
print("\nPose error values (x, y, z in mm; Rx, Ry, Rz in rad):")
for pose, errors in zip(poses, pose_errors):
    formatted_errors = ", ".join(f"{error:.12f}" for error in errors)
    print(f"{pose['name']}: {formatted_errors}")

#### graphing #####

import matplotlib.pyplot as plt


def plot_pose_errors(poses):
    labels = []
    labels = [pose["name"] for pose in poses]

    components = ["x", "y", "z", "Rx", "Ry", "Rz"]
    units = ["mm", "mm", "mm", "rad", "rad", "rad"]
    figure, axis = plt.subplots(figsize=(10, 6))

    for index, (component, unit) in enumerate(zip(components, units)):
        errors = [pose_error[index] for pose_error in pose_errors]
        axis.plot(labels, errors, marker="o", label=f"{component} ({unit})")
        for label, error in zip(labels, errors):
            axis.annotate(
                f"{error:.9f}",
                (label, error),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Forward-kinematics error across poses")
    axis.set_xlabel("Pose")
    axis.set_ylabel("Signed error (mm for position, rad for rotation)")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()
    return figure, pose_errors

figure, errors = plot_pose_errors(poses)
plt.show()


