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
    """
    Return the 4x4 homogeneous transform T0to6 for a UR5e arm.

    Parameters
    ----------
    q1, q2, q3, q4, q5, q6 : float
        Joint angles in radians.

    Returns
    -------
    sympy.Matrix
        4x4 transformation matrix from base frame to tool frame.
    """
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


# Example usage:
T_1 = ur5e_fk(math.radians(40), math.radians(-60), math.radians(-60), math.radians(80), math.radians(40), math.radians(40))
print(T_1)

T_2 = ur5e_fk(math.radians(-40), math.radians(-40), math.radians(-80), math.radians(-80), math.radians(40), math.radians(40))
print(T_2)