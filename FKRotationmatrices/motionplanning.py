import sympy as sp
import math

# Symbols for joints 1–6 and their DH parameters
theta = sp.symbols("theta1:7")
a = sp.symbols("a1:7")
d = sp.symbols("d1:7")
alpha = sp.symbols("alpha1:7")


def dh_matrix(theta_i, a_i, d_i, alpha_i):
    """Standard DH transform from frame i-1 to frame i."""
    return sp.Matrix([
        [sp.cos(theta_i), -sp.sin(theta_i) * sp.cos(alpha_i),
         sp.sin(theta_i) * sp.sin(alpha_i), a_i * sp.cos(theta_i)],
        [sp.sin(theta_i), sp.cos(theta_i) * sp.cos(alpha_i),
         -sp.cos(theta_i) * sp.sin(alpha_i), a_i * sp.sin(theta_i)],
        [0, sp.sin(alpha_i), sp.cos(alpha_i), d_i],
        [0, 0, 0, 1],
    ])


# T_0_6 = T_0_1 * T_1_2 * ... * T_5_6
T_0_6 = sp.eye(4)
for params in zip(theta, a, d, alpha):
    T_0_6 = T_0_6 * dh_matrix(*params)

print(T_0_6)  # 4×4 symbolic transform

ur5e_dh = [
    (40 * sp.pi / 180,   0,                 sp.Rational(1625, 10000),  sp.pi / 2),
    (-60 * sp.pi / 180,  sp.Rational(-425, 1000), 0,                  0),
    (-60 * sp.pi / 180,  sp.Rational(-3922, 10000), 0,                 0),
    (80 * sp.pi / 180,   0,                 sp.Rational(1333, 10000),  sp.pi / 2),
    (40 * sp.pi / 180,   0,                 sp.Rational(997, 10000),  -sp.pi / 2),
    (40 * sp.pi / 180,   0,                 sp.Rational(996, 10000),   0),
]

ur5e_subs = {}
for symbols, values in zip(
    (theta, a, d, alpha),
    zip(*ur5e_dh),
):
    ur5e_subs.update(zip(symbols, values))

T_0_6_ur5e = T_0_6.subs(ur5e_subs)  # exact symbolic result for these parameters
T_0_6_decimal = T_0_6_ur5e.evalf(6) # optional decimal approximation

print(T_0_6_decimal)