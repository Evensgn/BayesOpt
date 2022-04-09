import math
from numpy import *
import numpy as np


test_funcs = {}
test_func_bounds = {}

# xi ∈ [-512, 512], for all i = 1, 2
# global min = -959.6407 at (512, 404.2319)
def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    a = sqrt(fabs(x2 + x1 / 2 + 47))
    b = sqrt(fabs(x1 - (x2 + 47)))
    c = -(x2 + 47) * sin(a) - x1 * sin(b)
    return c


test_funcs['eggholder'] = lambda x: -eggholder(x)
test_func_bounds['eggholder'] = np.array([[-512, 512], [-512, 512]])


# xi ∈ [-32.768, 32.768]
# global min = 0 at (0, 0)
def ackley(x1, x2):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - exp(
        0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2))) + e + 20


# xi ∈ [-10, 10]
# global min = 0 at (1, 1 / sqrt(2))
def dixon_price(x1, x2):
    return ((x1 - 1) ** 2) + (2 * (2 * x2 ** 2 - x1) ** 2)


# Styblinski-Tang Function
# xi ∈ [-5, 5]
# global min = -78.3323 at x = (-2.903534, -2.903534)
def s_tang(x1, x2):
    return 0.5 * ((x1 ** 4 + x2 ** 4) - 16 * (x1 ** 2 + x2 ** 2) + 5 * (x1 + x2))


# 4-d xi ∈ [-4, 5]
# global min = 0 at x = (0, 0, ... 0)
def powell_4d(x1, x2, x3, x4):
    return (x1 + 10 * x2) ** 2 + 5 * (x3 - x4) ** 2 + (x2 - 2 * x3) ** 4 + 10 * (x1 - x4) ** 4


# x1 ∈ [-5, 10], x2 ∈ [0, 15]
# global min = 0.397887 at x = (-pi, 12,275), (pi, 2.275) and (9.42478, 2.475)
def branin(x1, x2):
    a = 1;
    b = 5.1 / (4 * math.pi ** 2);
    c = 5 / math.pi;
    r = 6;
    s = 10;
    t = 1 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


# xi ∈ [-2, 2]
# global min = 3 at (0, -1)
def goldstein_price(x1, x2):
    l1 = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    l2 = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return l1 * l2


# xi ∈ [0, 1]
# global min = -3.86278 at (0.114614, 0.555649, 0.852547)
def hartman_3d(x1, x2, x3):
    # the hartmann3 function (3D)
    # https://www.sfu.ca/~ssurjano/hart3.html

    # parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])

    x = np.tile(np.array([x1, x2, x3]), (4, 1))
    B = x - P
    B = B ** 2
    exponent = (A * B).sum(axis=1)
    C = np.exp(-exponent)
    hm3 = -np.dot(C, alpha)
    return hm3
