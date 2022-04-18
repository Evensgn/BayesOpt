import math
from numpy import *
import numpy as np


eggholder_et_al_2d_list = ['eggholder', 'ackley', 'dixon_price', 's_tang', 'branin', 'goldstein_price']

bo_test_funcs = {}
bo_test_func_bounds = {}
bo_test_func_max = {}

# xi ∈ [-512, 512], for all i = 1, 2
# global min = -959.6407 at (512, 404.2319)
def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    a = sqrt(fabs(x2 + x1 / 2 + 47))
    b = sqrt(fabs(x1 - (x2 + 47)))
    c = -(x2 + 47) * sin(a) - x1 * sin(b)
    return c


bo_test_funcs['eggholder'] = lambda x: -eggholder(x)
bo_test_func_bounds['eggholder'] = np.array([[-512, 512], [-512, 512]])
bo_test_func_max['eggholder'] = 959.6407


# xi ∈ [-32.768, 32.768]
# global min = 0 at (0, 0)
def ackley(x):
    x1 = x[0]
    x2 = x[1]
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - exp(
        0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2))) + e + 20


bo_test_funcs['ackley'] = lambda x: -ackley(x)
bo_test_func_bounds['ackley'] = np.array([[-32.768, 32.768], [-32.768, 32.768]])
bo_test_func_max['ackley'] = 0


# xi ∈ [-10, 10]
# global min = 0 at (1, 1 / sqrt(2))
def dixon_price(x):
    x1 = x[0]
    x2 = x[1]
    return ((x1 - 1) ** 2) + (2 * (2 * x2 ** 2 - x1) ** 2)


bo_test_funcs['dixon_price'] = lambda x: -dixon_price(x)
bo_test_func_bounds['dixon_price'] = np.array([[-10, 10], [-10, 10]])
bo_test_func_max['dixon_price'] = 0


# Styblinski-Tang Function
# xi ∈ [-5, 5]
# global min = -78.3323 at x = (-2.903534, -2.903534)
def s_tang(x):
    x1 = x[0]
    x2 = x[1]
    return 0.5 * ((x1 ** 4 + x2 ** 4) - 16 * (x1 ** 2 + x2 ** 2) + 5 * (x1 + x2))


bo_test_funcs['s_tang'] = lambda x: -s_tang(x)
bo_test_func_bounds['s_tang'] = np.array([[-5, 5], [-5, 5]])
bo_test_func_max['s_tang'] = 78.3323


# 4-d xi ∈ [-4, 5]
# global min = 0 at x = (0, 0, ... 0)
def powell_4d(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return (x1 + 10 * x2) ** 2 + 5 * (x3 - x4) ** 2 + (x2 - 2 * x3) ** 4 + 10 * (x1 - x4) ** 4


bo_test_funcs['powell_4d'] = lambda x: -powell_4d(x)
bo_test_func_bounds['powell_4d'] = np.array([[-4, 5], [-4, 5], [-4, 5], [-4, 5]])
bo_test_func_max['powell_4d'] = 0


# x1 ∈ [-5, 10], x2 ∈ [0, 15]
# global min = 0.397887 at x = (-pi, 12,275), (pi, 2.275) and (9.42478, 2.475)
def branin(x):
    x1 = x[0]
    x2 = x[1]
    a = 1
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


bo_test_funcs['branin'] = lambda x: -branin(x)
bo_test_func_bounds['branin'] = np.array([[-5, 10], [0, 15]])
bo_test_func_max['branin'] = -0.397887


# xi ∈ [-2, 2]
# global min = 3 at (0, -1)
def goldstein_price(x):
    x1 = x[0]
    x2 = x[1]
    l1 = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    l2 = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return l1 * l2


bo_test_funcs['goldstein_price'] = lambda x: -goldstein_price(x)
bo_test_func_bounds['goldstein_price'] = np.array([[-2, 2], [-2, 2]])
bo_test_func_max['goldstein_price'] = -3


# xi ∈ [0, 1]
# global min = -3.86278 at (0.114614, 0.555649, 0.852547)
def hartman_3d(x):
    # the hartmann3 function (3D)
    # https://www.sfu.ca/~ssurjano/hart3.html
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
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


bo_test_funcs['hartman_3d'] = lambda x: -hartman_3d(x)
bo_test_func_bounds['hartman_3d'] = np.array([[0, 1], [0, 1], [0, 1]])
bo_test_func_max['hartman_3d'] = 3.86278
