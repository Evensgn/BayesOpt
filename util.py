''' The code in this file is based on https://github.com/fmfn/BayesianOptimization '''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


inf = 1e3
eps = 1e-5


def acq_max_grid(ac, gp, y_max, grid_points, remaining_budget):
    def ac_value(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        mean, std = gp.predict(x, return_std=True)
        return ac(mean, std, y_max, remaining_budget)

    ys = ac_value(grid_points)
    ys_max = ys.max()

    # not using ys_max_id = ys.argmax() because it is not deterministic
    ys_max_id = None
    for i in range(len(ys)):
        if ys[i] + eps >= ys_max:
            ys_max_id = i
            break

    x_max = grid_points[ys_max_id]
    return x_max


def acq_max(ac, gp, y_max, bounds, remaining_budget, random_state, n_warmup=500, n_iter=5, lbfgs_maxiter=100):
    """
    A function to find the maximum of the acquisition function
    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.
    Parameters
    ----------
    :param ac:
        The acquisition function.
    :param gp:
        A gaussian process fitted to the relevant data.
    :param y_max:
        The current maximum known value of the target function.
    :param bounds:
        The variables bounds to limit the search of the acq max.
    :param steps_remaining:
        The number of steps remaining in the Bayesian optimization process.
    :param random_state:
        instance of np.RandomState random number generator
    :param n_warmup:
        number of times to randomly sample the aquisition function
    :param n_iter:
        number of times to run scipy.minimize
    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """
    def ac_value(x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        mean, std = gp.predict(x, return_std=True)
        return ac(mean, std, y_max, remaining_budget)

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac_value(x_tries)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more thoroughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac_value(x),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       options=dict(maxiter=lbfgs_maxiter))

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        res_fun = np.squeeze(np.array([res.fun]))
        if max_acq is None or -res_fun >= max_acq:
            x_max = res.x
            max_acq = -res_fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


# pack a sequence of single-step acquisition functions
class SequenceAcquisitionFunction:
    def __init__(self, acq_funcs):
        self._acq_funcs = acq_funcs

    def __call__(self, mean, std, y_max, remaining_budget):
        assert len(self._acq_funcs) >= remaining_budget
        return self._acq_funcs[remaining_budget - 1](mean, std, y_max)


# wrap a static acquisition function
class StaticAcquisitionFunction:
    def __init__(self, acq_func) -> object:
        self._acq_func = acq_func

    def __call__(self, mean, std, y_max, remaining_budget):
        return self._acq_func(mean, std, y_max)


class NeuralAcquisitionFunction(nn.Module):
    def __init__(self):
        super(NeuralAcquisitionFunction, self).__init__()

        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

        self.total_num_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_state_dict_from_vector(self, x):
        current_idx = 0
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = torch.from_numpy(x[current_idx:current_idx+value.numel()].reshape(value.shape))
            current_idx += value.numel()
        self.load_state_dict(state_dict)

    def __call__(self, mean, std, y_max):
        n_samples = mean.shape[0]
        mean = mean.reshape([n_samples, 1])
        std = std.reshape([n_samples, 1])
        y_max = np.repeat(y_max, mean.shape[0]).reshape([n_samples, 1])
        ret = self.forward(torch.from_numpy(np.hstack((mean, std, y_max))).float()).detach().numpy()
        ret = np.squeeze(ret)
        return ret


class UCBAcquisitionFunction:
    def __init__(self, beta):
        self._beta = beta

    def __call__(self, mean, std, y_max):
        return mean + np.sqrt(self._beta) * std


class EIAcquisitionFunction:
    def __init__(self, xi):
        self._xi = xi

    def __call__(self, mean, std, y_max):
        a = mean - (y_max + self._xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)


class PIAcquisitionFunction:
    def __init__(self, xi):
        self._xi = xi

    def __call__(self, mean, std, y_max):
        z = (mean - (y_max + self._xi)) / std
        return norm.cdf(z)


class RandomAcquisitionFunction:
    def __init__(self):
        pass

    def __call__(self, mean, std, y_max):
        return np.random.random(mean.shape)


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

