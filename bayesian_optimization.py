''' The code in this file is based on https://github.com/fmfn/BayesianOptimization '''

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import util
from util import acq_max, acq_max_grid, ensure_rng, inf
import warnings
import numpy as np


class BayesianOptimization:
    def __init__(self, f, bounds, grid_points=None, gp_params=None, random_state=None):
        self._random_state = ensure_rng(random_state)
        self._f = f
        self._bounds = bounds
        self._observations_xs = []
        self._observations_ys = []
        self._x_max = None
        self._y_max = -inf

        # internal GP regressor
        if gp_params is None:
            self._gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self._random_state
            )
        else:
            self._gp = GaussianProcessRegressor(**gp_params)

        # grid mode
        self._grid_points = grid_points

    def initialize(self, observations_xs=None, observations_ys=None):
        if observations_xs is not None:
            self._observations_xs = observations_xs
            self._observations_ys = observations_ys
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gp.fit(self._observations_xs, self._observations_ys)
        else:
            self._observations_xs = []
            self._observations_ys = []
        self._x_max = None
        self._y_max = -inf

    def probe(self, x):
        y = self._f(x)

        self._observations_xs.append(list(x))
        self._observations_ys.append(y)

        if self._y_max is None or y > self._y_max:
            self._x_max = x
            self._y_max = y

    def suggest(self, acq_func, remaining_budget):
        if len(self._observations_xs) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gp.fit(self._observations_xs, self._observations_ys)

        if self._grid_points is not None:
            suggestion = acq_max_grid(
                ac=acq_func,
                gp=self._gp,
                y_max=self._y_max,
                grid_points=self._grid_points,
                remaining_budget=remaining_budget
            )
        else:
            suggestion = acq_max(
                ac=acq_func,
                gp=self._gp,
                y_max=self._y_max,
                bounds=self._bounds,
                remaining_budget=remaining_budget,
                random_state=self._random_state
            )

        return suggestion

    def maximize(self, acq_func, n_init_points=0, budget=20, return_history=False):
        # generate the initial points
        if self._grid_points is None:
            init_points = self._random_state.uniform(self._bounds[:, 0], self._bounds[:, 1],
                                                     size=(n_init_points, self._bounds.shape[0]))
        else:
            init_points_idx = np.random.choice(self._grid_points.shape[0], size=n_init_points)
            init_points = self._grid_points[init_points_idx]

        for x in init_points:
            self.probe(x)

        if return_history:
            x_max_list = []
            y_max_list = []

        # iterations
        for i in range(budget):
            x = self.suggest(acq_func, budget - i)
            self.probe(x)
            if return_history:
                x_max_list.append(self._x_max)
                y_max_list.append(self._y_max)

        if return_history:
            return self._x_max, self._y_max, x_max_list, y_max_list, self._observations_xs, self._observations_ys
        else:
            return self._x_max, self._y_max

    def set_bounds(self, bounds):
        self._bounds = bounds

    def get_gp(self):
        return self._gp

    def set_gp_params(self, **params):
        self._gp.set_params(**params)

