''' The code in this file is based on https://github.com/fmfn/BayesianOptimization '''

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import util
from util import acq_max, ensure_rng
import warnings
import numpy as np


class BayesianOptimization:
    def __init__(self, f, bounds, random_state=None):
        self._random_state = ensure_rng(random_state)
        self._f = f
        self._bounds = bounds
        self._observations_xs = []
        self._observations_ys = []
        self._x_max = None
        self._y_max = None

        # internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

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
        self._y_max = None

    def generate_init_observations_for_pesudo_target_func(self, len_init_observations=20):
        self.initialize()
        init_observation_points = self._random_state.uniform(self._bounds[:, 0], self._bounds[:, 1],
                                                             size=(len_init_observations, self._bounds.shape[0]))
        for x in init_observation_points:
            self.probe(x, fit=True)
        return self._observations_xs, self._observations_ys

    def probe(self, x, fit=False):
        if self._f is None:
            # using the GP posterior to sample a target function
            if fit and len(self._observations_xs) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._gp.fit(self._observations_xs, self._observations_ys)
            mean, std = self._gp.predict(x.reshape(1, -1), return_std=True)
            y = np.random.normal(mean, std)
            y = np.squeeze(y)
        else:
            y = self._f(x)

        self._observations_xs.append(list(x))
        self._observations_ys.append(y)

        if self._y_max is None or y > self._y_max:
            self._x_max = x
            self._y_max = y

    def suggest(self, acq_func, remaining_budget):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._observations_xs, self._observations_ys)

        suggestion = acq_max(
            ac=acq_func,
            gp=self._gp,
            y_max=self._y_max,
            bounds=self._bounds,
            remaining_budget=remaining_budget,
            random_state=self._random_state
        )

        return suggestion

    def maximize(self, acq_func, n_init_points=5, budget=25):
        # generate the initial points
        init_points = self._random_state.uniform(self._bounds[:, 0], self._bounds[:, 1],
                                                 size=(n_init_points, self._bounds.shape[0]))
        for x in init_points:
            self.probe(x, fit=True)

        # iterations
        for i in range(budget):
            x = self.suggest(acq_func, budget - i)
            self.probe(x)

        return self._x_max, self._y_max


