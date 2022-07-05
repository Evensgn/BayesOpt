import numpy as np

from util import *
from bayesian_optimization import BayesianOptimization
from sklearn.gaussian_process.kernels import Matern, RBF
from bo_test_functions import bo_test_funcs, bo_test_func_bounds, bo_test_func_max, eggholder_et_al_2d_list
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import argparse
from multiprocessing import Process, Queue
from pathos.multiprocessing import ProcessingPool
import matplotlib.pyplot as plt
import math


class GridFunction:
    def __init__(self, grid_points, func_values):
        self._grid_points = grid_points
        self._func_values = func_values
        self._dict = {}

        for i in range(len(grid_points)):
            self._dict[tuple(grid_points[i])] = func_values[i]
        self._max_func_value = max(func_values)

    def __call__(self, x):
        return self._dict[tuple(x)]

    def max_func_value(self):
        return self._max_func_value

    def func_values(self):
        return self._func_values


# sample target functions from gp
def generate_samples_of_target_func(gp_params, grid_points, n_samples, random_state):
    if gp_params is None:
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state
        )
    else:
        gp = GaussianProcessRegressor(**gp_params)

    func_values = gp.sample_y(grid_points, n_samples, random_state=random_state)
    func_list = []
    max_func_values = []
    for i in range(n_samples):
        grid_func_i = GridFunction(grid_points, func_values[:, i])
        func_list.append(grid_func_i)
        max_func_values.append(grid_func_i.max_func_value())
    return func_list, max_func_values


class EstimatedGP:
    def __init__(self, grid_points):
        self.grid_points = grid_points
        self.id_dict = {}
        for i in range(len(grid_points)):
            self.id_dict[tuple(grid_points[i])] = i
        self.mu_estimate = None
        self.k_estimate = None
        self.X_t = None
        self.Y_t = None
        self.N = None
        self.M = len(grid_points)
        self.t = None

    def fit_prior(self, Y_dataset):
        # print('Y_dataset.shape', Y_dataset.shape)
        N = Y_dataset.shape[0]
        self.N = N
        self.mu_estimate = np.matmul(Y_dataset.T, np.ones((N, 1))) / N
        # print('mu_estimate.shape', self.mu_estimate.shape)
        diff = Y_dataset - np.matmul(np.ones((N, 1)), self.mu_estimate.T)
        self.k_estimate = np.matmul(diff.T, diff) / (N - 1)

    def _convert_to_one_hot(self, X):
        X_ids = []
        for i in range(len(X)):
            X_ids.append(self.id_dict[tuple(X[i])])
        X_one_hot = np.zeros((len(X), self.M))
        # print('np.arange(self.M):', np.arange(self.M))
        # print('X_ids:', X_ids)
        # print('X_one_hot.shape:', X_one_hot.shape)
        X_one_hot[np.arange(len(X)), X_ids] = 1.0
        return X_one_hot

    def fit(self, X_t, Y_t):
        self.t = len(X_t)
        self.X_t = self._convert_to_one_hot(X_t)
        self.Y_t = np.array([Y_t]).T

    def _mu(self, X):
        return np.matmul(X, self.mu_estimate)

    def _kernel(self, X_1, X_2):
        return np.matmul(X_1, np.matmul(self.k_estimate, X_2.T))

    def predict(self, X, return_std=True):
        X = self._convert_to_one_hot(X)
        k_X_Xt = self._kernel(X, self.X_t)
        k_Xt_Xt = self._kernel(self.X_t, self.X_t)
        # print('k_Xt_Xt:', k_Xt_Xt)
        inv_k_Xt_Xt = np.linalg.pinv(k_Xt_Xt)
        mean = self._mu(X) + np.matmul(k_X_Xt, np.matmul(inv_k_Xt_Xt, self.Y_t - self._mu(self.X_t)))
        if return_std:
            std = []
            for i in range(X.shape[0]):
                X_i = np.expand_dims(X[i], 0)
                k_Xi_Xi = self._kernel(X_i, X_i)
                k_Xi_Xt = self._kernel(X_i, self.X_t)
                k_Xt_Xi = self._kernel(self.X_t, X_i)
                k_i = (k_Xi_Xi - np.matmul(k_Xi_Xt, np.matmul(inv_k_Xt_Xt, k_Xt_Xi))) * (self.N - 1) / (self.N - self.t - 1)
                # print('k_i:', k_i)
                # check why k_i can be negative
                std_i = max(k_i, 0.0) ** 0.5
                std.append(std_i)
            # print(mean.T.squeeze(), np.array(std).T.squeeze())
            return mean.T.squeeze(), np.array(std).T.squeeze()
        else:
            return mean.T.squeeze()


def run_bo_2(run_args):
    (f, bound, acq_func, gp, grid_points, n_init_points, budget, noise_sigma, return_history, random_state) = run_args
    # random_state set to constant 0 to make sure the BO runs are deterministic (gp.fit() uses random_state)
    bo = BayesianOptimization(f, bound, grid_points=grid_points, ready_gp=gp, noise_sigma=noise_sigma, random_state=random_state)
    bo.initialize()
    if return_history:
        _, y_max, _, _, _, y_max_list, _, _ = \
            bo.maximize(acq_func, n_init_points=n_init_points, return_history=return_history, budget=budget)
        return y_max, y_max_list
    else:
        _, y_max, _, _ = bo.maximize(acq_func, n_init_points=n_init_points, return_history=return_history, budget=budget)
        return y_max


def test_gp(test_funcs, test_func_bounds, test_func_max_values, acq_func, gp, grid_points, n_init_points, budget,
            noise_sigma, n_bo_runs=50, return_history=False, random_state=None):
    global pool
    task_list = []
    for i, test_func in enumerate(test_funcs):
        for j in range(n_bo_runs):
            task_list.append((test_func, test_func_bounds[i], acq_func, gp, grid_points, n_init_points, budget, noise_sigma, return_history, random_state))

    task_outputs = pool.map(run_bo_2, task_list)

    # task_outputs = []
    # for task in task_list:
    #    task_outputs.append(run_bo_2(task))

    regrets = []
    if return_history:
        regret_histories = []
    for i, test_func in enumerate(test_funcs):
        for j in range(n_bo_runs):
            if return_history:
                y_max_ij = task_outputs[i][0]
                y_max_history_ij = task_outputs[i][1]
                regret_history_ij = [test_func_max_values[i] - x for x in y_max_history_ij]
                regret_histories.append(regret_history_ij)
            else:
                y_max_ij = task_outputs[i]
            regret_ij = test_func_max_values[i] - y_max_ij
            regrets.append(regret_ij)
    regrets_stats = (np.mean(regrets), np.std(regrets), regrets)
    if return_history:
        regret_histories_stats = (np.mean(regret_histories, axis=0), np.std(regret_histories, axis=0), regret_histories)
        return regrets_stats, regret_histories_stats
    else:
        return regrets_stats


if __name__ == '__main__':
    n_workers = 90
    n_dim = 2
    n_grid_each_dim = 10
    n_dataset_funcs = 1000
    n_test_funcs = 100
    budget = 20
    n_bo_runs = 10
    return_history = True
    n_init_points = 1
    noise_sigma = 0.3
    beta = 1.0

    random_state = ensure_rng(0)

    pool = ProcessingPool(nodes=n_workers)

    gp_params = {
        'kernel': Matern(nu=2.5, length_scale=1.0),
        'alpha': 1e-6,
        'normalize_y': True,
        'n_restarts_optimizer': 5,
        'random_state': random_state
    }
    gp_correct = GaussianProcessRegressor(**gp_params)

    # generate grid points
    grid_dims = []
    bounds = np.array([[-1.0, 1.0]] * n_dim)
    for i in range(len(bounds)):
        grid_dims.append(np.linspace(bounds[i][0], bounds[i][1], n_grid_each_dim))
    mesh_dims = np.meshgrid(*grid_dims)
    grid_points = np.vstack(list(map(np.ravel, mesh_dims))).T

    # generate dataset and test functions
    gen_funcs, gen_func_max_values = generate_samples_of_target_func(
        gp_params, grid_points, n_dataset_funcs+n_test_funcs, random_state
    )

    dataset_funcs = gen_funcs[:n_dataset_funcs]
    Y_dataset = []
    for func in dataset_funcs:
        Y_dataset.append(func.func_values())
    Y_dataset = np.array(Y_dataset)

    test_funcs = gen_funcs[n_dataset_funcs:]
    test_func_max_values = gen_func_max_values[n_dataset_funcs:]
    test_func_bounds = [bounds] * n_test_funcs

    # fit the estimated prior
    gp_estimate = EstimatedGP(grid_points)
    gp_estimate.fit_prior(Y_dataset)

    '''
    # run BO tests with correct GP and estimated GP
    result_list = []
    for beta in [0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6, 1e7]:
        acq_func = StaticAcquisitionFunction(UCBAcquisitionFunction(beta))
        regrets_stats_correct_gp, regret_histories_stats_correct_gp = test_gp(
            test_funcs, test_func_bounds, test_func_max_values, acq_func, gp_correct, grid_points, n_init_points, budget,
            noise_sigma,
            n_bo_runs=n_bo_runs, return_history=return_history, random_state=random_state
        )
        regrets_stats_estimated_gp, regret_histories_stats_estimated_gp = test_gp(
            test_funcs, test_func_bounds, test_func_max_values, acq_func, gp_estimate, grid_points, n_init_points, budget,
            noise_sigma,
            n_bo_runs=n_bo_runs, return_history=return_history, random_state=random_state
        )

        print('regrets_stats_correct_gp:', regrets_stats_correct_gp[0], regrets_stats_correct_gp[1])
        print('regrets_stats_estimated_gp:', regrets_stats_estimated_gp[0], regrets_stats_estimated_gp[1])

        result = {}
        result['beta'] = beta
        result['regrets_stats_correct_gp'] = regrets_stats_correct_gp
        result['regret_histories_stats_correct_gp'] = regret_histories_stats_correct_gp
        result['regrets_stats_estimated_gp'] = regrets_stats_estimated_gp
        result['regret_histories_stats_estimated_gp'] = regret_histories_stats_estimated_gp

        result_list.append(result)

    np.save('results/results_estimated_prior_2.npy', result_list)
    '''

    result = {}

    acq_func = StaticAcquisitionFunction(RandomAcquisitionFunction())
    regrets_stats_random, regret_histories_stats_random = test_gp(
        test_funcs, test_func_bounds, test_func_max_values, acq_func, gp_correct, grid_points, n_init_points, budget,
        noise_sigma,
        n_bo_runs=n_bo_runs, return_history=return_history, random_state=random_state
    )

    result['regrets_stats_random'] = regrets_stats_random
    result['regret_histories_stats_random'] = regret_histories_stats_random

    # test either overestimation or underestimation
    mean_correct_avg, std_correct_avg, mean_estimated_avg, std_estimated_avg = 0, 0, 0, 0
    for func in test_funcs:
        # sample 10 random points
        x_list = grid_points[random_state.choice(len(grid_points), size=10)]
        y_list = [func(x) for x in x_list]
        gp_correct.fit(x_list, y_list)
        gp_estimate.fit(x_list, y_list)
        # sample another 10 points to test on
        x_list = grid_points[random_state.choice(len(grid_points), size=10)]
        mean_correct, std_correct = gp_correct.predict(x_list, return_std=True)
        mean_estimated, std_estimated = gp_estimate.predict(x_list, return_std=True)
        mean_correct_avg += np.mean(mean_correct)
        std_correct_avg += np.mean(std_correct)
        mean_estimated_avg += np.mean(mean_estimated)
        std_estimated_avg += np.mean(std_estimated)
    mean_correct_avg /= n_test_funcs
    std_correct_avg /= n_test_funcs
    mean_estimated_avg /= n_test_funcs
    std_estimated_avg /= n_test_funcs
    result['mean_correct_avg'] = mean_correct_avg
    result['std_correct_avg'] = std_correct_avg
    result['mean_estimated_avg'] = mean_estimated_avg
    result['std_estimated_avg'] = std_estimated_avg

    np.save('results/results_estimated_prior_4.npy', result)
