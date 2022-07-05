import util
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


def run_bo(run_args):
    (f, bound, acq_func, grid_points, n_init_points, budget, return_history, random_state) = run_args
    # random_state set to constant 0 to make sure the BO runs are deterministic (gp.fit() uses random_state)
    bo = BayesianOptimization(f, bound, grid_points=grid_points, random_state=0)
    bo.initialize()
    if return_history:
        _, y_max, _, _, _, y_max_list, _, _ = \
            bo.maximize(acq_func, n_init_points=n_init_points, return_history=return_history, budget=budget)
        return y_max, y_max_list
    else:
        _, y_max, _, _ = bo.maximize(acq_func, n_init_points=n_init_points, return_history=return_history, budget=budget)
        return y_max


def test_acq_func(test_funcs, test_func_bounds, test_func_max_values, acq_func, grid_points, n_init_points, budget,
                  n_bo_runs=50, return_history=False, random_state=None):
    global pool
    task_list = []
    for i, test_func in enumerate(test_funcs):
        for j in range(n_bo_runs):
            task_list.append((test_func, test_func_bounds[i], acq_func, grid_points, n_init_points, budget, return_history, random_state))

    '''
    task_outputs = []
    for task in task_list:
        task_outputs.append(run_bo(task))
    '''

    task_outputs = pool.map(run_bo, task_list)

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


def test_single_step_acq_func(test_funcs, test_func_bounds, test_func_max_values, single_step_acq_func, later_step_acq_funcs, grid_points,
                              n_init_points, remaining_budget, n_bo_runs=50, random_state=None):
    acq_funcs = later_step_acq_funcs + [single_step_acq_func]
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return test_acq_func(test_funcs, test_func_bounds, test_func_max_values, acq_func, grid_points, n_init_points, remaining_budget,
                         n_bo_runs=n_bo_runs, random_state=random_state)


def parameterized_ucb_linear_schedule_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        beta_i = x[0] + (x[1] - x[0]) * i / (max_budget - 1)
        acq_funcs.append(UCBAcquisitionFunction(beta_i))
        params.append(beta_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


def parameterized_ucb_exponential_schedule_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        beta_i = x[0] * (x[1] ** (i / (max_budget - 1)))
        acq_funcs.append(UCBAcquisitionFunction(beta_i))
        params.append(beta_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


def parameterized_ucb_polynomial_schedule_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        beta_i = x[0] + x[1] * ((i + 1) / max_budget) ** x[2]
        acq_funcs.append(UCBAcquisitionFunction(beta_i))
        params.append(beta_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


def parameterized_ucb_constant_beta_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        beta_i = x[0]
        acq_funcs.append(UCBAcquisitionFunction(beta_i))
        params.append(beta_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


def parameterized_ei_linear_schedule_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        xi_i = x[0] + (x[1] - x[0]) * i / (max_budget - 1)
        acq_funcs.append(EIAcquisitionFunction(xi_i))
        params.append(xi_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


def parameterized_ei_exponential_schedule_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        xi_i = x[0] * (x[1] ** (i / (max_budget - 1)))
        acq_funcs.append(EIAcquisitionFunction(xi_i))
        params.append(xi_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


def parameterized_ei_polynomial_schedule_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        xi_i = x[0] + (x[1] - x[0]) * i / (max_budget - 1)
        acq_funcs.append(EIAcquisitionFunction(xi_i))
        params.append(xi_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


def parameterized_ei_constant_xi_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        xi_i = x[0]
        acq_funcs.append(EIAcquisitionFunction(xi_i))
        params.append(xi_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs)
    return acq_func, params


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


class OptimizeAcqFuncObjective:
    def __init__(self, n_sample_target_funcs, inner_gp_params, domain_bounds,
                 target_func_list, target_func_max_values, target_func_bounds, grid_points,
                 n_inner_bo_runs, n_inner_init_points, max_budget, random_state):
        self._target_func_list = target_func_list
        self._target_func_max_values = target_func_max_values
        self._target_func_bounds = target_func_bounds
        self._grid_points = grid_points
        self._n_sample_target_funcs = n_sample_target_funcs
        self._inner_gp_params = inner_gp_params
        self._n_inner_bo_runs = n_inner_bo_runs
        self._n_inner_init_points = n_inner_init_points
        self._max_budget = max_budget
        self._domain_bounds = domain_bounds
        self._random_state = random_state

    def mean_bo_regret(self, x):
        # generate samples of target functions
        if self._target_func_list is None:
            target_func_list, target_func_max_values = generate_samples_of_target_func(
                self._inner_gp_params, self._grid_points, self._n_sample_target_funcs, self._random_state
            )
        else:
            target_func_list = self._target_func_list
            target_func_max_values = self._target_func_max_values

        acq_func, params = conversion_func(x, self._max_budget)
        regret, _, _ = test_acq_func(
            target_func_list, self._target_func_bounds, target_func_max_values,
            acq_func, self._grid_points,
            self._n_inner_init_points, self._max_budget, n_bo_runs=self._n_inner_bo_runs, random_state=self._random_state
        )
        return regret


def optimize_parameterized_acq_func(param_bounds, conversion_func, grid_points,
                                    n_inner_init_points, max_budget, domain_bounds,
                                    n_sample_target_funcs=20, n_inner_bo_runs=50,
                                    n_outer_init_points=10, outer_maxiter=100,
                                    inner_gp_params=None, outer_gp_params=None, outer_acq_func='ei',
                                    random_state=None,
                                    target_func_list=None, target_func_max_values=None,
                                    fix_target_funcs=False):
    random_state = ensure_rng(random_state)

    # generate samples of target functions
    if target_func_list is None and fix_target_funcs:
        target_func_list, target_func_max_values = generate_samples_of_target_func(
            inner_gp_params, grid_points, n_sample_target_funcs, random_state
        )
    target_func_bounds = [domain_bounds] * n_sample_target_funcs

    mean_bo_regret = OptimizeAcqFuncObjective(
        n_sample_target_funcs, inner_gp_params, domain_bounds,
        target_func_list, target_func_max_values, target_func_bounds, grid_points,
        n_inner_bo_runs, n_inner_init_points, max_budget, random_state
    ).mean_bo_regret

    if outer_gp_params is None:
        outer_gp_params = {
            'kernel': Matern(nu=2.5),
            'alpha': 1e-6,
            'normalize_y': True,
            'n_restarts_optimizer': 5,
            'random_state': 0
        }

    # set random_state to constant 0 to make sure BO run is deterministic (gp.fit() uses random_state)
    bo = BayesianOptimization(lambda x: -mean_bo_regret(x), param_bounds, gp_params=outer_gp_params, random_state=0)
    if outer_acq_func == 'ei':
        outer_acq_func = StaticAcquisitionFunction(EIAcquisitionFunction(0.0))
    elif outer_acq_func == 'pi':
        outer_acq_func = StaticAcquisitionFunction(PIAcquisitionFunction(0.0))
    elif outer_acq_func == 'ucb':
        outer_acq_func = StaticAcquisitionFunction(UCBAcquisitionFunction(1.0))
    else:
        raise ValueError('Unknown outer_acq_func: {}'.format(outer_acq_func))
    x_max, y_max, x_infer, y_infer, x_max_history, y_max_history, observation_xs, observation_ys = bo.maximize(
        outer_acq_func, n_init_points=n_outer_init_points, budget=outer_maxiter, return_history=True,
    )

    regret_history = [-y for y in y_max_history]
    acq_func, params = conversion_func(x_max, max_budget)
    acq_func_infer, params_infer = conversion_func(x_infer, max_budget)
    acq_func_history = []
    for x in x_max_history:
        acq_func_t, _ = conversion_func(x, max_budget)
        acq_func_history.append(acq_func_t)

    history = (x_max_history, regret_history, acq_func_history, observation_xs, observation_ys)
    target_funcs = (target_func_list, target_func_max_values, target_func_bounds)
    return acq_func, params, acq_func_infer, params_infer, history, target_funcs


def benchmark_acq_func(test_funcs, test_func_bounds, test_func_max_values, acq_func, n_init_points,
                       test_budget_list, n_bo_runs, random_state=None):
    regret_means = []
    regret_stds = []
    regrets_list = []
    regret_history_means = []
    regret_history_stds = []
    regret_histories_list = []
    for i in test_budget_list:
        regrets_stats_i, regret_histories_stats_i = test_acq_func(
            test_funcs, test_func_bounds, test_func_max_values, acq_func, grid_points,
            n_init_points=n_init_points, budget=i, n_bo_runs=n_bo_runs,
            return_history=True, random_state=random_state
        )
        regret_mean_i, regret_std_i, regrets_i = regrets_stats_i
        regret_history_mean_i, regret_history_std_i, regret_histories_i = regret_histories_stats_i
        regret_means.append(regret_mean_i)
        regret_stds.append(regret_std_i)
        regrets_list.append(regrets_i)
        regret_history_means.append(regret_history_mean_i)
        regret_history_stds.append(regret_history_std_i)
        regret_histories_list.append(regret_histories_i)
    return (regret_means, regret_stds, regrets_list), (regret_history_means, regret_history_stds, regret_histories_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimization Experiments')

    parser.add_argument('--save_id', type=str, default='default', help='Save id')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--n_dim', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--n_grid_each_dim', type=int, default=100, help='Size of each dimension of the grid')
    parser.add_argument('--n_sample_target_funcs', type=int, default=100, help='Number of sampled target functions')
    parser.add_argument('--n_inner_bo_runs', type=int, default=1, help='Number of inner BO runs')
    parser.add_argument('--n_inner_init_points', type=int, default=0, help='Number of initial points for inner BO')
    parser.add_argument('--max_budget', type=int, default=5, help='Maximum budget for BO')
    parser.add_argument('--outer_maxiter', type=int, default=100, help='outer loop maxiter')
    parser.add_argument('--n_outer_init_points', type=int, default=0, help='Number of initial points for outer BO')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--n_test_funcs', type=int, default=100, help='Number of test functions')
    parser.add_argument('--n_test_init_points', type=int, default=0, help='Number of test initial points')
    parser.add_argument('--n_test_bo_runs', type=int, default=1, help='Number of test BO runs')
    parser.add_argument('--inner_gp_nu', type=float, default=2.5, help='Nu parameter for inner GP')
    parser.add_argument('--inner_gp_lambda', type=float, default=1.0, help='Lambda parameter for inner GP')
    parser.add_argument('--inner_gp_length_scale', type=float, default=1.0, help='Length scale parameter for inner GP')
    parser.add_argument('--acq_func_types', type=str, default='param-ucb-linear-schedule-10.0',
                        help='Type of acquisition function to optimize')
    parser.add_argument('--baseline_acq_func_types', type=str, default='static-ei,static-ucb-1.0,static-ucb-inf,static-pi,theory-ucb-delta-0.1-uniform-pi,theory-ucb-delta-0.5-uniform-pi',
                        help='Type of acquisition function to optimize')
    parser.add_argument('--fix_train_funcs', action='store_true', help='Fix training functions')
    parser.add_argument('--overfit_test_funcs', action='store_true', help='Use the test functions as training functions')
    # parser.add_argument('--additional_test_budget', type=int, nargs='+', default='[]')
    parser.add_argument('--outer_acq_func', type=str, default='ei', help='Type of outer acquisition function')
    parser.add_argument('--outer_gp_length_scale_factor', type=float, default=1.0, help='Length scale factor for outer GP')


    args = parser.parse_args()

    results = {}
    results['args'] = args

    # test_budget_list = [args.max_budget] + args.additional_test_budget
    test_budget_list = [args.max_budget]
    results['test_budget_list'] = test_budget_list

    pool = ProcessingPool(nodes=args.n_workers)

    random_state = ensure_rng(args.seed)

    if args.inner_gp_nu > 500:
        inner_gp_kernel = args.inner_gp_lambda * RBF(length_scale=args.inner_gp_length_scale)
    else:
        inner_gp_kernel = args.inner_gp_lambda * Matern(nu=args.inner_gp_nu, length_scale=args.inner_gp_length_scale)

    inner_gp_params = {
        'kernel': inner_gp_kernel,
        'alpha': 1e-6,
        'normalize_y': True,
        'n_restarts_optimizer': 5,
        'random_state': random_state
    }

    # generate grid points
    grid_dims = []
    bounds = np.array([[-1.0, 1.0]] * args.n_dim)
    for i in range(len(bounds)):
        grid_dims.append(np.linspace(bounds[i][0], bounds[i][1], args.n_grid_each_dim))
    mesh_dims = np.meshgrid(*grid_dims)
    grid_points = np.vstack(list(map(np.ravel, mesh_dims))).T

    results['grid_points'] = grid_points
    
    # generate test functions
    test_funcs, test_func_max_values = generate_samples_of_target_func(
        inner_gp_params, grid_points, args.n_test_funcs, random_state
    )
    test_func_bounds = [bounds] * args.n_test_funcs

    test_func_save = (test_funcs, test_func_max_values, test_func_bounds)
    results['test_func_save'] = test_func_save

    acq_key_list = []

    # optimize parameterized acq func
    acq_func_types = args.acq_func_types.split(',')
    for acq_func_type in acq_func_types:
        bound_value = float(acq_func_type.split('-')[-1])
        if acq_func_type.startswith('param-ucb-linear-schedule'):
            param_bounds = np.array([[0.0, bound_value]] * 2)
            conversion_func = parameterized_ucb_linear_schedule_conversion_func
            outer_gp_length_scale = bound_value / 2.0 * args.outer_gp_length_scale_factor
        elif acq_func_type.startswith('param-ucb-exponential-schedule'):
            param_bounds = np.array([[0.0, bound_value], [0.01, 100.0]])
            conversion_func = parameterized_ucb_exponential_schedule_conversion_func
            outer_gp_length_scale = [bound_value / 2.0 * args.outer_gp_length_scale_factor, 100.0 / 2.0 * args.outer_gp_length_scale_factor]
        elif acq_func_type.startswith('param-ucb-polynomial-schedule'):
            param_bounds = np.array([[0.0, bound_value]] * 3)
            conversion_func = parameterized_ucb_polynomial_schedule_conversion_func
            outer_gp_length_scale = bound_value / 2.0 * args.outer_gp_length_scale_factor
        elif acq_func_type.startswith('param-ucb-constant-beta'):
            param_bounds = np.array([[0.0, bound_value]] * 1)
            conversion_func = parameterized_ucb_constant_beta_conversion_func
            outer_gp_length_scale = bound_value / 2.0 * args.outer_gp_length_scale_factor
        elif acq_func_type.startswith('param-ei-linear-schedule'):
            param_bounds = np.array([[-bound_value, bound_value]] * 2)
            conversion_func = parameterized_ei_linear_schedule_conversion_func
            outer_gp_length_scale = bound_value * args.outer_gp_length_scale_factor
        elif acq_func_type.startswith('param-ei-exponential-schedule'):
            param_bounds = np.array([[-bound_value, bound_value], [0.01, 100.0]])
            conversion_func = parameterized_ei_exponential_schedule_conversion_func
            outer_gp_length_scale = [bound_value * args.outer_gp_length_scale_factor, 100.0 / 2.0 * args.outer_gp_length_scale_factor]
        elif acq_func_type.startswith('param-ei-polynomial-schedule'):
            param_bounds = np.array([[-bound_value, bound_value]] * 3)
            conversion_func = parameterized_ei_polynomial_schedule_conversion_func
            outer_gp_length_scale = bound_value * args.outer_gp_length_scale_factor
        elif acq_func_type.startswith('param-ei-constant-xi'):
            param_bounds = np.array([[-bound_value, bound_value]] * 1)
            conversion_func = parameterized_ei_constant_xi_conversion_func
            outer_gp_length_scale = bound_value * args.outer_gp_length_scale_factor
        else:
            raise ValueError('Unknown acquisition function type:', acq_func_type)

        outer_gp_params = {
            'kernel': Matern(nu=2.5, length_scale=outer_gp_length_scale),
            'alpha': 1e-6,
            'normalize_y': True,
            'n_restarts_optimizer': 5,
            'random_state': 0
        }

        acq_key_list.append(acq_func_type)

        if args.overfit_test_funcs:
            target_func_list = test_funcs
            target_func_max_values = test_func_max_values
        else:
            target_func_list = None
            target_func_max_values = None

        param_ucb_acq_func, param_ucb_params, param_ucb_infer_acq_func, param_ucb_infer_params, \
            param_ucb_history, param_ucb_target_funcs = \
            optimize_parameterized_acq_func(
                param_bounds, conversion_func, grid_points,
                n_inner_init_points=0, max_budget=args.max_budget,
                domain_bounds=bounds, n_sample_target_funcs=args.n_sample_target_funcs,
                n_inner_bo_runs=args.n_inner_bo_runs,
                n_outer_init_points=args.n_outer_init_points, outer_maxiter=args.outer_maxiter,
                inner_gp_params=inner_gp_params, outer_gp_params=outer_gp_params, outer_acq_func=args.outer_acq_func,
                random_state=random_state,
                target_func_list=target_func_list, target_func_max_values=target_func_max_values,
                fix_target_funcs=args.fix_train_funcs
            )
        param_ucb_best_x_history, param_ucb_train_regret_history, param_ucb_acq_func_history, param_ucb_observation_xs, \
            param_ucb_observation_ys = param_ucb_history
        print('{} param list: {}'.format(acq_func_type, param_ucb_params))
        print('{} param list (infer): {}'.format(acq_func_type, param_ucb_infer_params))
        print('best_x_history: {}'.format(param_ucb_best_x_history))
        print('{} train regret: {}'.format(acq_func_type, param_ucb_train_regret_history))
        results['{}_history'.format(acq_func_type)] = param_ucb_history
        results['{}_target_funcs'.format(acq_func_type)] = param_ucb_target_funcs
        results['{}_params'.format(acq_func_type)] = param_ucb_params
        results['{}_params_infer'.format(acq_func_type)] = param_ucb_infer_params

        # test optimized parameterized acq_func
        param_ucb_test_regrets, param_ucb_test_regret_histories = benchmark_acq_func(
            test_funcs, test_func_bounds, test_func_max_values, param_ucb_acq_func,
            args.n_test_init_points, test_budget_list, args.n_test_bo_runs, random_state
        )
        param_ucb_regret_mean, param_ucb_regret_std, parama_ucb_regrets = param_ucb_test_regrets
        param_ucb_regret_history_mean, param_ucb_regret_history_std, param_ucb_regret_histories = param_ucb_test_regret_histories
        results['{}_regret_mean'.format(acq_func_type)] = param_ucb_regret_mean
        results['{}_regret_std'.format(acq_func_type)] = param_ucb_regret_std
        results['{}_regrets'.format(acq_func_type)] = parama_ucb_regrets
        results['{}_regret_history_mean'.format(acq_func_type)] = param_ucb_regret_history_mean
        results['{}_regret_history_std'.format(acq_func_type)] = param_ucb_regret_history_std
        results['{}_regret_histories'.format(acq_func_type)] = param_ucb_regret_histories
        print('{} regret mean: {}, std: {}'.format(acq_func_type, param_ucb_regret_mean, param_ucb_regret_std))

        param_ucb_infer_test_regrets, param_ucb_infer_test_regret_histories = benchmark_acq_func(
            test_funcs, test_func_bounds, test_func_max_values, param_ucb_infer_acq_func,
            args.n_test_init_points, test_budget_list, args.n_test_bo_runs, random_state
        )
        param_ucb_infer_regret_mean, param_ucb_infer_regret_std, parama_ucb_infer_regrets = param_ucb_infer_test_regrets
        param_ucb_infer_regret_history_mean, param_ucb_infer_regret_history_std, param_ucb_infer_regret_histories = \
            param_ucb_infer_test_regret_histories
        results['{}_infer_regret_mean'.format(acq_func_type)] = param_ucb_infer_regret_mean
        results['{}_infer_regret_std'.format(acq_func_type)] = param_ucb_infer_regret_std
        results['{}_infer_regrets'.format(acq_func_type)] = parama_ucb_infer_regrets
        results['{}_infer_regret_history_mean'.format(acq_func_type)] = param_ucb_infer_regret_history_mean
        results['{}_infer_regret_history_std'.format(acq_func_type)] = param_ucb_infer_regret_history_std
        results['{}_infer_regret_histories'.format(acq_func_type)] = param_ucb_infer_regret_histories
        print('{} infer regret mean: {}, std: {}'.format(acq_func_type, param_ucb_infer_regret_mean, param_ucb_infer_regret_std))

        param_ucb_test_regret_mean_history = []
        param_ucb_test_regret_std_history = []
        param_ucb_test_regrets_history = []
        for i, acq_func in enumerate(param_ucb_acq_func_history):
            if i > 0 and np.array_equal(param_ucb_best_x_history[i], param_ucb_best_x_history[i - 1]):
                param_ucb_test_regret_mean_history.append(param_ucb_test_regret_mean_history[-1])
                param_ucb_test_regret_std_history.append(param_ucb_test_regret_std_history[-1])
                continue

            (param_ucb_regret_mean, param_ucb_regret_std, param_ucb_regrets), _ = benchmark_acq_func(
                test_funcs, test_func_bounds, test_func_max_values, acq_func,
                args.n_test_init_points, test_budget_list, args.n_test_bo_runs, random_state
            )
            param_ucb_test_regret_mean_history.append(param_ucb_regret_mean)
            param_ucb_test_regret_std_history.append(param_ucb_regret_std)
            param_ucb_test_regrets_history.append(param_ucb_regrets)
        print('{} test regret mean: {}, std: {}'.format(
            acq_func_type, param_ucb_test_regret_mean_history, param_ucb_test_regret_std_history))
        results['{}_test_regret_mean_history'.format(acq_func_type)] = param_ucb_test_regret_mean_history
        results['{}_test_regret_std_history'.format(acq_func_type)] = param_ucb_test_regret_std_history
        results['{}_test_regrets_history'.format(acq_func_type)] = param_ucb_test_regrets_history

    baseline_acq_func_types = args.baseline_acq_func_types.split(',')
    for acq_func_type in baseline_acq_func_types:
        if acq_func_type == 'random':
            acq_func = StaticAcquisitionFunction(RandomAcquisitionFunction())
        elif acq_func_type == 'static-ei':
            acq_func = StaticAcquisitionFunction(EIAcquisitionFunction(0.0))
        elif acq_func_type == 'static-ucb-1.0':
            acq_func = StaticAcquisitionFunction(UCBAcquisitionFunction(1.0))
        elif acq_func_type == 'static-ucb-inf':
            acq_func = StaticAcquisitionFunction(UCBAcquisitionFunction(1e5))
        elif acq_func_type == 'static-pi':
            acq_func = StaticAcquisitionFunction(PIAcquisitionFunction(0.0))
        elif acq_func_type == 'theory-ucb-delta-0.1-uniform-pi':
            theory_ucb_acq_funcs = []
            delta = 0.1
            for i in range(args.max_budget):
                t = args.max_budget - i
                pi = args.max_budget
                beta = 2 * np.log(pi / delta)
                theory_ucb_acq_funcs = [UCBAcquisitionFunction(beta)] + theory_ucb_acq_funcs
            acq_func = SequenceAcquisitionFunction(theory_ucb_acq_funcs)
        elif acq_func_type == 'theory-ucb-delta-0.5-uniform-pi':
            theory_ucb_acq_funcs = []
            delta = 0.5
            for i in range(args.max_budget):
                t = args.max_budget - i
                pi = args.max_budget
                beta = 2 * np.log(pi / delta)
                theory_ucb_acq_funcs = [UCBAcquisitionFunction(beta)] + theory_ucb_acq_funcs
            acq_func = SequenceAcquisitionFunction(theory_ucb_acq_funcs)
        else:
            raise ValueError('Unknown baseline acq func type: {}'.format(acq_func_type))

        acq_key_list.append(acq_func_type)

        ei_test_regrets, ei_test_regret_histories = benchmark_acq_func(
            test_funcs, test_func_bounds, test_func_max_values, acq_func,
            args.n_test_init_points, test_budget_list, args.n_test_bo_runs, random_state
        )
        ei_regret_mean, ei_regret_std, ei_regrets = ei_test_regrets
        ei_regret_history_mean, ei_regret_history_std, ei_regret_histories = ei_test_regret_histories
        results['{}_regret_mean'.format(acq_func_type)] = ei_regret_mean
        results['{}_regret_std'.format(acq_func_type)] = ei_regret_std
        results['{}_regrets'.format(acq_func_type)] = ei_regrets
        results['{}_regret_history_mean'.format(acq_func_type)] = ei_regret_history_mean
        results['{}_regret_history_std'.format(acq_func_type)] = ei_regret_history_std
        results['{}_regret_histories'.format(acq_func_type)] = ei_regret_histories
        print('{} regret mean: {}, std: {}'.format(acq_func_type, ei_regret_mean, ei_regret_std))

    results['acq_key_list'] = acq_key_list
    np.save('results/results_{}.npy'.format(args.save_id), results)

