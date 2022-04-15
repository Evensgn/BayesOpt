import util
from util import *
from bayesian_optimization import BayesianOptimization
from bo_test_functions import test_funcs, test_func_bounds, test_func_max, quick_test_funcs
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import argparse
from multiprocessing import Process, Queue
from pathos.multiprocessing import ProcessingPool


def run_bo(run_args):
    (f, bound, acq_func, grid_points, n_init_points, budget) = run_args
    bo = BayesianOptimization(f, bound, grid_points=grid_points)
    bo.initialize()
    _, y_max = bo.maximize(acq_func, n_init_points=n_init_points, budget=budget)
    return y_max


def test_acq_func(test_funcs, test_func_bounds, test_func_max_values, acq_func, grid_points, n_init_points, budget, n_bo_runs=50, random_state=None):
    global pool
    task_list = []
    for i, test_func in enumerate(test_funcs):
        for j in range(n_bo_runs):
            task_list.append((test_func, test_func_bounds[i], acq_func, grid_points, n_init_points, budget))
    '''
    task_outputs = []
    for task in task_list:
        task_outputs.append(run_bo(task))
    '''
    task_outputs = pool.map(run_bo, task_list)

    regrets = []
    total_regret = 0.0
    for i, test_func in enumerate(test_funcs):
        y_max_i = 0.0
        for j in range(n_bo_runs):
            y_max_i += task_outputs[i]
        mean_y_max_i = y_max_i / n_bo_runs
        regret_i = test_func_max_values[i] - mean_y_max_i
        regrets.append(regret_i)
        total_regret += regret_i

    return total_regret / len(test_funcs), regrets


def test_single_step_acq_func(test_funcs, test_func_bounds, test_func_max_values, single_step_acq_func, later_step_acq_funcs, grid_points,
                              n_init_points, remaining_budget, n_bo_runs=50, random_state=None):
    acq_funcs = later_step_acq_funcs + [single_step_acq_func]
    seq_acq_func = SequenceAcquisitionFunction(acq_funcs)
    acq_func = seq_acq_func.acq_func
    return test_acq_func(test_funcs, test_func_bounds, test_func_max_values, acq_func, grid_points, n_init_points, remaining_budget,
                         n_bo_runs=n_bo_runs, random_state=random_state)


def parameterized_ucb_conversion_func(x, max_budget):
    params = []
    acq_funcs = []
    for i in range(max_budget):
        beta_i = x[0] * i + x[1]
        acq_funcs.append(UCBAcquisitionFunction(beta_i).acq_func)
        params.append(beta_i)
    acq_func = SequenceAcquisitionFunction(acq_funcs).acq_func
    return acq_func, params


class GridFunction:
    def __init__(self, grid_points, func_values):
        self._grid_points = grid_points
        self._func_values = func_values
        self._dict = {}

        for i in range(len(grid_points)):
            self._dict[tuple(grid_points[i])] = func_values[i]
        self._max_func_value = max(func_values)

    def func(self, x):
        return self._dict[tuple(x)]

    def max_func_value(self):
        return self._max_func_value


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
        func_list.append(grid_func_i.func)
        max_func_values.append(grid_func_i.max_func_value())
    return func_list, max_func_values


def optimize_parameterized_acq_func(param_bounds, conversion_func, grid_points,
                                    n_inner_init_points, max_budget, domain_bounds,
                                    n_sample_target_funcs=20, n_inner_bo_runs=50,
                                    outer_maxiter=100, inner_gp_params=None, random_state=None,
                                    target_func_list=None, target_func_max_values=None):
    random_state = ensure_rng(random_state)

    # generate samples of target functions
    if target_func_list is None:
        target_func_list, target_func_max_values = generate_samples_of_target_func(
            inner_gp_params, grid_points, n_sample_target_funcs, random_state
        )
    target_func_bounds = [domain_bounds] * n_sample_target_funcs

    # optimize the acquisition functions in a backward induction way
    acq_funcs = []
    param_list = []

    def mean_bo_regret(x):
        acq_func, params = conversion_func(x, max_budget)
        regret, _ = test_acq_func(
            target_func_list, target_func_bounds, target_func_max_values,
            acq_func, grid_points,
            n_inner_init_points, max_budget, n_bo_runs=n_inner_bo_runs, random_state=random_state
        )
        return regret

    bo = BayesianOptimization(lambda x: -mean_bo_regret(x), param_bounds)
    ei_acq_func = StaticAcquisitionFunction(util.expected_improvement).acq_func
    x_max, y_max = bo.maximize(ei_acq_func, n_init_points=0, budget=outer_maxiter)

    acq_func, params = conversion_func(x_max, max_budget)

    '''
    for i in range(1, max_budget+1):
        print('Optimizing step {} acq func'.format(i))

        def mean_bo_regret(x):
            single_step_acq_func = conversion_func(x)
            regret, _ = test_single_step_acq_func(
                target_func_list, target_func_bounds, target_func_max_values,
                single_step_acq_func, acq_funcs, grid_points,
                n_inner_init_points, i, n_bo_runs=n_inner_bo_runs, random_state=random_state
            )
            return regret

        bo = BayesianOptimization(lambda x: -mean_bo_regret(x), param_bounds)
        ei_acq_func = StaticAcquisitionFunction(util.expected_improvement).acq_func
        x_max, y_max = bo.maximize(ei_acq_func, n_init_points=0, budget=outer_maxiter)

        single_step_acq_func = conversion_func(x_max)
        acq_funcs.append(single_step_acq_func)
        param_list.append(x_max)
    '''

    return acq_func, params


def benchmark_acq_func(test_funcs, test_func_bounds, test_func_max_values, acq_func, n_init_points,
                       max_budget, n_bo_runs, random_state=None):
    regrets = []
    for i in range(max_budget, max_budget+1):
        regret_i, _ = test_acq_func(
            test_funcs, test_func_bounds, test_func_max_values, acq_func, grid_points,
            n_init_points=n_init_points, budget=i, n_bo_runs=n_bo_runs,
            random_state=random_state
        )
        regrets.append(regret_i)
    return regrets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimization Experiments')

    parser.add_argument('--save_id', type=int, default=0, help='Save id')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    parser.add_argument('--n_dim', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--n_grid_each_dim', type=int, default=100, help='Size of each dimension of the grid')
    parser.add_argument('--n_sample_target_funcs', type=int, default=100, help='Number of sampled target functions')
    parser.add_argument('--n_inner_bo_runs', type=int, default=1, help='Number of inner BO runs')
    parser.add_argument('--n_inner_init_points', type=int, default=0, help='Number of initial points for inner BO')
    parser.add_argument('--max_budget', type=int, default=5, help='Maximum budget for BO')
    parser.add_argument('--outer_maxiter', type=int, default=100, help='outer loop maxiter')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--n_test_funcs', type=int, default=100, help='Number of test functions')
    parser.add_argument('--n_test_init_points', type=int, default=0, help='Number of test initial points')
    parser.add_argument('--n_test_bo_runs', type=int, default=1, help='Number of test BO runs')

    args = parser.parse_args()

    pool = ProcessingPool(nodes=args.n_workers)

    random_state = ensure_rng(args.seed)

    # generate grid points
    grid_dims = []
    bounds = np.array([[-1.0, 1.0]] * args.n_dim)
    for i in range(len(bounds)):
        grid_dims.append(np.linspace(bounds[i][0], bounds[i][1], args.n_grid_each_dim))
    mesh_dims = np.meshgrid(*grid_dims)
    grid_points = np.vstack(list(map(np.ravel, mesh_dims))).T

    # generate test functions
    test_funcs, test_func_max_values = generate_samples_of_target_func(
        None, grid_points, args.n_test_funcs, random_state
    )
    test_func_bounds = [bounds] * args.n_test_funcs

    results = {}

    # optimize parameterized UCB
    param_bounds = np.array([[-100.0, 100.0]] * 2)
    param_ucb_acq_func, param_ucb_params = optimize_parameterized_acq_func(
        param_bounds, parameterized_ucb_conversion_func, grid_points,
        n_inner_init_points=0, max_budget=args.max_budget,
        domain_bounds=bounds, n_sample_target_funcs=args.n_sample_target_funcs,
        n_inner_bo_runs=args.n_inner_bo_runs, outer_maxiter=args.outer_maxiter,
        # target_func_list=test_funcs, target_func_max_values=test_func_max_values,
        random_state=random_state
    )
    print('parameterized UCB param list: {}'.format(param_ucb_params))

    # test optimized parameterized UCB
    param_ucb_regret = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, param_ucb_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    results['param_ucb_regret'] = param_ucb_regret
    print('parameterized UCB regret: {}'.format(param_ucb_regret))

    # test static EI
    ei_acq_func = StaticAcquisitionFunction(util.expected_improvement).acq_func
    ei_acq_test_results = {}
    ei_regret = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, ei_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    results['ei_acq_regret'] = ei_regret
    print('static EI regret: {}'.format(ei_regret))

    # test static UCB
    static_ucb_acq_func = StaticAcquisitionFunction(UCBAcquisitionFunction(1.0).acq_func).acq_func
    static_ucb_regret = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, static_ucb_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    results['static_ucb_regret'] = static_ucb_regret
    print('static UCB regret: {}'.format(static_ucb_regret))

    # test decay UCB
    decay_ucb_acq_funcs = []
    beta = 1.0
    decay = 0.9
    for i in range(args.max_budget):
        decay_ucb_acq_funcs = [UCBAcquisitionFunction(beta).acq_func] + decay_ucb_acq_funcs
        beta *= decay
    decay_ucb_acq_func = SequenceAcquisitionFunction(decay_ucb_acq_funcs).acq_func
    decay_ucb_regret = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, decay_ucb_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    results['decay_ucb_regret'] = decay_ucb_regret
    print('decay UCB regret: {}'.format(decay_ucb_regret))

    np.save('results_{}.npy'.format(args.save_id), results)

