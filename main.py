import util
from util import *
from bayesian_optimization import BayesianOptimization
from bo_test_functions import bo_test_funcs, bo_test_func_bounds, bo_test_func_max, eggholder_et_al_2d_list
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import argparse
from multiprocessing import Process, Queue
from pathos.multiprocessing import ProcessingPool
import matplotlib.pyplot as plt


def run_bo(run_args):
    (f, bound, acq_func, grid_points, n_init_points, budget, return_history, random_state) = run_args
    # random_state set to constant 0 to make sure the BO runs are deterministic (gp.fit() uses random_state)
    bo = BayesianOptimization(f, bound, grid_points=grid_points, random_state=0)
    bo.initialize()
    if return_history:
        _, y_max, _, y_max_list, _, _ = \
            bo.maximize(acq_func, n_init_points=n_init_points, return_history=return_history, budget=budget)
        return y_max, y_max_list
    else:
        _, y_max = bo.maximize(acq_func, n_init_points=n_init_points, return_history=return_history, budget=budget)
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
        beta_i = x[0] * i + x[1]
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

    def mean_bo_regret(x):
        acq_func, params = conversion_func(x, max_budget)
        regret, _, _ = test_acq_func(
            target_func_list, target_func_bounds, target_func_max_values,
            acq_func, grid_points,
            n_inner_init_points, max_budget, n_bo_runs=n_inner_bo_runs, random_state=random_state
        )
        return regret

    # set random_state to constant 0 to make sure BO run is deterministic (gp.fit() uses random_state)
    bo = BayesianOptimization(lambda x: -mean_bo_regret(x), param_bounds, random_state=0)
    ei_acq_func = StaticAcquisitionFunction(EIAcquisitionFunction(0.0))
    x_max, y_max, x_max_history, y_max_history, observation_xs, observation_ys = bo.maximize(
        ei_acq_func, n_init_points=0, budget=outer_maxiter, return_history=True
    )

    regret_history = [-y for y in y_max_history]
    acq_func, params = conversion_func(x_max, max_budget)
    acq_func_history = []
    for x in x_max_history:
        acq_func_t, _ = conversion_func(x, max_budget)
        acq_func_history.append(acq_func_t)

    history = (x_max_history, regret_history, acq_func_history, observation_xs, observation_ys)
    target_funcs = (target_func_list, target_func_max_values, target_func_bounds)
    return acq_func, params, history, target_funcs

    '''
    # optimize the acquisition functions in a backward induction way
    acq_funcs = []
    param_list = []
    
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


def benchmark_acq_func(test_funcs, test_func_bounds, test_func_max_values, acq_func, n_init_points,
                       max_budget, n_bo_runs, random_state=None):
    regret_means = []
    regret_stds = []
    regrets_list = []
    regret_history_means = []
    regret_history_stds = []
    regret_histories_list = []
    for i in range(max_budget, max_budget+1):
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
    parser.add_argument('--acq_func_type', type=str, default='param-ucb-linear-schedule',
                        help='Type of acquisition function to optimize')

    args = parser.parse_args()

    results = {}
    results['args'] = args

    pool = ProcessingPool(nodes=args.n_workers)

    random_state = ensure_rng(args.seed)

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
        None, grid_points, args.n_test_funcs, random_state
    )
    test_func_bounds = [bounds] * args.n_test_funcs

    # eggholder et al.
    eggholder_et_al_test_funcs = []
    eggholder_et_al_test_func_bounds = []
    eggholder_et_al_test_func_max_values = []
    for func in eggholder_et_al_2d_list:
        eggholder_et_al_test_funcs.append(bo_test_funcs[func])
        eggholder_et_al_test_func_bounds.append(bo_test_func_bounds[func])
        eggholder_et_al_test_func_max_values.append(bo_test_func_max[func])

    test_func_save = (test_funcs, test_func_max_values, test_func_bounds)
    results['test_func_save'] = test_func_save

    # optimize parameterized UCB
    if args.acq_func_type == 'param-ucb-linear-schedule':
        param_bounds = np.array([[-100.0, 100.0]] * 2)
        conversion_func = parameterized_ucb_linear_schedule_conversion_func
    elif args.acq_func_type == 'param-ucb-polynomial-schedule':
        param_bounds = np.array([[-100.0, 100.0]] * 3)
        conversion_func = parameterized_ucb_polynomial_schedule_conversion_func
    elif args.acq_func_type == 'param-ucb-constant-beta':
        param_bounds = np.array([[-100.0, 100.0]] * 1)
        conversion_func = parameterized_ucb_constant_beta_conversion_func
    else:
        raise ValueError('Unknown acquisition function type')
    param_ucb_acq_func, param_ucb_params, param_ucb_history, param_ucb_target_funcs = \
        optimize_parameterized_acq_func(
            param_bounds, conversion_func, grid_points,
            n_inner_init_points=0, max_budget=args.max_budget,
            domain_bounds=bounds, n_sample_target_funcs=args.n_sample_target_funcs,
            n_inner_bo_runs=args.n_inner_bo_runs, outer_maxiter=args.outer_maxiter,
            random_state=random_state
        )
    param_ucb_best_x_history, param_ucb_train_regret_history, param_ucb_acq_func_history, param_ucb_observation_xs, \
        param_ucb_observation_ys = param_ucb_history
    print('parameterized UCB param list: {}'.format(param_ucb_params))
    print('best_x_history: {}'.format(param_ucb_best_x_history))
    print('parameterized UCB train regret: {}'.format(param_ucb_train_regret_history))
    results['param_ucb_history'] = param_ucb_history
    results['param_ucb_target_funcs'] = param_ucb_target_funcs

    # test optimized parameterized UCB
    param_ucb_test_regrets, param_ucb_test_regret_histories = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, param_ucb_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    param_ucb_regret_mean, param_ucb_regret_std, parama_ucb_regrets = param_ucb_test_regrets
    param_ucb_regret_history_mean, param_ucb_regret_history_std, param_ucb_regret_histories = param_ucb_test_regret_histories
    results['param_ucb_regret_mean'] = param_ucb_regret_mean
    results['param_ucb_regret_std'] = param_ucb_regret_std
    results['param_ucb_regrets'] = parama_ucb_regrets
    results['param_ucb_regret_history_mean'] = param_ucb_regret_history_mean
    results['param_ucb_regret_history_std'] = param_ucb_regret_history_std
    results['param_ucb_regret_histories'] = param_ucb_regret_histories
    print('parameterized UCB regret mean: {}, std: {}'.format(param_ucb_regret_std, param_ucb_regret_std))

    param_ucb_test_regret_mean_history = []
    param_ucb_test_regret_std_history = []
    param_ucb_test_regrets_history = []
    param_ucb_test_eggholder_et_al_regret_mean_history = []
    param_ucb_test_eggholder_et_al_regret_std_history = []
    param_ucb_test_eggholder_et_al_regrets_history = []
    for i, acq_func in enumerate(param_ucb_acq_func_history):
        if i > 0 and np.array_equal(param_ucb_best_x_history[i], param_ucb_best_x_history[i - 1]):
            param_ucb_test_regret_mean_history.append(param_ucb_test_regret_mean_history[-1])
            param_ucb_test_regret_std_history.append(param_ucb_test_regret_std_history[-1])
            param_ucb_test_eggholder_et_al_regret_mean_history.append(param_ucb_test_eggholder_et_al_regret_mean_history[-1])
            param_ucb_test_eggholder_et_al_regret_std_history.append(param_ucb_test_eggholder_et_al_regret_std_history[-1])
            continue

        (param_ucb_regret_mean, param_ucb_regret_std, param_ucb_regrets), _ = benchmark_acq_func(
            test_funcs, test_func_bounds, test_func_max_values, acq_func,
            args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
        )
        param_ucb_test_regret_mean_history.append(param_ucb_regret_mean)
        param_ucb_test_regret_std_history.append(param_ucb_regret_std)
        param_ucb_test_regrets_history.append(param_ucb_regrets)

        (param_ucb_eggholder_et_al_regret_mean, param_ucb_eggholder_et_al_regret_std, param_ucb_eggholder_et_al_regrets), _ = benchmark_acq_func(
            eggholder_et_al_test_funcs, eggholder_et_al_test_func_bounds, eggholder_et_al_test_func_max_values,
            acq_func,
            args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
        )
        param_ucb_test_eggholder_et_al_regret_mean_history.append(param_ucb_eggholder_et_al_regret_mean)
        param_ucb_test_eggholder_et_al_regret_std_history.append(param_ucb_eggholder_et_al_regret_std)
        param_ucb_test_eggholder_et_al_regrets_history.append(param_ucb_eggholder_et_al_regrets)
    print('parameterized UCB test regret mean: {}, std: {}'.format(
        param_ucb_test_regret_mean_history, param_ucb_test_regret_std_history))
    results['param_ucb_test_regret_mean_history'] = param_ucb_test_regret_mean_history
    results['param_ucb_test_regret_std_history'] = param_ucb_test_regret_std_history
    results['param_ucb_test_regrets_history'] = param_ucb_test_regrets_history
    print('parameterized UCB eggholder et al. test regret mean: {}, std: {}'.format(
        param_ucb_test_eggholder_et_al_regret_mean_history, param_ucb_test_eggholder_et_al_regret_std_history))
    results['param_ucb_test_eggholder_et_al_regret_mean_history'] = param_ucb_test_eggholder_et_al_regret_mean_history
    results['param_ucb_test_eggholder_et_al_regret_std_history'] = param_ucb_test_eggholder_et_al_regret_std_history
    results['param_ucb_test_eggholder_et_al_regrets_history'] = param_ucb_test_eggholder_et_al_regrets_history

    plt.xlabel('outer loop iteration')
    plt.ylabel('regret')
    plt.plot(param_ucb_train_regret_history, label='train regret')
    plt.plot(param_ucb_test_regret_mean_history, label='test regret mean')
    plt.legend()
    plt.savefig('results/param_ucb_history_{}.pdf'.format(args.save_id))
    plt.close()

    # test static EI
    ei_acq_func = StaticAcquisitionFunction(EIAcquisitionFunction(0.0))
    ei_test_regrets, ei_test_regret_histories = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, ei_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    ei_regret_mean, ei_regret_std, ei_regrets = ei_test_regrets
    ei_regret_history_mean, ei_regret_history_std, ei_regret_histories = ei_test_regret_histories
    results['ei_acq_regret_mean'] = ei_regret_mean
    results['ei_acq_regret_std'] = ei_regret_std
    results['ei_acq_regrets'] = ei_regrets
    results['ei_acq_regret_history_mean'] = ei_regret_history_mean
    results['ei_acq_regret_history_std'] = ei_regret_history_std
    results['ei_acq_regret_histories'] = ei_regret_histories
    print('static EI regret mean: {}, std: {}'.format(ei_regret_mean, ei_regret_std))

    # test static PI
    pi_acq_func = StaticAcquisitionFunction(PIAcquisitionFunction(0.0))
    pi_test_regrets, pi_test_regret_histories = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, pi_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    pi_regret_mean, pi_regret_std, pi_regrets = pi_test_regrets
    pi_regret_history_mean, pi_regret_history_std, pi_regret_histories = pi_test_regret_histories
    results['pi_acq_regret_mean'] = pi_regret_mean
    results['pi_acq_regret_std'] = pi_regret_std
    results['pi_acq_regrets'] = pi_regrets
    results['pi_acq_regret_history_mean'] = pi_regret_history_mean
    results['pi_acq_regret_history_std'] = pi_regret_history_std
    results['pi_acq_regret_histories'] = pi_regret_histories
    print('static PI regret mean: {}, std: {}'.format(pi_regret_mean, pi_regret_std))

    # test static UCB
    static_ucb_acq_func = StaticAcquisitionFunction(UCBAcquisitionFunction(1.0))
    static_ucb_test_regrets, static_ucb_test_regret_histories = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, static_ucb_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    static_ucb_regret_mean, static_ucb_regret_std, static_ucb_regrets = static_ucb_test_regrets
    static_ucb_regret_history_mean, static_ucb_regret_history_std, static_ucb_regret_histories = static_ucb_test_regret_histories
    results['static_ucb_acq_regret_mean'] = static_ucb_regret_mean
    results['static_ucb_acq_regret_std'] = static_ucb_regret_std
    results['static_ucb_acq_regrets'] = static_ucb_regrets
    results['static_ucb_acq_regret_history_mean'] = static_ucb_regret_history_mean
    results['static_ucb_acq_regret_history_std'] = static_ucb_regret_history_std
    results['static_ucb_acq_regret_histories'] = static_ucb_regret_histories
    print('static UCB regret mean: {}, std: {}'.format(static_ucb_regret_mean, static_ucb_regret_std))

    # test decay UCB
    decay_ucb_acq_funcs = []
    beta = 1.0
    decay = 0.9
    for i in range(args.max_budget):
        decay_ucb_acq_funcs = [UCBAcquisitionFunction(beta)] + decay_ucb_acq_funcs
        beta *= decay
    decay_ucb_acq_func = SequenceAcquisitionFunction(decay_ucb_acq_funcs)
    decay_ucb_test_regrets, decay_ucb_test_regret_histories = benchmark_acq_func(
        test_funcs, test_func_bounds, test_func_max_values, decay_ucb_acq_func,
        args.n_test_init_points, args.max_budget, args.n_test_bo_runs, random_state
    )
    decay_ucb_regret_mean, decay_ucb_regret_std, decay_ucb_regrets = decay_ucb_test_regrets
    decay_ucb_regret_history_mean, decay_ucb_regret_history_std, decay_ucb_regret_histories = decay_ucb_test_regret_histories
    results['decay_ucb_acq_regret_mean'] = decay_ucb_regret_mean
    results['decay_ucb_acq_regret_std'] = decay_ucb_regret_std
    results['decay_ucb_acq_regrets'] = decay_ucb_regrets
    results['decay_ucb_acq_regret_history_mean'] = decay_ucb_regret_history_mean
    results['decay_ucb_acq_regret_history_std'] = decay_ucb_regret_history_std
    results['decay_ucb_acq_regret_histories'] = decay_ucb_regret_histories
    print('decay UCB regret mean: {}, std: {}'.format(decay_ucb_regret_mean, decay_ucb_regret_std))

    np.save('results/results_{}.npy'.format(args.save_id), results)

