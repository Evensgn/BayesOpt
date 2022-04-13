import util
from util import *
from bayesian_optimization import BayesianOptimization
from bo_test_functions import test_funcs, test_func_bounds, test_func_max, quick_test_funcs
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import argparse
from multiprocessing import Process, Queue
from pathos.multiprocessing import ProcessingPool
from time import sleep
import dill as pickle


def run_bo(run_args):
    (bo, observations_xs, observations_ys, acq_func, n_init_points, budget) = run_args
    bo.initialize(observations_xs=observations_xs, observations_ys=observations_ys)
    _, y_max = bo.maximize(acq_func, n_init_points=n_init_points, budget=budget)
    return y_max


def test_acq_func(f, bound, f_max, acq_func, n_init_points, budget, init_observations_list=None, n_bo_runs=50):
    pool = ProcessingPool(nodes=args.n_workers)

    bo = BayesianOptimization(f, bound)
    mean_y_max = 0.0

    task_list = []
    if f is None:
        for observations_xs, observations_ys in init_observations_list:
            for i in range(n_bo_runs):
                task_list.append((bo, observations_xs, observations_ys, acq_func, n_init_points, budget))

        mean_y_max /= n_bo_runs * len(init_observations_list)
    else:
        for i in range(n_bo_runs):
            task_list.append((bo, None, None, acq_func, n_init_points, budget))
    task_outputs = pool.map(run_bo, task_list)
    for output in task_outputs:
        mean_y_max += output
    mean_y_max /= len(task_outputs)

    # return the simple regret
    return f_max - mean_y_max


def test_single_step_acq_func(f, bound, f_max, single_step_acq_func, later_step_acq_funcs, n_init_points,
                              remaining_budget, init_observations_list=None, n_bo_runs=50):
    acq_funcs = later_step_acq_funcs + [single_step_acq_func]
    seq_acq_func = SequenceAcquisitionFunction(acq_funcs)
    acq_func = seq_acq_func.acq_func
    return test_acq_func(f, bound, f_max, acq_func, n_init_points, remaining_budget,
                         init_observations_list=init_observations_list,
                         n_bo_runs=n_bo_runs)


'''
def fitness(x):
    model = NeuralAcquisitionFunction()
    model.load_state_dict_from_vector(x)
    bound = np.array([[-1.0, 1.0]] * 2)
    return -test_single_step_acq_func(None, bound, model.acq_func, 0, 2, [util.expected_improvement], n_bo_runs=100)
'''


def parameterized_ucb_conversion_func(x):
    return UCBAcquisitionFunction(x).acq_func


def optimize_parameterized_acq_func(param_bound, conversion_func, x0, n_init_points, max_budget, domain_bound, len_init_observations=20,
                                    n_init_observations=10, n_bo_runs=50, lbfgs_maxiter=100):
    acq_funcs = []
    param_list = []

    init_observations_list = []
    bo = BayesianOptimization(None, domain_bound)
    for i in range(n_init_observations):
        observations_xs, observations_ys = bo.generate_init_observations_for_pesudo_target_func(len_init_observations)
        init_observations_list.append((observations_xs, observations_ys))

    for i in range(1, max_budget+1):
        print('Optimizing step {} acq func'.format(i))

        def objective_func(x):
            single_step_acq_func = conversion_func(x)
            return test_single_step_acq_func(None, domain_bound, 0.0, single_step_acq_func, acq_funcs, n_init_points, i,
                                             init_observations_list=init_observations_list,
                                             n_bo_runs=n_bo_runs)

        res = minimize(objective_func,
                       x0,
                       bounds=param_bound,
                       method='L-BFGS-B',
                       options=dict(maxiter=lbfgs_maxiter))

        single_step_acq_func = conversion_func(res.x)
        acq_funcs.append(single_step_acq_func)
        param_list.append(res.x)

    return acq_funcs, param_list


# multiprocessing Worker and dispatchWorker based on https://gist.github.com/prempv/717270a4470a10146c6776820e8e3cbc
class Worker(Process):
    def __init__(self, input_queue, output_queue):
        Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            if self.input_queue.empty():
                break
            else:
                run_args = self.input_queue.get()
                ret = run_bo(*run_args)
                self.output_queue.put(ret)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimization Experiments')
    parser.add_argument('--len_init_observations', type=int, default=20, help='Length of initial observations')
    parser.add_argument('--n_init_observations', type=int, default=10, help='Number of initial observations')
    parser.add_argument('--n_bo_runs', type=int, default=20, help='Number of BO runs')
    parser.add_argument('--max_budget', type=int, default=5, help='Maximum budget for BO')
    parser.add_argument('--acq_param_lbfgs_maxiter', type=int, default=10,
                        help='Maximum number of iterations for L-BFGS-B for parameterized acquisition function optimization')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--save_id', type=int, default=0, help='Save id')

    parser.add_argument('--n_test_init_points', type=int, default=5, help='Number of test initial points')
    parser.add_argument('--n_test_bo_runs', type=int, default=300, help='Number of test BO runs')
    args = parser.parse_args()

    # optimize parameterized UCB
    param_bound = np.array([[0.0, 10.0]])
    x0 = np.array([[1.0]])
    domain_bound = np.array([[-1.0, 1.0]] * 2)
    ucb_acq_funcs, ucb_param_list = optimize_parameterized_acq_func(
        param_bound, parameterized_ucb_conversion_func, x0,
        n_init_points=0, max_budget=args.max_budget,
        domain_bound=domain_bound,
        len_init_observations=args.len_init_observations,
        n_init_observations=args.n_init_observations,
        n_bo_runs=args.n_bo_runs,
        lbfgs_maxiter=args.acq_param_lbfgs_maxiter
    )
    print('parameterized UCB param list: {}'.format(ucb_param_list))

    # test optimized parameterized UCB
    param_ucb_acq_func = SequenceAcquisitionFunction(ucb_acq_funcs).acq_func
    param_ucb_test_results = {}
    for func in quick_test_funcs:
        print('function: {}'.format(func))
        param_ucb_test_results[func] = []
        for i in range(1, 6):
            regret = test_acq_func(test_funcs[func], test_func_bounds[func],
                                   test_func_max[func], param_ucb_acq_func,
                                   n_init_points=args.n_test_init_points, budget=i, n_bo_runs=args.n_test_bo_runs)
            param_ucb_test_results[func].append(regret)
            print('Parameterized UCB when budget = {}: {}'.format(i, regret))

    # test static EI
    ei_acq_func = StaticAcquisitionFunction(util.expected_improvement).acq_func
    ei_acq_test_results = {}
    for func in quick_test_funcs:
        print('function: {}'.format(func))
        ei_acq_test_results[func] = []
        for i in range(1, 6):
            regret = test_acq_func(test_funcs[func], test_func_bounds[func],
                                   test_func_max[func], ei_acq_func,
                                   n_init_points=args.n_test_init_points, budget=i, n_bo_runs=args.n_test_bo_runs)
            ei_acq_test_results[func].append(regret)
            print('EI when budget = {}: {}'.format(i, regret))

    results = {}
    results['ucb_param_list'] = ucb_param_list
    results['param_ucb_test_results'] = param_ucb_test_results
    results['ei_acq_test_results'] = ei_acq_test_results
    np.save('results_{}.npy'.format(args.save_id), results)





