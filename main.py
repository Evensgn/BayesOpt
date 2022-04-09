import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from util import SequenceAcquisitionFunction, StaticAcquisitionFunction
from bayesian_optimization import BayesianOptimization
from bo_test_functions import test_funcs, test_func_bounds
from scipy.optimize import differential_evolution


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


def test_single_step_neural_acq_func(x, remaining_budget, later_step_acq_funcs, n_bo_runs=100):
    model = NeuralAcquisitionFunction()
    model.load_state_dict_from_vector(x)

    def current_step_acq_func(mean, std, y_max):
        n_samples = mean.shape[0]
        mean = mean.reshape([n_samples, 1])
        std = std.reshape([n_samples, 1])
        y_max = np.repeat(y_max, mean.shape[0]).reshape([n_samples, 1])
        ret = model(torch.from_numpy(np.hstack((mean, std, y_max))).float()).detach().numpy()
        ret = np.squeeze(ret)
        return ret

    acq_funcs = later_step_acq_funcs + [current_step_acq_func]
    seq_acq_func = SequenceAcquisitionFunction(acq_funcs)
    acq_func = seq_acq_func.acq_func

    bo = BayesianOptimization(None, np.array([[-1.0, 1.0]] * 2))
    mean_y_max = 0.0
    for i in range(n_bo_runs):
        bo.initialize()
        _, y_max = bo.maximize(acq_func, n_init_points=5, budget=remaining_budget)
        mean_y_max += y_max
    mean_y_max /= n_bo_runs
    return mean_y_max


def test_acq_func(f, bound, acq_func, budget, n_bo_runs=100):
    bo = BayesianOptimization(f, bound)
    mean_y_max = 0.0
    for i in range(n_bo_runs):
        bo.initialize()
        _, y_max = bo.maximize(acq_func, n_init_points=5, budget=budget)
        mean_y_max += y_max
    mean_y_max /= n_bo_runs
    return mean_y_max


def fitness(x):
    return -test_single_step_neural_acq_func(x, 2, [util.expected_improvement], n_bo_runs=100)


if __name__ == '__main__':
    ei_acq_func = StaticAcquisitionFunction(util.expected_improvement).acq_func

    # test EI when budget = 1
    print('EI when budget = 1:', test_acq_func(test_funcs['eggholder'], test_func_bounds['eggholder'],
                                               ei_acq_func, budget=1))
    # test EI when budget = 2
    print('EI when budget = 2:', test_acq_func(test_funcs['eggholder'], test_func_bounds['eggholder'],
                                               ei_acq_func, budget=2))
    # test EI when budget = 3
    print('EI when budget = 3:', test_acq_func(test_funcs['eggholder'], test_func_bounds['eggholder'],
                                               ei_acq_func, budget=3))

    # test second-to-last-step neural acq_func optimized with differential evolution
    last_step_model = NeuralAcquisitionFunction()
    bounds = [(-1.0, 1.0)] * last_step_model.total_num_parameters

    def test_second_to_last_step_neural_acq_func(x):
        model = NeuralAcquisitionFunction()
        model.load_state_dict_from_vector(x)

        def second_to_last_step_acq_func(mean, std, y_max):
            n_samples = mean.shape[0]
            mean = mean.reshape([n_samples, 1])
            std = std.reshape([n_samples, 1])
            y_max = np.repeat(y_max, mean.shape[0]).reshape([n_samples, 1])
            ret = model(torch.from_numpy(np.hstack((mean, std, y_max))).float()).detach().numpy()
            ret = np.squeeze(ret)
            return ret

        acq_func = SequenceAcquisitionFunction([util.expected_improvement, second_to_last_step_acq_func]).acq_func
        return test_acq_func(test_funcs['eggholder'], test_func_bounds['eggholder'],
                             acq_func, budget=2, n_bo_runs=100)

    def de_callback(x, convergence):
        pseudo_y = test_single_step_neural_acq_func(x, remaining_budget=2, later_step_acq_funcs=[util.expected_improvement], n_bo_runs=100)
        test_y = test_second_to_last_step_neural_acq_func(x)
        print('pseudo y: {}, test y: {}, convergence: {}'.format(pseudo_y, test_y, convergence))

    result = differential_evolution(fitness, bounds,
                                    maxiter=2, workers=1, callback=de_callback)

    print('test result: {}'.format(test_second_to_last_step_neural_acq_func(result.x)))


