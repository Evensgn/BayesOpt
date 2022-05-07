from matplotlib import pyplot as plt
import numpy as np
from util import *
from main import *
import argparse


def plot_performance_curve(ax, data, label, time_list, log_plot):
    '''
    middle = np.median(data, axis=0)
    err_low = np.percentile(data, q=70, axis=0)
    err_high = np.percentile(data, q=30, axis=0)
    '''
    middle = np.mean(data, axis=0)
    err_low = np.mean(data, axis=0) - 0.5 * np.std(data, axis=0)
    err_high = np.mean(data, axis=0) + 0.5 * np.std(data, axis=0)

    if log_plot:
        line = ax.semilogy(time_list, middle, label=label)[0]
    else:
        line = ax.plot(time_list, middle, label=label)[0]
    ax.fill_between(time_list, err_low, err_high, alpha=0.2, color=line.get_color())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimization Plotter')
    parser.add_argument('--load_id', type=str, default='default', help='Load id')
    parser.add_argument('--log_plot', action='store_true', help='Plot log scale')
    args = parser.parse_args()

    results = np.load('results/results_{}.npy'.format(args.load_id), allow_pickle=True).item()

    # plot the learning curve
    exp_args = results['args']
    acq_func_types = exp_args.acq_func_types.split(',')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('outer loop iteration')
    ax.set_ylabel('average simple regret')

    for acq_func_type in acq_func_types:
        param_ucb_history = results['{}_history'.format(acq_func_type)]
        param_ucb_best_x_history, param_ucb_train_regret_history, param_ucb_acq_func_history, param_ucb_observation_xs, \
            param_ucb_observation_ys = param_ucb_history
        param_ucb_test_regret_mean_history = results['{}_test_regret_mean_history'.format(acq_func_type)]
        ax.plot(param_ucb_train_regret_history, label='{} train regret'.format(acq_func_type))
        ax.plot(param_ucb_test_regret_mean_history, label='{} test regret'.format(acq_func_type))

    ax.legend()
    fig.savefig('results/learned_acq_learning_curve_{}.pdf'.format(args.load_id))
    plt.close(fig)

    # plot the performance curve for all acquisition functions
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('BO iteration')
    ax.set_ylabel('simple regret')

    max_budget = exp_args.max_budget
    time_list = range(1, max_budget + 1)

    for acq_func_type in results['acq_key_list']:
        ei_regret_histories = results['{}_regret_histories'.format(acq_func_type)]
        plot_performance_curve(ax, ei_regret_histories[0], acq_func_type, time_list, args.log_plot)

    ax.legend()
    fig.savefig('results/performance_curves_{}.pdf'.format(args.load_id))
    plt.close(fig)

