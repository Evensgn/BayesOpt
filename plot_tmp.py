from matplotlib import pyplot as plt
import numpy as np
from util import *
from main import *
import argparse
import seaborn as sns


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
    # ax.fill_between(time_list, err_low, err_high, alpha=0.2, color=line.get_color())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimization Plotter')
    parser.add_argument('--load_id', type=str, default='default', help='Load id')
    parser.add_argument('--log_plot', action='store_true', help='Plot log scale')
    args = parser.parse_args()

    # results = np.load('results/results_{}.npy'.format(args.load_id), allow_pickle=True).item()
    results = np.load(
        'results/results_{}.npy'.format(
            '58-dim-2-grid-20-budget-20-tfunc-100-oiter-100-nu-10.0-lambda-1.0-lenscale-0.3-outer-ei-fix-train'
        ),
        allow_pickle=True
    ).item()
    results_2 = np.load(
        'results/results_{}.npy'.format(
            '59-dim-2-grid-20-budget-20-tfunc-100-oiter-100-nu-10.0-lambda-1.0-lenscale-0.3-outer-ei-fix-train'
        ),
        allow_pickle=True
    ).item()
    results_3 = np.load(
        'results/results_{}.npy'.format(
            '62-dim-2-grid-20-budget-20-tfunc-100-oiter-100-nu-10.0-lambda-1.0-lenscale-0.3-outer-ei-fix-train'
        ),
        allow_pickle=True
    ).item()
    results_4 = np.load(
        'results/results_{}.npy'.format(
            '64-dim-2-grid-20-budget-20-tfunc-100-oiter-100-nu-10.0-lambda-1.0-lenscale-0.3-outer-ei-fix-train'
        ),
        allow_pickle=True
    ).item()
    results_5 = np.load(
        'results/results_{}.npy'.format(
            '65-dim-2-grid-20-budget-20-tfunc-300-oiter-1-nu-10.0-lambda-1.0-lenscale-0.3-outer-ei-fix-train'
        ),
        allow_pickle=True
    ).item()

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
    fig.savefig('results/learned_acq_learning_curve_ucb_exponential.pdf')
    plt.close(fig)

    exp_args = results_2['args']
    acq_func_types = exp_args.acq_func_types.split(',')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('outer loop iteration')
    ax.set_ylabel('average simple regret')

    for acq_func_type in acq_func_types:
        param_ucb_history = results_2['{}_history'.format(acq_func_type)]
        param_ucb_best_x_history, param_ucb_train_regret_history, param_ucb_acq_func_history, param_ucb_observation_xs, \
        param_ucb_observation_ys = param_ucb_history
        param_ucb_test_regret_mean_history = results_2['{}_test_regret_mean_history'.format(acq_func_type)]
        ax.plot(param_ucb_train_regret_history, label='{} train regret'.format(acq_func_type))
        ax.plot(param_ucb_test_regret_mean_history, label='{} test regret'.format(acq_func_type))

    ax.legend()
    fig.savefig('results/learned_acq_learning_curve_ucb_linear.pdf')
    plt.close(fig)

    exp_args = results_3['args']
    acq_func_types = exp_args.acq_func_types.split(',')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('outer loop iteration')
    ax.set_ylabel('average simple regret')

    for acq_func_type in acq_func_types:
        param_ucb_history = results_3['{}_history'.format(acq_func_type)]
        param_ucb_best_x_history, param_ucb_train_regret_history, param_ucb_acq_func_history, param_ucb_observation_xs, \
        param_ucb_observation_ys = param_ucb_history
        param_ucb_test_regret_mean_history = results_3['{}_test_regret_mean_history'.format(acq_func_type)]
        ax.plot(param_ucb_train_regret_history, label='{} train regret'.format(acq_func_type))
        ax.plot(param_ucb_test_regret_mean_history, label='{} test regret'.format(acq_func_type))

    ax.legend()
    fig.savefig('results/learned_acq_learning_curve_ei_exponential.pdf')
    plt.close(fig)

    exp_args = results_4['args']
    acq_func_types = exp_args.acq_func_types.split(',')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('outer loop iteration')
    ax.set_ylabel('average simple regret')

    for acq_func_type in acq_func_types:
        param_ucb_history = results_4['{}_history'.format(acq_func_type)]
        param_ucb_best_x_history, param_ucb_train_regret_history, param_ucb_acq_func_history, param_ucb_observation_xs, \
        param_ucb_observation_ys = param_ucb_history
        param_ucb_test_regret_mean_history = results_4['{}_test_regret_mean_history'.format(acq_func_type)]
        ax.plot(param_ucb_train_regret_history, label='{} train regret'.format(acq_func_type))
        ax.plot(param_ucb_test_regret_mean_history, label='{} test regret'.format(acq_func_type))

    ax.legend()
    fig.savefig('results/learned_acq_learning_curve_ei_linear.pdf')
    plt.close(fig)

    exp_args = results['args']
    acq_func_types = exp_args.acq_func_types.split(',')

    # plot the performance curve for all acquisition functions
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('BO iteration')
    ax.set_ylabel('simple regret')

    # colors = sns.color_palette("hls", 11)
    # ax.set_prop_cycle('color', colors)

    max_budget = exp_args.max_budget
    time_list = range(1, max_budget + 1)

    ei_regret_histories = results_5['{}_regret_histories'.format('random')]
    plot_performance_curve(ax, ei_regret_histories[0], 'random', time_list, args.log_plot)

    for acq_func_type in results['acq_key_list']:
        if acq_func_type == 'param-ucb-exponential-schedule-1000.0':
            continue

        '''
        if acq_func_type == 'random':
            continue
        elif acq_func_type == 'static-ei':
            continue
        elif acq_func_type == 'static-pi':
            continue
        '''
        if acq_func_type == 'theory-ucb-delta-0.5-uniform-pi':
            pass
        else:
            if 'ucb' in acq_func_type:
                continue

        ei_regret_histories = results['{}_regret_histories'.format(acq_func_type)]
        plot_performance_curve(ax, ei_regret_histories[0], acq_func_type, time_list, args.log_plot)

    ei_regret_histories = results['{}_regret_histories'.format('param-ucb-exponential-schedule-1000.0')]
    plot_performance_curve(ax, ei_regret_histories[0], 'param-ucb-exponential-schedule-1000.0', time_list, args.log_plot)

    '''
    ei_regret_histories = results_2['{}_regret_histories'.format('param-ucb-linear-schedule-1000.0')]
    plot_performance_curve(ax, ei_regret_histories[0], 'param-ucb-linear-schedule-1000.0', time_list, args.log_plot)
    '''

    ei_regret_histories = results_3['{}_regret_histories'.format('param-ei-exponential-schedule-0.1')]
    plot_performance_curve(ax, ei_regret_histories[0], 'param-ei-exponential-schedule-0.1', time_list, args.log_plot)

    ei_regret_histories = results_4['{}_regret_histories'.format('param-ei-linear-schedule-0.1')]
    plot_performance_curve(ax, ei_regret_histories[0], 'param-ei-linear-schedule-0.1', time_list, args.log_plot)

    ax.legend()
    ax.set_xlim(15, 20)
    ax.set_ylim(0.7, 1.0)
    fig.savefig('results/performance_curves_full.pdf')
    # fig.savefig('results/performance_curves_{}.pdf'.format(args.load_id))
    plt.close(fig)

