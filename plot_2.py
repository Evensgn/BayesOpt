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
    result_more = np.load('results/results_estimated_prior_matern1.5_100.npy', allow_pickle=True).item()
    results = result_more['result_list']

    print(result_more['regrets_stats_random'])
    print('mean_correct_avg:', result_more['mean_correct_avg'])
    print('std_correct_avg:', result_more['std_correct_avg'])
    print('mean_estimated_avg:', result_more['mean_estimated_avg'])
    print('std_estimated_avg:', result_more['std_estimated_avg'])

    # regret bound vs beta
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('beta')
    ax.set_ylabel('average best sample simple regret')

    ax.set_xscale('log')

    betas = []
    correct_gp_regrets = []
    correct_gp_regrets_upper = []
    correct_gp_regrets_lower = []
    estimated_gp_regrets = []
    estimated_gp_regrets_upper = []
    estimated_gp_regrets_lower = []

    for result in results:
        betas.append(result['beta'])
        correct_gp_regrets.append(result['regrets_stats_correct_gp'][0])
        correct_gp_regrets_upper.append(result['regrets_stats_correct_gp'][0] + result['regrets_stats_correct_gp'][1])
        correct_gp_regrets_lower.append(result['regrets_stats_correct_gp'][0] - result['regrets_stats_correct_gp'][1])
        estimated_gp_regrets.append(result['regrets_stats_estimated_gp'][0])
        estimated_gp_regrets_upper.append(result['regrets_stats_estimated_gp'][0] + result['regrets_stats_estimated_gp'][1])
        estimated_gp_regrets_lower.append(result['regrets_stats_estimated_gp'][0] - result['regrets_stats_estimated_gp'][1])

    line = ax.plot(betas, correct_gp_regrets, label='correct_gp')[0]
    ax.fill_between(betas, correct_gp_regrets_lower, correct_gp_regrets_upper, alpha=0.2, color=line.get_color())
    line = ax.plot(betas, estimated_gp_regrets, label='estimated_gp')[0]
    ax.fill_between(betas, estimated_gp_regrets_lower, estimated_gp_regrets_upper, alpha=0.2, color=line.get_color())

    ax.legend()
    fig.savefig('results/regret_vs_beta_matern1.5_100.pdf')
    plt.close(fig)

    # plot the performance curves
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xlabel('BO iteration')
    ax.set_ylabel('average best sample simple regret')

    max_budget = 100
    time_list = range(1, max_budget + 1)

    result = results[2]
    correct_gp_regret_history = result['regret_histories_stats_correct_gp'][0]
    correct_gp_regret_history_std = result['regret_histories_stats_correct_gp'][1]
    correct_gp_regret_history_upper = [correct_gp_regret_history[i] + correct_gp_regret_history_std[i] for i in range(max_budget)]
    correct_gp_regret_history_lower = [correct_gp_regret_history[i] - correct_gp_regret_history_std[i] for i in range(max_budget)]
    line = ax.plot(time_list, correct_gp_regret_history, label='correct_gp')[0]
    ax.fill_between(time_list, correct_gp_regret_history_lower, correct_gp_regret_history_upper, alpha=0.2, color=line.get_color())

    estimated_gp_regret_history = result['regret_histories_stats_estimated_gp'][0]
    estimated_gp_regret_history_std = result['regret_histories_stats_estimated_gp'][1]
    estimated_gp_regret_history_upper = [estimated_gp_regret_history[i] + estimated_gp_regret_history_std[i] for i in range(max_budget)]
    estimated_gp_regret_history_lower = [estimated_gp_regret_history[i] - estimated_gp_regret_history_std[i] for i in range(max_budget)]
    line = ax.plot(time_list, estimated_gp_regret_history, label='estimated_gp')[0]
    ax.fill_between(time_list, estimated_gp_regret_history_lower, estimated_gp_regret_history_upper, alpha=0.2, color=line.get_color())

    '''
    estimated_gp_regret_history_small_dataset = result['regret_histories_stats_estimated_gp_small_dataset'][0]
    estimated_gp_regret_history_small_dataset_std = result['regret_histories_stats_estimated_gp_small_dataset'][1]
    estimated_gp_regret_history_small_dataset_upper = [estimated_gp_regret_history_small_dataset[i] + estimated_gp_regret_history_small_dataset_std[i] for i in range(max_budget)]
    estimated_gp_regret_history_small_dataset_lower = [estimated_gp_regret_history_small_dataset[i] - estimated_gp_regret_history_small_dataset_std[i] for i in range(max_budget)]
    line = ax.plot(time_list, estimated_gp_regret_history_small_dataset, label='estimated_gp_small_dataset')[0]
    ax.fill_between(time_list, estimated_gp_regret_history_small_dataset_lower, estimated_gp_regret_history_small_dataset_upper, alpha=0.2, color=line.get_color())

    estimated_gp_regret_history_large_dataset = result['regret_histories_stats_estimated_gp_large_dataset'][0]
    estimated_gp_regret_history_large_dataset_std = result['regret_histories_stats_estimated_gp_large_dataset'][1]
    estimated_gp_regret_history_large_dataset_upper = [estimated_gp_regret_history_large_dataset[i] + estimated_gp_regret_history_large_dataset_std[i] for i in range(max_budget)]
    estimated_gp_regret_history_large_dataset_lower = [estimated_gp_regret_history_large_dataset[i] - estimated_gp_regret_history_large_dataset_std[i] for i in range(max_budget)]
    line = ax.plot(time_list, estimated_gp_regret_history_large_dataset, label='estimated_gp_large_dataset')[0]
    ax.fill_between(time_list, estimated_gp_regret_history_large_dataset_lower, estimated_gp_regret_history_large_dataset_upper, alpha=0.2, color=line.get_color())
    '''

    random_gp_regret_history = result_more['regret_histories_stats_random'][0]
    random_gp_regret_history_std = result_more['regret_histories_stats_random'][1]
    random_gp_regret_history_upper = [random_gp_regret_history[i] + random_gp_regret_history_std[i] for i in
                                         range(max_budget)]
    random_gp_regret_history_lower = [random_gp_regret_history[i] - random_gp_regret_history_std[i] for i in
                                         range(max_budget)]
    line = ax.plot(time_list, random_gp_regret_history, label='random')[0]
    ax.fill_between(time_list, random_gp_regret_history_lower, random_gp_regret_history_upper, alpha=0.2,
                    color=line.get_color())

    ax.legend()
    fig.savefig('results/regret_vs_steps_beta_matern1.5_100.pdf')
    plt.close(fig)

