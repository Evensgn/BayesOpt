import os
from matplotlib import pyplot as plt
import numpy as np


# all_results: list of evaluation results [result1, result2, ...]
# result_i is a dict with {"policy" : 'EI'/'Random'/etc..., 
#                          "rewards": [t1_exp1, t1_exp2, ... t1_expM, ... tN_expM] OR [(t1_exp1, 1), ... (tN_expM, N)],
#                          "T": N }
def plot_rewards(all_results, plot_name, file_path, logplot = False):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # do the plot
    for result in all_results:
        # prepare rewards_dict
        rewards_dict = {}
        for i, rew in enumerate(result["rewards"]):
            if isinstance(rew, tuple):
                t = rew[1]
                reward = rew[0]
            else:
                t = i % result['T'] + 1
                reward = rew

            if str(t) in rewards_dict:
                rewards_dict[str(t)].append(reward)
            else:
                rewards_dict[str(t)] = [reward]

        t_vec, loc, err_low, err_high = [], [], [], []
        for key, val in rewards_dict.items():
            t_vec.append(int(key))
            cur_loc = np.median(val)
            cur_err_low = np.percentile(val, q=70)
            cur_err_high = np.percentile(val, q=30)
            loc.append(cur_loc)
            err_low.append(cur_err_low)
            err_high.append(cur_err_high)

        t_vec, loc, err_low, err_high = np.array(t_vec), np.array(loc), np.array(err_low), np.array(err_high)
        # sort the arrays according to T
        sort_idx = np.argsort(t_vec)
        t_vec = t_vec[sort_idx]
        loc = loc[sort_idx]
        err_low = err_low[sort_idx]
        err_high = err_high[sort_idx]

        if not logplot:
            line = ax.plot(t_vec, loc, label=result['policy'])[0]
            ax.fill_between(t_vec, err_low, err_high, alpha=0.2, facecolor=line.get_color())
        else:
            line = ax.semilogy(t_vec, loc, label=result['policy'])[0]
            ax.fill_between(t_vec, err_low, err_high, alpha=0.2, facecolor=line.get_color())

    fig.suptitle(plot_name)
    ax.grid(alpha=0.3)
    ax.set_xlabel("t", labelpad=0)
    ax.set_ylabel("simple regret")
    ax.legend()

    fig.savefig(fname=os.path.join(file_path, "plot.png"))
    plt.close(fig)

