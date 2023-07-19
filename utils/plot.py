import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_rewards(accum, WINDOW, run_path):
    plt.figure()
    plt.plot(accum['steps'], accum['rew'])
    plt.ylim(-500, 200)
    plt.savefig(run_path + '/rew.png')
    plt.close()

    plt.plot(accum['eval_rew'])
    plt.ylim(-500, 200)
    plt.savefig(run_path + '/eval_rew.png')
    plt.close()

    plt.plot(accum['steps'][WINDOW-1:], np.convolve(accum['rew'], np.ones(WINDOW), 'valid') / WINDOW)
    plt.ylim(-500, 200)
    plt.savefig(run_path + '/roll_rew.png')
    plt.close()

    plt.plot(np.convolve(accum['eval_rew'], np.ones(WINDOW), 'valid') / WINDOW)
    plt.ylim(-500, 200)
    plt.savefig(run_path + '/roll_eval_rew.png')
    plt.close()
    return