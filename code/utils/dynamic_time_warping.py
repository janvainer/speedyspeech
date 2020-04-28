import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_dtw_1d(s1, s2, path, cost):
    """

    :param s1: 1d numpy array
    :param s2: 1d numpy array
    :param path: 2d numpy array
    :param cost: cost of path= dtw[-1, -1]
    :return:
    """
    plt.plot(np.arange(len(s1)), s1)
    plt.plot(np.arange(len(s2)), s2)

    for i in range(len(path)):
        x1, x2 = path[i]
        plt.plot(path[i], [s1[x1], s2[x2]], color='k', linestyle='--', linewidth=1)

    plt.title(f'cost: {cost}')
    plt.show()
