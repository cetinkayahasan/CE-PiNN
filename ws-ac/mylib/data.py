import os
import numpy as np
from scipy.io import loadmat

from mylib import env


def get_folder_name(exp_name):
    path_folder = os.path.join(env.DIR_EXPERIMENTS, exp_name)
    return path_folder


def get_test_data(rho, lmbda=0.001):

    if isinstance(rho, float) and rho.is_integer():
        rho_str = f"{rho:.0f}"
    else:
        rho_str = str(rho)

    filepath = os.path.join(env.DIR_DATASETS, f"ac_n{lmbda}_r{rho}.mat")
    data = loadmat(filepath)

    t = data["t"]
    x = data["x"]
    u = data["usol"]

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y
