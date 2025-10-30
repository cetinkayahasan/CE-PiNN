import os
import numpy as np
from scipy.io import loadmat

from mylib import env


def get_folder_name(exp_name):
    path_folder = os.path.join(env.DIR_EXPERIMENTS, exp_name)
    return path_folder


def get_test_data(gamma1,gamma2):

    print(f"::: Loading test data for gamma1={gamma1}, gamma2={gamma2} :::")
    filepath = os.path.join(env.DIR_DATASETS, f"kdv_lambda1_{gamma1}_lambda2_{gamma2}.mat")
    data = loadmat(filepath)

    t = data["t"]
    x = data["x"]
    u = data["usol"]
    # t = data["tt"]
    # x = data["x"]
    # u = data["uu"].T

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y
