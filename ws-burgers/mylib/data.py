import os
import numpy as np
from scipy.io import loadmat

from mylib import env


def get_folder_name(exp_name):
    path_folder = os.path.join(env.DIR_EXPERIMENTS, exp_name)
    return path_folder


def get_test_data(nu):

    if isinstance(nu, float) and nu.is_integer():
        nu_str = f"{nu:.0f}"
    else:
        nu_str = str(nu)

    #filepath = os.path.join(env.DIR_DATASETS, f"burgers_nu_{nu_str}.mat")
    # data = np.load(filepath)
    filepath = os.path.join(env.DIR_DATASETS, f"burgers_nu_{nu_str}.mat")
    data = loadmat(filepath)
    

    t = data["t"]
    x = data["x"]
    u = data["usol"]
    # u = data["usol"].T # Transpose if needed .npz file
    
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y
