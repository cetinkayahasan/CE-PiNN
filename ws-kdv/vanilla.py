"""
Vanilla Algorithm - Basic PINNs implementation (KdV with periodic features)
"""

import os
import json
import logging
import numpy as np
import deepxde as dde
from deepxde.backend import torch

from mylib import env
from mylib import data as dt, model as md

# get the logger
logger = env.logger


# 1. Output Transformation (matches "dirichlet" style in TF code)
def output_transform(x, y):
    """Hard-enforce IC: u(x,0) = cos(pi x)"""
    x_in = x[:, 0:1]  # spatial coordinate
    t_in = x[:, 1:2]  # temporal coordinate
    return 2 * t_in * y + torch.cos(np.pi * x_in)


# 2. Feature Transformation (matches "periodic" style in TF code)
def feature_transform(x):
    """Fourier features for periodicity in x"""
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return torch.cat(
        [
            torch.cos(np.pi * x_in),
            torch.sin(np.pi * x_in),
            torch.cos(2 * np.pi * x_in),
            torch.sin(2 * np.pi * x_in),
            t_in,
        ],
        dim=1,
    )


# 3. PDE Residual (KdV)
def pde_KdV(x, y, lambda1, lambda2):
    """KdV equation: u_t + λ₁ u u_x + λ₂ u_xxx = 0"""
    u = y
    du_x = dde.grad.jacobian(y, x, i=0, j=0)   # u_x
    du_t = dde.grad.jacobian(y, x, i=0, j=1)   # u_t
    du_xx = dde.grad.hessian(y, x, i=0, j=0)   # u_xx
    du_xxx = dde.grad.jacobian(du_xx, x, i=0, j=0)  # u_xxx
    return du_t + lambda1 * u * du_x + lambda2 * du_xxx


def solve(
    path_output,
    path_debug,
    gamma1,
    gamma2,
    n_epochs1=10000,
    n_collocation=2000,
    lr=0.001,
    test_period=1000,
):
    logger.info(f"::: gamma1={gamma1} and gamma2={gamma2} :::")
    logger.info(f"::: n_collocation={n_collocation}")
    logger.info(f"::: n_epochs1={n_epochs1}")
    logger.info(f"::: learning rate={lr}")

    # PDE
    pde = lambda x, y: pde_KdV(x, y, gamma1, gamma2)

    # Geometry
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Data
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [],  # IC/BC enforced via output_transform
        num_domain=n_collocation,
        num_test=10000,
        train_distribution="pseudo",
    )

    # Neural net
    net = dde.maps.FNN([5] + [100] * 5 + [1], "tanh", "Glorot normal")
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    # Test data
    X_test, y_test = dt.get_test_data(gamma1, gamma2)
    eval_callback = md.EvaluateModelCallback(
        X_test,
        y_test,
        pde,
        period=test_period,
        dir_exp=path_output,
        save_models=False,
    )

    logger.info("::: First training phase...")
    losshistory, train_state = model.train(
        epochs=n_epochs1,
        callbacks=[eval_callback],
    )

    logger.info("::: Continue with L-BFGS training phase...")
    model.compile("L-BFGS")
    losshistory, train_state = model.train(
        epochs=n_epochs1,
        callbacks=[eval_callback],
    )

    last_epoch_time = eval_callback.history["epoch"][-1]
    eval_callback.evaluate(last_epoch_time)
    eval_callback.save_results()
    # save model and history
    eval_callback.save_results()
    # final results
    last_epoch = eval_callback.history["epoch"][-1]
    final_loss = eval_callback.history["train_loss"][-1]
    final_val_loss = eval_callback.history["val_loss"][-1]
    final_l2_error = eval_callback.history["l2_error"][-1]

    # ===== JSON LOGGING =====
    results = {
        
        "gamma1": float(gamma1),
        "epochs": int(last_epoch),
        "n_collocation": n_collocation,
        "lr": lr,
        "algo": algo,
        "final_loss": float(final_loss),
        "val_loss": float(final_val_loss),
        "l2_error": float(final_l2_error),
    }
    # save to parent directory of path_output
    parent_dir = os.path.dirname(path_output)
    json_file = os.path.join(parent_dir, "results_log_kdv_vanilla.json")

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(results)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Results saved to {json_file}")
    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    print("::: Vanilla algorithm for KdV equation :::")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma2", type=float, default=0.0025)
    parser.add_argument("--gamma1", type=float)
    parser.add_argument("--nf", type=int, default=5000)
    args = parser.parse_args()

    gamma1 = args.gamma1
    gamma2 = args.gamma2
    nf = args.nf

    algo = "vanilla"
    exp_folder_name = f"{algo}-gamma1_{gamma1}-gamma2_{gamma2}-nf{nf}"

    path_output = dt.get_folder_name(exp_folder_name)
    os.makedirs(path_output, exist_ok=True)
    path_debug = os.path.join(path_output, "debug")
    os.makedirs(path_debug, exist_ok=True)

    filename = os.path.basename(__file__).split(".")[0]
    os.system(f"cp {__file__} {os.path.join(path_output, f'{filename}.bkp.py')}")

    log_path = os.path.join(path_output, "output.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(env.log_formatter)
    logger.addHandler(file_handler)

    logger.info(f"::: Running '{algo.upper()}' algorithm :::")

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        logger.info("Running on GPU")
    else:
        logger.warning("GPU is not available, running on CPU")

    n_epochs1 = 15000
    n_collocation = nf

    solve(
        path_output=path_output,
        path_debug=path_debug,
        gamma1=gamma1,
        gamma2=gamma2,
        n_epochs1=n_epochs1,
        n_collocation=n_collocation,
        lr=0.001,
    )
