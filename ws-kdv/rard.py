"""
CE-RAR Algorithm - Cirruculum Learning + Residual Based Adaptive Refinement 
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

# SOME FIXED PARAMETERS
RATIO = 0.005


# 1. Output Transformation (enforces initial condition)
def output_transform(x, y):
    """Hard-enforce u(0,x) = cos(πx)"""
    x_in = x[:, 0:1]  # Spatial coordinate (x ∈ [-1, 1])
    t_in = x[:, 1:2]  # Temporal coordinate (t ∈ [0, T])
    # return t_in * (1 - x_in**2)  * y + torch.cos(np.pi * x_in)  # u(t,x) = t·NN(t,x) + cos(πx) check!
    return t_in * y + torch.cos(np.pi * x_in)  # u(t,x) = t·NN(t,x) + cos(πx)

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

# 2. PDE Residual (corrected implementation)
def pde_KdV(x, y, lambda1, lambda2):
    """KdV equation: u_t + λ₁ u u_x + λ₂ u_xxx = 0"""
    # DeepXDE handles x as [x, t] by default
    u = y
    
    # First derivatives
    du_x = dde.grad.jacobian(y, x, i=0, j=0)  # u_x (spatial)
    du_t = dde.grad.jacobian(y, x, i=0, j=1)  # u_t (temporal)
    
    # Second derivative
    du_xx = dde.grad.jacobian(du_x, x, i=0, j=0)
    
    # Third derivative
    du_xxx = dde.grad.jacobian(du_xx, x, i=0, j=0)
    
    return du_t + lambda1 * u * du_x + lambda2 * du_xxx





def solve(
    path_output,
    path_debug,
    gamma1,
    gamma2,
    k=2.0,
    c=0.0,
    n_epochs1=10000,
    n_epochs2=1000,
    n_loop=90,
    n_collocation=2000,
    n_sampling=100000,
    lr=0.001,
    test_period=1000,
):
    # print log
    logger.info(f"::: gamma1={gamma1} and gamma2={gamma2}:::")
    logger.info(f"::: k={k} and c={c}:::")
    logger.info(f"::: n_collacation={n_collocation}, n_sampling={n_sampling}")
    logger.info(f"::: n_epochs1={n_epochs1}, n_epochs2={n_epochs2}, n_loop={n_loop}")
    logger.info(f"::: learning rate={lr}")

    # create the pde function
    pde = lambda x, y: pde_KdV(x, y, gamma1, gamma2)

    # create the geometry
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # create the data
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [],
        num_domain=n_collocation,
        num_test=10000,
        train_distribution="pseudo",
    )

    # create the neural network
    net = dde.maps.FNN([5] + [100] * 5 + [1], "tanh", "Glorot normal")
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    # get the test data
    X_test, y_test = dt.get_test_data(gamma1, gamma2)
    # create the callback
    eval_callback = md.EvaluateModelCallback(
        X_test,
        y_test,
        pde,
        period=test_period,
        dir_exp=path_output,
        save_models=False,
    )


    # start the training
    logger.info("::: First training phase...")
    # train model with calculating number of milliseconds passed
    losshistory, train_state = model.train(
        epochs=n_epochs1,
        callbacks=[eval_callback],
    )

    # continue with LBFGS training
    dde.optimizers.config.set_LBFGS_options( maxiter=n_epochs1)
    logger.info("::: Continue with L-BFGS training phase...")
    model.compile("L-BFGS")
    losshistory, train_state = model.train(
        epochs=n_epochs1,
        callbacks=[eval_callback],
    )

    # last epoch time
    last_epoch_time = eval_callback.history["epoch"][-1]    
    eval_callback.evaluate(last_epoch_time)


    logger.info("::: Second training  phase RARD...")
    dde.optimizers.config.set_LBFGS_options( maxiter=n_epochs2)
    for ii in range(n_loop):
        # get random points
        X = geomtime.random_points(n_sampling)
        # get the prediction
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        # calculate residuals
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]

        # add new samples top k high residual error points
        n_new_samples = int(n_collocation * RATIO)
        X_ids = np.random.choice(
            a=len(X), size=n_new_samples, replace=False, p=err_eq_normalized
        )
        X_selected = X[X_ids]
        data.add_anchors(X_selected)

        # enable plotting only in the last iteration
        losshistory, train_state = model.train(
            epochs=n_epochs2,
            callbacks=[eval_callback],
        )

        # continue with LBFGS training
        model.compile("L-BFGS")
        losshistory, train_state = model.train(
            epochs=n_epochs2,
            callbacks=[eval_callback],
        )
        # last epoch time
        last_epoch_time = eval_callback.history["epoch"][-1]
        eval_callback.evaluate(last_epoch_time)

    # save model and history
    eval_callback.save_results()

    # final results
    last_epoch = eval_callback.history["epoch"][-1]
    final_loss = eval_callback.history["train_loss"][-1]
    final_val_loss = eval_callback.history["val_loss"][-1]
    final_l2_error = eval_callback.history["l2_error"][-1]

    # ===== JSON LOGGING =====
    results = {
        "gamma2": float(gamma2),
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
    json_file = os.path.join(parent_dir, "results_log_kdv_rard.json")

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


# Run the algorithm
if __name__ == "__main__":

    # add an argument parser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma2", type=float, default=0.0025)
    parser.add_argument("--gamma1", type=float)
    parser.add_argument("--nf", type=int, default=5000)


    args = parser.parse_args()
    gamma1 = args.gamma1
    gamma2 = args.gamma2
    nf = args.nf
  

    # algorithm is simply the file name
    algo = "rard"
    exp_folder_name = f"{algo}-gamma1_{gamma1}-gamma2_{gamma2}-nf{nf}"


    # set the output path
    path_output = dt.get_folder_name(exp_folder_name)
    os.makedirs(path_output, exist_ok=True)
    path_debug = os.path.join(path_output, "debug")
    os.makedirs(path_debug, exist_ok=True)

    # create a copy of this current file to the output folder (as bkp.py file)
    # get file name
    filename = os.path.basename(__file__).split(".")[0]
    os.system(f"cp {__file__} {os.path.join(path_output, f'{filename}.bkp.py')}")

    # set the logger file handler
    # File handler
    log_path = os.path.join(path_output, "output.log")
    file_handler = logging.FileHandler(log_path, mode="w")  # Append logs to file
    file_handler.setFormatter(env.log_formatter)  # Set format for file logs
    logger.addHandler(file_handler)  # Add file handler to logger

    # print the first log
    logger.info(f"::: Running '{algo.upper()}' algorithm :::")

    # Check if GPU is available
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        logger.info("Running on GPU")
    else:
        logger.warning("GPU is not available, running on CPU")

    # epoch numbers
    n_epochs1 = 15000
    n_epochs2 = 1000
    n_loop = 50

    # number of collacation points
    n_collocation = nf
    # number of samples
    n_sampling = 100000

    solve(
        path_output=path_output,
        path_debug=path_debug,
        gamma1=gamma1,
        gamma2=gamma2,
        k=2.0,
        c=0.0,
        n_epochs1=n_epochs1,
        n_epochs2=n_epochs2,
        n_loop=n_loop,
        n_collocation=n_collocation,
        n_sampling=n_sampling,
        lr=0.001,
    )
