"""
CE-RAD Algorithm - Cirruculum Learning + Residual Based Adaptive Refinement with Distributional Sampling
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
LAMBDA = 0.001
RATIO = 0.005


def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return t_in * (1 + x_in) * (1 - x_in) * y + torch.square(x_in) * torch.cos(
        torch.pi * x_in
    )


def pde_AC(x, y, rho):
    u = y
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_t = dde.grad.jacobian(y, x, j=1)
    return du_t - LAMBDA * du_xx + rho * (u**3 - u)


def solve(
    path_output,
    path_debug,
    path_pre_model,
    rho,
    k=1.0,
    c=1.0,
    n_epochs1=10000,
    n_epochs2=1000,
    n_loop=90,
    n_collocation=2000,
    n_sampling=100000,
    lr=0.001,
    test_period=1000,
):
    # print log
    logger.info(f"::: Pre-trained model: {path_pre_model}")
    logger.info(f"::: rho={rho} and lambda={LAMBDA} :::")
    logger.info(f"::: k={k}, c={c}")
    logger.info(f"::: n_collacation={n_collocation}, n_sampling={n_sampling}")
    logger.info(f"::: n_epochs1={n_epochs1}, n_epochs2={n_epochs2}, n_loop={n_loop}")
    logger.info(f"::: learning rate={lr}")

    # create the pde function
    pde = lambda x, y: pde_AC(x, y, rho)

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
    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    # get the test data
    X_test, y_test = dt.get_test_data(rho)
    # create the callback
    eval_callback = md.EvaluateModelCallback(
        X_test,
        y_test,
        pde,
        period=test_period,
        dir_exp=path_output,
        save_models=False,
    )

    # Restore only the model weights (not the optimizer state)
    checkpoint = torch.load(path_pre_model)
    model.net.load_state_dict(checkpoint["model_state_dict"])
    logger.warning("!!! Model weights restored !!!")

    # model.compile("adam", lr=lr)
    logger.warning("!!! Model restored !!!")

   
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


    logger.info("::: Second training  phase...")
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

    logger.info("Done!")


# Run the algorithm
if __name__ == "__main__":

    # add an argument parser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=9)
    parser.add_argument("--nf", type=int, default=5000)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--path_pre_model", type=str, required=True)

    args = parser.parse_args()
    rho = args.rho
    nf = args.nf
    k = float(args.k)
    c = float(args.c)
    path_pre_model = args.path_pre_model

    # check if the path exists
    # if not, raise an error
    if not os.path.exists(path_pre_model):
        raise FileNotFoundError(f"Pre-trained model not found: {path_pre_model}")

    # algorithm is simply the file name
    algo = "ce_rard"
    exp_folder_name = f"{algo}-r{rho}-nf{nf}"

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
        path_pre_model=path_pre_model,
        rho=rho,
        k=k,
        c=c,
        n_epochs1=n_epochs1,
        n_epochs2=n_epochs2,
        n_loop=n_loop,
        n_collocation=n_collocation,
        n_sampling=n_sampling,
        lr=0.001,
    )
