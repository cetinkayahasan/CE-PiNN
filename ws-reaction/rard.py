"""
RARD Algorithm - Residual Based Adaptive Refinement with Distributional Sampling
k=2.0, c=0

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
# ratio of new samples
RATIO = 0.005




def output_transform(x, y):
    x0 = torch.pi
    sigma = torch.pi / 4
    # return torch.exp(-torch.pow((x[:, 0:1] - x0) / sigma, 2.) / 2.) + x[:, 1:2] * x[:, 0:1]*(2*torch.pi - x[:, 0:1]) * y
    return torch.exp(-torch.pow((x[:, 0:1] - x0) / sigma, 2.) / 2.) + x[:, 1:2] * y


# def output_transform(x, y):
    
#     return 1 - torch.sin(x[:, 0:1]) + x[:, 1:2] * x[:, 0:1]*(2*torch.pi - x[:, 0:1]) * y



def pde_reaction(x, y, rho):
    """Expresses the PDE residual of the reaction equation."""
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t  - rho * (y - y**2)



def solve(
    path_output,
    path_debug,
    rho,
    k,
    c,
    n_epochs1=10000,
    n_epochs2=1000,
    n_loop=90,
    n_collocation=2000,
    n_sampling=100000,
    lr=0.001,
    test_period=1000,
):
    # print log
    logger.info(f"::: rho={rho} :::")
    logger.info(f"::: n_collacation={n_collocation}, n_sampling={n_sampling}")
    logger.info(f"::: n_epochs1={n_epochs1}, n_epochs2={n_epochs2}, n_loop={n_loop}")
    logger.info(f"::: learning rate={lr}")

    # create the pde function
    pde = lambda x, y: pde_reaction(x, y, rho)

    # create the geometry
    geom = dde.geometry.Interval(0, 2*np.pi)
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
    

    logger.info("::: First training phase...with ADAM optimizer....")
    # train model with calculating number of milliseconds passed
    losshistory, train_state = model.train(
        epochs=n_epochs1,
        callbacks=[eval_callback],
    )


    # continue training with LBFGS
    logger.info("::: Continue with L-BFGS training phase...")
    model.compile("L-BFGS")
    losshistory, train_state = model.train(
        epochs=n_epochs1,
        callbacks=[eval_callback],
    )

    logger.info("::: Second training phase... RARD algorithm")
    dde.optimizers.config.set_LBFGS_options( maxiter=n_epochs2)
    for ii in range(n_loop):
       
        X = geomtime.random_points(n_sampling)
        # get the prediction
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
         # calculate residuals
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]

        # add new samples top k high residual error points
        X_ids = np.random.choice(
            a=len(X), size=int(n_collocation*RATIO), replace=False, p=err_eq_normalized
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
        "rho": float(rho),
        "epochs": int(last_epoch),
        "n_collocation": n_collocation,
        "lr": lr,
        "algo": algo,
        "k": k,
        "c": c,
        "final_loss": float(final_loss),
        "val_loss": float(final_val_loss),
        "l2_error": float(final_l2_error),
        "rar_epochs": n_epochs2,
        "rar_loops": n_loop,
        "rar_n_sampling": n_sampling,
        "rar_ratio": RATIO,
    }
    # save to parent directory of path_output
    parent_dir = os.path.dirname(path_output)
    json_file = os.path.join(parent_dir, "results_log_reaction_rard.json")

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
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--nf", type=int, default=5000)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--c", type=int, default=0)
    

    args = parser.parse_args()
    rho = args.rho
    nf = args.nf
    k = args.k
    c = args.c
    # algorithm is simply the file name
    algo = "rard"
    exp_folder_name = f"{algo}-r{rho}-nf{nf}-k{k}-c{c}"

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

    # rard
    k = 2.0
    c = 0
    solve(
        path_output=path_output,
        path_debug=path_debug,
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