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
NU = 0.001
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

# ===================== Adaptive RAR helpers =====================
def _results_json_path(path_output_parent):
    return os.path.join(path_output_parent, "results_log_kdv_ce_rarg_adaptive.json")


def _load_prev_result(json_file, current_gamma1):
    """Return the entry with the largest gamma1 < current_gamma1, or None."""
    if not os.path.exists(json_file):
        return None
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        candidates = []
        for d in data:
            try:
                r = float(d.get("gamma1", -1e5))
                if r < float(current_gamma1):
                    candidates.append(d)
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates, key=lambda d: float(d["gamma1"]))
    except Exception as e:
        logger.warning(f"Could not load previous results from {json_file}: {e}")
        return None

def _estimate_adaptive_loops2(
    prev_entry,
    pre_curr,
    default_loops=50,
    min_loops=10,
    max_loops=100,
    growth_alpha=1.0,  # exponent for proportional scaling
):
    """
    Decide RAR loop count for the current gamma1 using only pre-RAR residuals.

    Strategy:
    - If previous run exists and has valid pre-RAR residual and loop count:
      L = ceil(prev_loops * (pre_curr / prev_pre)^alpha)
      then clip to [min_loops, max_loops].
    - Otherwise, use default_loops.
    """
    try:
        if prev_entry is None:
            logger.info("No previous entry found; using default loops.")
            return default_loops

        prev_pre = float(prev_entry.get("pre_rar_mean_residual", np.nan))
        prev_loops = int(prev_entry.get("rar_loops", default_loops))

        if not np.isfinite(prev_pre) or prev_pre <= 0:
            logger.info("Previous pre-RAR residual not valid; using default loops.")
            return default_loops

        if not np.isfinite(pre_curr) or pre_curr <= 0:
            logger.info("Current pre-RAR residual not valid; using default loops.")
            return default_loops

        if prev_loops <= 0:
            logger.info("Previous loop count not valid; using default loops.")
            prev_loops = default_loops

        ratio = float(pre_curr) / prev_pre
        if not np.isfinite(ratio) or ratio <= 1.0:
            logger.info("Computed invalid ratio for pre_curr/prev_pre; using default loops.")
            return default_loops

        L_prop = int(np.ceil(prev_loops * (ratio ** growth_alpha)))
        L = int(np.clip(L_prop, min_loops, max_loops))

        logger.info(
            f"Adaptive loops (pre-only): prev_loops={prev_loops}, "
            f"pre_curr={pre_curr:.6e}, prev_pre={prev_pre:.6e}, "
            f"ratio={ratio:.4f}, growth_alpha={growth_alpha} -> n_loop={L}"
        )
        return L
    except Exception as e:
        logger.warning(f"Adaptive loop estimation failed; using default loops. Error: {e}")
        return default_loops


def _estimate_adaptive_loops(
    prev_entry,
    pre_curr,
    default_loops=50,
    min_loops=10,
    max_loops=100,
    growth_alpha=1.0,  # exponent for proportional scaling
):
    """
    Decide RAR loop count for the current gamma1.

    Strategy:
    1) If previous run has a valid contraction r < 1 (prev_post < prev_pre), compute loops L_contr
       to reach prev_post from pre_curr.
    2) Compute proportional loops L_prop = ceil(prev_loops * (pre_curr / prev_pre)^alpha).
       This guarantees an increase if pre_curr > prev_pre.
    3) Use max(L_contr, L_prop) if both valid; otherwise use whichever is valid.
    4) Clip to [min_loops, max_loops].
    """
    if prev_entry is None:
        logger.info("No previous entry found; using default loops.")
        return default_loops

    try:
        prev_pre = float(prev_entry.get("pre_rar_mean_residual", np.nan))
        prev_loops = int(prev_entry.get("rar_loops", default_loops))
        prev_post = prev_entry.get("post_rar_mean_residual", None)
        prev_post = None if prev_post is None else float(prev_post)

        if not np.isfinite(prev_pre) or prev_pre <= 0:
            logger.info("Previous pre-RAR residual not valid; using default loops.")
            return default_loops

        # Proportional scaling rule (always available given prev_pre and prev_loops)
        ratio = pre_curr / prev_pre
        L_prop = int(np.ceil(prev_loops * (ratio ** growth_alpha)))
        L_prop = int(np.clip(L_prop, min_loops, max_loops))
        logger.info(f"Proportional loop estimate: prev_loops={prev_loops}, "
                    f"pre_curr/prev_pre={ratio:.4f} -> L_prop={L_prop}")

        # Contraction-based rule (only if prev_post is valid and implies contraction)
        L_contr = None
        if prev_post is not None and np.isfinite(prev_post) and prev_post > 0 and prev_post < prev_pre and prev_loops > 0:
            r = (prev_post / prev_pre) ** (1.0 / prev_loops)
            logger.info(f"Estimated per-loop contraction r from previous gamma1: r={r:.6f}")
            if r < 1.0:
                if pre_curr <= prev_post:
                    L_contr = max(min_loops // 2, 1)
                else:
                    L_contr = int(np.ceil(np.log(prev_post / pre_curr) / np.log(r)))
                    L_contr = max(L_contr, min_loops)
                L_contr = int(np.clip(L_contr, min_loops, max_loops))
                logger.info(f"Contraction-based loop estimate L_contr={L_contr}")
            else:
                logger.warning("Estimated contraction r >= 1; ignoring contraction rule.")
        else:
            if prev_post is None:
                logger.info("No previous post-RAR residual found; skipping contraction rule.")
            elif not (prev_post < prev_pre):
                logger.info("Previous post-RAR residual is not smaller than pre; skipping contraction rule.")

        # Choose final L
        if L_contr is not None:
            L = max(L_prop, L_contr)  # conservative: never smaller than proportional
        else:
            L = L_prop

        L = int(np.clip(L, min_loops, max_loops))
        return L

    except Exception as e:
        logger.warning(f"Adaptive loop estimation failed: {e}")
        return default_loops
# ===============================================================


def solve(
    path_output,
    path_debug,
    path_pre_model,
    gamma1,
    gamma2,
    n_epochs1=10000,
    n_epochs2=1000,
    n_loop=90,                 # default/fallback loops (used for the first gamma1 or if no prev info)
    n_collocation=2000,
    n_sampling=100000,
    lr=0.001,
    test_period=1000,
    adaptive_rar=True,
    min_loops=10,
    max_loops=100,
    n_residual_eval=100000,    # how many random points to estimate mean residual
    predict_batch=None,        # batch size for model.predict during residual evaluation
):
    logger.info(f"::: Pre-trained model: {path_pre_model}")
    logger.info(f"::: gamma1={gamma1} and gamma2={gamma2} :::")
    logger.info(f"::: n_collacation={n_collocation}, n_sampling={n_sampling}")
    logger.info(f"::: n_epochs1={n_epochs1}, n_epochs2={n_epochs2}, baseline n_loop={n_loop}")
    logger.info(f"::: learning rate={lr}")

    pde = lambda x, y: pde_KdV(x, y, gamma1, gamma2)

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [],
        num_domain=n_collocation,
        num_test=10000,
        train_distribution="pseudo",
    )

    net = dde.maps.FNN([5] + [100] * 5 + [1], "tanh", "Glorot normal")
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    X_test, y_test = dt.get_test_data(gamma1, gamma2)
    eval_callback = md.EvaluateModelCallback(
        X_test,
        y_test,
        pde,
        period=test_period,
        dir_exp=path_output,
        save_models=False,
    )

    checkpoint = torch.load(path_pre_model)
    model.net.load_state_dict(checkpoint["model_state_dict"])
    logger.warning("!!! Model weights restored !!!")
    logger.warning("!!! Model restored !!!")

    logger.info("::: First training phase with ADAM optimizer...")
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

    last_epoch_time = eval_callback.history["epoch"][-1]
    eval_callback.evaluate(last_epoch_time)

    # ---------- Pre-RAR mean PDE residual ----------
    X_eval_resid = geomtime.random_points(n_residual_eval)
    if predict_batch is None:
        resid_eval = np.abs(model.predict(X_eval_resid, operator=pde))[:, 0]
    else:
        resid_eval = np.abs(model.predict(X_eval_resid, operator=pde, batch_size=predict_batch))[:, 0]
    pre_rar_mean_residual = float(np.mean(resid_eval))
    logger.info(f"Pre-RAR mean PDE residual (first-column value) = {pre_rar_mean_residual:.8f}")

    # ---------- Adaptive loop count ----------
    parent_dir = os.path.dirname(path_output)
    json_file = _results_json_path(parent_dir)
    n_loop_adapted = n_loop
    if adaptive_rar:
        prev_entry = _load_prev_result(json_file, gamma1)
        n_loop_adapted = _estimate_adaptive_loops(
            prev_entry=prev_entry,
            pre_curr=pre_rar_mean_residual,
            default_loops=n_loop,
            min_loops=min_loops,
            max_loops=max_loops,
        )
        logger.info(f"Adaptive RAR: using n_loop={n_loop_adapted} for gamma1={gamma1}")

    # ---------- RAR phase ----------
    logger.info("::: Starting RAR training phase...")
    dde.optimizers.config.set_LBFGS_options(maxiter=n_epochs2)
    for ii in range(n_loop_adapted):
        X = geomtime.random_points(n_sampling)
        if predict_batch is None:
            Y = np.abs(model.predict(X, operator=pde))[:, 0]
        else:
            Y = np.abs(model.predict(X, operator=pde, batch_size=predict_batch))[:, 0]
        err_eq = torch.tensor(Y)

        num_new_samples = int(n_collocation * RATIO)
        X_ids = torch.topk(err_eq, num_new_samples, dim=0)[1].cpu().numpy()
        data.add_anchors(X[X_ids])

        losshistory, train_state = model.train(
            epochs=n_epochs2,
            callbacks=[eval_callback],
        )
        model.compile("L-BFGS")
        losshistory, train_state = model.train(
            epochs=n_epochs2,
            callbacks=[eval_callback],
        )
        last_epoch_time = eval_callback.history["epoch"][-1]
        eval_callback.evaluate(last_epoch_time)

    # ---------- Post-RAR mean PDE residual ----------
    X_eval_resid = geomtime.random_points(n_residual_eval)
    if predict_batch is None:
        resid_eval = np.abs(model.predict(X_eval_resid, operator=pde))[:, 0]
    else:
        resid_eval = np.abs(model.predict(X_eval_resid, operator=pde, batch_size=predict_batch))[:, 0]
    post_rar_mean_residual = float(np.mean(resid_eval))
    logger.info(f"Post-RAR mean PDE residual = {post_rar_mean_residual:.8f}")

    eval_callback.save_results()

    last_epoch = eval_callback.history["epoch"][-1]
    final_loss = eval_callback.history["train_loss"][-1]
    final_val_loss = eval_callback.history["val_loss"][-1]
    final_l2_error = eval_callback.history["l2_error"][-1]

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
        "rar_epochs": n_epochs2,
        "rar_loops": int(n_loop_adapted),
        "rar_n_sampling": n_sampling,
        "rar_ratio": RATIO,
        "pre_rar_mean_residual": float(pre_rar_mean_residual),
        "post_rar_mean_residual": float(post_rar_mean_residual),
    }

    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data_log = json.load(f)
        except Exception:
            data_log = []
    else:
        data_log = []

    data_log.append(results)
    with open(json_file, "w") as f:
        json.dump(data_log, f, indent=4)
    logger.info(f"Results saved to {json_file}")
    logger.info("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma2", type=float, default=0.0025)
    parser.add_argument("--gamma1", type=float)
    parser.add_argument("--nf", type=int, default=5000)
    parser.add_argument("--path_pre_model", type=str, required=True)
    parser.add_argument("--adaptive_rar", action="store_true", help="Enable adaptive RAR loops")
    parser.add_argument("--min_loops", type=int, default=10)
    parser.add_argument("--max_loops", type=int, default=100)
    parser.add_argument("--n_residual_eval", type=int, default=100000)
    parser.add_argument("--predict_batch", type=int, default=None, help="Batch size for predict during residual eval")
    args = parser.parse_args()

    gamma1 = args.gamma1
    gamma2 = args.gamma2
    nf = args.nf
    path_pre_model = args.path_pre_model
    print(f"::: Using pre-trained model: {path_pre_model}")
    if not os.path.exists(path_pre_model):
        raise FileNotFoundError(f"Pre-trained model not found: {path_pre_model}")

    algo = "ce_rarg_adaptive"
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
    n_epochs2 = 1000
    n_loop_default = 50
    n_collocation = nf
    n_sampling = 100000

    solve(
        path_output=path_output,
        path_debug=path_debug,
        path_pre_model=path_pre_model,
        gamma1=gamma1,
        gamma2=gamma2,
        n_epochs1=n_epochs1,
        n_epochs2=n_epochs2,
        n_loop=n_loop_default,
        n_collocation=n_collocation,
        n_sampling=n_sampling,
        lr=0.001,
        adaptive_rar=True,
        min_loops=args.min_loops,
        max_loops=args.max_loops,
        n_residual_eval=args.n_residual_eval,
        predict_batch=args.predict_batch,
    )