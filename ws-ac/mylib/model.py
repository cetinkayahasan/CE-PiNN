import os
import time
import json
import numpy as np
from deepxde.callbacks import Callback
import deepxde as dde

from mylib import env, disp

logger = env.logger


class EvaluateModelCallback(Callback):

    def __init__(
        self,
        X_test,
        y_test,
        pde,
        period=1000,
        dir_exp=None,
        save_models=False,
    ):
        super().__init__()

        self.X_test = X_test
        self.y_test = y_test
        self.pde = pde
        self.period = period
        self.save_models = save_models

        # experiment directory
        self.dir_exp = dir_exp
        self.dir_models = os.path.join(self.dir_exp, "models")
        os.makedirs(self.dir_models, exist_ok=True)
        self.path_model = os.path.join(self.dir_models, "model")

        self.dir_plots = os.path.join(self.dir_exp, "plots")
        os.makedirs(self.dir_plots, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "l2_error": [],
            "mean_abs_error": [],
            "max_abs_error": [],
            "mean_residual": [],
            "max_residual": [],
            "epoch": [],
            "time": [],
        }

        # keep last epoch start time
        self.last_epoch_start_time = None
        # a flag for evaluation at the beginning
        self._first_eval = False

    def on_train_begin(self):
        if not self._first_eval:
            self._first_eval = True
            self.evaluate(0)

    # def on_train_end(self):
    #     self.last_epoch_start_time = time.perf_counter()

    def on_epoch_begin(self):
        # start the timer if it is the first epoch or the previous period is reached
        if self.model.train_state.epoch % self.period == 1:
            self.last_epoch_start_time = time.perf_counter()

    def on_epoch_end(self):
        # skip if period is not reached
        if self.model.train_state.epoch % self.period != 0:
            return

        # calculate the elapsed time
        elapsed_time = time.perf_counter() - self.last_epoch_start_time

        # test the model
        self.evaluate(elapsed_time)

        # save the model
        if self.save_models:
            self.model.save(self.path_model)

    def save_results(self):

        try:
            logger.info("Saving results...")

            # save the model
            self.model.save(self.path_model)

            # save the history
            with open(os.path.join(self.dir_exp, "history.json"), "w") as f:
                json.dump(self.history, f, indent=4)

            # get model's predictions on the test data
            y_pred = self.model.predict(self.X_test)
            # get the model's pde residuals
            residuals = self.model.predict(self.X_test, operator=self.pde)
            # now, save the X_test, y_test, y_pred, residuals
            np.savez(
                os.path.join(self.dir_exp, "final_state.npz"),
                X_test=self.X_test,
                y_test=self.y_test,
                y_pred=y_pred,
                residuals=residuals,
            )

            # plot and save the results
            disp.plot_metric(self.history, "loss", plot_dir=self.dir_plots)
            disp.plot_metric(self.history, "l2_error", plot_dir=self.dir_plots)
            disp.plot_metric(self.history, "mean_abs_error", plot_dir=self.dir_plots)
            disp.plot_metric(self.history, "mean_abs_error", plot_dir=self.dir_plots)
            disp.plot_metric(self.history, "max_abs_error", plot_dir=self.dir_plots)
            disp.plot_metric(self.history, "mean_residual", plot_dir=self.dir_plots)
            disp.plot_metric(self.history, "max_residual", plot_dir=self.dir_plots)

            disp.plot_model_results(
                self.X_test,
                self.y_test,
                y_pred,
                residuals,
                self.dir_plots,
                heatmap=True,
                slices=True,
            )

        except Exception as e:
            logger.exception(f"Error plotting history: {e}")

    def evaluate(self, elapsed_time):

        # get the loss history
        train_state = self.model.train_state
        # get epoch
        epoch = self.model.train_state.epoch

        # check if train or test losses have more than one element
        if len(train_state.loss_train) > 1 or len(train_state.loss_test) > 1:
            logger.warning(
                "More than one element in train or test losses. Taking the mean..."
            )
            train_loss = np.mean(train_state.loss_train)
            val_loss = np.mean(train_state.loss_test)
        else:
            train_loss = train_state.loss_train[0]
            val_loss = train_state.loss_test[0]

        # get the prediction
        y_pred = self.model.predict(self.X_test)
        # calculate l2 error
        l2_error = dde.metrics.l2_relative_error(self.y_test, y_pred)
        # calculate mean absolute error
        mean_abs_error = np.mean(np.absolute(self.y_test - y_pred))
        # calculate max absolute error
        max_abs_error = np.max(np.absolute(self.y_test - y_pred))

        # calculate mean residual
        f = self.model.predict(self.X_test, operator=self.pde)
        # calculate mean residual error
        mean_residual = np.mean(np.absolute(f))
        # calculate max residual error
        max_residual = np.max(np.absolute(f))

        # keep epoch and elapsed time in the results dictionary
        self.history["epoch"].append(epoch)
        self.history["time"].append(elapsed_time)

        # fill the results dictionary
        self.history["l2_error"].append(l2_error)
        self.history["mean_abs_error"].append(mean_abs_error)
        self.history["max_abs_error"].append(max_abs_error)
        self.history["mean_residual"].append(mean_residual)
        self.history["max_residual"].append(max_residual)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        logger.info("*" * 50)

        # iteration difference
        n_iter_diff = (
            epoch - self.history["epoch"][-2]
            if len(self.history["epoch"]) > 1
            else epoch
        )
        # total elapsed time
        total_time_sec = sum(self.history["time"])

        # print iteration number and elapsed time
        logger.info(f">>> Iteration             : {epoch} [+{n_iter_diff}]")
        logger.info(
            f">>> Elapsed Time          : {total_time_sec:.2f} [+{elapsed_time:.2f}] seconds"
        )

        # log the results
        logger.info(f">>> METRICS:")
        logger.info(f"  > Train Loss            : {train_loss}")
        logger.info(f"  > Validation Loss       : {val_loss}")
        logger.info(f"  > L2 Error              : {l2_error}")
        logger.info(f"  > Mean Absolute Error   : {mean_abs_error}")
        logger.info(f"  > Max Absolute Error    : {max_abs_error}")
        logger.info(f"  > Mean Residual         : {mean_residual}")
        logger.info(f"  > Max Residual          : {max_residual}")

        logger.info("*" * 50)
