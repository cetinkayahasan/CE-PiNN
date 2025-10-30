"""
Automating CE-RARG algorithm execution.
"""

import os
import subprocess
import argparse
import numpy as np

from mylib import env

def format_float(value):
    if value == int(value):
        return str(int(value))
    else:
        return str(value)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nf", type=int, default=5000)

    args = parser.parse_args()
    nf = args.nf

    algo = "ce_rard"
    pyfile = os.path.join(env.DIR_BASE, f"{algo}.py")

    gamma2= 0.0025
    # for each different rho value
    gamma1_values = [0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 1.9, 2.0]
    
    for gamma1 in gamma1_values[1:]:
        if gamma1 == 0.7:
            # get previous rho value in the list
            prev_gamma1 = gamma1_values[gamma1_values.index(gamma1) - 1]
            path_pre_exp = os.path.join(
                env.DIR_EXPERIMENTS, f"ce_vanilla-gamma1_{prev_gamma1}-gamma2_{gamma2}-nf{nf}", "models"
            )

        else:
            # get the previous rho value
            prev_gamma1 = gamma1_values[gamma1_values.index(gamma1) - 1]
            path_pre_exp = os.path.join(
                 env.DIR_EXPERIMENTS, f"{algo}-gamma1_{prev_gamma1}-gamma2_{gamma2}-nf{nf}", "models"
            )   
           

        # get all model files in the directory
        model_files = [f for f in os.listdir(path_pre_exp) if f.endswith(".pt")]
        # split file name and iteration number
        model_files = [
            (fname, int(fname.split("-")[-1].split(".")[0])) for fname in model_files
        ]
        # sort model files based on the iteration number
        model_files = sorted(model_files, key=lambda x: x[1])
        # get the model with the highest iteration number
        model_file = model_files[-1][0]
        # get the model file path
        path_pre_model = os.path.join(path_pre_exp, model_file)

        # call the python file with the pre-trained model
        subprocess.run(
            [
                "python",
                f"{algo}.py",
                "--gamma1",
                format_float(gamma1),
                "--nf",
                str(nf),
                "--path_pre_model",
                path_pre_model,
            ]
        )