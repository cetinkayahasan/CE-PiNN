"""
Automating CE-RARG algorithm execution.
"""

import os
import subprocess
import argparse
import numpy as np

from mylib import env

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nf", type=int, default=5000)

    args = parser.parse_args()
    nf = args.nf

    algo = "ce_rarg2"
    pyfile = os.path.join(env.DIR_BASE, f"{algo}.py")

   # for each different rho value
    rho_values = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for rho in rho_values[1:]:
        if rho == 6.0:
            # get previous rho value in the list
            prev_rho = rho_values[rho_values.index(rho) - 1]
            path_pre_exp = os.path.join(
                #env.DIR_EXPERIMENTS, f"ce_vanilla-r{prev_rho}-nf{nf}", "models"
                env.DIR_EXPERIMENTS, f"ce_rarg-r{prev_rho}-nf{nf}", "models"
            )

        else:
            # get the previous rho value
            prev_rho = rho_values[rho_values.index(rho) - 1]
            path_pre_exp = os.path.join(
                # env.DIR_EXPERIMENTS, f"{algo}-r{prev_rho}-nf{nf}", "models"
                env.DIR_EXPERIMENTS, f"ce_rarg-r{prev_rho}-nf{nf}", "models"
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
                "--rho",
                str(rho),
                "--nf",
                str(nf),
                "--path_pre_model",
                path_pre_model,
            ]
        )
