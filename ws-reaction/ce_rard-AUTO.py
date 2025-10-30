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

    algo = "ce_rard"
    pyfile = os.path.join(env.DIR_BASE, f"{algo}.py")

    # rho values to iterate over
    # rho_list = [2.0, 3.0, 5.0, 10.0, 15.0, 20.0]  
    rho_list =[15.0, 20.0, 25.0, 30.0, 40.0, 50.0] 
    
    # for each different nu value
    for rho in rho_list[2:]:
        # get the previous rho value
        prev_rho = rho_list[rho_list.index(rho) - 1]
        if rho == 20.0:
            prev_rho = rho_list[rho_list.index(rho) - 1]
            path_pre_exp = os.path.join(
                env.DIR_EXPERIMENTS, f"ce_vanilla-r{prev_rho}-nf{nf}", "models"
            )

        else:
            
            path_pre_exp = os.path.join(
                env.DIR_EXPERIMENTS, f"{algo}-r{prev_rho}-nf{nf}", "models"
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
