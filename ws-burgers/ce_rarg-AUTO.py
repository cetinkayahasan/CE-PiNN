"""
Automating CE-VANILLA algorithm execution with different parameters.
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
   
    algo = "ce_rarg4"
    pyfile = os.path.join(env.DIR_BASE, f"{algo}.py")

   
    nu_list = [0.09, 0.07, 0.05, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003] 

    # for each different nu value
    for nu in nu_list[6:]:
        # get the previous nu value
        prev_nu = nu_list[nu_list.index(nu) - 1]
    
        if nu == 0.009:
            
            path_pre_exp = os.path.join(
                env.DIR_EXPERIMENTS, f"ce_vanilla-nu{prev_nu}-nf{nf}", "models"
            )

        else:
            
            path_pre_exp = os.path.join(
                env.DIR_EXPERIMENTS, f"{algo}-nu{prev_nu}-nf{nf}", "models"
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
                "--nu",
                str(nu),
                "--nf",
                str(nf),
                "--path_pre_model",
                path_pre_model,
            ]
        )