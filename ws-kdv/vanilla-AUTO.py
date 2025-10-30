"""
Automating Vanilla algorithm execution with different parameters.
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

    algo = "vanilla"
    # for each different rho value
    pyfile = os.path.join(env.DIR_BASE, f"{algo}.py")

    # gamma1_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
    gamma1_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 1.9, 2.0, 2.5]
    
    for gamma1 in gamma1_values[10:]:
       
        gamma1 = format_float(gamma1)
        cmds = [
            "python",
            pyfile,
            "--gamma1",
            gamma1,
            "--nf",
            str(nf),
        ]

        subprocess.run(cmds)