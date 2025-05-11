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
    rho_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for rho in rho_list:
        rho = format_float(rho)
        cmds = [
            "python",
            pyfile,
            "--rho",
            rho,
            "--nf",
            str(nf),
        ]

        subprocess.run(cmds)
    
