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

    algo = "rarg" # vanilla, rarg, etc.
    # for each different nu value
    pyfile = os.path.join(env.DIR_BASE, f"{algo}.py")
    # nu list in descending order   
    nu_list = [0.009, 0.008, 0.007, 0.006, 0.005, 0.003]
    for nu in nu_list[:]:
        nu = format_float(nu)
        cmds = [
            "python",
            pyfile,
            "--nu",
            nu,
            "--nf",
            str(nf),
        ]

        subprocess.run(cmds)


