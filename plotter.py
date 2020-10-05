#!/share/apps/python/anaconda/bin/python


import sys
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

import os

def make_plot(args):
    "Plot results of timing trials"
    for arg in args.suffix:
        df = pd.read_csv(os.path.join(args.csv_dir, "timing-{0}.csv".format(arg)))
        plt.plot(df['size'], df['mflop']/1e3, label=arg)
    plt.xlabel('Dimension')
    plt.ylabel('Gflop/s')

def show(runs):
    "Show plot of timing runs (for interactive use)"
    make_plot(args)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def main(args):
    "Show plot of timing runs (non-interactive)"
    make_plot(args)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if not args.no_show:
        plt.show()
    plt.savefig(args.save_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='', help='directory that has the csv files')
    parser.add_argument('--suffix', type=str, nargs='+', help='prefix for the csv files')
    parser.add_argument('--save_file', type=str, default='timing.pdf',
                        help='filename for the plot')
    parser.add_argument('--no_show', action='store_true', help='whether to show the plot or not')
    args = parser.parse_args()
    main(args)
