import pandas as pd
import argparse
import numpy as np

def main(args):
    baseline = pd.read_csv(args.baseline)
    target = pd.read_csv(args.target_csv)

    print('Average Improvement (%): ', (target['mflop'] / baseline['mflop']).mean() * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile average improvement')
    parser.add_argument('--baseline', type=str, help='baseline csv to compared to')
    parser.add_argument('--target_csv', type=str, help='method csv') 
    args = parser.parse_args()
    main(args)   