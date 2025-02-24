"""
Script used to plot the distribution of eurValue in Ransomwise.

Usage:
```
python -m plots.distribution_eurValue [--args]

Arguments:
- `--dataset_file`: The path to the CSV file containing the dataset. Default is specified in `config.ini`.
- `--descriptor_file`: The path to the JSON file containing the descriptor. Default is specified in `config.ini`.
```

"""

import json
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Local modules
sys.path.append('.')
from src.path_utils import DATASET_FILE, DESCRIPTOR_FILE

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default=DATASET_FILE)
    parser.add_argument("--descriptor_file", type=str, default=DESCRIPTOR_FILE)
    args = parser.parse_args()

    dataset_file = args.dataset_file
    descriptor_file = args.descriptor_file
        

    df = pd.read_csv(dataset_file)

    with open(descriptor_file, 'r') as f: 
        descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']

    df_good = df[~df[target_column]].reset_index(drop=True)
    df_bad = df[df[target_column]].reset_index(drop=True)
    
    feature = 'eurValue_mean'
    plt.figure()
    bins = np.linspace(np.percentile(df[feature], 5), np.percentile(df[feature], 95))
    plt.hist([df_good[feature], df_bad[feature]], bins=bins)
    plt.ylabel(feature)
    plt.legend(['benign', 'malicious'])
    plt.draw()

    plt.show()
