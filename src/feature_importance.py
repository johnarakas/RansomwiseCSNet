""""
Script used to plot the rules learned by a decision tree / random forest model trained on Ransomwise.

Usage:
```
python plot_tree_rules.py [--args]
```

Arguments:
- `--model_save_file`: The path to the pickle file containing the model. Default is "models/tree_d3_gini.sav".
- `--descriptor_file`: The path to the JSON file containing the descriptor. Default is "data/descriptor.json".
- `--max_depth`: The maximum depth of the tree. Default is 3.
- `--plot_save_path`: The path to save the plot. Default is None.
"""

import json
import pickle
import os
# import sys

# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Local modules
# sys.path.append(".")
from path_utils import MODEL_SAVE_PATH, DESCRIPTOR_FILE

MODEL_SAVE_FILE = os.path.join(MODEL_SAVE_PATH, 'tree_d3_gini.pkl')
MAX_DEPTH = 3

def plot_feature_importances(model, feature_names: list, show=False):

    importances = model.feature_importances_

    plt.figure()
    x_range = np.arange(len(feature_names))
    plt.bar(x_range, height=importances)
    plt.xticks(x_range, feature_names, rotation=90)
    plt.tight_layout()
    plt.draw()

    if show:
        plt.show()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_file', type=str, default=MODEL_SAVE_FILE)
    parser.add_argument('--descriptor_file', type=str, default=DESCRIPTOR_FILE)
    parser.add_argument('--max_depth', type=int, default=MAX_DEPTH)
    parser.add_argument('--plot_save_path', type=str, default=None)
    args = parser.parse_args()

    model_save_file = args.model_save_file
    descriptor_file = args.descriptor_file

    with open(model_save_file, 'rb') as f:
        model = pickle.load(f)

    with open(descriptor_file, 'r') as f: 
        descriptor = json.load(f)

    feature_names = descriptor['feature_columns']

    plot_feature_importances(model, feature_names, show=True)