"""
Script used to plot the rules learned by a decision tree model trained on Ransomwise.

Usage:
```
python -m plots.tree_rules [--model_save_file MODEL_SAVE_FILE] [--descriptor_file DESCRIPTOR_FILE] [--max_depth MAX_DEPTH] [--plot_save_path PLOT_SAVE_PATH] [--show_plot SHOW_PLOT]
```

Arguments:
- `--model_save_file`: The path to the pickle file containing the model. Default is "models/tree_d3_gini.sav".
- `--descriptor_file`: The path to the JSON file containing the descriptor. Default is "data/descriptor.json".
- `--max_depth`: The maximum depth of the tree. Default is 3.
- `--plot_save_path`: The path to save the plot. Default is None.

Figure index:
- In CSNet24: Fig. 2
"""

import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# sys.path.append(".")
from src.path_utils import MODEL_SAVE_PATH, DESCRIPTOR_FILE

MODEL_SAVE_FILE = os.path.join(MODEL_SAVE_PATH, 'tree_d3_gini.pkl')
MAX_DEPTH = 3

def plot_tree_rules(model, descriptor: dict, max_depth: int=3, plot_save_path: str=None, show_plot: bool=False):

    feature_names = descriptor['feature_columns']
    
    plt.figure(figsize=(12,6))
    plot_tree(model, max_depth=max_depth, feature_names=feature_names, fontsize=6, class_names=['benign', 'malicious'])
    plt.draw()

    if plot_save_path is not None:
        plt.savefig(plot_save_path, dpi=100)  # NOTE: DPI are used only for pixel-based formats

    if show_plot:
        plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_file', type=str, default=MODEL_SAVE_FILE)
    parser.add_argument('--descriptor_file', type=str, default=DESCRIPTOR_FILE)
    parser.add_argument('--max_depth', type=int, default=MAX_DEPTH)
    parser.add_argument('--plot_save_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.model_save_file, 'rb') as f:
        model = pickle.load(f)

    with open(args.descriptor_file, 'r') as f: descriptor = json.load(f)

    plot_tree_rules(model, descriptor, max_depth=args.max_depth, show_plot=True)