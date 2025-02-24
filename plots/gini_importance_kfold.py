"""
Script used to plot the importance of features in tree-based models trained on Ransomwise with k-fold cross-validation.

Usage:
```
python -m plots.gini_importance_kfold [--models_file] [--descriptor_file DESCRIPTOR_FILE]
```

Arguments:
- `--models_file`: The path to the directory containing the models. Default is `{MODEL_SAVE_PATH}/models_cv.pkl`.
- `--descriptor_file`: The path to the JSON file containing the descriptor. Default is specified in `config.ini`.
"""

import os
# import sys

import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

# local modules
# sys.path.append('.')
from src.path_utils import DESCRIPTOR_FILE, MODEL_SAVE_PATH, FIGURES_PATH

MODELS_FILE = os.path.join(MODEL_SAVE_PATH, "models_cv.pkl")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_file', type=str, default=MODELS_FILE)
    parser.add_argument('--descriptor_file', type=str, default=DESCRIPTOR_FILE)
    args = parser.parse_args()

    with open(args.descriptor_file, 'r') as f: 
        descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']


    with open(args.models_file, 'rb') as f:
        saved_models = pickle.load(f)

    for model in saved_models['Decision tree (d=3)']:
        plt.figure()
        plot_tree(model, max_depth=3, feature_names=feature_names, fontsize=6, class_names=['benign', 'malicious'])
        plt.draw()

    for model_name in saved_models.keys():
        importances = [model.feature_importances_ for model in saved_models[model_name]]
        importances = np.asarray(importances)
        x_range = np.arange(importances.shape[-1])
        y_vals = importances.mean(axis=0)
        
        fig, ax = plt.subplots()

        ax.bar(
            x_range, y_vals, 
            yerr=[y_vals-importances.min(axis=0), importances.max(axis=0)-y_vals], 
            align='center', alpha=0.5, ecolor='black', capsize=5)

        ax.set_xticks(x_range)
        ax.set_xticklabels(feature_names, rotation=90)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_PATH}/gini_importance_kfold_{model_name}.pdf', dpi=100)
        plt.draw()

    plt.show()