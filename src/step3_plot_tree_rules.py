import json
import os
import pickle
import sys

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

sys.path.append(".")
from path_utils import model_path, data_path

if __name__ == "__main__":

    with open(os.path.join(model_path, 'tree_d3_gini.sav'), 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('data/aggregated.csv', nrows=10)
    descriptor_path = os.path.join(data_path, "descriptor.json")
    with open(descriptor_path, 'r') as f: descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']

    plt.figure(figsize=(12,6))
    plot_tree(model, max_depth=3, feature_names=feature_names, fontsize=6, class_names=['benign', 'malicious'])
    plt.draw()

    plt.show()