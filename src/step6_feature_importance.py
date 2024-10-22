import os
import sys

import json
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

# local modules
sys.path.append('.')
from path_utils import data_path, model_path

if __name__ == "__main__":

    descriptor_path = os.path.join(data_path, "descriptor.json")
    with open(descriptor_path, 'r') as f: descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']

    with open(os.path.join(model_path, 'tree_d3_gini.sav'), 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('data/aggregated.csv', nrows=10)

    importances = model.feature_importances_

    plt.figure()
    x_range = np.arange(len(feature_names))
    # print(x_range)
    # exit()
    plt.bar(x_range, height=importances)
    plt.xticks(x_range, feature_names, rotation=90)
    plt.tight_layout()
    plt.draw()

    plt.show()