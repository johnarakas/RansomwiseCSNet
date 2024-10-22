import os
import sys

import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree

# local modules
sys.path.append('.')
from path_utils import data_path, model_path, results_path

if __name__ == "__main__":

    descriptor_path = os.path.join(data_path, "descriptor.json")
    with open(descriptor_path, 'r') as f: descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']


    with open(os.path.join(model_path, "models_cv.pkl"), 'rb') as f:
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
        # plt.savefig(f'fig/gini_importance_kfold_{model_name}.pdf', dpi=100)
        plt.draw()

    plt.show()