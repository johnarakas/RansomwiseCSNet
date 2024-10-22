import os
import sys

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree

# local modules
sys.path.append('.')
from path_utils import data_path, model_path, results_path


if __name__ == "__main__":

    exp0_file = f"{results_path}/exp0_results.csv"

    df = pd.read_csv(exp0_file)

    year_cols = [f'{year}' for year in range(2015,2025)]
    metric_cols = ['accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg', 'balanced_accuracy']

    for model_name in df['model'].unique():
        print('\n')
        print(model_name)

        dft = df[df['model']==model_name].reset_index(drop=True)

        fig, ax = plt.subplots()

        for metric_name in metric_cols:
            metric_mean = dft[metric_name].mean()
            metric_std = dft[metric_name].std()
            print(f"KFold {metric_name}: {metric_mean:.4f} ({metric_std:.4f})")

        x_range = np.arange(len(year_cols))
        y_vals = dft[year_cols].mean()
        y_max = dft[year_cols].max()
        y_min = dft[year_cols].min()
        
        ax.bar(
            x_range, y_vals, 
            yerr=[y_vals-y_min, y_max-y_vals], 
            color='tab:gray',
            align='center', alpha=0.5, ecolor='black', capsize=5)

        ax.set_xticks(x_range)
        ax.set_xticklabels(year_cols, rotation=0)
        # ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'fig/misclassification_kfold_{model_name}.pdf', dpi=100)
        plt.draw()

    plt.show()
