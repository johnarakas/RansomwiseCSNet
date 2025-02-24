# TODO: Refactor

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metric_cols = ['accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg', 'balanced_accuracy']

# Local modules
sys.path.append('.')
from src.path_utils import RESULTS_PATH, FIGURES_PATH

CV_RESULT_FILE = os.path.join(RESULTS_PATH, 'results_cv_d3.csv')

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default=RESULTS_PATH)
    parser.add_argument("--cv_result_file", type=str, default=CV_RESULT_FILE)
    args = parser.parse_args()

    results_path = args.results_path
    cv_result_file = args.cv_result_file

    df = pd.read_csv(cv_result_file)

    year_cols = [col for col in df.columns if col.startswith('n_wrong_')]
    years = [int(col.removeprefix('n_wrong_')) for col in year_cols]
    

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
            align='center', alpha=0.5, ecolor='black', capsize=5
        )

        ax.set_xticks(x_range)
        ax.set_xticklabels(years, rotation=0)
        # ax.set_ylim([0, 1])
        plt.tight_layout()
        outfile = os.path.join(FIGURES_PATH, f'misclassification_by_year_{model_name}.pdf')
        plt.savefig(outfile, dpi=100)
        plt.draw()

    plt.show()
