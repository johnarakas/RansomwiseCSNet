import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.path_utils import RESULTS_PATH, FIGURES_PATH

RESULTS_FILE = os.path.join(RESULTS_PATH, 'leave_one_source_out.csv') 

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default=RESULTS_FILE)
    parser.add_argument('--figure_save_path', type=str, default=FIGURES_PATH)
    args = parser.parse_args()

    results_file = args.results_file
    figure_save_path = args.figure_save_path

    df = pd.read_csv(results_file)

    n_models = len(df['model'].unique())
    n_sources = len(df['source'].unique())
    x_range = np.arange(n_sources)

    plt.figure(figsize=(8,3))

    shifts = np.linspace(-0.2, 0.2, n_models)
    for i, model in enumerate(df['model'].unique()):
        df_model = df[df['model'] == model]

        plt.bar(x_range+shifts[i], df_model['recall'], width=0.4/(n_models-1), label=model, alpha=0.5, edgecolor='black')

    plt.xticks(x_range, df['source'].unique())
    plt.legend()
    plt.tight_layout()

    outfile = os.path.join(figure_save_path, 'leave_one_source_out.pdf')
    plt.savefig(outfile)
    plt.show()