import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# local modules
sys.path.append('.')
from path_utils import data_path, model_path, results_path


if __name__ == "__main__":
    

    df = pd.read_csv(os.path.join(data_path, 'aggregated.csv'))


    df_good = df[~df['blackList']].reset_index(drop=True)
    df_bad = df[df['blackList']].reset_index(drop=True)
    
    feature = 'eurValue_mean'
    # feature = 'btcValue_mean'
    # feature = 'transaction_count'
    plt.figure()
    # plt.boxplot([df_good[feature], df_bad[feature]], showfliers=False)
    bins = np.linspace(np.percentile(df[feature], 5), np.percentile(df[feature], 95))
    plt.hist([df_good[feature], df_bad[feature]], bins=bins)
    # plt.xticks([1, 2], ['benign', 'ransomware'])
    plt.ylabel(feature)
    plt.legend(['benign', 'malicious'])
    plt.draw()

    plt.show()
