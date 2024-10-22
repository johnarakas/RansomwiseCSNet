import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from path_utils import data_path, model_path, results_path

if __name__ == "__main__":

    df = pd.read_csv(os.path.join(data_path, 'aggregated.csv'))

    wallet_name = df['walletOfInterest'].to_numpy()

    feature_names = list(df.columns[1:-1])
    X = df[feature_names]#.to_numpy()
    y = 1*df[df.columns[-1]]#.to_numpy()

    x_range = np.arange(len(X))

    features_to_plot = ['usdValue_rc_mean', 'usdValue_sn_mean']
    # features_to_plot = feature_names  # all features
    # for feature in features_to_plot:
    #     x = X[feature]
    #     plt.figure()
    #     # plt.scatter(x_range, X['usdValue_sn_mean'], c=y, marker='.')
    #     plt.plot(x_range[y == 1], x[y == 1], '^', label='malicious')
    #     plt.plot(x_range[y == 0], x[y == 0], '.', label='benign')
    #     # plt.yscale('log')
    #     plt.ylim([0, max(x)/1_000.])  # for better visualization
    #     plt.title(feature)
    #     plt.legend()
    #     plt.draw()

    plt.figure()
    plt.plot(df.loc[y==0, 'usdValue_rc_mean'], df.loc[y==0, 'usdValue_sn_mean'], 
             '.', color='C0', label='benign')
    plt.plot(df.loc[y==1, 'usdValue_rc_mean'], df.loc[y==1, 'usdValue_sn_mean'], '.', 
             color='C1', label='bad')
    plt.xlim([0, max(df['usdValue_rc_mean'])/1_000.])  # for better visualization
    plt.ylim([0, max(df['usdValue_sn_mean'])/1_000.])  # for better visualization
    plt.legend()
    plt.draw()

    plt.show()