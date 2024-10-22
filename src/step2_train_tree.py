import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# local modules
sys.path.append('.')
from path_utils import data_path, model_path, results_path
from ml_utils import compute_binary_classification_metrics, temporal_train_test_split

"""
Hyperparameters:
- n_estimators: If 1, trains a single decision tree classifier. If >1 trains a random forest.
- max_depth: Maximum depth of each decision tree (more depth = more complex model).
- criterion: Split based on gini index or entropy.
For more info, see https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

hparams = {
    'n_estimators': 1,
    'max_depth': 3,
    'criterion': 'gini',
}

metrics = ['accuracy', 'precision', 'recall', 'specificity', 'balanced_accuracy']

# other training specifics
test_size = 0.3  # fraction of dataset used for test
save_results = True  # set to false when making changes

use_time_split = False

def check_time_split(ts_train, ts_test):
    print(f"Min ts train: {ts_train.min()}")
    print(f"Max ts train: {ts_train.max()}")
    print(f"Min ts test: {ts_test.min()}")
    print(f"Max ts test: {ts_test.max()}")

if __name__ == "__main__":
    
    np.random.seed(42)

    df = pd.read_csv(os.path.join(data_path, 'aggregated.csv'))

    wallet_name = df['walletOfInterest'].to_numpy()

    descriptor_path = os.path.join(data_path, "descriptor.json")
    with open(descriptor_path, 'r') as f: descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']
    
    if use_time_split:
        df_train, df_test = temporal_train_test_split(df, test_size=test_size, time_column=time_column)
        check_time_split(df_train[time_column], df_test[time_column])
    else:
        # random split
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)

    df_train[['walletOfInterest']].to_csv("data/wallet_train.csv", index=False)

    X_train = df_train[feature_names]
    y_train = df_train[target_column]
    X_test = df_test[feature_names]
    y_test = df_test[target_column]

    if hparams['n_estimators'] > 1:
        model = RandomForestClassifier(n_estimators=hparams['n_estimators'], 
                                       max_depth=hparams['max_depth'], 
                                       criterion=hparams['criterion'], 
                                       random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=hparams['max_depth'], 
                                       criterion=hparams['criterion'],
                                       random_state=42)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    n_bad_train = len(y_train[y_train==1])
    n_good_train = len(y_train[y_train==0])
    n_bad_test = len(y_test[y_test==1])
    n_good_test = len(y_test[y_test==0])

    to_print = [
        f"Number of blackListed wallets (train): {n_bad_train}",
        f"Number of non-blackListed wallets (train): {n_good_train}",
        f"Number of blackListed wallets (test): {n_bad_test}",
        f"Number of non-blackListed wallets (test): {n_good_test}",
        ""
        ]

    for s in to_print:
        print(s)
    
    _, to_print_train = compute_binary_classification_metrics(y_train, y_pred_train, metrics=metrics,
                                          is_train=True, 
                                          verbose=1, 
                                          positive_class_name=target_column)
    
    print("")

    _, to_print_test = compute_binary_classification_metrics(y_test, y_pred_test, metrics=metrics,
                                          is_train=False, 
                                          verbose=1, 
                                          positive_class_name=target_column)

    to_print += to_print_train + "\n" + to_print_test

    if not save_results:
        exit(0)

    print("Saving results.")

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(results_path, f'n{hparams["n_estimators"]}_d{hparams["max_depth"]}_{hparams["criterion"]}.txt'), 'w') as f:
        f.write("-"*32+'\n')
        f.write('# Hyperparameters\n\n')
        for param, val in hparams.items():
            f.write(f"{param}: {val}\n")
        f.write('\n'+"-"*32+'\n')
        f.write('# Results\n\n')
        f.write("\n".join(to_print))

    model_filename = os.path.join(model_path, 'model.sav')
    if hparams['n_estimators'] > 1:
         model_filename = os.path.join(model_path, f'forest_n{hparams["n_estimators"]}_d{hparams["max_depth"]}_{hparams["criterion"]}.sav')
    else:
         model_filename = os.path.join(model_path, f'tree_d{hparams["max_depth"]}_{hparams["criterion"]}.sav')

    print(f"Saving model to path: {model_filename}")

    with open( model_filename, 'wb') as f:
        pickle.dump(model, f)
