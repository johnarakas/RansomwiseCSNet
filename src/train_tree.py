import json
import os
import pickle
# import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Local modules
# sys.path.append('.')
from src.path_utils import DATASET_FILE, DESCRIPTOR_FILE, MODEL_SAVE_PATH, RESULTS_PATH
from src.ml_utils import compute_binary_classification_metrics, temporal_train_test_split

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
TEST_SIZE = 0.3  # fraction of dataset used for test
save_results = True  # set to false when making changes

USE_TIME_SPLIT = False

def check_time_split(ts_train, ts_test):
    print(f"Min ts train: {ts_train.min()}")
    print(f"Max ts train: {ts_train.max()}")
    print(f"Min ts test: {ts_test.min()}")
    print(f"Max ts test: {ts_test.max()}")

def get_treebased_model(hparams: dict):
    if hparams['n_estimators'] > 1:
        model = RandomForestClassifier(n_estimators=hparams['n_estimators'], 
                                       max_depth=hparams['max_depth'], 
                                       criterion=hparams['criterion'], 
                                       random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=hparams['max_depth'], 
                                       criterion=hparams['criterion'],
                                       random_state=42)
    return model


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_time_split", default=USE_TIME_SPLIT, type=bool)
    parser.add_argument("--test_size", default=TEST_SIZE, type=float)
    parser.add_argument("--dataset_file", type=str, default=DATASET_FILE)
    parser.add_argument("--descriptor_file", type=str, default=DESCRIPTOR_FILE)
    parser.add_argument("--results_path", type=str, default=RESULTS_PATH)
    parser.add_argument("--model_save_path", type=str, default=MODEL_SAVE_PATH)
    args = parser.parse_args()

    dataset_file = args.dataset_file
    descriptor_file = args.descriptor_file
    results_file = args.results_path
    model_save_path = args.model_save_path
    use_time_split = args.use_time_split
    test_size = args.test_size
    
    np.random.seed(42)

    df = pd.read_csv(dataset_file)

    with open(descriptor_file, 'r') as f: 
        descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']
    
    if use_time_split:
        df_train, df_test = temporal_train_test_split(df, test_size=test_size, time_column=time_column)
        check_time_split(df_train[time_column], df_test[time_column])
    else:
        # random split
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)

    X_train = df_train[feature_names]
    y_train = df_train[target_column]
    X_test = df_test[feature_names]
    y_test = df_test[target_column]

    model = get_treebased_model(hparams)
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
    
    metrics_train = compute_binary_classification_metrics(
        y_train, y_pred_train, 
        metrics=metrics,
        is_train=True, 
        verbose=1, 
        positive_class_name=target_column
    )
    
    metrics_test = compute_binary_classification_metrics(
        y_test, y_pred_test, 
        metrics=metrics,
        is_train=False, 
        verbose=1, 
        positive_class_name=target_column
    )

    to_print_train = []
    to_print_test = []

    for metric in metrics_train.keys():
        to_print_train.append(f"Train {metric}: {metrics_train[metric]}")
        to_print_test.append(f"Test {metric}: {metrics_test[metric]}")

    to_print_train = "\n".join(to_print_train)
    to_print_test = "\n".join(to_print_test)

    to_print.append(to_print_train + "\n" + to_print_test)

    if not save_results:
        exit(0)

    print("Saving results.")

    results_file = os.path.join(results_file, f'n{hparams["n_estimators"]}_d{hparams["max_depth"]}_{hparams["criterion"]}.txt')
    with open(results_file, 'w') as f:
        f.write("-"*32+'\n')
        f.write('# Hyperparameters\n\n')
        for param, val in hparams.items():
            f.write(f"{param}: {val}\n")
        f.write('\n'+"-"*32+'\n')
        f.write('# Results\n\n')
        f.write("\n".join(to_print))

    print("Retraining model on all data.")

    model = get_treebased_model(hparams)

    X = df[feature_names]
    y = df[target_column]

    model.fit(X, y)

    model_filename = os.path.join(model_save_path, 'model.pkl')
    if hparams['n_estimators'] > 1:
         model_filename = os.path.join(model_save_path, f'forest_n{hparams["n_estimators"]}_d{hparams["max_depth"]}_{hparams["criterion"]}.pkl')
    else:
         model_filename = os.path.join(model_save_path, f'tree_d{hparams["max_depth"]}_{hparams["criterion"]}.pkl')

    print(f"Saving model to path: {model_filename}")

    with open( model_filename, 'wb') as f:
        pickle.dump(model, f)

    wallet_filename = os.path.join(MODEL_SAVE_PATH, 'wallets_train.txt')
    df_train[['walletOfInterest']].to_csv(wallet_filename, index=False)