import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# local modules
from src.ml_utils import compute_binary_classification_metrics
from src.path_utils import DATASET_FILE, DESCRIPTOR_FILE, MODEL_SAVE_PATH, RESULTS_PATH

N_FOLDS = 5
MAX_DEPTH = 3

metrics = ['accuracy', 'precision', 'recall', 'balanced_accuracy']

def find_misclassified_by_year(
        year_series: pd.Series, 
        y_true: np.ndarray, y_pred: np.ndarray
    ):

    idx_wrong = np.argwhere(y_true != y_pred).T[0]
    frac_wrong_by_year = year_series.iloc[idx_wrong].value_counts().sort_index()#/year_series.value_counts().sort_index()
    frac_wrong_by_year = frac_wrong_by_year.to_dict()

    frac_wrong_by_year = {
        f"n_wrong_{year}": n_wrong for year, n_wrong in frac_wrong_by_year.items()
    }

    return frac_wrong_by_year


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default=DATASET_FILE)
    parser.add_argument('--n_folds', type=int, default=N_FOLDS)
    parser.add_argument('--max_depth', type=int, default=MAX_DEPTH)
    parser.add_argument('--descriptor_file', type=str, default=DESCRIPTOR_FILE)
    parser.add_argument('--model_save_path', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--results_path', type=str, default=RESULTS_PATH)
    args = parser.parse_args()

    dataset_file = args.dataset_file
    n_folds = args.n_folds
    max_depth = args.max_depth
    descriptor_file = args.descriptor_file
    model_save_path = os.path.abspath(args.model_save_path)
    results_path = os.path.abspath(args.results_path)

    print("Model save path:", model_save_path)
    print("Results path:", results_path)

    assert n_folds >= 2
    assert max_depth >= 1

    assert os.path.isfile(dataset_file), "Dataset file does not exist"
    assert os.path.isfile(descriptor_file), "Descriptor file does not exist"
    assert os.path.isdir(os.path.dirname(model_save_path)), "Parent directory of model save path does not exist"
    assert os.path.isdir(os.path.dirname(results_path)), "Parent directory of results path does not exist"

    # Create directories if they don't exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.results_path):
        os.makedirs(args.results_path)

    dataset_file = args.dataset_file
    n_folds = args.n_folds
    max_depth = args.max_depth
    descriptor_file = args.descriptor_file

    with open(descriptor_file, 'r') as f: 
        descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']

    # fix random seed for reproducibility
    np.random.seed(42)

    model_dict = {
        f'Decision tree (d={max_depth})': lambda: DecisionTreeClassifier(max_depth=max_depth),
        f'Random forest (n_trees=100, d={max_depth})': lambda: RandomForestClassifier(max_depth=max_depth),
        f"Gradient boosting (n_estimators=100, learning_rate=0.1, max_depth={max_depth})": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=max_depth)
    }

    df = pd.read_csv(dataset_file)
    df['year'] = df[time_column].apply(lambda x: datetime.datetime.fromtimestamp(x).year)
    wallet_names = df['walletOfInterest'].to_numpy()
    

    # Shuffle dataset (only for cross-validation!)
    df = df.sample(frac=1).reset_index(drop=True)

    results = []
    models_to_save = {model_name: [] for model_name in model_dict.keys()}

    with open(descriptor_file, 'r') as f: descriptor = json.load(f)
    feature_columns = descriptor['feature_columns']

    X = df[feature_columns].to_numpy()
    y = df[target_column].to_numpy()

    kf = KFold(n_splits=n_folds)
    for i_fold, (i_train, i_test)in enumerate(kf.split(X)):

        # create train and test sets
        X_train = X[i_train]
        y_train = y[i_train]
        X_test = X[i_test]
        y_test = y[i_test]

        n_bad_train = len(y_train[y_train==1])
        n_good_train = len(y_train[y_train==0])
        n_bad_test = len(y_test[y_test==1])
        n_good_test = len(y_test[y_test==0])

        to_print = [
            "-"*32,
            f"Fold {i_fold+1}/{n_folds}",
            f"Number of blackListed wallets (train): {n_bad_train}",
            f"Number of non-blackListed wallets (train): {n_good_train}",
            f"Number of blackListed wallets (test): {n_bad_test}",
            f"Number of non-blackListed wallets (test): {n_good_test}",
            ""
        ]
        
        print("\n".join(to_print))

        for model_name, model_fn in model_dict.items():
            model = model_fn()

            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)

            res_dict = {
                'fold': i_fold+1,
                'n_good_train': n_good_train,
                'n_bad_train': n_bad_train,
                'n_good_test': n_good_test, 
                'n_bad_test': n_bad_test,
                'model': model_name,
            }
            models_to_save[model_name].append(model)

            metrics_dict = compute_binary_classification_metrics(
                y_test, y_pred_test, 
                metrics=metrics,
                is_train=False, 
                verbose=1, 
                positive_class_name=target_column,
            )

            years = df['year'].iloc[i_test].reset_index(drop=True)  # year values for test set
            years_dict = find_misclassified_by_year(years, y_test, y_pred_test)

            res_dict.update(metrics_dict)
            res_dict.update(years_dict)

            results.append(res_dict)


    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{results_path}/results_cv_d{max_depth}.csv", index=False)

    with open(os.path.join(model_save_path, "models_cv.pkl"), 'wb') as f:
        pickle.dump(models_to_save, f)

        