import os
import sys
import datetime

import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold

# local modules
sys.path.append('.')
from path_utils import data_path, model_path, results_path
from ml_utils import compute_binary_classification_metrics

n_folds = 5

max_depth = 3

model_dict = {
    f'Decision tree (d={max_depth})': lambda: DecisionTreeClassifier(max_depth=max_depth),
    f'Random forest (n_trees=100, d={max_depth})': lambda: RandomForestClassifier(max_depth=max_depth),
    # f'Extra tree (d={max_depth})': lambda: ExtraTreesClassifier(max_depth=max_depth),
    # 'Naive Bayes': lambda: GaussianNB(),
    # 'Logistic Regression': lambda: LogisticRegression(),
    # f'SVM (RBF kernel)': lambda: SVC(kernel='rbf')
}

metrics = ['accuracy', 'precision', 'recall', 'balanced_accuracy']
time_column = 'lastUnixTime'
# time_column = 'firstUnixTime'
target_column = 'blackList'
not_feature_columns = ['walletOfInterest', 'blackList', 'firstUnixTime', 'lastUnixTime']

def find_misclassified_by_year(
        year_series: pd.Series, 
        y_true: np.ndarray, y_pred: np.ndarray
    ):
    idx_wrong = np.argwhere(y_true != y_pred).T[0]
    frac_wrong_by_year = year_series.iloc[idx_wrong].value_counts().sort_index()#/year_series.value_counts().sort_index()
    frac_wrong_by_year = frac_wrong_by_year.to_dict()

    return frac_wrong_by_year


if __name__ == "__main__":

    # fix random seed for reproducibility
    np.random.seed(42)

    df = pd.read_csv(os.path.join(data_path, 'aggregated.csv'))
    wallet_names = df['walletOfInterest'].to_numpy()
    feature_names = [col for col in df.columns if not col in not_feature_columns]
    
    df['year'] = df[time_column].apply(lambda x: datetime.datetime.fromtimestamp(x).year)

    # Remove wallets which don't have transactions after 2014 
    df = df[df['year'] > 2014].reset_index(drop=True)

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    results = []
    models_to_save = {model_name: [] for model_name in model_dict.keys()}

    X = df[feature_names].to_numpy()
    y = df[target_column].to_numpy()

    kf = KFold(n_splits=n_folds)
    for i_fold, (i_train, i_test)in enumerate(kf.split(X)):

        # X_train, y_train, X_test, y_test

        X_train = X[i_train]
        y_train = y[i_train]
        X_test = X[i_test]
        y_test = y[i_test]

        n_bad_train = len(y_train[y_train==1])
        n_good_train = len(y_train[y_train==0])
        n_bad_test = len(y_test[y_test==1])
        n_good_test = len(y_test[y_test==0])

        to_print = [
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

            # y_pred_train = model.predict(X_train)
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
                return_results_string=False,
            )

            years = df['year'].iloc[i_test].reset_index(drop=True)
            years_dict = find_misclassified_by_year(years, y_test, y_pred_test)

            res_dict.update(metrics_dict)
            res_dict.update(years_dict)

            results.append(res_dict)


    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{results_path}/exp0_results.csv", index=False)

    with open(os.path.join(model_path, "models_cv.pkl"), 'wb') as f:
        pickle.dump(models_to_save, f)

        
