import os
import sys

import json
import pickle

import numpy as np
import pandas as pd

# Local modules
sys.path.append('.')

from path_utils import data_path, model_path, results_path

from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score

max_depth = 3

model_dict = {
    f'Decision tree (d={max_depth})': lambda: DecisionTreeClassifier(max_depth=max_depth),
    f'Random forest (n_trees=100, d={max_depth})': lambda: RandomForestClassifier(max_depth=max_depth),

}


time_column = 'lastUnixTime'
# time_column = 'firstUnixTime'
target_column = 'blackList'
not_feature_columns = ['walletOfInterest', 'blackList', 'firstUnixTime', 'lastUnixTime']


if __name__ == "__main__":

    data_file = "data/aggregated.csv"

    df = pd.read_csv(data_file)
    df_good = df[~df[target_column]].reset_index()

    feature_names = [col for col in df.columns if not col in not_feature_columns]

    with open("data/json/sources/Abuses.json", 'r') as f:
        source_dict = json.load(f)

    source_df_dict = dict()
    for source_name, wallet_dict in source_dict.items():
        
        if not 'Ransomware' in wallet_dict.keys():
            continue

        source_wallets = wallet_dict['Ransomware']
        source_df = df[df['walletOfInterest'].isin(source_wallets)].reset_index(drop=True)
        source_df_dict[source_name] = source_df

    del df

    # print(source_df_dict['Ransomwhere'])
    results = []
    models_to_save = {model_name: [] for model_name in model_dict.keys()}

    # source_df_dict.pop('EconomicSignificanceofRansomwareCampaigns')

    # print(source_df_dict.keys())
    # exit()

    for test_source_name in source_df_dict.keys():
        df_train = pd.concat([source_df for source_name, source_df in source_df_dict.items() if source_name != test_source_name])
        df_train = pd.concat([df_train, df_good])  # add benign wallets
        df_test_all = source_df_dict[test_source_name]
        df_test_unique = df_test_all[~df_test_all['walletOfInterest'].isin(df_train['walletOfInterest'])].reset_index(drop=True)

        # if len(df_test_unique) == 0: 
        #     print(test_source_name)
        #     continue

        X_train = df_train[feature_names].to_numpy()
        y_train = df_train[target_column].to_numpy()

        X_test_all = df_test_all[feature_names].to_numpy()
        y_test_all = df_test_all[target_column].to_numpy()

        if len(df_test_unique) > 0:
            X_test_unique = df_test_unique[feature_names].to_numpy()
            y_test_unique = df_test_unique[target_column].to_numpy()

        n_bad_train = len(y_train[y_train==1])
        n_good_train = len(y_train[y_train==0])
        n_bad_test_all = len(y_test_all[y_test_all==1])
        n_good_test_all = len(y_test_all[y_test_all==0])
        if len(df_test_unique) > 0:
            n_bad_test_unique = len(y_test_unique[y_test_unique==1])
            n_good_test_unique = len(y_test_unique[y_test_unique==0])

        to_print = [
            f'',
            f"Test source: {test_source_name}",
            f"Number of blackListed wallets (train): {n_bad_train}",
            f"Number of non-blackListed wallets (train): {n_good_train}",
            f"Number of blackListed wallets (test all): {n_bad_test_all}",
            f"Number of non-blackListed wallets (test all): {n_good_test_all}",
            ""
        ]

        if len(df_test_unique) > 0:
            to_print += [
                f"Number of blackListed wallets (test unique): {n_bad_test_unique}",
                f"Number of non-blackListed wallets (test unique): {n_good_test_unique}",
            ]
        
        print("\n".join(to_print))
   
        for model_name, model_fn in model_dict.items():
            model = model_fn()

            model.fit(X_train, y_train)

            # y_pred_train = model.predict(X_train)
            y_pred_test_all = model.predict(X_test_all)
            

            res_dict = {
                'source': test_source_name,
                'n_good_train': n_good_train,
                'n_bad_train': n_bad_train,
                # 'n_good_test_all': n_good_test_all, 
                'n_bad_test_all': n_bad_test_all,
                # 'n_good_test_unique': n_good_test_unique, 
                'n_bad_test_unique': n_bad_test_unique,
                'model': model_name,
            }
            models_to_save[model_name].append(model)

            metrics_dict = {'recall_all': recall_score(y_test_all, y_pred_test_all)}

            if len(df_test_unique[df_test_unique[target_column]]) > 0:
                y_pred_test_unique = model.predict(X_test_unique)
                metrics_dict['recall_unique'] = recall_score(y_test_unique, y_pred_test_unique)
            else:
                metrics_dict['recall_unique'] = np.nan

            res_dict.update(metrics_dict)
            results.append(res_dict)

    
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{results_path}/exp1_results.csv", index=False)

    with open(os.path.join(model_path, "models_cross_source.pkl"), 'wb') as f:
        pickle.dump(models_to_save, f)