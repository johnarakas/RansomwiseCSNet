#---------- Python libraries ------------
import json
import os
import pickle
import sys


import pandas as pd

#------------ Local modules -------------
import sys
sys.path.append(".")
from data_utils import preprocess_sources
from ml_utils import compute_binary_classification_metrics

sources = [
    # 'data/Alienvault.csv',
    # 'data/Behas.csv',
    # 'data/ChainAbuse.csv',
    # 'data/Irvine.csv',
    # 'data/KillingTheBear.csv',
    # 'data/Ransomlook.csv',
    # 'data/Ransomwhere.csv',
    # 'data/SophosLab.csv',
    # 'data/Tessii.csv',
    # 'data/Traceer.csv',
    # 'data/benignTransactions.csv',
    # 'data/darknet.csv',
    # 'data/ofac.csv'
    # 'data/bitcoinAbuseTotal.csv'
    # 'data/spam.csv'
    'data/benignTransactions.csv',
]

from path_utils import model_path, data_path

if __name__ == "__main__":

    metrics = ['accuracy']
    discard_seen_wallets = True

    # Load model and make predictions
    with open(os.path.join(model_path, 'tree_d3_gini.sav'), 'rb') as f: model = pickle.load(f)
    
    descriptor_path = os.path.join(data_path, "descriptor.json")
    with open(descriptor_path, 'r') as f: descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']

    wallets_train = pd.read_csv("data/wallet_train.csv")['walletOfInterest'].to_list()

    verbose = 1

    for source in sources:

        print(f"Source: {source}")

        df, source_wallet_dict = preprocess_sources([source], verbose=0, return_source_wallet_dict=True)

        wallets_source = df['walletOfInterest'].unique().tolist()
        
        print(f"Number of wallets: {len(wallets_source)}")

        wallets_shared = list(set(wallets_train).intersection(set(wallets_source)))
        print(f"Number of wallets that were seen during training: {len(wallets_shared)}")

        print("Before discarding common wallets:")

        y_true = df['blackList'].to_numpy()
        y_pred = model.predict(df[feature_names])

        _, _ = compute_binary_classification_metrics(y_true, y_pred, metrics=metrics,
                                            is_train=False, 
                                            verbose=1, 
                                            positive_class_name='blacklisted')

        if discard_seen_wallets:
            df = df[~df['walletOfInterest'].isin(wallets_shared)].reset_index(drop=True)

        print("After discarding common wallets:")

        y_true = df['blackList'].to_numpy()
        y_pred = model.predict(df[feature_names])

        _, _ = compute_binary_classification_metrics(y_true, y_pred, metrics=metrics,
                                            is_train=False, 
                                            verbose=1, 
                                            positive_class_name='blacklisted')
        
        print("")