import datetime
import json
import os
import pickle

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score


# local modules
from src.path_utils import DATASET_FILE, DESCRIPTOR_FILE, MODEL_SAVE_PATH, RESULTS_PATH, SOURCES_FILE

MAX_DEPTH = 3

exclude_sources = ["Ransomlook", "Behas", "Boulevard"]


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default=DATASET_FILE)
    parser.add_argument('--max_depth', type=int, default=MAX_DEPTH)
    parser.add_argument('--descriptor_file', type=str, default=DESCRIPTOR_FILE)
    parser.add_argument('--model_save_path', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--results_path', type=str, default=RESULTS_PATH)
    parser.add_argument('--sources_file', type=str, default=SOURCES_FILE)
    args = parser.parse_args()

    dataset_file = args.dataset_file
    max_depth = args.max_depth
    descriptor_file = args.descriptor_file
    model_save_path = os.path.abspath(args.model_save_path)
    results_path = os.path.abspath(args.results_path)
    sources_file = os.path.abspath(args.sources_file)

    assert max_depth >= 1

    print("Model save path:", model_save_path)
    print("Results path:", results_path)

    with open(descriptor_file, 'r') as f: 
        descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']

    # fix random seed for reproducibility
    np.random.seed(42)

    model_dict = {
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=max_depth),
        "Random Forest": lambda: RandomForestClassifier(max_depth=max_depth),
        "Gradient Boosted Trees": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=max_depth)
    }

    result_list = []

    df = pd.read_csv(dataset_file)

    with open(sources_file, 'r') as f: 
        sources = json.load(f)

    sources = {name: sources[name] for name in sources if name not in exclude_sources}
    
    all_wallets = sum(sources.values(), [])
    wallet_counts = Counter(all_wallets)
    common_wallets = [wallet for wallet in all_wallets if wallet_counts[wallet] > 1]
    
    common_df = df[df['walletOfInterest'].isin(common_wallets) | ~df[target_column]]
    X_train = common_df[feature_names]
    y_train = common_df[target_column]

    for model_name, model_factory in model_dict.items():
        model = model_factory()

        model.fit(X_train, y_train)

        for name, source_list in sources.items():
            print(f"\nProcessing {name}...")
            source_df = df[df['walletOfInterest'].isin(source_list)]
            source_df = source_df[~source_df['walletOfInterest'].isin(common_wallets)]

            X_test = source_df[feature_names]
            y_test = source_df[target_column]

            y_pred = model.predict(X_test)

            recall = recall_score(y_test, y_pred)

            print(f"Recall: {recall}")

            result_list.append({
                'model': model_name,
                'source': name,
                'recall': recall,
            })

    result_df = pd.DataFrame(result_list)
    result_df.to_csv(os.path.join(results_path, 'leave_one_source_out.csv'), index=False)