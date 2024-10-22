#---------- Python libraries ------------
import os
import pickle
import json

#------------ Local modules -------------
import sys
sys.path.append(".")
from path_utils import model_path, results_path, data_path
from data_utils import preprocess_sources

# List of sources to be read
sources = [
    'data/benignTransactions.csv',
    'data/maliciousTransactions.csv',
]
# sources = [f"data/{filename}" for filename in os.listdir('data/sources') if filename.endswith('.csv')]

outfile = os.path.join(results_path, 'predictions.json')


if __name__ == "__main__":

    verbose = 1

    results_dict = dict()

    descriptor_path = os.path.join(data_path, "descriptor.json")
    with open(descriptor_path, 'r') as f: descriptor = json.load(f)
    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']
    
    # Preprocess sources and get source_wallet_dict
    # entries of source_wallet dict are e.g. ('3cc49v...', ['source1.csv', 'source2.csv'])
    df, source_wallet_dict = preprocess_sources(sources, verbose=verbose, return_source_wallet_dict=True)

    if verbose >= 1:
        print("Predicting blacklisted/not for each wallet.")

    # Load model and make predictions
    with open(os.path.join(model_path, 'tree_d3_gini.sav'), 'rb') as f: model = pickle.load(f)

    df['prediction'] = model.predict(df[feature_names])
    wallets_fp = df[(df['prediction'])&(~df[target_column])]['walletOfInterest']
    wallets_fn = df[~(df['prediction'])&(df[target_column])]['walletOfInterest']

    # print(df[df['walletOfInterest'].isin(wallets_fp)][target_column])
    # exit()

    wallets_fp.to_csv(f'{results_path}/false_positives.txt', index=False)
    wallets_fn.to_csv(f'{results_path}/false_negatives.txt', index=False)

    # if verbose >= 1:
    #     print(f"Saving results in {outfile}.")

    # for _, entry in df.iterrows():
    #     source_list = []
    #     if entry['walletOfInterest'] in source_wallet_dict:
    #         source_list = source_wallet_dict[entry['walletOfInterest']]

    #     prediction = 'blackListed' if entry['blackList'] else 'not blackListed'

    #     results_dict[entry['walletOfInterest']] = {
    #         'prediction': prediction,
    #         'sources': source_list,
    #     }

    # with open(outfile, 'w') as f: f.write(json.dumps(results_dict))

    # if verbose >= 1:
    #     print("Terminated successfully.")
    