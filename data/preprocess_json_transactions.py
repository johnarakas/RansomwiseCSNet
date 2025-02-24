"""
This script is used to convert transactions in JSON format to CSV format.

Usage:
```
python preprocess_json_transactions.py [--outpath OUTPATH] [--use_existing_metadata]
```

Arguments:
- `--outpath`: The path to the output directory for the CSV files. Default is "data/transactions".
- `--use_existing_metadata`: If set, the script will use the existing metadata files in the output directory.

Since the benign wallets are scraped at random from the Bitcoin blockchain, there might be some overlap
between the benign and ransomware wallets.
This is reported at the end of the script.

"""


import os
import json
import pandas as pd
import warnings

CSV_TRANSACTIONS_PATH = "data/transactions"
USE_EXISTING_METADATA = False

BENIGN_SOURCES = [
    "RANDOM",
    "KAGGLE",
    "LARGE_TRANSACTIONS",
]
RANSOMWARE_SOURCES = [
    "CSNET24",
    "UPDATED",
]

ORIGINAL_RANSOMWARE_SOURCES = [
    "data/transactions/json/maliciousTransactions.json",
]

UPDATED_RANSOMWARE_SOURCES = [
    "data/transactions/json/ransomwareTransactions.json",
]

KAGGLE_SOURCES = [
    "data/transactions/json/kaggleTransactions.json",
]

RANDOM_SOURCES = [
    "data/transactions/json/randomBenignTransactions.json",
]

LARGE_TRANSACTION_SOURCES = [
    "data/transactions/json/benignTransactions.json",
]

def get_json_files(benign_sources: list, ransomware_sources: list):

    benign_json_files = []
    ransomware_json_files = []

    for source in benign_sources:
        match source:
            case "KAGGLE":
                benign_json_files.extend(KAGGLE_SOURCES)
            case "RANDOM":
                benign_json_files.extend(RANDOM_SOURCES)
            case "LARGE_TRANSACTIONS":
                benign_json_files.extend(LARGE_TRANSACTION_SOURCES)
            case _:
                raise ValueError(f"Unknown benign source: {source}")
            
    for source in ransomware_sources:
        match source:
            case "CSNET24":
                ransomware_json_files.extend(ORIGINAL_RANSOMWARE_SOURCES)
            case "UPDATED":
                ransomware_json_files.extend(UPDATED_RANSOMWARE_SOURCES)
            case _:
                raise ValueError(f"Unknown ransomware source: {source}")

    return benign_json_files, ransomware_json_files
        

def preprocess_json(
        sourcefile: str, 
        outpath: str, 
        ransomware: bool, 
        use_existing_metadata: bool=False
    ) -> list:

    data: dict = {}
    with open(sourcefile, 'r') as f:
        data = json.load(f)

    sourcefilename = sourcefile.split('/')[-1].replace('.json', '.csv')
    outfile = f"{outpath}/{sourcefilename}"
    print(outfile)

    if os.path.isfile(outfile) and use_existing_metadata:
        warnings.warn("Using existing metadata. To overwrite, set `use_existing_metadata=False` or delete existing metadata.")
        df = pd.read_csv(outfile)
    else:
        tx_list: list = []
        for wallet in data.keys():
            wallet_data = data[wallet]

            # NOTE: If a wallet has no transactions, it does not appear in the final DataFrame
            for inout in ['in', 'out']:
                for ts, eur, btc in zip(wallet_data[f'{inout}_timestamps'], 
                                        wallet_data[f'{inout}coming_transactions_euro'], 
                                        wallet_data[f'{inout}coming_transactions_btc']):
                    
                    tx = dict()  #dictionary describing the transaction
                    tx['walletOfInterest'] = wallet
                    tx['unixTime'] = ts
                    tx['eurValue'] = eur
                    tx['btcValue'] = btc
                    tx['isReceiver'] = True if inout == 'in' else False 
                    # tx['isSender'] = not tx['isReceiver']
                    tx['blackList'] = ransomware
                    tx_list.append(tx)

        df: pd.DataFrame = pd.DataFrame(tx_list)
        df.to_csv(outfile, index=False)
    return df['walletOfInterest'].unique().tolist()



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default=CSV_TRANSACTIONS_PATH)
    parser.add_argument('--use_existing_metadata', type=bool, default=USE_EXISTING_METADATA)
    # TODO: Add argument for different source configurations used in the paper
    args = parser.parse_args()

    outpath = args.outpath
    use_existing_metadata = args.use_existing_metadata

    benign_json_files, ransomware_json_files = get_json_files(BENIGN_SOURCES, RANSOMWARE_SOURCES)
    
    bad_wallets = set()
    for filename in ransomware_json_files:
        wallets = preprocess_json(filename, outpath, ransomware=True)
        bad_wallets = bad_wallets.union(set(wallets))
    bad_wallets = list(bad_wallets)

    print(f'Total ransomware wallets: {len(bad_wallets)}')
        
    good_wallets = set()
    for filename in benign_json_files:
        wallets = preprocess_json(filename, outpath, ransomware=False)
        good_wallets = good_wallets.union(set(wallets))
    good_wallets = list(good_wallets)

    print(f'Total benign wallets: {len(good_wallets)}')

    overlap = list(set(good_wallets).intersection(set(bad_wallets)))
    print(f"Overlap between benign and ransomware wallets: {len(overlap)}")


    