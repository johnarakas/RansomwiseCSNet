"""
This script can be used to validate the transaction data contained in a JSON file.

Constants:
    JSON_FILE (str): The path to the JSON file containing the transaction data.
    START_YEAR (int): The minimum valid year for a transaction.
    END_YEAR (int): The maximum valid year for a transaction.
    MIN_EURO_BTC_RATIO (float): The minimum ratio of Euro to Bitcoin for a valid conversion.
    MAX_EURO_BTC_RATIO (float): The maximum ratio of Euro to Bitcoin for a valid conversion.
    PRINT_DATA_SUMMARY (bool): Whether to print the summary of the transaction data. IMPORTANT: This requires Pandas (`pip install pandas>=2.2`).
"""


# NOTE: Please use Python >= 3.10

# JSON_FILE = "data/json/benignTransactions.json"
JSON_FILE = "../data/ransomware/sources/ransomwareTransactions.json"
START_YEAR = 2010
END_YEAR = 2024
MIN_EURO_BTC_RATIO = 200
MAX_EURO_BTC_RATIO = 100_000

PRINT_DATA_SUMMARY = True  # NOTE: Requires Pandas

######################################################################

import json
from datetime import datetime

def check_valid_bitcoin_address(wallet_id: str) -> bool:
    """
    TODO: Find better rules to validate the address
    """

    if not isinstance(wallet_id, str):
        return False
        
    if (len(wallet_id) > 42) or (len(wallet_id) < 2):
        return False

    return True

def check_matching_entry_lengths(wallet_history: dict, keys: list[str]) -> bool:
    """
    Checks if all entries in the wallet_history dictionary corresponding to 
    the provided keys have matching lengths.

    Args:
        wallet_history (dict): A dictionary containing transaction history data.
        keys (list[str]): A list of keys to check for matching entry lengths.

    Returns:
        bool: True if all specified entries have the same length, False otherwise.
    """
    return all(len(wallet_history[key]) == len(wallet_history[keys[0]]) for key in keys)

def check_valid_timestamps(timestamps: list[int, float], start_year: int, end_year: int) -> bool:
    """
    Checks if all timestamps in the given list are valid, i.e. lie within the given year range.
    
    Args:
        timestamps (list[int] | list[float]): A list of timestamps in seconds.
        start_year (int): The start year of the valid time range.
        end_year (int): The end year of the valid time range.
    
    Returns:
        bool: True if all timestamps are valid, False otherwise.
    """

    check_valid_year = lambda year: (year >= start_year) and (year <= end_year)
    return all([check_valid_year(datetime.fromtimestamp(timestamp).year)  for timestamp in timestamps])

def check_non_zero(wallet_history: dict[str, list], key: str) -> bool:
    """
    Checks if the given wallet history's values are non-zero.
    """
    return all([value != 0 for value in wallet_history[key]])

def check_valid_btc_euro_conversion(wallet_history: dict[str, list], btc_key: str, euro_key: str) -> bool:
    """
    Checks if the given wallet history's btc and euro values are valid,
    which is the case if:
    - all EUR / BTC ratios are between MIN_EURO_BTC_RATIO and MAX_EURO_BTC_RATIO
    """
    check_valid_conversion_f = lambda btc, euro: (euro / btc > MIN_EURO_BTC_RATIO) and  (euro / btc < MAX_EURO_BTC_RATIO)
    return all([check_valid_conversion_f(btc, euro) for (btc, euro) in zip(wallet_history[btc_key], wallet_history[euro_key])])


def assert_valid_transaction_history(
        wallet_id: str, wallet_history: dict[str, list],
        year_range: tuple[int, int],
    ) -> None:
    """
    Asserts that the given transaction history is valid.
    
    A transaction history is considered valid if:
    - The wallet ID is a valid Bitcoin address
    - The lengths of all incoming transaction history lists are equal
    - The lengths of all outcoming transaction history lists are equal
    - All timestamps are within the given year range
    - All BTC to EUR conversions are valid

    Args:
        wallet_id (str): The wallet ID.
        wallet_history (dict[str, list]): The transaction history data.
        year_range (tuple[int, int]): The start and end years of the valid time range.
    """
    
    assert check_valid_bitcoin_address(wallet_id), "Invalid wallet address"

    is_matching_in_transaction_length = check_matching_entry_lengths(wallet_history, ['in_timestamps', 'incoming_transactions_euro', 'incoming_transactions_btc'])
    assert is_matching_in_transaction_length, "Lengths of in_timestamps and incoming_transactions_euro and incoming_transactions_btc are not equal"
    
    is_valid_timestamp_list = check_valid_timestamps(wallet_history['in_timestamps'], start_year=year_range[0], end_year=year_range[1])
    assert is_valid_timestamp_list, "Invalid timestamps in in_timestamps"

    is_valid_timestamp_list = check_valid_timestamps(wallet_history['out_timestamps'], start_year=year_range[0], end_year=year_range[1])
    assert is_valid_timestamp_list, "Invalid timestamps in out_timestamps"

    is_matching_out_transaction_length = check_matching_entry_lengths(wallet_history, ['out_timestamps', 'outcoming_transactions_euro', 'outcoming_transactions_btc'])
    assert is_matching_out_transaction_length, "Lengths of out_timestamps and outcoming_transactions_euro and outcoming_transactions_btc are not equal"

    assert check_non_zero(wallet_history, 'incoming_transactions_btc'), "Zero values in incoming_transactions_btc"
    assert check_non_zero(wallet_history, 'incoming_transactions_euro'), "Zero values in incoming_transactions_euro"
    assert check_non_zero(wallet_history, 'outcoming_transactions_btc'), "Zero values in outcoming_transactions_btc"
    assert check_non_zero(wallet_history, 'outcoming_transactions_euro'), "Zero values in outcoming_transactions_euro"

    is_valid_btc_euro_conversion = check_valid_btc_euro_conversion(wallet_history, 'incoming_transactions_btc', 'incoming_transactions_euro')
    assert is_valid_btc_euro_conversion, "Invalid BTC to EUR conversion in incoming transactions"

    is_valid_btc_euro_conversion = check_valid_btc_euro_conversion(wallet_history, 'outcoming_transactions_btc', 'outcoming_transactions_euro')
    assert is_valid_btc_euro_conversion, "Invalid BTC to EUR conversion in outcoming transactions"


def print_data_summary(transaction_data: dict[str, list]) -> None:
    """
    Prints a summary of the transaction data.

    The summary includes the total number of transactions, and the total amount of
    Bitcoin and Euro transferred, grouped by year.
    """
    import pandas as pd
    dfs = []
    for direction in ['in', 'out']:
        tx_data = {
            key.removeprefix(f"{direction}").removeprefix("coming_transactions").lstrip("_"): value
            for key, value in transaction_data.items() 
            if key.startswith(direction)
        }
        df = pd.DataFrame(tx_data)
        df['transactions'] = 1
        df['year'] = df['timestamps'].apply(lambda x: datetime.fromtimestamp(x).year)        
        df = df.groupby('year').sum()[['transactions', 'euro', 'btc']]
        dfs.append(df)

    df = pd.merge(dfs[0], dfs[1], how='outer', on='year', suffixes=('_in', '_out'))
    df = df.fillna(0)

    print(32 * '-')
    print("Transaction data summary by year:")
    print(df)
    print(32 * '-')
    print("Overall transaction data summary:")
    print(df.sum())
    print(32 * '-')


if __name__ == '__main__':

    json_file = JSON_FILE
    year_range = (START_YEAR, END_YEAR)

    with open(json_file, 'r') as f:
        transactions = json.load(f)

    # print(transactions['18A7kZU23Fb26RSaiCD9dax27bHjcEHTxU'])
    # exit()

    transaction_data: dict[str, list] = {key: [] for key in list(transactions.values())[0].keys()}
    
    n_corrupted = 0
    for wallet_id, wallet_history in transactions.items():
        try:
            assert_valid_transaction_history(wallet_id, wallet_history, year_range)
        except AssertionError as e:
            print(f"Invalid transaction history for wallet {wallet_id}")
            print('\t', e)
            n_corrupted += 1
            continue

        for key in transaction_data.keys():
            transaction_data[key] += wallet_history[key]

    print(32 * '-')
    print(f"Number of corrupted wallets: {n_corrupted}")

    if PRINT_DATA_SUMMARY:
        print_data_summary(transaction_data)