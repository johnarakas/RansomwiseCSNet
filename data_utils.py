from typing import Optional

import pandas as pd

def drop_corrupted_entries(df):
    """
    Drop entries with EUR or BTC value equal to zero.
    """
    df = df[(df['eurValue'] != 0)&(df['btcValue'] != 0)]#.reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def compute_aggregate_col(
        df_aggr: pd.DataFrame, 
        df: pd.DataFrame, 
        col: str, 
        col_suffix: Optional[str]=None  # example: 'rc', 'sn'
    ):
    dfa = df_aggr
    dfa = dfa.merge(df.groupby('walletOfInterest')[col].mean().reset_index(drop=False), 
                    how='left', on='walletOfInterest').fillna(0)
    
    col_name = f'{col}_mean'
    if col_suffix is not None:
        col_name = f'{col}_{col_suffix}_mean'
    dfa = dfa.rename(columns={col: col_name})

    dfa = dfa.merge(df.groupby('walletOfInterest')[col].std().reset_index(drop=False), 
                    how='left', on='walletOfInterest').fillna(0)
    
    col_name = f'{col}_std'
    if col_suffix is not None:
        col_name = f'{col}_{col_suffix}_std'
    dfa = dfa.rename(columns={col: col_name})

    return dfa

def extract_wallet_features(df):
    """
    Extracts wallet-level features.

    TODO: Implement data augmentation option (Consider all subsequences of transactions for each wallet). 
    """

    # overall dataframe, dataframe with only transactions as sender, dataframe with only transactions as receiver
    df = df.sort_values(by=['walletOfInterest', 'unixTime'])

    df_sn = df[~df['isReceiver']].reset_index(drop=True)

    df_rc = df[df['isReceiver']].reset_index(drop=True)

    # calculate inter-transaction times
    df['timeDelta'] = df.groupby('walletOfInterest')['unixTime'].diff().fillna(0)
    df_sn['timeDelta'] = df_sn.groupby('walletOfInterest')['unixTime'].diff().fillna(0)
    df_rc['timeDelta'] = df_rc.groupby('walletOfInterest')['unixTime'].diff().fillna(0)

    # keep first and last transaction time
    dfa = df.groupby('walletOfInterest')['unixTime'].min().rename('firstUnixTime').reset_index(drop=False)
    dfa = dfa.merge(
        df.groupby('walletOfInterest')['unixTime'].max().rename('lastUnixTime').reset_index(drop=False),
        how='left', on='walletOfInterest'
    )

    # count total number of transactions
    dfa = dfa.merge(
        df.groupby('walletOfInterest')['unixTime'].count().rename('transaction_count').reset_index(drop=False),
        how='left', on='walletOfInterest'
    )
    df.groupby('walletOfInterest')['unixTime'].count()

    # count transactions as sender
    dfa = dfa.merge(
        df_sn.groupby('walletOfInterest')['unixTime'].count().reset_index(drop=False), 
        how='left', on='walletOfInterest'
    ).fillna(0).rename(columns={'unixTime': 'transaction_count_sn'})

    # count transactions as receiver
    dfa = dfa.merge(
        df_rc.groupby('walletOfInterest')['unixTime'].count().reset_index(drop=False), 
        how='left', on='walletOfInterest'
    ).fillna(0).rename(columns={'unixTime': 'transaction_count_rc'})

    # basic statistics: mean, std, mean_rc, std_rc, mean_sd, std_sd for each feature in the list
    for col in ['eurValue', 'btcValue', 'timeDelta']:
        dfa = compute_aggregate_col(dfa, df, col)
        dfa = compute_aggregate_col(dfa, df_sn, col, col_suffix='sn')
        dfa = compute_aggregate_col(dfa, df_rc, col, col_suffix='rc')


    # if at least one entry is blacklisted, the entire wallet is blacklisted
    dfa['blackList'] = df.groupby('walletOfInterest')['blackList'].agg(lambda x: x.any()).values

    dfa = dfa.fillna(0)
    dfa = dfa.reset_index(drop=True)
    return dfa



def preprocess_sources(
        sources:list, 
        verbose:int=0, 
        return_source_wallet_dict:bool=False,
        start_year: Optional[int]=None,
        end_year: Optional[int]=None,
    ):
    """
    Extracts wallet-level features from multiple sources.
    Can also return a dictionary of (wallet, source list) pairs if the
    `return_source_wallet_dict` flag is set to True.
    
    Args:
        sources (int): list of sources (complete absolute or relative path)
        verbose (int): determines whether progress is printed (0: no prints; >=1: print) 
        return_source_wallet_dict (bool): returns a dictionary of (wallet, source list) pairs if set to True
        start_year (int, optional): transactions before `start_year` are discarded
        end_year (int, optional): transactions after `end_year` are discarded
    """
    assert verbose >= 0

    if verbose >= 1:
        print("Building source list for each wallet.")

    if return_source_wallet_dict:
        source_wallet_dict = dict()
    
    dft_list = []
    #  Read each CSV and build dictionary of wallets
    for source in sources:
        if verbose >= 1:
            print(f'Preprocessing source: {source}')
        dft = pd.read_csv(source)
        dft = drop_corrupted_entries(dft)
        dft_list.append(dft)
        if return_source_wallet_dict:
            unique_wallets = dft['walletOfInterest'].unique()
            for wallet in unique_wallets:
                if not wallet in source_wallet_dict.keys():
                    # initialize empty source list for unseen wallets
                    source_wallet_dict[wallet] = []
                source_wallet_dict[wallet].append(source)
        

    if verbose >= 1:
        print("Extracting features for each wallet.")

    df = pd.concat(dft_list)
    df = df.sort_values(by=['walletOfInterest', 'blackList']).reset_index(drop=False)

    df = df.drop_duplicates(subset=['walletOfInterest']).reset_index(drop=True)

    df['year'] = pd.DatetimeIndex(pd.to_datetime(df['unixTime'], unit='s')).year
    if (start_year is not None) and (end_year is not None):
        assert start_year <= end_year
    if start_year is not None:
        df = df[df['year'] >= start_year]
    if end_year is not None:
        df = df[df['year'] <= end_year]
    df = df.drop(columns=['year'])

    df = extract_wallet_features(df)

    if return_source_wallet_dict:
        return df, source_wallet_dict
    return df
    