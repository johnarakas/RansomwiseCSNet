import os
import json
import pandas as pd

from datetime import datetime


def drop_corrupted_entries(df):
    """
    Drop entries with EUR or BTC value equal to zero.
    """
    df = df[(df['eurValue'] != 0)&(df['btcValue'] != 0)]#.reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def save_descriptor(descriptor_path: str, features: list):

    descriptor = {
        "id_column": "walletOfInterest",
        "feature_columns": features,
        "time_column": "lastUnixTime",
        "target_column": "blackList"
    }
    assert os.path.isdir(descriptor_path), "descriptor_path is not a valid path."

    with open(os.path.join(descriptor_path, 'descriptor.json'), 'w') as f:
        json.dump(descriptor, f)



def extract_wallet_features(df: pd.DataFrame, start_year: int=None, end_year: int=None, descriptor_path: str | None=None) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): DataFrame of transactions (each row is a transaction).
        start_year (int | None): Start year for transactions (transactions before start_year are discarded). If None, there is no lower limit to the transaction year.
        end_year (int | None): End year for transactions (transactions after end_year are discarded). If None, there is no upper limit to the transaction year.
        descriptor_path (str | None): Path (directory) where the `descriptor.json` file will be saved. If None, the descriptor will not be saved.
    """

    df = drop_corrupted_entries(df)
    df = df.sort_values(by=['walletOfInterest', 'unixTime']).reset_index(drop=True)

    df['year'] = df['unixTime'].apply(lambda unixtime: datetime.fromtimestamp(unixtime).year)

    if start_year is None: start_year = df['year'].min()
    if end_year is None: end_year = df['year'].max()

    df = df[df['year'] >= start_year]
    df = df[df['year'] <= end_year]
    df = df.reset_index(drop=True)

    dfa = pd.DataFrame()

    # extract overall features
    dfa = df.groupby('walletOfInterest').agg(
        firstUnixTime=pd.NamedAgg(column="unixTime", aggfunc='min'),
        lastUnixTime=pd.NamedAgg(column="unixTime", aggfunc='max'),
        transaction_count=pd.NamedAgg(column="unixTime", aggfunc=lambda x: len(x)),
        timeDelta_mean=pd.NamedAgg(column='unixTime', aggfunc=lambda x: x.diff().mean()),
        timeDelta_std=pd.NamedAgg(column='unixTime', aggfunc=lambda x: x.diff().std()),
        eurValue_mean=pd.NamedAgg(column='eurValue', aggfunc='mean'),
        btcValue_mean=pd.NamedAgg(column='btcValue', aggfunc='mean'),
        eurValue_std=pd.NamedAgg(column='eurValue', aggfunc='std'),
        btcValue_std=pd.NamedAgg(column='btcValue', aggfunc='std'),
        blackList=pd.NamedAgg(column='blackList', aggfunc=lambda x: x.mode())
    )


    # extract sender features
    dfa_sn = df[~df['isReceiver']].groupby('walletOfInterest').agg(
        transaction_count_sn=pd.NamedAgg(column="unixTime", aggfunc=lambda x: len(x)),
        timeDelta_sn_mean=pd.NamedAgg(column='unixTime', aggfunc=lambda x: x.diff().mean()),
        timeDelta_sn_std=pd.NamedAgg(column='unixTime', aggfunc=lambda x: x.diff().std()),
        eurValue_sn_mean=pd.NamedAgg(column='eurValue', aggfunc='mean'),
        btcValue_sn_mean=pd.NamedAgg(column='btcValue', aggfunc='mean'),
        eurValue_sn_std=pd.NamedAgg(column='eurValue', aggfunc='std'),
        btcValue_sn_std=pd.NamedAgg(column='btcValue', aggfunc='std'),
    )

    # extract receiver features
    dfa_rc = df[df['isReceiver']].groupby('walletOfInterest').agg(
        transaction_count_rc=pd.NamedAgg(column="unixTime", aggfunc=lambda x: len(x)),
        timeDelta_rc_mean=pd.NamedAgg(column='unixTime', aggfunc=lambda x: x.diff().mean()),
        timeDelta_rc_std=pd.NamedAgg(column='unixTime', aggfunc=lambda x: x.diff().std()),
        eurValue_rc_mean=pd.NamedAgg(column='eurValue', aggfunc='mean'),
        btcValue_rc_mean=pd.NamedAgg(column='btcValue', aggfunc='mean'),
        eurValue_rc_std=pd.NamedAgg(column='eurValue', aggfunc='std'),
        btcValue_rc_std=pd.NamedAgg(column='btcValue', aggfunc='std'),
    )

    dfa = pd.merge(dfa, dfa_sn, how='left', on='walletOfInterest')
    dfa = pd.merge(dfa, dfa_rc, how='left', on='walletOfInterest')

    dfa = dfa[dfa['transaction_count'] > 0]

    dfa = dfa.fillna(0)
    dfa = dfa.reset_index(drop=False)

    features = [
            col for col in dfa.columns if col not in ('walletOfInterest', 'blackList', 'firstUnixTime', 'lastUnixTime')
        ]

    # set blackList as last column
    dfa = dfa[
        ['walletOfInterest', 'firstUnixTime', 'lastUnixTime'] + features + ['blackList']
    ]

    if descriptor_path is not None:
        save_descriptor(descriptor_path, features)

    return dfa


def preprocess_sources(sources: list[str], start_year: int | None=None, end_year: int | None=None, descriptor_path: str | None=None):
    """
    Args:
        sources (list[str]): List of paths to preprocessed transaction files in CSV format.
        start_year (int | None): Start year for transactions (transactions before start_year are discarded). If None, there is no lower limit to the transaction year.
        end_year (int | None): End year for transactions (transactions after end_year are discarded). If None, there is no upper limit to the transaction year.
        descriptor_path (str | None): Path (directory) where the `descriptor.json` file will be saved. If None, the descriptor will not be saved.
    """
    
    dfa_list = []
    for source in sources:
        df = pd.read_csv(source)
        dfa_s = extract_wallet_features(df, start_year, end_year, descriptor_path)
        dfa_list.append(dfa_s)
    dfa = pd.concat(dfa_list)

    dfa = dfa.drop_duplicates(subset='walletOfInterest')
    dfa = dfa.reset_index(drop=True)
    return dfa
