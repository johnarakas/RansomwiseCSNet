from datetime import datetime

import pandas as pd


def drop_corrupted_entries(df):
    """
    Drop entries with EUR or BTC value equal to zero.
    """
    df = df[(df['eurValue'] != 0)&(df['btcValue'] != 0)]#.reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def extract_wallet_features(df: pd.DataFrame, start_year: int=None, end_year: int=None) -> pd.DataFrame:

    df = drop_corrupted_entries(df)

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

    dfa = dfa.fillna(0)
    dfa = dfa.reset_index(drop=False)

    # set blackList as last column
    dfa = dfa[
        ['walletOfInterest'] + [
            col for col in dfa.columns if col not in ('walletOfInterest', 'blackList')
        ] + ['blackList']
    ]

    return dfa

if __name__ == "__main__":

    filename = "data/randomBenignTransactions.csv"
    start_year = 2015
    end_year = 2024

    df = pd.read_csv(filename)

    dfa = extract_wallet_features(df)
    print(dfa)
