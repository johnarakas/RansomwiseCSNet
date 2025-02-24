import pandas as pd

if __name__ == "__main__":

    sources = [
        # 'data/transactions/benignTransactions.csv',
        'data/transactions/kaggleTransactions.csv',
        'data/transactions/randomBenignTransactions.csv'
    ]

    source_wallets = {source: [] for source in sources}
    for source in sources:
        df = pd.read_csv(source)
        wallet_list = df['walletOfInterest'].unique().tolist()
        source_wallets[source] = wallet_list
        print(f"{source}: {len(wallet_list)} wallets")

    intersection = list(set(source_wallets[sources[0]]).intersection(set(source_wallets[sources[1]])))
    print(f"Overlap: {len(intersection)} wallets")

    
        
