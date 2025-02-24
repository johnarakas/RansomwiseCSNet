import pandas as pd

filename_random = "data/transactions/randomBenignTransactions.csv"
filename_big = "data/transactions/benignTransactions.csv"

if __name__ == "__main__":

    df_rnd = pd.read_csv(filename_random)
    df_big = pd.read_csv(filename_big)

    df_rnd = df_rnd[~df_rnd['walletOfInterest'].isin(df_big['walletOfInterest'])]

    df_rnd.to_csv(filename_random, index=False)