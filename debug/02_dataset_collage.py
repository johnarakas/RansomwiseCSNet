"""
Script to make a collage of datasets since someone screwed up the ransomware data.
"""

import pandas as pd


filename_bad = "data/ransomwise.csv"
filename_good = "data/all_benign_aggregated.csv"

outfile_name = "data/ransomwise_allbenign.csv"

df_bad = pd.read_csv(filename_bad)
df_bad = df_bad[df_bad["blackList"]]  # just an extra check

df_good = pd.read_csv(filename_good)
df_good = df_good[~df_good["blackList"]]

# ensure no ransomware wallets are in the good list
df_good = df_good[~df_good["walletOfInterest"].isin(df_bad["walletOfInterest"])]

# remove wallets before start_year
import datetime

start_year = 2015
df_bad = df_bad[df_bad["lastUnixTime"].apply(lambda unixtime: datetime.datetime.fromtimestamp(unixtime).year >= start_year)]
df_good = df_good[df_good["lastUnixTime"].apply(lambda unixtime: datetime.datetime.fromtimestamp(unixtime).year >= start_year)]

# print(df_bad)
# print(df_good)

assert all(df_bad["blackList"])
assert all(~df_good["blackList"])

df = pd.concat([df_good, df_bad]).reset_index(drop=True)
df.to_csv(outfile_name, index=False)