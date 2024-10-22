#---------- Python libraries ------------
import os
import sys

#------------ Local modules -------------
sys.path.append('.')
from path_utils import data_path
from data_utils import preprocess_sources

# paths to the sources to be included in the aggregated dataset
# sources = [
#     # 'data/Alienvault.csv',
#     # 'data/Behas.csv',
#     # 'data/ChainAbuse.csv',
#     # 'data/Irvine.csv',
#     # 'data/KillingTheBear.csv',
#     # 'data/Ransomlook.csv',
#     # 'data/Ransomwhere.csv',
#     # 'data/SophosLab.csv',
#     # 'data/Tessii.csv',
#     # 'data/Traceer.csv',
#     'data/maliciousTransactions.csv',
#     # 'data/benignTransactions.csv',
#     'data/kaggle.csv',
#     # 'data/ofac.csv'
#     'data/newBenignTransactions.csv',
#     # 'bitcoinAbuseTotal.csv'
# ]

sources = [
    'data/maliciousTransactions.csv',
    'data/kaggle.csv',
    'data/newBenignTransactions.csv',
]

outfile = 'aggregated.csv'  # will be saved in data_path/outfile
# outfile = 'bitcoinAbuse.csv'  # will be saved in data_path/outfile

if __name__ == "__main__":

    # df = preprocess_sources(sources)
    df = preprocess_sources(sources, start_year=2015, end_year=2024)

    
    print(df['blackList'].value_counts())

    df.to_csv(os.path.join(data_path, outfile), index=False)

    