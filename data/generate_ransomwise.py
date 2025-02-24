"""
Script to generate ransomwise dataset.

Usage:
```
python generate_ransomwise.py [--datapath DATAPATH] [--outpath OUTPATH]
```

Arguments:
- `--datapath`: The path to the data directory. Default is "data/transactions".
- `--outpath`: The path to the output file. Default is "data/ransomwise.csv".
- `--start_year`: The start year for the dataset. The dataset will include transactions starting from start_year (included). Default is 2015.
- `--end_year`: The end year for the dataset. The dataset will include transactions up to end_year (included). Default is 2024.
"""

import os
import sys

# sys.path.append(".")
from src.data_utils import preprocess_sources

DATA_PATH = 'data/transactions'
OUTFILE = 'data/ransomwise.csv'
START_YEAR = 2015
END_YEAR = 2024

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default=DATA_PATH)
    parser.add_argument('--outfile', type=str, default=OUTFILE)
    parser.add_argument('--start_year', type=int, default=START_YEAR)
    parser.add_argument('--end_year', type=int, default=END_YEAR)
    args = parser.parse_args()

    data_path = args.datapath
    outfile = args.outfile
    start_year = args.start_year
    end_year = args.end_year

    assert os.path.isdir(data_path), "Data path does not exist"
    assert os.path.isdir(os.path.dirname(outfile)), "Output directory does not exist"

    sources = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.csv')]
    df = preprocess_sources(sources, start_year=start_year, end_year=end_year, descriptor_path="data")
    
    print("Class distribution:")
    print(df['blackList'].value_counts())

    print(f"Saving dataset to {outfile}")
    df.to_csv(outfile, index=False)
    print(f"Dataset successfully saved to {outfile}")
