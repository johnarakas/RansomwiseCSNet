# Ransomwise: A Transaction-Based Dataset for Ransomware Wallet Identification 

## Dataset

The dataset can be found in [data/ransomwise_csnet24.csv](data/ransomwise_csnet24.csv)

## Project structure

```
.
├── data
│   ├── json
│   ├── transactions
│   ├── descriptor.json
│   ├── generate_ransomwise.py
│   ├── preprocess_json_transactions.py
│   └── ransomwise.csv
├── evaluation
│   ├── cross_validation.py
│   └── temporal.py
├── figures
├── plots
│   ├── distribution_eurValue.py
│   ├── gini_importance_kfold.py
│   ├── misclassification_by_year.py
│   ├── precision_recall.py
│   └── tree_rules.py
├── results
├── saved_models
├── scripts
│   ├── clear_results.sh
│   └── init_dirs.sh
├── src
│   ├── data_utils.py
│   ├── evaluation_protocols.py
│   ├── feature_importance.py
│   ├── ml_utils.py
│   ├── path_utils.py
│   └── train_tree.py
├── CHANGELOG.md
├── conda_env.txt
├── config.ini
├── README.md
├── requirements.txt
└── run.sh
```

## Python environment 

Using [Conda](https://conda-forge.org/):
```bash
conda create --name ransomwise --file conda_env.txt
```

Using [PIP](https://pypi.org/project/pip/):
```bash
pip install -r requirements.txt
```


## Generate dataset from transaction data

From JSON transactions (see [README](data/transactions/README.md)):
- place the JSON files in `data/transaction/json`
- modify the `data/preprocess_json_transactions.py`
- while in the root directory of the project, run: 
```bash
python -m data.preprocess_json_transactions
```
- verify that CSV files have been created in the `data/transactions` directory (see [README](data/transactions/README.md))

From CSV transactions: 
- while in the root directory of the project, run:
```bash
python -m data.generate_ransomwise
```
- verify that `data/ransomwise.csv` has been created
- verify that `data/descriptor.json` has been created

## Run evaluation protocols

K-fold cross-validation:
```bash
python -m evaluation.cross_validation [--dataset_file DATASET_FILE]
```

Temporal protocols:
```bash
python -m evaluation.temporal [--dataset_file DATASET_FILE]
```

To reproduce the results of the paper published in CSNet '24, set `ransomwise_csnet24.csv` as dataset file.


## Citation

To cite this work, please use the following BibTeX entry:

```tex
@inproceedings{arakas2024ransomwise,
  title={Ransomwise: A Transaction-Based Dataset for Ransomware Bitcoin Wallet Detection},
  author={Arakas, Ioannis and Myrtakis, Nikolaos and Marchioro, Thomas and Markatos, Evangelos},
  booktitle={2024 8th Cyber Security in Networking Conference (CSNet)},
  pages={100--107},
  year={2024},
  organization={IEEE}
}
```

### Notes on data integrity checks

- `data.preprocess_json_transactions`:
    - measure overlap between benign and ransomware wallets

- `data.preprocess_json_transactions`:
    - drop corrupted transactions (euro or btc value == 0)
    - drop transactions before start_year and after end_year (if these are not None)
    - NaNs are filled with 0 (in aggregated wallet data)
    - drop wallets with 0 transaction_count
- `src.data_utils.preprocess_sources`: (runs `data_utils.extract_wallet_features` on multiple files and concats them)
    - drop duplicate wallets

