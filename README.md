# Ransomwise

This repo contains data and scripts that were used for the paper "RANSOMWISE: A Transaction-Based Dataset for Ransomware Bitcoin Wallet Detection" 


## Preliminary steps
- Clone the repo and navigate to the main directory
- Install the required libraries using `pip install -r requirements.txt` (Recommended: use a virtual environment https://docs.python.org/3/library/venv.html)
- Run `bash init_dirs.sh`: this will create the directories `data/json`, `data/json/train`, `data/json/train/benign`, `data/json/malicious`, and `data/json/predict`
- Copy the training data (in JSON format) for benign and malicious wallets in the directories `data/json/train/benign` and `data/json/malicious`
- Copy the data (in JSON format) for the wallets to be classified in `data/json/predict`

## Run prediction
- Run `python predict_wallets.py`. This should produce an output such as:
```
Total benign wallets used for training: 8604
Total malicious wallets used for training: 25397
Total wallets to be predicted: 1000
Number of overlapping wallets between training and prediction: 0 (0.00%)
Number of wallets predicted as malicious: 999 (99.90%)
Predictions saved in: data/results/predictions.csv
```

## Notes on metadata
- By default, many intermediate steps are stored as metadata to speed up the prediction process. If you need to overwrite the metadata (e.g., because some files were modified), you should either delete the `metadata` directory or set the variable `USE_EXISTING_METADATA` to `False` 