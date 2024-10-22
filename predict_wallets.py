import os
import sys
import warnings

# External libraries
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Local modules
sys.path.append('.')

from data.preprocess_json import preprocess_json

from data_utils import preprocess_sources
from ml_utils import compute_binary_classification_metrics


## USER PARAMETERS

USE_EXISTING_METADATA = True

hparams = {
    'n_estimators': 1,
    'max_depth': 100,
    'criterion': 'gini',
}

# other training specifics
test_size = 0.3  # fraction of dataset used for test
save_results = True  # set to false when making changes

metrics = ['accuracy', 'precision', 'recall', 'balanced_accuracy']
time_column = 'lastUnixTime'
target_column = 'blackList'
not_feature_columns = ['walletOfInterest', 'blackList', 'firstUnixTime', 'lastUnixTime']


# Paths
OUTPATH = "data/results"

TRAIN_BENIGN_PATH = "data/json/train/benign"
TRAIN_MALICIOUS_PATH = "data/json/train/malicious"
PREDICT_PATH = "data/json/predict"
METADATA_PATH = "data/metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_PATH, "train")
TRAIN_METADATA_BENIGN_PATH = os.path.join(TRAIN_METADATA_PATH, "benign")
TRAIN_METADATA_MALICIOUS_PATH = os.path.join(TRAIN_METADATA_PATH, "malicious")
PREDICT_METADATA_PATH = os.path.join(METADATA_PATH, "predict")



def init_paths():
    
    dirlist = [OUTPATH, METADATA_PATH, TRAIN_METADATA_PATH, TRAIN_METADATA_BENIGN_PATH, TRAIN_METADATA_MALICIOUS_PATH, PREDICT_METADATA_PATH]

    for dirname in dirlist:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
    return


def get_wallets_from_path(sourcepath: str, metadatapath: str, malicious: bool, use_existing_metadata: bool=USE_EXISTING_METADATA):
    all_wallets = []
    for filename in os.listdir(sourcepath):
        if filename.endswith(".json"):
            wallets = preprocess_json(
                os.path.join(sourcepath, filename),
                metadatapath,
                malicious=malicious,
                use_existing_metadata=use_existing_metadata,
                )
            all_wallets += wallets
    all_wallets = list(set(all_wallets))
    return all_wallets


if __name__ == "__main__":

    init_paths()

    train_wallets_benign = get_wallets_from_path(TRAIN_BENIGN_PATH, TRAIN_METADATA_BENIGN_PATH, malicious=False)
    train_wallets_malicious = get_wallets_from_path(TRAIN_MALICIOUS_PATH, TRAIN_METADATA_MALICIOUS_PATH, malicious=True)
    overlap_benign_malicious = list(set(train_wallets_benign).intersection(set(train_wallets_malicious)))
    if len(overlap_benign_malicious) > 0:
        print("Overlapping between benign and malicious wallets in the training data:", overlap_benign_malicious)

    # set label to malicious by default, this should be ignored!
    predict_wallets = get_wallets_from_path(PREDICT_PATH, PREDICT_METADATA_PATH, malicious=True)  

    
    train_sources = [os.path.join(TRAIN_METADATA_BENIGN_PATH, filename) for filename in os.listdir(TRAIN_METADATA_BENIGN_PATH)]
    train_sources += [os.path.join(TRAIN_METADATA_MALICIOUS_PATH, filename) for filename in os.listdir(TRAIN_METADATA_MALICIOUS_PATH)]

    pred_sources = [os.path.join(PREDICT_METADATA_PATH, filename) for filename in os.listdir(PREDICT_METADATA_PATH)]

    aggregated_train_file = os.path.join(TRAIN_METADATA_PATH, "aggregated_train.csv")
    aggregated_predict_file = os.path.join(TRAIN_METADATA_PATH, "aggregated_predict.csv")

    if os.path.isfile(aggregated_train_file) and USE_EXISTING_METADATA:
        warnings.warn("Using existing metadata for aggregated training dataframe")
        df_train = pd.read_csv(aggregated_train_file)
    else:
        df_train = preprocess_sources(train_sources)
        df_train.to_csv(aggregated_train_file, index=False)

    if os.path.isfile(aggregated_predict_file) and USE_EXISTING_METADATA:
        warnings.warn("Using existing metadata for aggregated prediction dataframe")
        df_pred = pd.read_csv(aggregated_predict_file)
    else:
        df_pred = preprocess_sources(pred_sources)
        df_pred.to_csv(aggregated_predict_file, index=False)

    # TODO: Add routine for model evaluation
        
    feature_names = [col for col in df_train.columns if not col in not_feature_columns]
    assert target_column not in feature_names  # bug-checker

    X_train = df_train[feature_names]
    y_train = df_train[target_column]
    X_pred = df_pred[feature_names]

    if hparams['n_estimators'] > 1:
        model = RandomForestClassifier(n_estimators=hparams['n_estimators'], 
                                       max_depth=hparams['max_depth'], 
                                       criterion=hparams['criterion'], 
                                       random_state=42)
    else:
        model = DecisionTreeClassifier(max_depth=hparams['max_depth'], 
                                       criterion=hparams['criterion'],
                                       random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_pred)

    n_pred_malicious = len(y_pred[y_pred==1])
    n_pred = len(y_pred)

    print(f'Total benign wallets used for training: {len(df_train[~df_train[target_column]])}')
    print(f'Total malicious wallets used for training: {len(df_train[df_train[target_column]])}')
    print(f'Total wallets to be predicted: {len(df_pred)}')

    df_pred[target_column] = y_pred
    df_pred['inTraining'] = df_pred['walletOfInterest'].isin(df_train['walletOfInterest'])
    n_overlap = len(df_pred[df_pred['inTraining']])
    print(f'Number of overlapping wallets between training and prediction: {n_overlap} ({n_overlap/len(df_pred)*100:.2f}%)')


    print(f"Number of wallets predicted as malicious: {n_pred_malicious} ({n_pred_malicious/n_pred*100:.2f}%)")
    outfile = os.path.join(OUTPATH, 'predictions.csv')
    
    df_pred[['walletOfInterest', 'inTraining', target_column]].to_csv(outfile, index=False)
    print(f"Predictions saved in: {outfile}")

