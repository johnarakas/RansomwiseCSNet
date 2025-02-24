import json
import os

# External libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Local modules
from src.path_utils import DATASET_FILE, DESCRIPTOR_FILE, MODEL_SAVE_PATH, RESULTS_PATH, FIGURES_PATH
from sklearn.ensemble import GradientBoostingClassifier
from src.ml_utils import temporal_train_test_split

"""
Hyperparameters Tested:

- Gradient Boosting:
    learning_rate: [0.05, 0.1, 0.2, 1.0]
    max_depth: [1, 5, 10]
    n_estimators: [50, 100]

The optimal hyperparameters, based on a 10% validation set, among the combinations above
are shown in the classifier initialization below.
"""

# other training specifics
TEST_SIZE = 0.3  # fraction of dataset used for test

USE_TIME_SPLIT = False

metrics = ['accuracy', 'precision', 'recall', 'specificity', 'balanced_accuracy']

# TODO: Try different hyperparameters to achieve better precision-recall tradeoff

models = [
    MLPClassifier(
        hidden_layer_sizes=(64,32), 
        max_iter=100, 
        batch_size=512,
        activation="relu",
        random_state=42,
    ),
    GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=10, 
        random_state=0
    ),
]

def plot_pr_curve(model, X, y , save_path=None):
    model_name = model.__class__.__name__

    y_scores = model.predict_proba(X)[:, 1]

    precision, recall, _ = precision_recall_curve(y, y_scores)
    average_precision = average_precision_score(y, y_scores)

    print('Average precision score: {0:0.2f}'.format(average_precision))

    plt.plot(recall, precision, 'k-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.grid(linestyle=':')
    # plt.title(f'Precision-Recall Curve {model_name}')
    

    if save_path is not None:
        outfile = os.path.join(save_path, f'pr_curve_{model_name}.pdf')
        plt.savefig(outfile)
    plt.show()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_time_split", default=USE_TIME_SPLIT, type=bool)
    parser.add_argument("--test_size", default=0.3, type=float)
    parser.add_argument("--dataset_file", type=str, default=DATASET_FILE)
    parser.add_argument("--descriptor_file", type=str, default=DESCRIPTOR_FILE)
    parser.add_argument("--results_path", type=str, default=RESULTS_PATH)
    parser.add_argument("--model_save_path", type=str, default=MODEL_SAVE_PATH)
    args = parser.parse_args()


    dataset_file = args.dataset_file
    descriptor_file = args.descriptor_file
    model_save_path = args.model_save_path
    results_path = args.results_path
    use_time_split = args.use_time_split
    test_size = args.test_size

    np.random.seed(42)

    df = pd.read_csv(dataset_file)
    with open(descriptor_file, 'r') as f: 
        descriptor = json.load(f)

    feature_names = descriptor['feature_columns']
    time_column = descriptor['time_column']
    target_column = descriptor['target_column']


    if use_time_split:
        df_train, df_test = temporal_train_test_split(df, time_column, test_size=test_size)

    else:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)

    X_train = df_train[feature_names]
    y_train = df_train[target_column]

    X_test = df_test[feature_names]
    y_test = df_test[target_column]

    for model in models:
        print("Testing model:", model.__class__.__name__)

        model.fit(X_train, y_train)
        plot_pr_curve(model, X_test, y_test, save_path=FIGURES_PATH)
    
    


    
    