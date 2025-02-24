import os

import pandas as pd

from src.path_utils import RESULTS_PATH

CV_RESULT_FILE = os.path.join(RESULTS_PATH, 'results_cv_d3.csv')

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=RESULTS_PATH)
    parser.add_argument("--cv_result_file", type=str, default=CV_RESULT_FILE)
    args = parser.parse_args()

    output_path = args.output_path
    cv_result_file = args.cv_result_file

    df = pd.read_csv(cv_result_file)

    dfa = df.groupby('model').agg(
        mean_accuracy=pd.NamedAgg(column='accuracy', aggfunc='mean'),
        mean_precision_pos=pd.NamedAgg(column='precision_pos', aggfunc='mean'),
        mean_precision_neg=pd.NamedAgg(column='precision_neg', aggfunc='mean'),
        mean_recall_pos=pd.NamedAgg(column='recall_pos', aggfunc='mean'),
        mean_recall_neg=pd.NamedAgg(column='recall_neg', aggfunc='mean'),
        mean_balanced_accuracy=pd.NamedAgg(column='balanced_accuracy', aggfunc='mean'),
        std_accuracy=pd.NamedAgg(column='accuracy', aggfunc='std'),
        std_precision_pos=pd.NamedAgg(column='precision_pos', aggfunc='std'),
        std_precision_neg=pd.NamedAgg(column='precision_neg', aggfunc='std'),
        std_recall_pos=pd.NamedAgg(column='recall_pos', aggfunc='std'),
        std_recall_neg=pd.NamedAgg(column='recall_neg', aggfunc='std'),
        std_balanced_accuracy=pd.NamedAgg(column='balanced_accuracy', aggfunc='std')
    )

    dfstats = pd.DataFrame()


    metrics = ['accuracy', 'precision_pos', 'precision_neg', 'recall_pos', 'recall_neg', 'balanced_accuracy']

    for metric in metrics:
        dfstats[metric.replace('_', ' ')] = dfa.apply(lambda x: f'{x[f"mean_{metric}"]:.4f} ({x[f"std_{metric}"]:.4f})', axis=1)


    outfile = cv_result_file.replace('.csv', '_stats.tex').split('/')[-1]
    dfstats.to_latex(os.path.join(output_path, outfile), index=True)

