from pathlib import Path
from typing import List

# External libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score
import seaborn as sns

# Local modules
from src.evaluation_protocols import temp_split
from src.path_utils import DATASET_FILE

metrics = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score, 'balanced_accuracy': balanced_accuracy_score}
time_column = 'lastUnixTime'
target_column = 'blackList'
not_feature_columns = ['walletOfInterest', 'blackList', 'firstUnixTime', 'lastUnixTime', 'year']


def get_valid_ids_from_years(years: List[int], df: pd.DataFrame):
	ids = []
	for y in years:
		ids.extend(df[df['year'] == y].index)
	return ids

def line_plot_per_year(df_per_year, xlabel, ylabel, fname, ylim=None):
	df_per_year.plot.line(rot=60, marker='o')
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	if ylim is not None:
		plt.ylim(ylim)
	plt.tight_layout()
	folder = 'figures'
	Path(folder).mkdir(exist_ok=True, parents=True)
	plt.savefig(Path(folder, fname))
	plt.close()


def heatmap_per_year(df_per_year, xlabel, ylabel, fname):
	sns.heatmap(df_per_year)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.xticks(rotation=30)
	plt.tight_layout()
	folder = 'figures'
	Path(folder).mkdir(exist_ok=True, parents=True)
	plt.savefig(Path(folder, fname))
	plt.close()


def eval_models_per_year(train_ids, test_ids, perfs, importances):

	rf = RandomForestClassifier(max_depth=3, random_state=42)
	dt = DecisionTreeClassifier(max_depth=3, random_state=42)
	xgb = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=42)

	train_year_df = df.loc[train_ids]
	test_year_df = df.loc[test_ids]

	train_years = np.unique(train_year_df['year'])
	if len(train_years) > 1:
		train_years_key = str(train_years[0]) + '-' + str(train_years[-1])
	else:
		train_years_key = str(train_years[0])
	test_year_key = str(np.unique(test_year_df['year'])[0])

	print('Train in years:', train_years, 'Test in years:', test_year_key)

	X_curr, y_curr = train_year_df.drop(columns=not_feature_columns), train_year_df[target_column]
	X_next, y_next = test_year_df.drop(columns=not_feature_columns), test_year_df[target_column]

	rf.fit(X_curr, y_curr)
	dt.fit(X_curr, y_curr)
	xgb.fit(X_curr, y_curr)

	preds_rf = rf.predict(X_next)
	preds_dt = dt.predict(X_next)
	preds_xgb = xgb.predict(X_next)

	perfs[train_years_key + '/' + test_year_key] = {'Random Forest': balanced_accuracy_score(y_next, preds_rf),
													'Decision Tree': balanced_accuracy_score(y_next, preds_dt),
													'Gradient Boosting': balanced_accuracy_score(y_next, preds_xgb)}

	importances.append(pd.DataFrame(dt.feature_importances_, columns=[train_years_key], index=X_curr.columns))

	importances_df = pd.concat(importances, axis=1)
	return pd.DataFrame(perfs), importances_df



if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_file', type=str, default=DATASET_FILE)
	args = parser.parse_args()

	train_years_per_fold = [1, 2, 3]

	df = pd.read_csv(args.dataset_file)

	cumulative_rolling_cv = False

	fname_suffix = 'agg' if cumulative_rolling_cv else 'non_agg'

	df['year'] = pd.DatetimeIndex(pd.to_datetime(df[time_column], unit='s')).year

	for years_per_fold_num in train_years_per_fold:

		perfs = {}
		importances = []
		aggregated_ids = []

		train_win_ids, test_win_ids = temp_split(years=df['year'], window_length=years_per_fold_num, horizon=[1])

		for curr_train_ids, curr_test_ids in zip(train_win_ids, test_win_ids):

			curr_train_ids = [*aggregated_ids, *curr_train_ids]

			if cumulative_rolling_cv:
				aggregated_ids = curr_train_ids.copy()

			perfs_df, importances_df = eval_models_per_year(train_ids=curr_train_ids, test_ids=curr_test_ids, perfs=perfs, importances=importances)
			line_plot_per_year(perfs_df.T, xlabel = 'Train Year(s) / Test Year', ylabel = 'Balanced Accuracy',
							   fname = f'bal_acc_w{years_per_fold_num}_1_{fname_suffix}.pdf', ylim = (0, 1))
			heatmap_per_year(importances_df, xlabel='Train Year(s)', ylabel='Feature Importance',
							 fname=f'feat_imp_w{years_per_fold_num}_1_{fname_suffix}.pdf')
