import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_protocols import temp_split, leave_one_year_out

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, average_precision_score


metrics = ['accuracy', 'precision', 'recall', 'balanced_accuracy']
time_column = 'lastUnixTime'
target_column = 'blackList'
not_feature_columns = ['walletOfInterest', 'blackList', 'firstUnixTime', 'lastUnixTime', 'year']

if __name__ == '__main__':

	df = pd.read_csv('data/aggregated.csv')

	df['year'] = pd.DatetimeIndex(pd.to_datetime(df[time_column], unit='s')).year

	# Bar plot for sample size per year
	sample_size_per_year = df['year'].value_counts().sort_index()


	# draw_bar_plot(class_count_per_year, y_label='Class Count', legend=True)

	# train_ids, test_ids = temp_split(df, 'year', window_length=3, horizon=[1])
	train_ids, test_ids = leave_one_year_out(df, 'year')

	lag = 1 # the lag is measured in years

	models = {}
	perfs = {}

	columns = None

	max_depth = 3

	for train_ids, test_ids in zip(train_ids, test_ids):
		rf = RandomForestClassifier(max_depth=max_depth, random_state=42)
		dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

		train_year_df = df.loc[train_ids]
		test_year_df = df.loc[test_ids]

		train_years = np.unique(train_year_df['year'])
		train_years_key = str(train_years[0]) + '-' + str(train_years[-1])
		test_year_key = str(np.unique(test_year_df['year'])[0])

		X_curr, y_curr = train_year_df.drop(columns=not_feature_columns), train_year_df[target_column]
		X_next, y_next = test_year_df.drop(columns=not_feature_columns), test_year_df[target_column]

		if columns is None:
			columns = X_curr.columns

		rf.fit(X_curr, y_curr)
		dt.fit(X_curr, y_curr)

		preds_rf = rf.predict(X_next)
		preds_dt = dt.predict(X_next)

		models[train_years_key + '/' + test_year_key] = {'rf': rf, 'dt': dt}
		perfs[train_years_key + '/' + test_year_key] = {'rf': roc_auc_score(y_next, preds_rf), 'dt': roc_auc_score(y_next, preds_dt)}

		importances = dt.feature_importances_
		#
		# t_id = test_ids[counter]
		# train_ids = np.where(train_labels == y_preds[t_id])[0]
		# axes[i, j].hist(inf_mat[train_ids, t_id])
		# axes[i, j].set_title(y_preds[t_id])


		# x_range = np.arange(X_curr.shape[1])
		# plt.bar(x_range, height=importances)
		# plt.xticks(x_range, X_curr.columns, rotation=90)
		# plt.title(f'Train: {curr_year}, Test: {next_year}')
		# plt.tight_layout()
		# plt.draw()

		# plt.show()
	print()

	fig, axes = plt.subplots(3, 4, figsize=(18, 12))
	# models_iter = iter(models)
	# for i in range(4):
	# 	for j in range(4):
	# 		curr_year = next(models_iter, None)
	# 		if curr_year is None:
	# 			break
	# 		x_range = np.arange(len(columns))
	# 		axes[i, j].bar(x_range, height=models[curr_year]['dt'].feature_importances_)
	# 		axes[i, j].set_xticks(x_range, columns, rotation=90)
	# 		axes[i, j].set_title(f'Train: {curr_year}, Test: {curr_year + 1}, AUC {np.around(perfs[curr_year]["dt"],2)}')
	# plt.suptitle('Feature Importance per year')
	# plt.tight_layout()
	# plt.savefig('fimp.png', dpi=300)
	# plt.show()

	kind = 'line'

	perfs_df = pd.DataFrame.from_dict(perfs).T
	perfs_df['Random Classifier'] = 0.5
	if kind == 'bar':
		ax = perfs_df.plot(rot=90, kind=kind, color=['tab:blue', 'tab:orange', 'tab:red'])
	else:
		ax = perfs_df.plot(rot=90, kind=kind, marker='o', color=['tab:blue', 'tab:orange', 'tab:red'])
		ax.get_lines()[-1].set_marker('')
		ax.get_lines()[-1].set_color('red')
	plt.ylabel('AUC')
	plt.title(f'Max Depth per Tree: {max_depth}')
	plt.xlabel('Train Start-End Year / Test Year')
	plt.ylim((0, 1))
	# plt.xticks(list(perfs.keys()))
	plt.tight_layout()
	plt.show()
