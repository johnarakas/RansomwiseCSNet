import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_protocols import rolling_cv

def draw_bar_plot(df_to_plot, y_label, legend=True, ylim=None):
	df_to_plot.plot.bar(rot=0)
	if legend:
		plt.legend()
	plt.ylabel(y_label)
	plt.yscale('log')
	if ylim is not None:
		plt.ylim(ylim)
	plt.show()

metrics = ['accuracy', 'precision', 'recall', 'balanced_accuracy']
time_column = 'lastUnixTime'
target_column = 'blackList'
exclude_columns = ['walletOfInterest']

if __name__ == '__main__':
	test_size = 0.3  # fraction of dataset used for test

	df = pd.read_csv('data/aggregated.csv')

	df['year'] = pd.DatetimeIndex(pd.to_datetime(df[time_column], unit='s')).year

	# Bar plot for sample size per year
	sample_size_per_year = df['year'].value_counts().sort_index()

	draw_bar_plot(sample_size_per_year, y_label='Sample Size', legend=False)

	# Bar plot for class count per year
	class_count_per_year = df.groupby(by='year')[target_column].value_counts().reset_index().set_index('year')
	is_mal = class_count_per_year[target_column].values
	mal_count_per_year = class_count_per_year[is_mal][['count']].rename(columns={'count': 'Malicious'})
	ben_count_per_year = class_count_per_year[~is_mal][['count']].rename(columns={'count': 'Benign'})
	class_count_per_year = pd.concat([mal_count_per_year, ben_count_per_year], axis=1)

	draw_bar_plot(class_count_per_year, y_label='Class Count', legend=True)

	parts = rolling_cv(
		df,
		time_col='year',
		num_splits=-1,
		time_split_values=np.unique(df['year']),
		trainset_expansion=False,
		test_size_frac=0,
		stratify=True,
		stratify_col_name=target_column
		)

	stats_train = {'mean': [], 'std': []}
	stats_test = {'mean': [], 'std': []}

	xlabels = []

	for p_train, p_test in parts:
		time_vals = [*p_train['year'].values, *p_test['year'].values]
		xlabels.append(min(time_vals))
		p_train = p_train.drop(columns=exclude_columns)
		p_test = p_test.drop(columns=exclude_columns)
		stats_train['mean'].append(p_train[p_train[target_column] == False].mean(axis=0))
		stats_test['mean'].append(p_train[p_train[target_column] == True].mean(axis=0))
		stats_train['std'].append(p_train[p_train[target_column] == False].std(axis=0))
		stats_test['std'].append(p_train[p_train[target_column] == True].std(axis=0))

	# columns_to_plot = ['usdValue_sn_mean', 'usdValue_sn_std', 'usdValue_rc_std', 'usdValue_rc_mean']
	columns_to_plot = ['btcValue_sn_mean', 'btcValue_sn_std', 'btcValue_rc_std', 'btcValue_rc_mean']

	mean_train_stats_df = pd.DataFrame(stats_train['mean'], index=xlabels)[columns_to_plot]
	std_train_stats_df = pd.DataFrame(stats_train['std'], index=xlabels)[columns_to_plot]
	mean_test_stats_df = pd.DataFrame(stats_test['mean'], index=xlabels)[columns_to_plot]
	std_test_stats_df = pd.DataFrame(stats_test['std'], index=xlabels)[columns_to_plot]
	print()

	# columns_to_plot()

	mean_train_stats_df.plot.line()
	plt.semilogy()
	plt.title('Mean Benign Values')
	plt.ylabel('Value (log scale)')
	plt.show()

	mean_test_stats_df.plot.line()
	plt.semilogy()
	plt.title('Mean Malicious Values')
	plt.ylabel('Value (log scale)')
	plt.show()

	std_train_stats_df.plot.line()
	plt.semilogy()
	plt.title('Std Malicious Values')
	plt.ylabel('Value (log scale)')
	plt.show()

	std_test_stats_df.plot.line()
	plt.semilogy()
	plt.title('Std Malicious Values')
	plt.ylabel('Value (log scale)')
	plt.show()