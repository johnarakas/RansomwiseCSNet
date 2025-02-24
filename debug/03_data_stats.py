import pandas as pd
import matplotlib.pyplot as plt

def format_filename_to_title(filename: str):
    return filename.split('/')[-1].capitalize().replace('_', ' ')

def remove_outliers(values: pd.Series, iqr_threshold: float=1.5):
    values = values.copy()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    delta = (q3 - q1) * iqr_threshold

    values = values[values >= q1 - delta]
    values = values[values <= q3 + delta]
    return values

if __name__ == "__main__":

    filename = "data/ransomwise_new.csv"
    df = pd.read_csv(filename)

    features_to_analyze = ['transaction_count', 'transaction_count_sn', 'transaction_count_rc',
       'eurValue_mean', 'eurValue_std', 'eurValue_sn_mean', 'eurValue_sn_std',
       'eurValue_rc_mean', 'eurValue_rc_std', 'btcValue_mean', 'btcValue_std',
       'btcValue_sn_mean', 'btcValue_sn_std', 'btcValue_rc_mean',
       'btcValue_rc_std', 
       'timeDelta_mean', 'timeDelta_std',
       'timeDelta_sn_mean', 'timeDelta_sn_std', 'timeDelta_rc_mean',
       'timeDelta_rc_std'
       ]


    print(filename)

    for feature in features_to_analyze:
        
        n_nonzero = len(df[df[feature] != 0])
        print(f"Wallets with non-zero {feature}: {n_nonzero}")

    for tx_threshold in (1, 2, 10, 100):
        n_multiple_tx = len(df[df['transaction_count'] > tx_threshold])

        print(f"Wallets with more than {tx_threshold} transactions: {n_multiple_tx}")

        val = df[feature]
        val = remove_outliers(val)
        
        plt.hist(val, bins=20)
        plt.ylabel(feature)
        plt.draw()

    plt.show()