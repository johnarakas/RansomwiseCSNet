import numpy as np
import pandas as pd
from typing import List, Tuple
from sktime.forecasting.base import ForecastingHorizon

from sktime.split import SlidingWindowSplitter

from itertools import chain

#################################
# Temporal Evaluation Protocols #
#################################

def get_windows(col_to_split, cv, col_split_values):
    """Generate windows"""
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(col_to_split)):
        train_ids = [col_split_values[col_split_values == col_to_split[year_id]].index for year_id in train]
        test_ids = [col_split_values[col_split_values == col_to_split[year_id]].index for year_id in test]
        train_windows.append(list(chain(*train_ids)))
        test_windows.append(list(chain(*test_ids)))
    return train_windows, test_windows

def temp_split(
        years: List[str],
        window_length: int = 3,
        horizon : List[int] = None
):
    if horizon is None:
        horizon = [1]

    fh = ForecastingHorizon(horizon)

    cv = SlidingWindowSplitter(window_length=window_length, fh=fh)

    col_to_split_unq_vals = np.unique(np.array(years))

    n_splits = cv.get_n_splits(col_to_split_unq_vals)
    print(f"Number of Folds = {n_splits}")

    train_windows_ids, test_windows_ids = get_windows(col_to_split_unq_vals, cv, years)

    return train_windows_ids, test_windows_ids

#####################################
# Non-Temporal Evaluation Protocols #
#####################################

def leave_one_year_out(
    df: pd.DataFrame,
    time_col: str,
):
    train_ids = []
    test_ids = []
    years = np.unique(df[time_col])
    for year in years:
        train_ids.append(list(chain(df[df[time_col] != year].index)))
        test_ids.append(list(chain(df[df[time_col] == year].index)))
    return train_ids, test_ids


