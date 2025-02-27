#---------- Python libraries ------------
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

from typing import Optional, Tuple, Union

def _parse_split(train_size: Optional[float]=None, test_size: Optional[float]=None) -> Tuple[float, float]:
    assert (train_size is not None) or (test_size is not None), "Either train_size or test_size must be specified"

    if (train_size is not None) and (test_size is not None):
        assert (train_size > 0.) and (test_size > 0.), "train_size and test_size must be positive"
        assert train_size + test_size <= 1., "train_size and test_size must sum to 1"
    elif train_size is None:
        train_size = 1. - test_size
    else:  # test_size is None
        test_size = 1. - train_size
    
    return train_size, test_size


def temporal_train_test_split(
        df: pd.DataFrame, 
        time_column: str, 
        train_size: Optional[float]=None, 
        test_size: Optional[float]=None
        ):
    
    assert time_column in df.columns

    train_size, test_size = _parse_split(train_size, test_size)

    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    df_train, df_test = train_test_split(
        df, train_size=train_size, test_size=test_size, shuffle=False,
    )

    return df_train, df_test



def compute_binary_classification_metrics(
        y_true, y_pred, 
        metrics:list, 
        verbose:int=0, 
        is_train:bool=False, 
        positive_class_name='ransomware',  # NOTE: used only for prints
        negative_class_name=None,
        ):
    """
    Args:   
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        metrics (list): List of metrics to compute
        verbose (int, optional): Verbosity level. Defaults to 0.
        is_train (bool, optional): Whether the data is for training or testing. Defaults to False.
        positive_class_name (str, optional): Name displayed for the positive class while printing. Defaults to 'ransomware'.
        negative_class_name (str, optional): Name displayed for the negative class while printing. If None, the negative class name will be 'non-{positive_class_name}'. Defaults to None.
    """

    metrics_dict = dict()

    to_print = []
    if verbose >= 1:
        train_or_test = 'train' if is_train else 'test'
        if negative_class_name is None:
            negative_class_name = f'non-{positive_class_name}'

    if 'accuracy' in metrics:
        metrics_dict['accuracy'] = (y_true == y_pred).mean()
        if verbose >= 1:
            to_print.append(f"{train_or_test.capitalize()} accuracy: {metrics_dict['accuracy']:.4f}")

    if 'precision' in metrics:
        metrics_dict['precision_pos'] = precision_score(y_true, y_pred, pos_label=1)
        metrics_dict['precision_neg'] = precision_score(y_true, y_pred, pos_label=0)
        if verbose >= 1:
            to_print.append(f"{train_or_test.capitalize()} precision for {positive_class_name}: {metrics_dict['precision_pos']:.4f}")
            to_print.append(f"{train_or_test.capitalize()} precision for non-{positive_class_name}: {metrics_dict['precision_neg']:.4f}")

    if 'recall' in metrics:
        metrics_dict['recall_pos'] = recall_score(y_true, y_pred, pos_label=1)
        metrics_dict['recall_neg'] = recall_score(y_true, y_pred, pos_label=0)
        if verbose >= 1:
            to_print.append(f"{train_or_test.capitalize()} recall for {positive_class_name}: {metrics_dict['recall_pos']:.4f}")
            to_print.append(f"{train_or_test.capitalize()} recall for non-{positive_class_name}: {metrics_dict['recall_neg']:.4f}")

    if 'balanced_accuracy' in metrics:
        metrics_dict['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        if verbose >= 1:
            to_print.append(f"{train_or_test.capitalize()} balanced accuracy: {metrics_dict['balanced_accuracy']:.4f}")


    if verbose >= 1:
        to_print = "\n".join(to_print)
        print(to_print)

    return metrics_dict
    