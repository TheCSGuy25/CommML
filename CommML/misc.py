import numpy as np
import random

def train_test_split(x, y, train_size=0.8, test_size=None, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    x : array-like
        Feature data
    y : array-like
        Target data
    train_size : float
        Proportion of data to include in the training set
    test_size : float, optional
        Proportion of data to include in the testing set
        If not specified, it is calculated as 1 - train_size
    random_state : int, optional
        Random seed for shuffling the data

    Returns
    -------
    x_train : array-like
        Feature data for the training set
    x_test : array-like
        Feature data for the testing set
    y_train : array-like
        Target data for the training set
    y_test : array-like
        Target data for the testing set
    """

    if test_size is None:
        test_size = 1 - train_size
    else:
        train_size = 1 - test_size

    x = np.array(x)
    y = np.array(y)

    n = len(x)
    indices = np.arange(n)

    np.random.seed(random_state)
    np.random.shuffle(indices)

    split_point = int(n * train_size)

    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
 

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    return x_train, x_test, y_train, y_test