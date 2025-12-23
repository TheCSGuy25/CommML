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

def min_max_scaler(x):
    """
    A scaling method that scales the values of X based on the 
    minumum and maximum value
    
    :param x: A variable feature length input
    :return: A variable min output
    """

    number_of_features = len(x[0]) if type(x) == list else 1
    if number_of_features == 1:
        return [(x[i] - min(x)) / (max(x) - min(x)) for i in range((len(x)))]
    else:
        minimum = [min(x[i][j] for i in range(len(x))) for j in range(number_of_features)]
        maximum = [max(x[i][j] for i in range(len(x))) for j in range(number_of_features)]

        if any(maximum[i] == minimum[i] for i in range(number_of_features)):
            raise ValueError("A minumum and maxium value is the same which leads to division by Zero")
        
        return [
                [(x[i][j] - minimum[j]) / (maximum[j] - minimum[j])
                for j in range(number_of_features)]
                for i in range(len(x))
            ]


def mean_scaler(x):
    number_of_features  = len(x[0]) if type(x[0]) == list else 1
    x = np.array(x)
    if number_of_features == 1:
        mean_x = np.mean(x)
        min_x, max_x = np.min(x), np.max(x)
        return ((x - mean_x) / (max_x - min_x)).tolist()
    else:
        mean_x = np.mean(x, axis=0)
        min_x = np.min(x, axis=0)
        max_x = np.max(x, axis=0)

        if np.any(max_x == min_x):
            raise ValueError("A feature has equal min and max; division by zero.")

        return ((x - mean_x) / (max_x - min_x)).tolist()