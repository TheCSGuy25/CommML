import numpy as np
import random

def train_test_split(x, y, train_size=0.8, test_size=None, random_state=42):
    if test_size is None:
        test_size = 1 - train_size
    else:
        train_size = 1 - test_size

    x = np.array(x)
    y = np.array(y)

    n = len(x)
    indices = np.arange(n)

    random.seed(random_state)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    split_point = int(n * train_size)

    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    return x_train, x_test, y_train, y_test
