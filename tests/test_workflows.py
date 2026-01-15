import numpy as np
import pytest
from CommML.workflows import *
from CommML.Linear_Models import *

def test_regression_workflow():
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    model, metrics = regression_workflow(x, y)
    assert isinstance(model, linear_regression)
    assert isinstance(metrics, dict)


def test_classification_workflow():
    x = np.array([
        [1.2, 2.1],
        [2.9, 3.8],
        [3.1, 2.7],
        [4.8, 5.2],
        [5.5, 4.9],
        [6.2, 6.1],
        [6.8, 7.4],
        [7.9, 7.1],
        [8.3, 8.9],
        [9.1, 8.2]
    ])

    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0])

    model, metrics = classification_workflow(x, y, epochs=1)
    assert isinstance(model, logistic_regression)
    assert isinstance(metrics, dict)