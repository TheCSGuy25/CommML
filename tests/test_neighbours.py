import numpy as np
import pytest
from CommML.Neighbours import KNN   

@pytest.fixture
def simple_knn():
    knn = KNN(n=3)
    X = [
        [1, 1],
        [2, 2],
        [3, 3],
        [6, 6],
        [7, 7]
    ]
    y = [0, 0, 0, 1, 1]
    knn.store(X, y)
    return knn


def test_knn_predict_euclidean(simple_knn):
    prediction = simple_knn.predict([2, 2], distance="euclidean")
    assert prediction == 0


def test_knn_predict_manhattan(simple_knn):
    prediction = simple_knn.predict([6, 6], distance="manhattan")
    assert prediction == 1


def test_knn_single_neighbor():
    knn = KNN(n=1)
    X = [[0, 0], [10, 10]]
    y = [0, 1]
    knn.store(X, y)

    prediction = knn.predict([9, 9])
    assert prediction == 1


def test_knn_tie_breaking():
    knn = KNN(n=2)
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    knn.store(X, y)

    prediction = knn.predict([0.5, 0.5])
    assert prediction in [0, 1] 


def test_knn_predict_returns_scalar_label(simple_knn):
    prediction = simple_knn.predict([3, 3])
    assert isinstance(prediction, (int, np.integer))
