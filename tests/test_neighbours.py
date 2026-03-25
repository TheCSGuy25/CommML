import numpy as np
import pytest
from CommML.neighbours import KNN


@pytest.fixture
def simple_knn():
    knn = KNN(k=3, metric="euclidean")
    X = [
        [1, 1],
        [2, 2],
        [3, 3],
        [6, 6],
        [7, 7]
    ]
    y = [0, 0, 0, 1, 1]

    knn.fit(X, y)
    knn.is_trained = True
    return knn


def test_knn_predict_euclidean(simple_knn):
    prediction = simple_knn.predict([[2, 2]])
    assert prediction[0] == 0


def test_knn_predict_manhattan():
    knn = KNN(k=3, metric="manhattan")

    X = [
        [1, 1],
        [2, 2],
        [3, 3],
        [6, 6],
        [7, 7]
    ]
    y = [0, 0, 0, 1, 1]

    knn.fit(X, y)
    knn.is_trained = True

    prediction = knn.predict([[6, 6]])
    assert prediction[0] == 1


def test_knn_single_neighbor():
    knn = KNN(k=1)

    X = [[0, 0], [10, 10]]
    y = [0, 1]

    knn.fit(X, y)
    knn.is_trained = True

    prediction = knn.predict([[9, 9]])
    assert prediction[0] == 1


def test_knn_tie_breaking():
    knn = KNN(k=2)

    X = [[0, 0], [1, 1]]
    y = [0, 1]

    knn.fit(X, y)
    knn.is_trained = True

    prediction = knn.predict([[0.5, 0.5]])
    assert prediction[0] in [0, 1]


def test_knn_predict_returns_numpy_array(simple_knn):
    prediction = simple_knn.predict([[3, 3]])
    assert isinstance(prediction, np.ndarray)