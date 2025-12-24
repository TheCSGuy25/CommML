from CommML.Metrics import *
import pytest
import math

@pytest.fixture
def binary_data():
    y = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    return y, y_pred


def test_get_confusion(binary_data):
    y, y_pred = binary_data
    matrix = get_confusion(y, y_pred)
    assert matrix == [[1, 1], [1, 1]]


def test_confusion_matrix_prints_output(binary_data, capsys):
    y, y_pred = binary_data
    confusion_matrix(y, y_pred)
    captured = capsys.readouterr()
    assert "Rows: Actual Label , Columns: Predicted Labels" in captured.out
    assert "[0, 1]" in captured.out


def test_get_accuracy(binary_data):
    y, y_pred = binary_data
    accuracy = get_accuracy(y, y_pred)
    assert accuracy == 0.5


def test_get_precision(binary_data):
    y, y_pred = binary_data
    precision = get_precision(y, y_pred)
    assert precision == 0.5


def test_get_recall(binary_data):
    y, y_pred = binary_data
    recall = get_recall(y, y_pred)
    assert recall == 0.5


def test_f1_score(binary_data):
    y, y_pred = binary_data
    f1 = f1_score(y, y_pred)
    assert f1 == 0.5


def test_mean_squared_error():
    y = [1, 2, 3]
    y_pred = [2, 2, 4]
    mse = mean_squared_error(y, y_pred)
    assert mse == pytest.approx(2 / 3)


def test_mean_absolute_error():
    y = [1, 2, 3]
    y_pred = [2, 2, 4]
    mae = mean_absolute_error(y, y_pred)
    assert mae == pytest.approx(2 / 3)


def test_r2_score():
    y = [1, 2, 3]
    y_pred = [2, 2, 4]
    r2 = r2_score(y, y_pred)
    assert r2 == pytest.approx(1 - (2 / 3))


def test_root_mean_squared_error():
    y = [1, 2, 3]
    y_pred = [2, 2, 4]
    rmse = root_mean_squared_error(y, y_pred)
    assert rmse == pytest.approx(math.sqrt(2 / 3))
