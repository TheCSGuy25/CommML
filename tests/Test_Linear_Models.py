import pytest
from CommML.linear_models import *

@pytest.mark.parametrize(
"x, y",
[
    ([[1,2],[2,3],[3,4]], [2,4,6]),
    ([[1012.42,1],[123.42,2],[123.42,3]], [202.84, 246.84, 246.84]),
]
)
def test_linear_regression_valid(x, y):
    model = linear_regression()
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred is not None


def test_linear_regression_1d_error():
    model = linear_regression()

    x = [1,2,3]
    y = [2,4,6]

    with pytest.raises(ValueError):
        model.fit(x, y)

@pytest.mark.parametrize(
    "x, y, degree",
    [
        ([[1, 2], [2, 3], [3, 4]], [2, 4, 6], 1),
        ([[1, 2], [2, 3], [3, 4]], [2, 4, 6], 2),
        ([[1012.42, 1], [123.42, 2], [123.42, 3]], [202.84, 246.84, 246.84], 1),
        ([[1012.42, 1], [123.42, 2], [123.42, 3]], [202.84, 246.84, 246.84], 2),
    ]
)
def test_polynomial_regression_valid(x, y, degree):
    model = polynomial_regression(degree=degree)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred is not None


def test_polynomial_regression_1d_error():
    model = polynomial_regression(degree=2)
    x = [1, 2, 3]
    y = [2, 4, 6]
    with pytest.raises(ValueError):
        model.fit(x, y)