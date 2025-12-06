import sys
print(sys.path)
import pytest
from CommML.Linear_Models import *



@pytest.mark.parametrize(
    "x , y , epochs",
    [
        ([1,2,3], [2,4,6], 20),
        ([1012.42, 123.42, 123.42], [202.84, 246.84, 246.84], 20),
        ([1012.42, 123.42, 123.42], [202.84, 246.84, 246.84], None),
        ([[1,2,3], [2,4,6]], [2,4,6], 20),
    ]
)
def test_linear_regression(x, y, epochs):
    model = linear_regression()
    if epochs:
        model.fit(x, y, epochs = epochs)
    else:
        model.fit(x, y)
    y_pred = model.predict(x[0])
    assert y_pred is not None



@pytest.mark.parametrize(
    "x , y , epochs",
    [
        ([1,2,3], [2,4,6], 20),
        ([1012.42, 123.42, 123.42], [202.84, 246.84, 246.84], 20),
        ([1012.42, 123.42, 123.42], [202.84, 246.84, 246.84], None),
        ([[1,2,3], [2,4,6]], [2,4,6], 20),
    ]
)
def test_polynomial_regression(x, y, epochs):
    model = polynomial_regression()
    if epochs:
        model.fit(x, y, epochs = epochs)
    else:
        model.fit(x, y)
    y_pred = model.predict(x[0])
    assert y_pred is not None