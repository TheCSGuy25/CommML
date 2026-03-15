import pytest
from CommML.linear_models import *



@pytest.mark.parametrize(
"x, y",
[
    ([[1],[2],[3]], [2,4,6]),
    ([[1012.42],[123.42],[123.42]], [202.84, 246.84, 246.84]),
]
)

def test_linear_regression(x, y):
    model = linear_regression()
    model.fit(x, y)
    y_pred = model.predict(x)
    assert y_pred is not None