import pytest
from CommML.scaler import *

@pytest.mark.parametrize(
    ("x"),
    [
        ([1,2,3]),
        ([1012.42, 123.42, 123.42]),
        ([1012.42, 123.42, 123.42]),
        ([[1,2,3], [2,4,6]]),
    ]
)
def test_normalization(x):
    scaled = normalization(x)
    assert scaled is not None


@pytest.mark.parametrize(
    ("x"),
    [
        ([1,2,3]),
        ([1012.42, 123.42, 123.42]),
        ([1012.42, 123.42, 123.42]),
        ([[1,2,3], [2,4,6]]),
    ]
)
def test_mean_scaler(x):
    scaled = mean_scaler(x)
    assert scaled is not None



@pytest.mark.parametrize(
    ("x"),
    [
        ([1,2,3]),
        ([1012.42, 123.42, 123.42]),
        ([1012.42, 123.42, 123.42]),
        ([[1,2,3], [2,4,6]]),
    ]
)
def test_standard_scaler(x):
    scaled = standard_scaler(x)
    assert scaled is not None

