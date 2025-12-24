import pytest
from CommML.misc import *



@pytest.mark.parametrize(
    ("x , y , train_ratio"),
    [
        ([1,2,3], [2,4,6], 0.8),
        ([1012.42, 123.42, 123.42], [202.84, 246.84, 246.84], 0.20),
        ([1012.42, 123.42, 123.42], [202.84, 246.84, 246.84], 0.45),
        ([[1,2,3], [2,4,6]], [2,4,6], 0.320),
    ]
)
def train_test_split_test(x, y, train_ratio):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_ratio)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == len(x)
    assert len(y_train) + len(y_test) == len(y)
