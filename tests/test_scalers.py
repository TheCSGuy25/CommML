import pytest
import numpy as np
from CommML.preprocessing import *


@pytest.mark.parametrize(
    "x",
    [
        [1, 2, 3],
        [1012.42, 123.42, 123.42],
        [[1, 2, 3], [2, 4, 6]],
    ],
)
def test_min_max_scaler(x):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(x)

    assert scaled is not None
    assert np.array(scaled).shape == np.array(x).shape



@pytest.mark.parametrize(
    "x",
    [
        [1, 2, 3],
        [1012.42, 123.42, 123.42],
        [[1, 2, 3], [2, 4, 6]],
    ],
)
def test_max_abs_scaler(x):
    scaler = MaxAbsScaler()
    scaled = scaler.fit_transform(x)

    assert scaled is not None
    assert np.array(scaled).shape == np.array(x).shape