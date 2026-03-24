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
def test_normalization(x):
    scaler = normalization()
    scaled = scaler.fit_transform(x)

    assert scaled is not None
    assert np.array(scaled).shape == np.array(x).shape