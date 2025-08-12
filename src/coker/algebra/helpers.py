from typing import Union
import numpy as np


def is_scalar(x: Union[int, float, np.ndarray]):
    if isinstance(x, np.ndarray):
        return all(s_i == 1 for s_i in x.shape)
    return isinstance(x, (int, float))
