import numpy as np

Vec3 = np.ndarray
Scalar = float | int
Array = np.ndarray


def array(*args):
    return np.array(*args)
