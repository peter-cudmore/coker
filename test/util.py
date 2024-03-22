import numpy as np


def is_close(a, b, tolerance=1e-16):
    assert a.shape == b.shape, "shapes don't match"
    return np.linalg.norm(a-b, ord=np.inf) < tolerance

