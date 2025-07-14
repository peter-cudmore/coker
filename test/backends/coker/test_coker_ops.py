import numpy as np
import pytest

from coker.backends.coker.op_impl import *
from coker.backends.coker.weights import MemorySpec, BilinearWeights

from coker import function, VectorSpace

x_symbol = BilinearWeights(
    MemorySpec(0, 3),
    linear=dok_ndarray.eye(3),
    shape=(3,)
)


def test_dot():

    a_np = np.array([1, 2, 3])
    b_np = np.array([4, 5, 6])
    dot_np = np.dot(a_np, b_np)

    assert dot(a_np, b_np) == dot_np

    a, b = [dok_ndarray.fromarray(o) for o in [a_np, b_np]]

    assert dot(a, b) == dot_np



