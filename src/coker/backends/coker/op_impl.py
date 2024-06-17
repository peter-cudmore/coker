from coker import OP
from coker.backends.coker.sparse_tensor import dok_ndarray, tensor_vector_product, is_constant
from coker.backends.coker.tensor_contants import hat, levi_civita_3_tensor
from coker.backends.coker.weights import BilinearWeights


def cross(x, y):
    if is_constant(x):
        Ax = hat(x)
        return Ax @ y
    if is_constant(y):
        Ay = -hat(y)
        return Ay @ x

    assert isinstance(x, BilinearWeights)
    # W_ix + b_i \cross W_jy + b_j
    return (levi_civita_3_tensor @ y)  @ x


def dot(x, y):
    if is_constant(x):
        return x.T @ y
    if is_constant(y):
        return y.T @ x

    assert isinstance(x, BilinearWeights)

    raise NotImplementedError


def transpose(x):
    if is_constant(x):
        return x.T

    assert isinstance(x, BilinearWeights)
    raise NotImplementedError

