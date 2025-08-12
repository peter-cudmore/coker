from coker import OP
from coker.backends.coker.sparse_tensor import (
    dok_ndarray,
    tensor_vector_product,
    is_constant,
)
from coker.backends.coker.tensor_contants import hat, levi_civita_3_tensor
from coker.backends.coker.weights import BilinearWeights


def cross(x, y):
    if is_constant(x):
        Ax = hat(x)
        return (Ax).toarray() @ y
    if is_constant(y):
        Ay = -hat(y)
        return Ay.toarray() @ x

    assert isinstance(x, BilinearWeights)
    # W_ix + b_i \cross W_jy + b_j
    return levi_civita_3_tensor @ (x, y)


def dot(x, y):
    if is_constant(x):
        if len(x.shape) == 1:
            (n,) = x.shape
            xt = x.T.reshape((1, n))
        else:
            n, m = x.shape
            assert m == 1
            xt = x.T
        return xt @ y
    if is_constant(y):
        if len(y.shape) == 1:
            (n,) = y.shape
            yT = y.T.reshape((1, n))
        else:
            n, m = y.shape
            assert m == 1
            yT = y.T
        return yT @ x

    if isinstance(x, BilinearWeights) and isinstance(y, BilinearWeights):
        return x.dot(y)

    raise NotImplementedError


def transpose(x):
    if is_constant(x):
        if len(x.shape) == 1:
            (n,) = x.shape
            return x.reshape((n, 1)).T

        return x.T

    if isinstance(x, BilinearWeights):
        return x.transpose()

    raise NotImplementedError(f"Cannot transpose {type(x)}, {x.shape}")


def is_scalar(x):
    if isinstance(x, (float, complex, int)):
        return True
    try:
        return all(s == 1 for s in x.shape)
    except AttributeError:
        pass
    if isinstance(x, BilinearWeights):
        return x.dimension == 1
    raise NotImplementedError


ops = {
    OP.MUL: lambda x, y: x * y,
    OP.DIV: lambda x, y: x / y,
    OP.ADD: lambda x, y: x + y,
    OP.SUB: lambda x, y: x - y,
    OP.MATMUL: lambda x, y: x @ y,
    OP.CROSS: cross,
    OP.DOT: dot,
    OP.TRANSPOSE: transpose,
}
