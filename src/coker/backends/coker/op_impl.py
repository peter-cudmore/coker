import numpy as np
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
    assert isinstance(y, BilinearWeights)
    assert (
        x.memory == y.memory
    ), "cross product requires BilinearWeights from same memory"
    assert (
        x.is_linear and y.is_linear
    ), "cross product of quadratic weights not supported"

    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1

    c_x = x.constant.toarray()  # shape (3,)
    c_y = y.constant.toarray()  # shape (3,)
    L_x = x.linear.toarray()  # shape (3, n)
    L_y = y.linear.toarray()  # shape (3, n)

    c_result = np.cross(c_x, c_y)  # shape (3,)
    # linear contribution: cross(L_x @ m, c_y) + cross(c_x, L_y @ m)
    #   = -hat(c_y) @ L_x @ m + hat(c_x) @ L_y @ m
    L_result = (
        -hat(c_y).toarray() @ L_x + hat(c_x).toarray() @ L_y
    )  # shape (3, n)
    # quadratic contribution: cross(L_x @ m, L_y @ m)
    Q_result = np.einsum("ijk,js,kt->ist", eps, L_x, L_y)  # shape (3, n, n)

    return BilinearWeights(
        x.memory,
        (3,),
        constant=dok_ndarray.fromarray(c_result),
        linear=dok_ndarray.fromarray(L_result),
        quadratic=dok_ndarray.fromarray(Q_result),
    )


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


def mul(x, y):
    return x * y


def div(x, y):
    return x / y


ops = {
    OP.MUL: mul,
    OP.DIV: div,
    OP.ADD: lambda x, y: x + y,
    OP.SUB: lambda x, y: x - y,
    OP.NEG: lambda x: -x,
    OP.MATMUL: lambda x, y: x @ y,
    OP.CROSS: cross,
    OP.DOT: dot,
    OP.TRANSPOSE: transpose,
}
