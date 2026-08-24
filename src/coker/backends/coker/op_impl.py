import numpy as np
from coker.algebra.ops import OP
from coker.backends.coker.sparse_tensor import dok_ndarray, is_constant
from coker.backends.coker.tensor_contants import hat
from coker.backends.coker.weights import BilinearWeights

_CROSS_EPSILON = np.zeros((3, 3, 3))
_CROSS_EPSILON[0, 1, 2] = _CROSS_EPSILON[1, 2, 0] = _CROSS_EPSILON[2, 0, 1] = 1
_CROSS_EPSILON[2, 1, 0] = _CROSS_EPSILON[0, 2, 1] = _CROSS_EPSILON[1, 0, 2] = (
    -1
)


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

    epsilon = (
        ((0, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((0, 0, -1), (0, 0, 0), (1, 0, 0)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 0)),
    )

    def accumulate(data, key, value):
        result = data.get(key, 0.0) + value
        if result:
            data[key] = result
        else:
            data.pop(key, None)

    constant = {}
    for (left_axis,), left_value in x.constant.keys.items():
        for (right_axis,), right_value in y.constant.keys.items():
            for output_axis, epsilon_value in enumerate(
                epsilon[left_axis][right_axis]
            ):
                if epsilon_value:
                    accumulate(
                        constant,
                        (output_axis,),
                        epsilon_value * left_value * right_value,
                    )

    linear = {}
    for (left_axis, memory_index), left_value in x.linear.keys.items():
        for (right_axis,), right_value in y.constant.keys.items():
            for output_axis, epsilon_value in enumerate(
                epsilon[left_axis][right_axis]
            ):
                if epsilon_value:
                    accumulate(
                        linear,
                        (output_axis, memory_index),
                        epsilon_value * left_value * right_value,
                    )
    for (left_axis,), left_value in x.constant.keys.items():
        for (right_axis, memory_index), right_value in y.linear.keys.items():
            for output_axis, epsilon_value in enumerate(
                epsilon[left_axis][right_axis]
            ):
                if epsilon_value:
                    accumulate(
                        linear,
                        (output_axis, memory_index),
                        epsilon_value * left_value * right_value,
                    )

    quadratic = {}
    for (left_axis, left_memory), left_value in x.linear.keys.items():
        for (right_axis, right_memory), right_value in y.linear.keys.items():
            for output_axis, epsilon_value in enumerate(
                epsilon[left_axis][right_axis]
            ):
                if epsilon_value:
                    accumulate(
                        quadratic,
                        (output_axis, left_memory, right_memory),
                        epsilon_value * left_value * right_value,
                    )

    return BilinearWeights.from_trusted_dok(
        x.memory,
        (3,),
        constant=dok_ndarray((3,), constant),
        linear=dok_ndarray((3, x.memory.count), linear),
        quadratic=dok_ndarray((3, x.memory.count, x.memory.count), quadratic),
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
