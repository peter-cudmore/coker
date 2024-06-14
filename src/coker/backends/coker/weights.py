from coker import Dimension
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.sparse_tensor import dok_ndarray, scalar, tensor_sum
from coker.backends.coker.core import to_vec
import numpy as np


class BilinearWeights:

    def __init__(self, memory: MemorySpec, constant=None, linear=None, quadratic=None):
        self.memory = memory

        out_shape = {m.shape[0] for m in [constant, linear, quadratic] if m is not None}
        assert out_shape, f"Cannot create weights with no arguments"
        assert len(out_shape) == 1, f"Weights have inconsistent shapes"
        s, = out_shape

        self.linear = dok_ndarray.from_maybe(linear, expected_shape=(s, memory.count))
        self.constant = dok_ndarray.from_maybe(constant, expected_shape=(memory.count, ))
        self.quadratic = dok_ndarray.from_maybe(quadratic, expected_shape=(s, memory.count, memory.count))

    def __call__(self, x):
        if isinstance(x, scalar):
            x_v = np.array([x])
        else:
            x_v = x

        qx = self.quadratic @ x_v
        ax = (self.linear @ x_v).toarray()
        qxx = (qx @ x_v).toarray()

        return self.constant.toarray() + ax + qxx

    def diff(self, x):
        dq = tensor_sum(self.quadratic, x, axis=1) + tensor_sum(self.quadratic, x, axis=2)


    @staticmethod
    def identity(arg):
        if isinstance(arg, Dimension):
            n = arg.flat()
        else:
            assert isinstance(arg, int)
            n = arg
        a = dok_ndarray.eye(n)
        b = np.zeros((n,))
        Q = dok_ndarray.zeros((n, n, n))
        return BilinearWeights(a, b, Q)

    def __mul__(self, other):
        if isinstance(other, scalar):
            constant = other * self.constant
            linear = other * self.linear
            quadratic = other * self.quadratic
            return BilinearWeights(self.memory, constant=constant, linear=linear, quadratic=quadratic)
        raise TypeError(f'Cannot multiply by {type(other)}')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, scalar):
            linear = self.linear.clone()
            constant = self.constant + other
            quadratic = self.quadratic.clone()
            return BilinearWeights(self.memory, constant=constant, linear=linear, quadratic=quadratic)
        elif isinstance(other, BilinearWeights):
            assert self.memory == other.memory
            return BilinearWeights(
                self.memory,
                self.constant + other.constant,
                self.linear + other.linear,
                self.quadratic + other.quadratic,
            )
        raise TypeError(f'Cannot add {type(other)}')

    def __sub__(self, other):
        if isinstance(other, scalar):
            linear = self.linear.clone()
            constant = self.constant - other
            quadratic = self.quadratic.clone()
            return BilinearWeights(self.memory, constant=constant, linear=linear, quadratic=quadratic)
        elif isinstance(other, BilinearWeights):
            assert self.memory == other.memory
            return BilinearWeights(
                self.memory,
                self.constant - other.constant,
                self.linear - other.linear,
                self.quadratic - other.quadratic,
            )
        raise TypeError(f'Cannot add {type(other)}')

    def __neg__(self):
        return BilinearWeights(-self.constant, -self.linear, -self.quadratic)
