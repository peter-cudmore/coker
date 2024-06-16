from coker import Dimension
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.sparse_tensor import dok_ndarray, scalar, tensor_vector_product
from coker.backends.coker.core import to_vec
import numpy as np

def dense_array_cast(x):
    if isinstance(x, scalar):
        return np.array([x])
    return x



class BilinearWeights:

    def __init__(self, memory: MemorySpec, constant=None, linear=None, quadratic=None):
        self.memory = memory

        out_shape = {m.shape[0] for m in [constant, linear, quadratic] if m is not None}
        assert out_shape, f"Cannot create weights with no arguments"
        assert len(out_shape) == 1, f"Weights have inconsistent shapes"
        s, = out_shape
        self.dimension = s
        self.linear = dok_ndarray.from_maybe(linear, expected_shape=(s, memory.count))
        self.constant = dok_ndarray.from_maybe(constant, expected_shape=(memory.count, ))
        self.quadratic = dok_ndarray.from_maybe(quadratic, expected_shape=(s, memory.count, memory.count))

    def __call__(self, x):
        x_v = dense_array_cast(x)
        qx = self.quadratic @ x_v
        ax = (self.linear @ x_v).toarray()
        qxx = (qx @ x_v).toarray()

        return self.constant.toarray() + ax + qxx

    def diff(self, x):
        dq = tensor_vector_product(self.quadratic, x, axis=1) + tensor_vector_product(self.quadratic, x, axis=2)

        return dq + self.linear

    def push_forwards(self, x, dx):
        x = dense_array_cast(x)
        dx = dense_array_cast(dx)

        dw = self.diff(x)
        qx = self.quadratic @ x
        lx = self.linear @ x
        w = self.constant.clone() + lx + qx @ x
        return w.toarray(), (dw @ dx).toarray()

    def is_scalar(self):
        return self.dimension == 1

    def is_constant(self):
        return not self.linear.keys and not self.quadratic.keys

    def is_linear(self):
        return not self.quadratic.keys

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

        if self.is_scalar() and isinstance(other, BilinearWeights):
            assert self.memory == other.memory, f"Cannot multiply weights with different source"
            if self.is_constant():
                return float(self.constant) * other
            if other.is_constant() and other.is_scalar():
                return float(other.constant) * self

            if self.is_linear() and other.is_linear():
                result = float(self.constant) * other
                result.quadratic = dok_ndarray((1,1,1), {(0,0,0): float(self.linear) * float(other.linear)})
                return result


        raise TypeError(f'Cannot multiply {self} by {type(other)}')

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
        elif isinstance(other, np.ndarray):
            return BilinearWeights(
                self.memory,
                self.constant + dok_ndarray.fromarray(other),
                self.linear.clone(),
                self.quadratic.clone()
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
        return BilinearWeights(self.memory, -self.constant, -self.linear, -self.quadratic)


    def __rmatmul__(self, other):
        if isinstance(other, (np.ndarray, dok_ndarray)):
            constant = other @ self.constant
            linear = other @ self.linear
            quadratic = other @ self.quadratic
            return BilinearWeights(self.memory, constant, linear, quadratic)

        raise TypeError(f'Cannot matmul {type(other)}')

    def __array_ufunc__(self, ufunc, method, args, out=None):
        if ufunc == np.matmul and method == '__call__':
            return self.__rmatmul__(args)

        raise NotImplementedError