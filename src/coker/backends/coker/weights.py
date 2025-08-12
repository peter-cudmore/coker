from typing import Tuple
from functools import reduce
from coker import Dimension
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.sparse_tensor import (
    dok_ndarray,
    scalar,
    tensor_vector_product,
    cast_vector,
)
import numpy as np


def dense_array_cast(x):
    if isinstance(x, scalar):
        return np.array([x])
    return x


class BilinearWeights(np.lib.mixins.NDArrayOperatorsMixin):

    def __init__(
        self,
        memory: MemorySpec,
        shape: Tuple[int, ...],
        constant=None,
        linear=None,
        quadratic=None,
    ):
        self.memory = memory

        assert isinstance(shape, tuple)
        self.shape = shape

        self.constant = dok_ndarray.from_maybe(constant, expected_shape=shape)
        self.linear = dok_ndarray.from_maybe(
            linear, expected_shape=(*shape, memory.count)
        )
        self.quadratic = dok_ndarray.from_maybe(
            quadratic, expected_shape=(*shape, memory.count, memory.count)
        )

    def transpose(self) -> "BilinearWeights":
        if len(self.shape) == 1:
            (n,) = self.shape
            return BilinearWeights(
                self.memory,
                shape=(1, n),
                constant=self.constant.T,
                linear=self.linear.swap_indices(0, 1),
                quadratic=self.quadratic.swap_indices(0, 1),
            )
        if len(self.shape) == 2:
            n, m = self.shape
            return BilinearWeights(
                self.memory,
                shape=(m, n),
                constant=self.constant.T,
                linear=self.linear.swap_indices(0, 1),
                quadratic=self.quadratic.swap_indices(0, 1),
            )

        raise NotImplementedError(
            f"Cannot transpose {len(self.shape)} dimensions"
        )

    def __call__(self, x):
        x_v = dense_array_cast(x)
        try:
            qxx = (self.quadratic @ (x_v, x_v)).toarray()
        except TypeError as ex:
            raise ex

        ax = (self.linear @ x_v).toarray()

        c = self.constant.toarray()
        return c + ax + qxx

    def diff(self, x):
        dq = tensor_vector_product(
            self.quadratic, x, axis=1
        ) + tensor_vector_product(self.quadratic, x, axis=2)

        return dq + self.linear

    def push_forwards(self, x, dx):
        x = dense_array_cast(x)
        dx = dense_array_cast(dx)

        dw = self.diff(x)
        qxx = self.quadratic @ (x, x)
        lx = self.linear @ x
        w = self.constant.clone() + lx + qxx
        return w.toarray(), (dw @ dx).toarray()

    def is_scalar(self):
        return self.shape == (1,)

    def is_constant(self):
        return not self.linear.keys and not self.quadratic.keys

    def is_linear(self):
        return not self.quadratic.keys

    def __mul__(self, other):
        if isinstance(other, scalar):
            constant = other * self.constant
            linear = other * self.linear
            quadratic = other * self.quadratic
            return BilinearWeights(
                self.memory,
                shape=self.shape,
                constant=constant,
                linear=linear,
                quadratic=quadratic,
            )

        try:
            assert all(s == 1 for s in other.shape) and not isinstance(
                other, BilinearWeights
            )
            return float(other) * self
        except (AttributeError, AssertionError):
            pass

        if self.is_scalar():
            if isinstance(other, BilinearWeights):
                assert (
                    self.memory == other.memory
                ), f"Cannot multiply weights with different source"
                if self.is_constant():
                    return float(self.constant) * other
                if other.is_constant() and other.is_scalar():
                    return float(other.constant) * self

                if self.is_linear() and other.is_linear():
                    result = float(self.constant) * other
                    result.quadratic = dok_ndarray(
                        (1, 1, 1),
                        {(0, 0, 0): float(self.linear) * float(other.linear)},
                    )
                    return result

            if isinstance(other, (np.ndarray, dok_ndarray)):
                other = dok_ndarray.fromarray(other)
                constants = float(self.constant) * other

                # Other : (l, m)
                # self.linear: Array(1, n),         ->          (l, m, n)
                # self.quadratic : Array(1, n, n)   ->          (l, m, n, n)
                # linear =
                linear = outer_product(other, self.linear)
                quadratic = outer_product(other, self.quadratic)
                return BilinearWeights(
                    self.memory, other.shape, constants, linear, quadratic
                )

        raise TypeError(f"Cannot multiply {self} by {type(other)}")

    def __rmul__(self, other):
        assert isinstance(other, scalar)
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, scalar):
            linear = self.linear.clone()
            constant = self.constant + other
            quadratic = self.quadratic.clone()
            return BilinearWeights(
                self.memory,
                self.shape,
                constant=constant,
                linear=linear,
                quadratic=quadratic,
            )
        elif isinstance(other, BilinearWeights):

            if self.linear.is_empty() and self.quadratic.is_empty():
                return BilinearWeights(
                    other.memory,
                    self.shape,
                    self.constant + other.constant,
                    other.linear,
                    other.quadratic,
                )
            if other.linear.is_empty() and other.quadratic.is_empty():
                return BilinearWeights(
                    self.memory,
                    self.shape,
                    self.constant + other.constant,
                    self.linear,
                    self.quadratic,
                )

            assert self.memory == other.memory, f"{self}, {other}"
            return BilinearWeights(
                self.memory,
                self.shape,
                self.constant + other.constant,
                self.linear + other.linear,
                self.quadratic + other.quadratic,
            )
        elif isinstance(other, np.ndarray):
            return BilinearWeights(
                self.memory,
                self.shape,
                self.constant + dok_ndarray.fromarray(other),
                self.linear.clone(),
                self.quadratic.clone(),
            )

        raise TypeError(f"Cannot add {type(other)}")

    def __sub__(self, other):
        if isinstance(other, scalar):
            linear = self.linear.clone()
            constant = self.constant - other
            quadratic = self.quadratic.clone()
            return BilinearWeights(
                self.memory,
                self.shape,
                constant=constant,
                linear=linear,
                quadratic=quadratic,
            )
        elif isinstance(other, BilinearWeights):
            assert self.memory == other.memory
            return BilinearWeights(
                self.memory,
                self.shape,
                self.constant - other.constant,
                self.linear - other.linear,
                self.quadratic - other.quadratic,
            )
        raise TypeError(f"Cannot add {type(other)}")

    def __neg__(self):
        return BilinearWeights(
            self.memory,
            self.shape,
            -self.constant,
            -self.linear,
            -self.quadratic,
        )

    def __rmatmul__(self, other):

        if isinstance(other, (np.ndarray, dok_ndarray)):
            constant = other @ self.constant
            try:

                linear = other @ self.linear
            except IndexError as ex:
                print(f"{other}, {(self.linear.shape, self.linear.keys)}")
                raise ex
            quadratic = other @ self.quadratic
            shape = constant.shape
            return BilinearWeights(
                self.memory, shape, constant, linear, quadratic
            )

        raise TypeError(f"Cannot matmul {type(other)}")

    def __matmul__(self, other):
        assert (
            isinstance(other, BilinearWeights) and other.memory is self.memory
        )
        assert (
            (self.is_linear() and other.is_linear())
            or self.is_constant()
            or other.is_constant()
        )
        constant = self.constant @ other.constant
        linear = self.constant @ other.linear + self.linear @ other.constant
        quadratic = (
            self.constant @ other.quadratic
            + self.quadratic @ other.quadratic
            + self.linear @ other.linear
        )
        return BilinearWeights(
            self.memory, constant.shape, constant, linear, quadratic
        )

    def clone(self):
        return BilinearWeights(
            self.memory,
            self.shape,
            self.constant.clone(),
            self.linear.clone(),
            self.quadratic.clone(),
        )

    def __array_ufunc__(self, ufunc, method, args, out=None):
        if ufunc == np.matmul and method == "__call__":
            return self.__rmatmul__(args)

        if ufunc == np.multiply and method == "__call__":
            if self.is_scalar() or isinstance(args, scalar):
                return self.__mul__(args)

        if ufunc == np.add and method == "__call__":
            return self.__add__(args)

        if ufunc == np.subtract and method == "__call__":
            if self.is_scalar() and isinstance(args, scalar):
                if self.constant.is_empty():
                    constant = dok_ndarray((1, 1), {(0, 0): -args})
                else:
                    constant = self.constant.clone()
                    constant[(0, 0)] = args - constant[(0, 0)]
                linear = -self.linear
                quadratic = -self.quadratic

                return BilinearWeights(
                    self.memory, self.shape, constant, linear, quadratic
                )

        raise NotImplementedError(f"{ufunc} not implemented")

    def __truediv__(self, other):
        if not isinstance(other, scalar):
            raise TypeError(f"Cannot divide {self} by {type(other)}")

        return BilinearWeights(
            self.memory,
            self.shape,
            self.constant / other,
            self.linear / other,
            self.quadratic / other,
        )

    def dot(self, rhs: "BilinearWeights"):

        assert self.memory == rhs.memory
        c = self.constant.T @ rhs.constant
        l = self.linear.T @ rhs.constant + self.constant.T @ rhs.linear
        q = (
            self.linear.T @ rhs.linear
            + self.constant.T @ rhs.quadratic
            + self.constant.T @ rhs.quadratic
        )

        #        cubic = self.linear.T @ rhs.quadratic + rhs.linear.T @ self.quadratic

        return BilinearWeights(self.memory, c.shape, c, l, q)

    @staticmethod
    def identity2(memory: MemorySpec):

        shape = (memory.count,)
        data = {}
        for i in range(memory.count):
            data[(i, i)] = 1

        linear = dok_ndarray((memory.count, memory.count), data)

        return BilinearWeights(memory, shape, linear=linear)


def outer_product(lhs: dok_ndarray, rhs: dok_ndarray):
    assert rhs.shape[0] == 1
    shape = (*lhs.shape, *rhs.shape[1:])
    data = {}
    for k_l, v_l in lhs.keys.items():
        for k_r, v_r in rhs.keys.items():
            key = tuple((*k_l, *k_r[1:]))
            data[key] = v_l * v_r

    return dok_ndarray(shape, data)
