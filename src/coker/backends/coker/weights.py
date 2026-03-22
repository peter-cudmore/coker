from typing import Tuple
from functools import reduce
from coker import Dimension
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.sparse_tensor import (
    dok_ndarray,
    scalar,
    tensor_vector_product,
    tensor_sum,
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
        result = c + ax + qxx
        return np.reshape(result, self.shape)

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
                if isinstance(other, np.ndarray) and len(other.shape) == 1:
                    result_shape = other.shape
                else:
                    result_shape = None
                other = dok_ndarray.fromarray(other)
                if result_shape is None:
                    result_shape = other.shape
                constants = float(self.constant) * other

                # Other : (l, m)
                # self.linear: Array(1, n),         ->          (l, m, n)
                # self.quadratic : Array(1, n, n)   ->          (l, m, n, n)
                # linear =
                linear = outer_product(other, self.linear)
                quadratic = outer_product(other, self.quadratic)
                return BilinearWeights(
                    self.memory, result_shape, constants, linear, quadratic
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
        # Contract self's last output axis with other's first output axis.
        # self.shape = (..., n), other.shape = (n, ...)
        # self.constant/linear/quadratic have shape (*self.shape, ...) — output axes first,
        # memory axes last.  We must contract over axis len(self.shape)-1 of self's tensors
        # and axis 0 of other's tensors, NOT the last (memory) axis.
        col = len(self.shape) - 1

        def _contract(lhs: dok_ndarray, rhs: dok_ndarray) -> dok_ndarray:
            """tensor_sum contracting lhs axis col with rhs axis 0.

            When rhs is a column vector (shape (n,1)) the trailing 1 is propagated
            into higher-dimensional results; we drop it for rank >= 3 outputs.
            """
            result = tensor_sum(lhs, rhs, l_index=col, r_index=0)
            if rhs.is_vector() and len(result.shape) > 2:
                # Drop the spurious trailing 1: (a, b, ..., 1) -> (a, b, ...)
                shape = result.shape[:-1]
                data = {k[:-1]: v for k, v in result.keys.items()}
                result = dok_ndarray(shape, data)
            return result

        constant = _contract(self.constant, other.constant)
        linear = _contract(self.constant, other.linear) + _contract(
            self.linear, other.constant
        )
        linear_linear_quadratic = tensor_sum(
            self.linear, other.linear, l_index=col, r_index=0
        )
        # tensor_sum produces (out_l..., mem_l, out_r..., mem_r).
        # BilinearWeights convention requires (out_l..., out_r..., mem_l, mem_r).
        # Move mem_l (at position col) past the M-1 out_r axes.
        mem_l_pos = col
        for _ in range(len(other.shape) - 1):
            linear_linear_quadratic = linear_linear_quadratic.swap_indices(
                mem_l_pos, mem_l_pos + 1
            )
            mem_l_pos += 1
        quadratic = (
            _contract(self.constant, other.quadratic)
            + _contract(self.quadratic, other.constant)
            + linear_linear_quadratic
        )
        shape = (*self.shape[:-1], *other.shape[1:])
        return BilinearWeights(self.memory, shape, constant, linear, quadratic)

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
                    constant = dok_ndarray((1, 1), {(0, 0): args})
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
        """Matrix multiplication of two bilinear weights.

        We assume that the total order of the result is 2.
        So that either;
        - self and RHS have order <= 1 (i.e. the quadratic terms are zero)
        - or one has order 2 and the other has order 0.

        If
        math::
            y_0 = c_0 + L_0x + Q_0(x,x)
            y_1 = c_1 + L_1x + Q_1(x,x)

            dot(y_0, y_1) = dot(c_0, c_1)
                             + ((c_0.T @ L_1 + c_1.T @ L_0 x)
                             + (c_0.T @ Q_1 + c_1.T @ Q_0 + L_0.T @ L_1) (x,x)


        """
        assert self.memory == rhs.memory
        assert (
            (self.is_linear() and rhs.is_linear())
            or (not self.is_linear() and rhs.is_constant())
            or (self.is_constant() and not rhs.is_linear())
        ), "dot requires both operands order<=1, or one order==2 and the other order==0"
        c = self.constant.T @ rhs.constant
        l = self.constant.T @ rhs.linear + rhs.constant.T @ self.linear
        m = self.memory.count
        ll = self.linear.T @ rhs.linear  # (m, m)
        ll_expanded = dok_ndarray(
            (1, m, m), {(0, *k): v for k, v in ll.keys.items()}
        )
        q = (
            ll_expanded
            + self.constant.T @ rhs.quadratic
            + rhs.constant.T @ self.quadratic
        )

        return BilinearWeights(self.memory, (1,), c, l, q)

    @staticmethod
    def identity2(memory: MemorySpec):

        shape = (memory.count,)
        data = {}
        for i in range(memory.count):
            data[(i, i)] = 1

        linear = dok_ndarray((memory.count, memory.count), data)

        return BilinearWeights(memory, shape, linear=linear)

    @staticmethod
    def reshape_identity(memory: MemorySpec, shape: tuple):
        """BilinearWeights that maps a flat vector of size memory.count to the given shape."""
        n = memory.count
        data = {}
        for k in range(n):
            multi_idx = np.unravel_index(k, shape, order="C")
            data[(*multi_idx, k)] = 1
        linear = dok_ndarray((*shape, n), data)
        return BilinearWeights(memory, shape, linear=linear)


def outer_product(lhs: dok_ndarray, rhs: dok_ndarray):
    assert rhs.shape[0] == 1
    if lhs.is_vector():
        # lhs has shape (n,1) — treat as (n,) so the result is (n, ...) not (n,1,...)
        shape = (lhs.shape[0], *rhs.shape[1:])
        data = {}
        for k_l, v_l in lhs.keys.items():
            for k_r, v_r in rhs.keys.items():
                key = (k_l[0], *k_r[1:])
                data[key] = v_l * v_r
        return dok_ndarray(shape, data)
    shape = (*lhs.shape, *rhs.shape[1:])
    data = {}
    for k_l, v_l in lhs.keys.items():
        for k_r, v_r in rhs.keys.items():
            key = tuple((*k_l, *k_r[1:]))
            data[key] = v_l * v_r

    return dok_ndarray(shape, data)
