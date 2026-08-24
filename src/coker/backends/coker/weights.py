from typing import Tuple

import numpy as np

from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.sparse_tensor import (
    dok_ndarray,
    scalar,
    tensor_sum,
    tensor_vector_product,
)


def dense_array_cast(x):
    if isinstance(x, scalar):
        return np.array([x])
    return x


def _scale_tensor_by_array(
    tensor: dok_ndarray, scale: np.ndarray, n_output_dims: int
) -> dok_ndarray:
    """Return a copy of tensor scaled by ``scale[output_indices]``.

    ``tensor`` has shape ``(*output_shape, *memory_shape)``. The first
    ``n_output_dims`` axes are the output axes, and ``scale`` must have
    exactly that shape.
    """
    new_keys = {}
    for k, v in tensor.keys.items():
        factor = scale[k[:n_output_dims]]
        new_keys[k] = v * float(factor)
    return dok_ndarray(tensor.shape, new_keys)


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

    @classmethod
    def from_trusted_dok(
        cls,
        memory: MemorySpec,
        shape: Tuple[int, ...],
        constant: dok_ndarray | None = None,
        linear: dok_ndarray | None = None,
        quadratic: dok_ndarray | None = None,
    ) -> "BilinearWeights":
        def _coerce(value, expected_shape: Tuple[int, ...]) -> dok_ndarray:
            if value is None:
                return dok_ndarray(expected_shape)
            if isinstance(value, dok_ndarray):
                return value
            if isinstance(value, np.ndarray):
                return dok_ndarray.fromarray(value)
            raise TypeError(
                f"Expected dok_ndarray or ndarray, got {type(value)}"
            )

        obj = cls.__new__(cls)
        obj.memory = memory
        obj.shape = shape
        obj.constant = _coerce(constant, shape)
        obj.linear = _coerce(linear, (*shape, memory.count))
        obj.quadratic = _coerce(
            quadratic, (*shape, memory.count, memory.count)
        )
        return obj

    def transpose(self) -> "BilinearWeights":
        if len(self.shape) == 1:
            (n,) = self.shape
            # constant has shape (n,); build the (1, n) transpose explicitly
            transposed_constant = dok_ndarray(
                (1, n), {(0, k[0]): v for k, v in self.constant.keys.items()}
            )
            return BilinearWeights(
                self.memory,
                shape=(1, n),
                constant=transposed_constant,
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

    def to_export_dict(self):
        output_count = int(np.prod(self.shape, dtype=int))
        homogeneous_count = self.memory.count + 1
        entries = []

        for key, value in sorted(self.constant.keys.items()):
            row_index = int(np.ravel_multi_index(key, self.shape, order="C"))
            entries.append({"index": [row_index, 0, 0], "value": float(value)})

        for key, value in sorted(self.linear.keys.items()):
            row_index = int(
                np.ravel_multi_index(key[:-1], self.shape, order="C")
            )
            entries.append(
                {
                    "index": [row_index, int(key[-1]) + 1, 0],
                    "value": float(value),
                }
            )

        for key, value in sorted(self.quadratic.keys.items()):
            row_index = int(
                np.ravel_multi_index(key[:-2], self.shape, order="C")
            )
            entries.append(
                {
                    "index": [
                        row_index,
                        int(key[-2]) + 1,
                        int(key[-1]) + 1,
                    ],
                    "value": float(value),
                }
            )

        return {
            "memory": self.memory.to_export_dict(),
            "shape": list(self.shape),
            "quadratic": {
                "shape": [output_count, homogeneous_count, homogeneous_count],
                "entries": entries,
            },
        }

    def is_scalar(self):
        return self.shape == (1,)

    @property
    def is_constant(self):
        return not self.linear.keys and not self.quadratic.keys

    @property
    def is_linear(self):
        return not self.quadratic.keys

    @property
    def is_quadratic(self):
        return bool(self.quadratic.keys)

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
            return other.flat[0] * self
        except (AttributeError, AssertionError):
            pass

        if isinstance(other, np.ndarray) and other.shape == self.shape:
            # Componentwise scaling: result[i...] = self[i...] * other[i...]
            n_output_dims = len(self.shape)
            constant = _scale_tensor_by_array(
                self.constant, other, n_output_dims
            )
            linear = _scale_tensor_by_array(self.linear, other, n_output_dims)
            quadratic = _scale_tensor_by_array(
                self.quadratic, other, n_output_dims
            )
            return BilinearWeights(
                self.memory, self.shape, constant, linear, quadratic
            )

        if self.is_scalar():
            if isinstance(other, BilinearWeights):
                assert (
                    self.memory == other.memory
                ), "Cannot multiply weights with different source"
                if self.is_constant:
                    return float(self.constant) * other
                if other.is_constant and other.is_scalar():
                    return float(other.constant) * self

                if self.is_linear and other.is_linear:
                    memory_count = self.memory.count
                    constant_value = self.constant[(0,)] * other.constant[(0,)]
                    linear_data = {}
                    for key, value in self.linear.keys.items():
                        contribution = other.constant[(0,)] * value
                        if contribution:
                            linear_data[key] = (
                                linear_data.get(key, 0.0) + contribution
                            )
                    for key, value in other.linear.keys.items():
                        contribution = self.constant[(0,)] * value
                        if contribution:
                            linear_data[key] = (
                                linear_data.get(key, 0.0) + contribution
                            )
                    quadratic_data = {}
                    for self_key, self_value in self.linear.keys.items():
                        for (
                            other_key,
                            other_value,
                        ) in other.linear.keys.items():
                            target = (0, self_key[-1], other_key[-1])
                            quadratic_data[target] = (
                                quadratic_data.get(target, 0.0)
                                + self_value * other_value
                            )
                    constant = (
                        dok_ndarray((1,), {(0,): constant_value})
                        if constant_value
                        else dok_ndarray((1,))
                    )
                    return BilinearWeights.from_trusted_dok(
                        self.memory,
                        (1,),
                        constant,
                        dok_ndarray((1, memory_count), linear_data),
                        dok_ndarray(
                            (1, memory_count, memory_count),
                            quadratic_data,
                        ),
                    )

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
                linear = outer_product(other, self.linear)
                quadratic = outer_product(other, self.quadratic)
                return BilinearWeights(
                    self.memory, result_shape, constants, linear, quadratic
                )

        raise TypeError(f"Cannot multiply {self} by {type(other)}")

    def __rmul__(self, other):
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
        raise TypeError(f"Cannot subtract {type(other)}")

    def __rsub__(self, other):
        return (-self).__add__(other)

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

            def project(tensor: dok_ndarray) -> dok_ndarray:
                if isinstance(other, np.ndarray):
                    return tensor.__rmatmul__(other)
                return other @ tensor

            constant = project(self.constant)
            linear = project(self.linear)
            quadratic = project(self.quadratic)
            return BilinearWeights.from_trusted_dok(
                self.memory, constant.shape, constant, linear, quadratic
            )

        raise TypeError(f"Cannot matmul {type(other)}")

    def __matmul__(self, other):
        if isinstance(other, (np.ndarray, dok_ndarray)):
            assert len(other.shape) >= 1
            assert self.shape[-1] == other.shape[0]
            col = len(self.shape) - 1
            vector = (
                dok_ndarray.fromarray(other)
                if isinstance(other, np.ndarray)
                else other
            )

            def contract(tensor: dok_ndarray, memory_rank: int) -> dok_ndarray:
                result = tensor_sum(tensor, vector, l_index=col, r_index=0)
                # tensor_sum places the left memory axes before right output
                # axes; move them to the trailing positions used by
                # BilinearWeights.
                right_rank = len(other.shape) - 1
                for memory_index in reversed(range(memory_rank)):
                    position = col + memory_index
                    for _ in range(right_rank):
                        result = result.swap_indices(position, position + 1)
                        position += 1
                return result

            shape = (*self.shape[:-1], *other.shape[1:])
            if not shape:
                shape = (1,)
            return BilinearWeights.from_trusted_dok(
                self.memory,
                shape,
                contract(self.constant, 0),
                contract(self.linear, 1),
                contract(self.quadratic, 2),
            )

        assert (
            isinstance(other, BilinearWeights) and other.memory is self.memory
        )
        assert (
            (self.is_linear and other.is_linear)
            or self.is_constant
            or other.is_constant
        )
        col = len(self.shape) - 1

        def _contract(
            lhs: dok_ndarray,
            rhs: dok_ndarray,
            lhs_memory_rank: int,
        ) -> dok_ndarray:
            result = tensor_sum(lhs, rhs, l_index=col, r_index=0)
            right_output_rank = len(other.shape) - 1
            for memory_index in reversed(range(lhs_memory_rank)):
                position = col + memory_index
                for _ in range(right_output_rank):
                    result = result.swap_indices(position, position + 1)
                    position += 1
            return result

        constant = _contract(self.constant, other.constant, 0)
        linear = _contract(self.constant, other.linear, 0) + _contract(
            self.linear, other.constant, 1
        )
        linear_linear_quadratic = _contract(self.linear, other.linear, 1)
        quadratic = (
            _contract(self.constant, other.quadratic, 0)
            + _contract(self.quadratic, other.constant, 2)
            + linear_linear_quadratic
        )
        shape = (*self.shape[:-1], *other.shape[1:])
        return BilinearWeights.from_trusted_dok(
            self.memory, shape, constant, linear, quadratic
        )

    def reshape(
        self, newshape: Tuple[int, ...], order="C"
    ) -> "BilinearWeights":
        if order != "C":
            raise NotImplementedError("Only C-order reshape is supported")
        assert int(np.prod(self.shape)) == int(np.prod(newshape))

        def _reshape_tensor(tensor: dok_ndarray) -> dok_ndarray:
            if tensor.is_empty():
                return dok_ndarray(
                    (*newshape, *tensor.shape[len(self.shape) :])
                )
            data = {}
            for key, value in tensor.keys.items():
                out_key = key[: len(self.shape)]
                mem_key = key[len(self.shape) :]
                flat_index = np.ravel_multi_index(
                    out_key, self.shape, order="C"
                )
                new_out_key = np.unravel_index(flat_index, newshape, order="C")
                data[(*new_out_key, *mem_key)] = value
            return dok_ndarray(
                (*newshape, *tensor.shape[len(self.shape) :]), data
            )

        return BilinearWeights.from_trusted_dok(
            self.memory,
            newshape,
            constant=_reshape_tensor(self.constant),
            linear=_reshape_tensor(self.linear),
            quadratic=_reshape_tensor(self.quadratic),
        )

    def extend_memory(self, memory: MemorySpec) -> "BilinearWeights":
        assert memory.location == 0
        assert self.memory.location == 0
        assert memory.count >= self.memory.count
        if memory.count == self.memory.count:
            return self.clone()

        linear = dok_ndarray(
            (*self.shape, memory.count),
            {k: v for k, v in self.linear.keys.items()},
        )
        quadratic = dok_ndarray(
            (*self.shape, memory.count, memory.count),
            {k: v for k, v in self.quadratic.keys.items()},
        )
        return BilinearWeights.from_trusted_dok(
            memory,
            self.shape,
            constant=self.constant.clone(),
            linear=linear,
            quadratic=quadratic,
        )

    def clone(self):
        return BilinearWeights.from_trusted_dok(
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
            return self.__mul__(args)

        if ufunc == np.add and method == "__call__":
            return self.__add__(args)

        if ufunc == np.subtract and method == "__call__":
            return self.__rsub__(args)

        raise NotImplementedError(f"{ufunc} not implemented")

    def __truediv__(self, other):
        if isinstance(other, scalar):
            return BilinearWeights.from_trusted_dok(
                self.memory,
                self.shape,
                self.constant / other,
                self.linear / other,
                self.quadratic / other,
            )

        if isinstance(other, np.ndarray):
            if other.size == 1:
                return self / float(other)
            return self * (1.0 / other)

        raise TypeError(f"Cannot divide {self} by {type(other)}")

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
            (self.is_linear and rhs.is_linear)
            or (not self.is_linear and rhs.is_constant)
            or (self.is_constant and not rhs.is_linear)
        ), (
            "dot requires both operands order<=1, or one order==2 and "
            "the other order==0"
        )
        memory_count = self.memory.count
        output_rank = len(self.shape)
        constant_value = 0.0
        linear_data = {}
        quadratic_data = {}

        for output_key, value in self.constant.keys.items():
            constant_value += value * rhs.constant[output_key]
        for key, value in self.linear.keys.items():
            output_key = key[:output_rank]
            memory_index = key[-1]
            contribution = rhs.constant[output_key] * value
            if contribution:
                target = (0, memory_index)
                linear_data[target] = (
                    linear_data.get(target, 0.0) + contribution
                )
        for key, value in rhs.linear.keys.items():
            output_key = key[:output_rank]
            memory_index = key[-1]
            contribution = self.constant[output_key] * value
            if contribution:
                target = (0, memory_index)
                linear_data[target] = (
                    linear_data.get(target, 0.0) + contribution
                )

        for source, other_constant in (
            (self.quadratic, rhs.constant),
            (rhs.quadratic, self.constant),
        ):
            for key, value in source.keys.items():
                output_key = key[:output_rank]
                contribution = other_constant[output_key] * value
                if contribution:
                    target = (0, key[-2], key[-1])
                    quadratic_data[target] = (
                        quadratic_data.get(target, 0.0) + contribution
                    )

        rhs_linear_by_output = {}
        for key, value in rhs.linear.keys.items():
            rhs_linear_by_output.setdefault(key[:output_rank], []).append(
                (key[-1], value)
            )
        for key, value in self.linear.keys.items():
            output_key = key[:output_rank]
            for rhs_memory_index, rhs_value in rhs_linear_by_output.get(
                output_key, ()
            ):
                target = (0, key[-1], rhs_memory_index)
                quadratic_data[target] = (
                    quadratic_data.get(target, 0.0) + value * rhs_value
                )

        constant = (
            dok_ndarray((1,), {(0,): constant_value})
            if constant_value
            else dok_ndarray((1,))
        )
        linear = dok_ndarray((1, memory_count), linear_data)
        quadratic = dok_ndarray(
            (1, memory_count, memory_count), quadratic_data
        )

        return BilinearWeights.from_trusted_dok(
            self.memory, (1,), constant, linear, quadratic
        )

    @staticmethod
    def identity2(memory: MemorySpec):
        shape = (memory.count,)
        data = {(i, i): 1 for i in range(memory.count)}
        linear = dok_ndarray((memory.count, memory.count), data)
        return BilinearWeights.from_trusted_dok(memory, shape, linear=linear)

    @staticmethod
    def project(memory: MemorySpec, spec: MemorySpec, shape: tuple):
        data = {}
        for k in range(spec.count):
            multi_idx = np.unravel_index(k, shape, order="C")
            data[(*multi_idx, spec.location + k)] = 1
        linear = dok_ndarray((*shape, memory.count), data)
        return BilinearWeights.from_trusted_dok(memory, shape, linear=linear)

    @staticmethod
    def reshape_identity(memory: MemorySpec, shape: tuple):
        return BilinearWeights.project(
            memory, MemorySpec(memory.location, memory.count), shape
        )


def compose_bilinear_weights(
    outer: BilinearWeights, inner: BilinearWeights
) -> BilinearWeights | None:
    """Substitute ``inner`` into ``outer`` while preserving degree two.

    ``outer`` is a polynomial in the coordinates produced by ``inner``.
    Composition is exact when the resulting polynomial has degree at most two;
    higher-order terms are rejected rather than silently discarded.
    """
    assert outer.memory.count == inner.shape[0]
    if len(outer.shape) != 1 or (outer.is_quadratic and inner.is_quadratic):
        return None
    memory = inner.memory
    output_shape = outer.shape
    constant = {}
    linear = {}
    quadratic = {}
    inner_linear_by_row = {}
    for (row, coordinate), value in inner.linear.keys.items():
        inner_linear_by_row.setdefault(row, []).append((coordinate, value))
    inner_quadratic_by_row = {}
    for (row, left, right), value in inner.quadratic.keys.items():
        inner_quadratic_by_row.setdefault(row, []).append((left, right, value))
    outer_linear_by_output = {}
    for (*output_index, coordinate), value in outer.linear.keys.items():
        outer_linear_by_output.setdefault(tuple(output_index), []).append(
            (coordinate, value)
        )
    outer_quadratic_by_output = {}
    for (*output_index, left, right), value in outer.quadratic.keys.items():
        outer_quadratic_by_output.setdefault(tuple(output_index), []).append(
            (left, right, value)
        )

    def add(table, key, value):
        if value:
            table[key] = table.get(key, 0.0) + value

    for output_index in np.ndindex(output_shape):
        c = outer.constant.keys.get(output_index, 0.0)
        l_terms = outer_linear_by_output.get(output_index, ())
        q_terms = outer_quadratic_by_output.get(output_index, ())
        for index, value in l_terms:
            c += value * inner.constant.keys.get((index,), 0.0)
        for left, right, value in q_terms:
            c += (
                value
                * inner.constant.keys.get((left,), 0.0)
                * inner.constant.keys.get((right,), 0.0)
            )
        if c:
            constant[output_index] = c

        for index, value in l_terms:
            for coordinate, inner_value in inner_linear_by_row.get(index, ()):
                add(
                    linear,
                    (*output_index, coordinate),
                    value * inner_value,
                )
            for left, right, inner_value in inner_quadratic_by_row.get(
                index, ()
            ):
                add(
                    quadratic,
                    (*output_index, left, right),
                    value * inner_value,
                )
        for left, right, value in q_terms:
            left_constant = inner.constant.keys.get((left,), 0.0)
            right_constant = inner.constant.keys.get((right,), 0.0)
            for coordinate, inner_value in inner_linear_by_row.get(left, ()):
                add(
                    linear,
                    (*output_index, coordinate),
                    value * inner_value * right_constant,
                )
            for coordinate, inner_value in inner_linear_by_row.get(right, ()):
                add(
                    linear,
                    (*output_index, coordinate),
                    value * inner_value * left_constant,
                )
            for i, j, inner_value in inner_quadratic_by_row.get(left, ()):
                add(
                    quadratic,
                    (*output_index, i, j),
                    value * inner_value * right_constant,
                )
            for i, j, inner_value in inner_quadratic_by_row.get(right, ()):
                add(
                    quadratic,
                    (*output_index, i, j),
                    value * inner_value * left_constant,
                )
            for i, left_value in inner_linear_by_row.get(left, ()):
                for j, right_value in inner_linear_by_row.get(right, ()):
                    add(
                        quadratic,
                        (*output_index, i, j),
                        value * left_value * right_value,
                    )

    # Drop exact cancellation, keeping sparse tensors compact.
    for table in (constant, linear, quadratic):
        for key in [key for key, value in table.items() if value == 0.0]:
            del table[key]

    return BilinearWeights.from_trusted_dok(
        memory,
        output_shape,
        constant=dok_ndarray((*output_shape,), constant),
        linear=dok_ndarray((*output_shape, memory.count), linear),
        quadratic=dok_ndarray(
            (*output_shape, memory.count, memory.count), quadratic
        ),
    )


def outer_product(lhs: dok_ndarray, rhs: dok_ndarray):
    assert rhs.shape[0] == 1
    shape = (*lhs.shape, *rhs.shape[1:])
    data = {}
    for k_l, v_l in lhs.keys.items():
        for k_r, v_r in rhs.keys.items():
            key = tuple((*k_l, *k_r[1:]))
            data[key] = v_l * v_r

    return dok_ndarray(shape, data)
