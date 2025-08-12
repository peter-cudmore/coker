import operator
from functools import reduce

import numpy as np
from typing import Optional, Dict, Tuple, Union, NewType

MultiIndex = NewType("MultiIndex", Union[Tuple[int, ...], int])

scalar = (float, int, np.int32, np.int64, np.float32, np.float64)
NDArray = NewType("NDArray", Union[np.ndarray, "dok_ndarray"])


def is_constant(a):
    return (
        isinstance(a, scalar)
        or isinstance(a, np.ndarray)
        or isinstance(a, dok_ndarray)
    )


def cast_vector(arg, dim):
    if len(arg.shape) == 1:
        if arg.shape[0] == dim:
            return arg
    elif len(arg.shape) == 2:
        if arg.shape[0] == dim and arg.shape[1] == 1:
            return arg.reshape((dim, 1))

    raise TypeError(f"Cannot cast {arg.shape} vector to a {(dim,)} vector")


class dok_ndarray(np.lib.mixins.NDArrayOperatorsMixin):
    """Sparse Multilinear map

    Understood as a vector of linear tensor fields.
    Some notation:

    - `e^j` always refers to the basis vectors for the V_i vector space (i.e. a column vector)
    - `E_i` always refers to the basis covectors for the V_j vector space (i.e. a row vector)


    M = \alpha^i_{jkm} E_i*e^j*e^k*e^m...

    Operations:
        - addition, subtraction, and scalar multiplication as usual
        - 'matmul' is tensor contraction along the last + first dimensions
           that is, if N @ M means that if
                N = n_{ab...di}E_a e^b ...e^d e^i
           and
                M = m_{ijk...}E_ie^je^k...
           then
                NM = (nm)_{ab...djk}E_a e^b ...  e^d * e^je^k ...
           with
                the coefficients (nm) being summed appropriately.

        -
    """

    def __init__(self, shape, data: Optional[Dict[MultiIndex, float]] = None):

        self.keys = {}
        if len(shape) == 1:

            shape = (shape[0], 1)
            if data is not None:
                for k in data.keys():
                    if len(k) == 2:
                        assert k[1] == 0
                        self.keys[k] = data[k]
                    elif len(k) == 1:
                        self.keys[(*k, 0)] = data[k]
                    else:
                        raise TypeError("Failed to store data ")
            else:
                self.keys = {}
        else:
            self.keys = data if data is not None else {}
        self.shape = shape

    def is_empty(self):
        return self.keys == {}

    def is_scalar(self):
        return all(s == 1 for s in self.shape)

    def is_vector(self):
        return len(self.shape) == 2 and self.shape[1] == 1

    def __setitem__(self, key, value):
        if isinstance(key, int):
            assert self.is_vector()
            key = (key, 0)
        if isinstance(key, tuple) and len(key) == 1:
            key = (key[0], 0)

        if value != 0:
            self.keys[key] = value
        else:
            if key in self.keys:
                del self.keys[key]

    def __getitem__(self, key):
        if isinstance(key, int):
            assert self.is_vector()
            key = (key, 0)

        try:
            return self.keys[key]
        except KeyError:
            return 0

    def clone(self) -> "dok_ndarray":
        return dok_ndarray(self.shape, self.keys.copy())

    def toarray(self):
        m = np.zeros(shape=self.shape)

        for k, v in self.keys.items():
            m[k] = v

        if self.is_vector():
            return m.flatten()

        return m

    def swap_indices(self, i, j):
        assert i != j and i < len(self.shape) and j < len(self.shape)
        if i > j:
            i, j = j, i

        data = {}
        for k, v in self.keys.items():
            k_i, k_j = k[i], k[j]
            k_prime = (*k[0:i], k_j, *k[i + j : j], k_i, *k[j + 1 :])
            data[k_prime] = v
        s_i, s_j = self.shape[i], self.shape[j]
        shape = (
            *self.shape[0:i],
            s_j,
            *self.shape[i + j : j],
            s_j,
            *self.shape[j + 1 :],
        )

        return dok_ndarray(shape=shape, data=data)

    @staticmethod
    def zeros(shape):
        if isinstance(shape, int):
            shape = (shape, 1)

        if len(shape) == 1:
            shape = (shape[0], 1)

        return dok_ndarray(shape)

    @staticmethod
    def eye(n):
        shape = (n, n)
        keys = {(i, i): 1 for i in range(n)}

        return dok_ndarray(shape, keys)

    @staticmethod
    def fromarray(other: np.ndarray):
        if len(other.shape) == 1:
            other = other.reshape((-1, 1))

        shape = other.shape

        keys = {}
        with np.nditer(
            other, op_flags=["readonly"], flags=["multi_index", "reduce_ok"]
        ) as it:
            for item in it:
                if item != 0:
                    index = tuple(it.multi_index)
                    keys[index] = float(item)

        return dok_ndarray(shape, keys)

    @staticmethod
    def from_maybe(
        arg: Optional[Union[np.ndarray, "dok_ndarray"]],
        expected_shape: Optional[Tuple[int, ...]] = None,
    ) -> "dok_ndarray":

        if isinstance(arg, np.ndarray):
            result = dok_ndarray.fromarray(arg)
            return result
        if isinstance(arg, dok_ndarray):

            return arg.clone()
        if arg is None and expected_shape is not None:
            return dok_ndarray(expected_shape)

        raise TypeError(
            f"Don't know how to turn {arg} into an array of shape {expected_shape}"
        )

    @property
    def T(self):

        if len(self.shape) != 2:
            raise NotImplementedError(
                "Can't unambiguously transpose a higher dimensional array"
            )
        return dok_ndarray(
            tuple(reversed(self.shape)),
            {(k2, k1): v for (k1, k2), v in self.keys.items()},
        )

    def __float__(self):
        if self.is_scalar():
            if len(self.keys) == 1:
                (v,) = self.keys.values()
                return float(v)
            else:
                assert len(self.keys) == 0
                return 0.0

        raise TypeError(f"Cannot cast a {self.shape} array to a float")

    def __neg__(self):
        keys = {k: -v for k, v in self.keys.items()}
        return dok_ndarray(self.shape, keys)

    def __mul__(self, other):
        if isinstance(other, (dok_ndarray, np.ndarray)):
            assert other.shape in {(1,), (1, 1)}
            other = float(other)
        else:
            assert isinstance(
                other, (float, int)
            ), f"Cannot multiply by {other}"

        if other == 0:
            return dok_ndarray(self.shape, {})

        keys = {k: v * other for k, v in self.keys.items()}
        return dok_ndarray(self.shape, keys)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if not other:
            return self
        if isinstance(other, scalar):
            if all(s == 1 for s in self.shape):
                k = tuple(0 for _ in self.shape)
                if self.keys:
                    data = {k: self.keys[k] + other}
                else:
                    data = {k: other}
                return dok_ndarray(self.shape, data)

        if isinstance(other, dok_ndarray):
            assert (
                other.shape == self.shape
            ), f"Cannot add tensors of shape {other.shape} to {self.shape}"
            keys = self.keys.copy()
            for k, v in other.keys.items():
                if k in keys:
                    keys[k] += v
                else:
                    keys[k] = v
            return dok_ndarray(self.shape, keys)

        raise NotImplementedError(
            f"Cannot add {other} to a tensor of shape {self.shape}"
        )

    def __sub__(self, other):
        if isinstance(other, dok_ndarray):
            keys = self.keys.copy()
            for k in other.keys.items():
                if k in keys:
                    keys[k] -= other[k]
                else:
                    keys[k] = -other[k]
        else:
            raise NotImplementedError()
        return dok_ndarray(self.shape, keys)

    def evaluate(self, *vectors: NDArray) -> NDArray:
        #        n_vector = len(vectors)

        vectors = [
            cast_vector(v, s)
            for v, s in zip(vectors, self.shape[-len(vectors) :])
        ]
        data = {}
        for k, v in self.keys.items():
            i, *k_prod = k
            a_k = v * reduce(
                operator.mul, (vec[j] for vec, j in zip(vectors, k_prod)), 1
            )
            if (i,) not in data:
                data[(i, 0)] = a_k
            else:
                data[(i, 0)] += a_k

        shape = self.shape[: -len(vectors)]
        return dok_ndarray(shape, data)

    def __matmul__(self, other: Union[NDArray, Tuple[NDArray]]):
        """SymbolicVector contraction

        it T = a_{ijk} e_i e^j e^k

        """
        if isinstance(other, tuple):
            return self.evaluate(*other)

        elif isinstance(other, (dok_ndarray, np.ndarray)):
            if isinstance(other, np.ndarray) and len(other.shape) == 1:
                other = other.reshape((-1, 1))

            if self.shape[-1] != other.shape[0]:
                #
                # self.shape == (3,)
                # other.shape == (1, 3)
                # -> reinterpret self.shape as (3,1)

                raise TypeError(
                    f"Cannot multiply {self.shape} @ {other.shape}"
                )
        else:
            return other.__rmatmul__(self)
        shape = (*self.shape[:-1], *other.shape[1:])

        if not self.keys:
            return dok_ndarray(shape, {})

        assert len(self.shape) > 1

        if isinstance(other, np.ndarray):
            return tensor_ndarray_product(self, other)
        else:
            return tensor_sum(self, other, l_index=len(self.shape) - 1)

        return dok_ndarray(shape, keys)

    def __rmatmul__(self, other):
        try:
            assert (
                other.shape[-1] == self.shape[0]
            ), f"Can't multiple {other.shape} x {self.shape}"
        except AssertionError as ex:
            raise ex

        shape = (*other.shape[:-1], *self.shape[1:])
        data = {}
        with np.nditer(
            other, flags=["multi_index"], op_flags=["readonly"]
        ) as it:
            for v in it:
                *key_1, i_1 = it.multi_index
                for (i_2, *key_2), v_2 in self.keys.items():
                    if i_1 != i_2:
                        continue
                    key = (*key_1, *key_2)
                    if key in data:
                        data[key] += v_2 * float(v)
                    else:
                        data[key] = v_2 * float(v)
        return dok_ndarray(shape, data)

    def __eq__(self, other):
        if isinstance(other, dok_ndarray):
            return self.shape == other.shape and self.keys == other.keys
        if isinstance(other, np.ndarray):
            raise NotImplementedError

        if isinstance(other, scalar) and self.is_scalar():
            return float(self) == float(other)

        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, args, out=None):

        if ufunc == np.matmul and method == "__call__":
            return self.__rmatmul__(args).toarray()
        if ufunc == np.dot and method == "__call__":
            return args.T @ self

        if ufunc == np.multiply and method == "__call__":
            if args == 1.0:
                return self
            if self.is_scalar() or isinstance(args, scalar):
                return args * self

        raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, dok_ndarray):
            other = float(other)
        assert isinstance(other, scalar)
        assert other != 0

        return dok_ndarray(
            self.shape, {k: v / other for k, v in self.keys.items()}
        )

    def reshape(self, shape):
        if len(shape) > len(self.shape) and all(
            s_i == s_n for s_i, s_n in zip(shape, (*self.shape, 1))
        ):
            self.keys = {(*k, 0): v for k, v in self.keys.items()}
            self.shape = (*shape[:-1], *shape[1:])
            return

        if len(shape) == len(self.shape) and all(
            s_i >= s_n for s_i, s_n in zip(shape, self.shape)
        ):
            self.shape = shape
            return
        raise NotImplementedError(
            f"Don't know how to reshape a {self.shape} tensor into a {shape} tensor"
        )


def tensor_vector_product(tensor: dok_ndarray, vector: np.ndarray, axis=1):
    assert isinstance(axis, int) and 0 <= axis < len(tensor.shape)

    shape = tuple(s for i, s in enumerate(tensor.shape) if i is not axis)

    new_data = {}
    for k, v in tensor.keys.items():
        i = k[axis]
        entry = float(v * vector[i])
        new_key = tuple(k_i for i, k_i in enumerate(k) if i is not axis)
        if new_key in new_data:
            new_data[new_key] += entry
        else:
            new_data[new_key] = entry

    return dok_ndarray(shape, new_data)


def tensor_ndarray_product(
    tensor: dok_ndarray, array: np.ndarray, l_axis=-1, r_axis=0
):

    assert isinstance(array, np.ndarray) and isinstance(tensor, dok_ndarray)
    assert abs(l_axis) < len(tensor.shape)
    assert abs(r_axis) < len(array.shape)
    assert array.shape[r_axis] == tensor.shape[l_axis]

    if l_axis == -1:
        l_axis = len(tensor.shape) - 1

    if r_axis == -1:
        r_axis = len(array.shape) - 1

    l_shape = tuple(s for i, s in enumerate(tensor.shape) if i != l_axis)
    r_shape = tuple(s for i, s in enumerate(array.shape) if i != r_axis)
    shape = tuple((*l_shape, *r_shape))
    data = {}
    with np.nditer(array, flags=["multi_index"], op_flags=["readonly"]) as it:
        for x in it:
            x_idx_front = it.multi_index[0:r_axis]
            x_idx_tail = it.multi_index[r_axis + 1 :]
            keys = (
                k for k in tensor.keys if k[l_axis] == it.multi_index[r_axis]
            )
            for k in keys:
                k_l = k[0:l_axis]
                k_r = k[l_axis + 1 :]

                key = tuple((*k_l, *k_r, *x_idx_front, *x_idx_tail))
                if key in data:
                    data[key] += float(x) * tensor.keys[k]
                else:
                    data[key] = float(x) * tensor.keys[k]

    return dok_ndarray(shape, data)


def tensor_sum(lhs: dok_ndarray, rhs, l_index=0, r_index=0):
    assert lhs.shape[l_index] == rhs.shape[r_index]

    new_shape = (
        *[s for i, s in enumerate(lhs.shape) if i is not l_index],
        *[s for i, s in enumerate(rhs.shape) if i is not r_index],
    )

    new_data = {}
    for k_l, v_l in lhs.keys.items():
        for k_r, v_r in rhs.keys.items():
            if k_l[l_index] == k_r[r_index]:
                new_key = (
                    *k_l[0:l_index],
                    *k_l[l_index + 1 :],
                    *k_r[0:r_index],
                    *k_r[r_index + 1 :],
                )

                v = v_l * v_r
                if new_key in new_data:
                    new_data[new_key] += v
                else:
                    new_data[new_key] = v

    return dok_ndarray(new_shape, new_data)
