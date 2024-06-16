import numpy as np
from typing import Optional, Dict, Tuple, Union, NewType

MultiIndex = NewType('MultiIndex', Union[Tuple[int, ...], int])

scalar = (float, int)

class dok_ndarray:
    def __init__(self, shape, data: Optional[Dict[MultiIndex, float]] = None):
        self.shape = shape
        self.keys = data if data is not None else {}

    def __setitem__(self, key, value):
        self.keys[key] = value

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key, )
        return self.keys[key]

    def clone(self) -> 'dok_ndarray':
        return dok_ndarray(self.shape, self.keys.copy())

    def toarray(self):
        m = np.zeros(shape=self.shape)
        for k, v in self.keys.items():
            m[k] = v

        return m

    @staticmethod
    def zeros(shape):
        return dok_ndarray(shape)

    @staticmethod
    def eye(n):
        shape = (n, n)
        keys = {(i, i): 1 for i in range(n)}

        return dok_ndarray(shape, keys)

    @staticmethod
    def fromarray(other: np.ndarray):
        shape = other.shape
        keys ={}
        with np.nditer(other, op_flags=['readonly'], flags=['multi_index','reduce_ok']) as it:
            for item in it:
                if item != 0:
                    index = tuple(it.multi_index)
                    keys[index] = float(item)

        return dok_ndarray(shape, keys)

    @staticmethod
    def from_maybe(arg: Optional[Union[np.ndarray, 'dok_ndarray']],
                        expected_shape: Optional[Tuple[int, ...]]=None) -> 'dok_ndarray':

        if isinstance(arg, np.ndarray):
            assert expected_shape is None or arg.shape == expected_shape
            return dok_ndarray.fromarray(arg)
        if isinstance(arg, dok_ndarray):
            assert expected_shape is None or arg.shape == expected_shape
            return arg
        if arg is None and expected_shape is not None:
            return dok_ndarray(expected_shape)

        raise TypeError(f"Don't know how to turn {arg} into an array of shape {expected_shape}")

    def __float__(self):
        if all(s == 1 for s in  self.shape):
            if len(self.keys) == 1:
                v, = self.keys.values()
                return float(v)
            else:
                assert len(self.keys) == 0
                return 0.0

        raise TypeError(F"Cannot cast a {self.shape} array to a float")

    def __neg__(self):
        keys = {k: -v for k,v in self.keys.items()}
        return dok_ndarray(self.shape, keys)

    def __mul__(self, other):
        if isinstance(other, (dok_ndarray, np.ndarray)):
            assert other.shape in {(1, ), (1, 1)}
            other = float(other)
        else:
            assert isinstance(other, (float, int)) , f"Cannot multiply by {other}"

        if other == 0:
            return dok_ndarray(self.shape, {})

        keys = {
            k: v * other for k, v in self.keys.items()
        }
        return dok_ndarray(self.shape, keys)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, scalar):
            if all(s == 1 for s in self.shape):
                k = tuple(0 for _ in self.shape)
                if self.keys:
                    data = {k: self.keys[k] + other}
                else:
                    data = {k: other}
                return dok_ndarray(self.shape, data)

        if isinstance(other, dok_ndarray):
            assert other.shape == self.shape, f"Cannot add tensors of shape {other.shape} to {self.shape}"
            keys = self.keys.copy()
            for k,v in other.keys.items():
                if k in keys:
                    keys[k] += v
                else:
                    keys[k] = v
            return dok_ndarray(self.shape, keys)

        raise NotImplementedError(f"Cannot add {other} to a tensor of shape {self.shape}")

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

    def __matmul__(self, other):
        assert self.shape[-1] == other.shape[0]
        keys = {}
        for k, v in self.keys.items():
            *k_prime, i = k
            k_prime = tuple(k_prime)
            if k_prime in keys:
                keys[k_prime] += v * other[i]
            else:
                keys[k_prime] = v * other[i]

        shape = (*self.shape[:-1], *other.shape[1:])

        return dok_ndarray(shape, keys)

    def __array_ufunc__(self, ufunc, method, args, out=None):
        if ufunc == np.matmul and method == '__call__':
            return args @ self.toarray()
        raise NotImplementedError

    def reshape(self, shape):
        if len(shape) > len(self.shape) and all(s_i == s_n for s_i, s_n in zip(shape, (*self.shape, 1))):
            self.keys = {(*k, 0): v for k, v in self.keys.items()}
            self.shape = (*shape[:-1], *shape[1:])
            return

        if len(shape) == len(self.shape) and all(s_i >= s_n for s_i, s_n in zip(shape, self.shape)):
            self.shape = shape
            return
        raise NotImplementedError(f"Don't know how to reshape a {self.shape} tensor into a {shape} tensor")


def tensor_vector_product(tensor: dok_ndarray, vector: np.ndarray, axis=1):
    assert isinstance(axis, int) and 0 <= axis < len(tensor.shape)

    shape = tuple(s for i,s in enumerate(tensor.shape) if i is not axis)
    new_data = {}
    for k, v in tensor.keys.items():
        i = k[axis]
        entry = float(v * vector[i])
        new_key = (k_i for i, k_i in enumerate(k) if i is not axis)
        if new_key in new_data:
            new_data[new_key] += entry
        else:
            new_data[new_key] = entry

    return dok_ndarray(shape, new_data)


def tensor_sum(lhs: dok_ndarray, rhs, l_index=0, r_index=0):
    assert lhs.shape[l_index] == rhs.shape[r_index]

    new_shape = (*[s for i, s in enumerate(lhs.shape) if i is not l_index],
                 *[s for i, s in enumerate(rhs.shape) if i is not r_index])

    new_data = {}
    for k_l, v_l in lhs.keys.items():
        for k_r, v_r in rhs.keys.items():
            if k_l[l_index] == k_r[r_index]:
                new_key = (
                    *k_l[0:l_index], *k_l[l_index + 1:],
                    *k_r[0:r_index], *k_r[r_index + 1:]
                )

                v = v_l * v_r
                if new_key in new_data:
                    new_data[new_key] += v
                else:
                    new_data[new_key] = v
    return dok_ndarray(new_shape, new_data)