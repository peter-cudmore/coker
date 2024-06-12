import numpy as np
from typing import Optional, Dict, Tuple, Union, NewType

MultiIndex = NewType('MultiIndex', Union[Tuple[int, ...], int])


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
    def fromarray(other: np.ndarray):
        shape = other.shape
        keys = {
            item.index: float(item[0])
            for item in np.nditer(other, flags=['multi_index'])
            if float(item[0]) != 0.0
        }
        return dok_ndarray(shape, keys)

    def __neg__(self):
        keys = {k: -v for k,v in self.keys.items()}
        return dok_ndarray(self.shape, keys)

    def __mul__(self, other):
        assert isinstance(other, (float, int))

        if other == 0:
            return dok_ndarray(self.shape, {})

        keys = {
            k: v * other for k, v in self.keys.items()
        }
        return dok_ndarray(self.shape, keys)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, dok_ndarray):
            keys = self.keys.copy()
            for k in other.keys.items():
                if k in keys:
                    keys[k] += other[k]
                else:
                    keys[k] = other[k]
        else:
            raise NotImplementedError()
        return dok_ndarray(self.shape, keys)

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

    def reshape(self, shape):
        if len(shape) > len(self.shape) and all(s_i == s_n for s_i, s_n in zip(shape, (*self.shape, 1))):
            self.keys = {(*k, 0): v for k, v in self.keys.items()}
            self.shape = (*shape[:-1], *shape[1:])
            return

        if len(shape) == len(self.shape) and all(s_i >= s_n for s_i, s_n in zip(shape, self.shape)):
            self.shape = shape
            return
        raise NotImplementedError(f"Don't know how to reshape a {self.shape} tensor into a {shape} tensor")
