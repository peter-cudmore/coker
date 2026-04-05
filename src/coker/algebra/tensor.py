import numpy as np


class SymbolicVector:
    """A mutable array that can hold a mix of constants and symbolic tracers.

    Use this inside an ``implementation`` passed to :func:`~coker.algebra.kernel.function`
    when you need to build an output vector by assigning to individual
    elements, some of which may be symbolic expressions and some constants.

    After all assignments are made, the vector is collapsed into a single
    tape node via :meth:`collapse`.  Coker calls ``collapse`` automatically
    when the implementation returns a ``SymbolicVector``.

    Example:
        >>> from coker import function, VectorSpace, SymbolicVector
        >>> def impl(x):
        ...     r = SymbolicVector((2,))
        ...     r[0] = 1.0          # constant
        ...     r[1] = x[0] + x[1]  # symbolic
        ...     return r
        >>> import numpy as np
        >>> f = function([VectorSpace("x", 2)], impl, backend="numpy")
        >>> f(np.array([3.0, 4.0]))
        array([1., 7.])
    """

    def __init__(self, shape, data=None, symbolics=None):

        self.shape = shape
        self.dense_consts = data if data is not None else np.zeros(shape)
        self.symbolics = symbolics if symbolics is not None else {}

    @staticmethod
    def from_array(array: np.ndarray):

        assert array.dtype != np.object_, f"{array}"
        return SymbolicVector(array.shape, array)

    @staticmethod
    def zeros(shape):
        return SymbolicVector(shape)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._set_row(key, value)
        else:
            raise NotImplementedError

    def __add__(self, other):
        if not self.symbolics:
            return self.dense_consts + other
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if not self.symbolics:
            return self.dense_consts - other
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if not self.symbolics:
            return self.dense_consts * other
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if not self.symbolics:
            return other * self.dense_consts
        else:
            raise NotImplementedError

    def _set_row(self, index: int, value):
        if isinstance(value, (float, int)):
            self.dense_consts[index] = value
        else:
            self.symbolics[index] = value

    def collapse(self):
        out = self.dense_consts
        for key, value in self.symbolics.items():
            p = projector_from_key(self.shape, key)
            out += value * p

        return out

    @staticmethod
    def from_list(data):
        shape = (len(data),)
        dense = np.zeros(shape)
        symbols = {}

        for row, val in enumerate(data):

            if isinstance(val, (float, int)):
                dense[row] = val

            elif isinstance(val, (list, tuple)):
                raise NotImplementedError
            else:
                symbols[row] = val

        if not symbols:
            return dense
        else:
            return SymbolicVector(shape, dense, symbols)


def projector_from_key(shape, key):
    assert isinstance(key, int)
    basis = np.zeros(shape)
    basis[key] = 1
    return basis
