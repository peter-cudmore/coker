import numpy as np
import scipy as scp
import sympy as sp

from coker.algebra.dimensions import Dimension


scalar_types = (
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    float,
    complex,
    int,
    bool,
    np.bool_,
)


def is_scalar_symbol(value):
    if isinstance(value, sp.MatrixSymbol):
        return value.shape == (1, 1)
    return isinstance(value, (sp.Symbol, sp.Expr))


def reshape_sympy_matrix(arg, shape):
    if len(shape) == 1:
        output_shape = (shape[0], 1)
    else:
        assert len(shape) == 2
        output_shape = shape

    if arg.shape == (*output_shape, 1):
        return arg[:, :, 0]

    old_cols = 1 if len(arg.shape) == 1 else arg.shape[1]

    def lookup(row_index, column_index):
        offset = row_index * output_shape[1] + column_index
        old_row = offset // old_cols
        old_col = offset % old_cols
        return arg[old_row, old_col]

    return sp.Matrix(*output_shape, lookup)


def reshape(arg, dim: Dimension):
    if dim.is_scalar():
        if isinstance(arg, scalar_types) or is_scalar_symbol(arg):
            return arg
        try:
            (inner,) = arg
        except ValueError as ex:
            raise TypeError(f"Expecting a scalar, got {arg}") from ex
        except TypeError as ex:
            raise TypeError(f"Expecting a scalar, got {arg}") from ex
        return reshape(inner, dim)

    if isinstance(
        arg, (sp.Matrix, sp.Array, sp.MatrixSlice, sp.ImmutableMatrix)
    ):
        if arg.shape == dim.dim:
            return arg
        return reshape_sympy_matrix(arg, dim.dim)
    if isinstance(arg, np.ndarray):
        return np.reshape(arg, dim.dim)
    if scp.sparse.issparse(arg):
        return np.reshape(arg.toarray(), dim.dim)
    if isinstance(arg, (float, int)):
        return np.array([arg]).reshape(dim.dim)
    if arg is None:
        return arg
    raise NotImplementedError(f"Dont know how to reshape {arg}")
