"""Symbolic sparse-matrix construction utilities."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import scipy.sparse

from coker.algebra.dimensions import VectorSpace
from coker.algebra.kernel import Tracer
from coker.algebra.tensor import SymbolicVector



@dataclass(frozen=True)
class SparseMatrixPattern:
    """Fixed CSC structure and source data for a symbolic sparse matrix."""

    shape: tuple[int, int]
    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    data: Tracer
class SparseMatrixBuilder:
    """Build matrices with a fixed compressed-sparse-column boolean pattern.

    The supplied pattern determines CSC column pointers and row indices. Data
    follows exactly ``scipy.sparse.csc_array(pattern).data`` ordering.
    """

    def __init__(self, pattern: np.ndarray | scipy.sparse.sparray):
        if scipy.sparse.issparse(pattern):
            compressed = scipy.sparse.csc_array(pattern, dtype=bool)
        else:
            dense = np.asarray(pattern)
            if dense.ndim != 2 or dense.dtype != bool:
                raise TypeError(
                    "pattern must be a two-dimensional boolean matrix"
                )
            compressed = scipy.sparse.csc_array(dense)
        compressed.eliminate_zeros()
        compressed.sort_indices()
        assert compressed.indptr is not None
        assert compressed.indices is not None
        self.shape = compressed.shape
        self.indptr = compressed.indptr.copy()
        self.indices = compressed.indices.copy()
        self._flat_indices = np.concatenate(
            [
                self.indices[self.indptr[column] : self.indptr[column + 1]]
                * self.shape[1]
                + column
                for column in range(self.shape[1])
            ]
        )

    @property
    def nnz(self) -> int:
        """Return the number of stored CSC values."""
        return int(self.indptr[-1])

    def data_space(self, name: str) -> VectorSpace:
        """Return the flat input space for this pattern's CSC data values."""
        return VectorSpace(name, self.nnz)

    def matrix(self, data: Tracer | np.ndarray) -> np.ndarray | Tracer:
        """Scatter CSC data into a matrix with this builder's fixed pattern."""
        if isinstance(data, Tracer):
            if data.dim.flat() != self.nnz:
                raise ValueError(
                    f"expected {self.nnz} CSC values, got {data.dim.flat()}"
                )
            matrix = SymbolicVector.zeros((self.shape[0] * self.shape[1],))
            for index, flat_index in enumerate(self._flat_indices):
                matrix[int(flat_index)] = data[index]
            result = np.reshape(matrix.collapse(), self.shape)
            result.sparse_matrix_pattern = SparseMatrixPattern(
                self.shape,
                tuple(int(value) for value in self.indptr),
                tuple(int(value) for value in self.indices),
                data,
            )
            return result

        values = np.asarray(data)
        if values.shape != (self.nnz,):
            raise ValueError(
                f"expected data shape ({self.nnz},), got {values.shape}"
            )
        return scipy.sparse.csc_array(
            (values, self.indices, self.indptr), shape=self.shape
        ).toarray()
