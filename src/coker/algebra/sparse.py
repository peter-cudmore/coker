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
    """Immutable CSC structure and symbolic source data for a sparse matrix.

    ``indptr`` and ``indices`` use the canonical CSC ordering: columns are
    visited from left to right and row indices within each column are sorted.
    ``data`` contains the symbolic values in exactly that order.
    """

    shape: tuple[int, int]
    indptr: tuple[int, ...]
    indices: tuple[int, ...]
    data: Tracer


class SparseMatrixBuilder:
    """Construct symbolic matrices from a fixed boolean CSC sparsity pattern.

    Pattern canonicalisation happens once during construction.  Every call to
    :meth:`matrix` then interprets its flat data vector in canonical CSC
    order, making the resulting sparsity structure stable across backends and
    runs.  The builder owns only the small structural arrays; callers own the
    data vectors.
    """

    def __init__(self, pattern: np.ndarray | scipy.sparse.sparray):
        """Canonicalise ``pattern`` and retain only its fixed CSC structure.

        Dense patterns must be boolean.  Sparse inputs are converted through
        boolean CSC semantics so numeric sparse masks remain convenient while
        duplicate entries and unsorted rows are normalised deterministically.
        """
        if scipy.sparse.issparse(pattern):
            if len(pattern.shape) != 2:
                raise TypeError("pattern must be two-dimensional")
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
        self.shape = (int(compressed.shape[0]), int(compressed.shape[1]))
        self.indptr = tuple(int(value) for value in compressed.indptr)
        self.indices = tuple(int(value) for value in compressed.indices)
        flat_indices = [
            row * self.shape[1] + column
            for column in range(self.shape[1])
            for row in self.indices[
                self.indptr[column] : self.indptr[column + 1]
            ]
        ]
        self._flat_indices = np.asarray(flat_indices, dtype=np.intp)

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
