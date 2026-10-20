import numpy as np
import pytest

import scipy.sparse

from coker import SparseMatrixBuilder, VectorSpace, function


def test_sparse_matrix_builder_retains_numeric_csc_data_order():
    builder = SparseMatrixBuilder(
        np.array([[True, False, True], [True, True, False]])
    )

    assert builder.data_space("A_data").dimension == 4
    matrix = builder.matrix(np.array([1.0, 2.0, 3.0, 4.0]))
    assert isinstance(matrix, scipy.sparse.csc_array)
    assert tuple(matrix.indptr) == builder.indptr
    assert tuple(matrix.indices) == builder.indices
    assert np.array_equal(
        matrix.toarray(),
        np.array([[1.0, 0.0, 4.0], [2.0, 3.0, 0.0]]),
    )


def test_fixed_csc_matrix_multiplies_traced_vector():
    builder = SparseMatrixBuilder(
        np.array([[True, False, True], [True, True, False]])
    )
    matrix = builder.matrix(np.array([1.0, 2.0, 3.0, 4.0]))
    apply = function(
        [VectorSpace("vector", 3)],
        lambda vector: matrix @ vector,
        backend="numpy",
    )

    assert np.allclose(apply(np.array([2.0, -1.0, 0.5])), [4.0, 1.0])


def test_sparse_matrix_builder_traces_csc_data():
    pattern = scipy.sparse.csr_array(
        np.array([[True, False, True], [True, True, False]])
    )
    builder = SparseMatrixBuilder(pattern)
    apply = function(
        [builder.data_space("A_data")],
        lambda data: builder.matrix(data) @ np.array([2.0, -1.0, 0.5]),
        backend="numpy",
    )

    assert np.allclose(apply(np.array([1.0, 2.0, 3.0, 4.0])), [4.0, 1.0])


def test_sparse_matrix_builder_validates_dense_patterns_and_empty_columns():
    with pytest.raises(TypeError, match="two-dimensional boolean"):
        SparseMatrixBuilder(np.ones((2, 2), dtype=float))

    builder = SparseMatrixBuilder(np.zeros((3, 0), dtype=bool))
    assert builder.shape == (3, 0)
    assert builder.nnz == 0
    assert builder.matrix(np.empty(0)).shape == (3, 0)
