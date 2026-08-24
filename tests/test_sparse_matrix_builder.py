import numpy as np
import scipy.sparse

from coker import SparseMatrixBuilder, function


def test_sparse_matrix_builder_uses_csc_data_order():
    builder = SparseMatrixBuilder(
        np.array([[True, False, True], [True, True, False]])
    )

    assert builder.data_space("A_data").dimension == 4
    assert np.array_equal(
        builder.matrix(np.array([1.0, 2.0, 3.0, 4.0])),
        np.array([[1.0, 0.0, 4.0], [2.0, 3.0, 0.0]]),
    )


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
