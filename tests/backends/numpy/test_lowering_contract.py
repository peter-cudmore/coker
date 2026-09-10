import numpy as np

from coker import Scalar, VectorSpace, function


def test_numpy_lowering_reuses_plan_for_constants_and_declares_thread_safety():
    bias = np.array([1.5, -2.0])
    compiled = function(
        [VectorSpace("x", 2)],
        lambda x: x + bias,
        backend="numpy",
    )

    lowered = compiled.lower()
    assert lowered is compiled.lower()
    assert lowered.capabilities.thread_safe is True
    assert lowered.capabilities.supports_prepare is False
    assert lowered.capabilities.eager_execution is True
    assert lowered.capabilities.symbolic_execution is True
    assert lowered.capabilities.autograd is False
    assert lowered.capabilities.supports_module_adapter is False

    first = lowered.execute([np.array([2.0, 4.0])])[0]
    second = lowered.execute([np.array([-1.0, 3.0])])[0]
    np.testing.assert_allclose(first, [3.5, 2.0])
    np.testing.assert_allclose(second, [0.5, 1.0])


def test_numpy_lowering_preserves_scalar_vector_and_matrix_results():
    compiled = function(
        [VectorSpace("x", (2, 2)), Scalar("s")],
        lambda x, s: (s, x[0, :], x @ x),
        backend="numpy",
    )
    lowered = compiled.lower()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    scalar, vector, result_matrix = lowered(matrix, 2.5)
    assert np.asarray(scalar).shape == ()
    assert np.asarray(vector).shape == (2,)
    assert np.asarray(result_matrix).shape == (2, 2)
    np.testing.assert_allclose(scalar, 2.5)
    np.testing.assert_allclose(vector, [1.0, 2.0])
    np.testing.assert_allclose(result_matrix, matrix @ matrix)

    public = compiled(matrix, 2.5)
    np.testing.assert_allclose(public[0], scalar)
    np.testing.assert_allclose(public[1], vector)
    np.testing.assert_allclose(public[2], result_matrix)
