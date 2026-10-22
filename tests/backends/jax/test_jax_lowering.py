import numpy as np
import pytest

from coker import VectorSpace, function

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


@pytest.mark.parametrize("packed", [False, True], ids=["public", "packed"])
def test_squared_norm_preserves_jax_differentiation(packed):
    compiled = function(
        [VectorSpace("x", 2)],
        lambda x: x[0] * x[0] + x[1] * x[1],
        backend="jax",
    )
    lowered = compiled.lower()
    evaluate = (lambda x: lowered.execute([x])[0]) if packed else compiled
    x = jnp.array([2.0, -3.0])

    np.testing.assert_allclose(jax.grad(evaluate)(x), [4.0, -6.0])
    result = evaluate(x)
    assert isinstance(result, jax.Array)
    assert result.shape == ()
    np.testing.assert_allclose(result, 13.0)


def test_packed_vector_matrix_and_absent_outputs_preserve_jax_gradients():
    compiled = function(
        [VectorSpace("x", 2), VectorSpace("K", (2, 2))],
        lambda x, K: (K @ x, K * K, None),
        backend="jax",
    )
    lowered = compiled.lower()
    x = jnp.array([1.0, 2.0])
    K = jnp.array([[4.0, -1.0], [-1.0, 2.0]])

    vector, matrix, absent = lowered.execute([x, K])
    assert isinstance(vector, jax.Array)
    assert isinstance(matrix, jax.Array)
    assert vector.shape == (2,)
    assert matrix.shape == (2, 2)
    assert absent is None
    np.testing.assert_allclose(vector, [2.0, 3.0])
    np.testing.assert_allclose(matrix, [[16.0, 1.0], [1.0, 4.0]])

    def objective(x, K):
        vector, matrix, _absent = lowered.execute([x, K])
        return jnp.sum(vector) + jnp.sum(matrix)

    gradient_x, gradient_K = jax.grad(objective, argnums=(0, 1))(x, K)
    np.testing.assert_allclose(gradient_x, [3.0, 1.0])
    np.testing.assert_allclose(gradient_K, [[9.0, 0.0], [-1.0, 6.0]])
