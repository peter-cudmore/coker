import numpy as np
import pytest

from coker.backends.coker.op_impl import cross, dot
from coker.backends.coker.sparse_tensor import dok_ndarray
from coker.backends.coker.weights import BilinearWeights, MemorySpec

MEMORY = MemorySpec(0, 3)

# Maps x -> A @ x (identity)
A = np.eye(3)
# Maps x -> B @ x (a rotation about z)
B = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

linear_a = BilinearWeights(MEMORY, shape=(3,), linear=dok_ndarray.fromarray(A))
linear_b = BilinearWeights(MEMORY, shape=(3,), linear=dok_ndarray.fromarray(B))

constant_vector = np.array([1.0, 2.0, 3.0])
constant_weights = BilinearWeights(
    MEMORY, shape=(3,), constant=dok_ndarray.fromarray(constant_vector)
)

# cross(linear_a, linear_b) produces a quadratic BilinearWeights
quadratic_weights = cross(linear_a, linear_b)

TEST_VECTOR = np.array([1.0, 2.0, 3.0])


def test_scalar_linear_product_supports_multi_value_memory():
    left = BilinearWeights(
        MEMORY,
        shape=(1,),
        constant=dok_ndarray((1,), {(0,): 2.0}),
        linear=dok_ndarray(
            (1, 3),
            {(0, 0): 1.0, (0, 2): -0.5},
        ),
    )
    right = BilinearWeights(
        MEMORY,
        shape=(1,),
        constant=dok_ndarray((1,), {(0,): -1.0}),
        linear=dok_ndarray(
            (1, 3),
            {(0, 1): 3.0},
        ),
    )

    product = left * right

    assert product.quadratic.shape == (1, 3, 3)
    expected = left(TEST_VECTOR) * right(TEST_VECTOR)
    assert np.allclose(product(TEST_VECTOR), expected)


# --- dot product tests ---


def test_dot_both_linear():
    result = dot(linear_a, linear_b)
    assert result.shape == (1,)
    assert np.isclose(
        result(TEST_VECTOR),
        np.dot(A @ TEST_VECTOR, B @ TEST_VECTOR),
    )


def test_dot_builds_sparse_coefficients_without_dense_round_trip(monkeypatch):
    dense_round_trips = []
    original_toarray = dok_ndarray.toarray

    def record_toarray(value):
        dense_round_trips.append(value.shape)
        return original_toarray(value)

    monkeypatch.setattr(dok_ndarray, "toarray", record_toarray)
    result = dot(linear_a, linear_b)

    assert dense_round_trips == []
    actual = original_toarray(result.quadratic).reshape(3, 3)
    assert np.allclose(actual, A.T @ B)


def test_dot_constant_and_quadratic():
    result = dot(constant_weights, quadratic_weights)
    assert result.shape == (1,)
    assert np.isclose(
        result(TEST_VECTOR),
        np.dot(constant_vector, quadratic_weights(TEST_VECTOR)),
    )


def test_dot_quadratic_and_constant():
    result = dot(quadratic_weights, constant_weights)
    assert result.shape == (1,)
    assert np.isclose(
        result(TEST_VECTOR),
        np.dot(quadratic_weights(TEST_VECTOR), constant_vector),
    )


def test_dot_linear_and_quadratic_raises():
    with pytest.raises(AssertionError):
        dot(linear_a, quadratic_weights)


def test_dot_both_quadratic_raises():
    with pytest.raises(AssertionError):
        dot(quadratic_weights, quadratic_weights)


# --- cross product tests ---


def test_cross_both_linear():
    result = cross(linear_a, linear_b)
    assert result.shape == (3,)
    assert np.allclose(
        result(TEST_VECTOR),
        np.cross(A @ TEST_VECTOR, B @ TEST_VECTOR),
    )


def test_cross_linear_and_quadratic_raises():
    with pytest.raises(AssertionError):
        cross(linear_a, quadratic_weights)


def test_cross_quadratic_and_linear_raises():
    with pytest.raises(AssertionError):
        cross(quadratic_weights, linear_a)
