import numpy as np
import pytest

ca = pytest.importorskip("casadi")

from coker import FunctionSpace, Scalar, VectorSpace, function


def test_casadi_lowering_uses_native_fast_path_and_preserves_native_values():
    compiled = function(
        [VectorSpace("x", (2, 2))],
        lambda x: (x + 1, x @ x),
        backend="casadi",
    )
    lowered = compiled.lower()

    assert lowered.ca_function is not None
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = lowered.execute([matrix])
    assert all(isinstance(value, (ca.DM, ca.MX)) for value in values)
    np.testing.assert_allclose(np.asarray(values[0]), matrix + 1)
    np.testing.assert_allclose(np.asarray(values[1]), matrix @ matrix)

    public = compiled(matrix)
    assert isinstance(public, tuple)
    np.testing.assert_allclose(np.asarray(public[0]), matrix + 1)
    np.testing.assert_allclose(np.asarray(public[1]), matrix @ matrix)


def test_casadi_function_space_and_none_inputs_use_evaluate_fallback():
    space = FunctionSpace("f", [Scalar("x")], [Scalar("y")])
    compiled = function(
        [space, VectorSpace("optional", 1)],
        lambda f, optional: (f(3.0), None),
        backend="casadi",
    )
    inner = function([Scalar("x")], lambda x: x * 2.0, backend="casadi")
    lowered = compiled.lower()

    assert lowered.ca_function is None
    values = lowered.execute([inner, None])
    assert values == (6.0, None)
    assert compiled(inner, None) == (6.0, None)
