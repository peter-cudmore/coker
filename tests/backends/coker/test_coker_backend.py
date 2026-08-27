import numpy as np
import pytest

from coker import Scalar, VectorSpace, function


def test_coker_backend_compiles_scalar_through_rust_runtime():
    symbolic_function = function(
        [Scalar("x")],
        implementation=lambda x: x,
        backend="coker",
    )

    value = 1.25
    assert symbolic_function(value) == pytest.approx(value)


def test_coker_backend_compiles_vector_operations_through_rust_runtime():
    symbolic_function = function(
        [VectorSpace("x", 3)],
        implementation=lambda x: x,
        backend="coker",
    )

    value = np.array([1.0, 2.0, 3.0])
    actual = symbolic_function(value)

    np.testing.assert_allclose(actual, value)


def test_coker_backend_artifact_is_stable_and_persistable(tmp_path):
    symbolic_function = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: x,
        backend="coker",
    )

    lowered = symbolic_function.lower()
    artifact = lowered.artifact
    assert isinstance(artifact, bytes)
    assert artifact
    destination = tmp_path / "ordinary.coker"
    destination.write_bytes(artifact)
    persisted = destination.read_bytes()
    assert persisted == artifact
    assert isinstance(persisted, bytes)
    np.testing.assert_allclose(lowered(np.array([2.0, -3.0]))[0], [2.0, -3.0])


def test_coker_backend_does_not_retain_python_graph_lowering():
    symbolic_function = function(
        [Scalar("x")], implementation=lambda x: x + 1.0, backend="coker"
    )

    lowered = symbolic_function.lower()
    assert lowered.__class__.__module__ == "coker.backends.coker.core"
    assert lowered(2.0) == [pytest.approx(3.0)]


def test_coker_backend_nested_module_keeps_outer_entry_first():
    inner = function(
        [Scalar("x")],
        implementation=lambda x: x + 10.0,
        backend="coker",
    )
    outer = function(
        [Scalar("x")],
        implementation=lambda x: inner(x) + 1.0,
        backend="coker",
    )

    # If recursive extraction emitted the callee first, module entry 0 would
    # return 12. The outer tape must remain entry 0 and return 13.
    assert outer(2.0) == pytest.approx(13.0)
