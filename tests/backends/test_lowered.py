from dataclasses import dataclass

import numpy as np
import pytest

from coker import VectorSpace, function
from coker.backends.lowered import (
    LoweredFunction,
    LoweringCapabilities,
    LoweringOptions,
)


@dataclass(frozen=True)
class BackendCase:
    name: str
    autograd: bool
    module_adapter: bool
    optional_module: str | None = None


BACKENDS = (
    BackendCase("numpy", autograd=False, module_adapter=False),
    BackendCase(
        "casadi",
        autograd=False,
        module_adapter=False,
        optional_module="casadi",
    ),
    BackendCase(
        "pytorch", autograd=True, module_adapter=True, optional_module="torch"
    ),
)


def _require_backend(case):
    if case.optional_module is not None:
        pytest.importorskip(case.optional_module)


def test_lowering_capabilities_default_to_unsupported():
    assert LoweringCapabilities() == LoweringCapabilities(
        eager_execution=False,
        symbolic_execution=False,
        autograd=False,
        serializable_artifact=False,
        caller_owned_workspace=False,
        supports_prepare=False,
        supports_module_adapter=False,
        thread_safe=False,
    )


@pytest.mark.parametrize("case", BACKENDS)
def test_lowered_contract_signature_abis_and_public_equivalence(case):
    _require_backend(case)
    backend_name = case.name
    compiled = function(
        [VectorSpace("x", 2)],
        lambda x: (x + 1, x * 2),
        backend=backend_name,
    )
    lowered = compiled.lower()
    value = np.array([2.0, 3.0])

    assert isinstance(lowered, LoweredFunction)
    assert lowered.backend_name == backend_name
    assert lowered.signature == compiled.signature
    assert compiled.signature is compiled.signature
    assert [spec.name for spec in compiled.signature.inputs] == ["x"]
    assert compiled.signature.inputs[0].space.name == "x"
    assert [spec.name for spec in compiled.signature.outputs] == [
        "output_0",
        "output_1",
    ]
    assert [spec.shape.dim for spec in compiled.signature.outputs] == [
        (2,),
        (2,),
    ]

    packed = lowered.execute([value])
    positional = lowered(value)
    public_result = compiled(value)
    assert isinstance(packed, tuple)
    assert isinstance(positional, tuple)
    assert isinstance(public_result, tuple)
    for actual, expected in zip(packed, ([3.0, 4.0], [4.0, 6.0])):
        np.testing.assert_allclose(np.asarray(actual).reshape(-1), expected)
    for actual, expected in zip(positional, packed):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))
    for actual, expected in zip(public_result, packed):
        np.testing.assert_allclose(
            np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
        )


@pytest.mark.parametrize("case", BACKENDS)
def test_lowered_none_output_policy(case):
    _require_backend(case)
    backend_name = case.name
    compiled = function(
        [VectorSpace("x", 2)],
        lambda x: (x + 1, None),
        backend=backend_name,
    )
    lowered = compiled.lower()
    value = np.array([2.0, 3.0])

    packed = lowered.execute([value])
    positional = lowered(value)
    public_result = compiled(value)
    assert isinstance(packed, tuple)
    assert len(packed) == 2
    np.testing.assert_allclose(np.asarray(packed[0]).reshape(-1), [3.0, 4.0])
    assert packed[1] is None
    assert positional[1] is None
    np.testing.assert_allclose(
        np.asarray(positional[0]).reshape(-1),
        np.asarray(packed[0]).reshape(-1),
    )
    assert public_result[1] is None
    np.testing.assert_allclose(
        np.asarray(public_result[0]).reshape(-1),
        np.asarray(packed[0]).reshape(-1),
    )
    assert lowered.signature.outputs[1].shape is None


@pytest.mark.parametrize("case", BACKENDS)
def test_lowered_cache_options_prepare_and_capabilities(case):
    _require_backend(case)
    backend_name = case.name
    compiled = function(
        [VectorSpace("x", 2)],
        lambda x: x + 1,
        backend=backend_name,
    )

    default_options = LoweringOptions()
    explicit_options = LoweringOptions(backend=backend_name)
    assert default_options != explicit_options
    assert default_options.backend is None
    with pytest.raises((AttributeError, TypeError)):
        default_options.backend = backend_name

    first = compiled.lower(default_options)
    value = np.array([2.0, 3.0])
    packed = first.execute([value])
    positional = first(value)
    assert isinstance(packed, tuple)
    assert len(packed) == 1
    np.testing.assert_allclose(np.asarray(positional), np.asarray(packed[0]))

    assert first is compiled.lower(default_options)
    assert first is not compiled.lower(explicit_options)
    first.prepare()
    first.prepare()
    first.close()
    first.close()

    capabilities = first.capabilities
    assert capabilities.eager_execution is True
    assert capabilities.symbolic_execution is True
    assert capabilities.autograd is case.autograd
    assert capabilities.serializable_artifact is False
    assert capabilities.caller_owned_workspace is False
    assert capabilities.supports_prepare is False
    assert capabilities.supports_module_adapter is case.module_adapter
    assert capabilities.thread_safe is True


@pytest.mark.parametrize("case", BACKENDS)
def test_lowered_zero_width_output_preserves_declared_shape(case):
    _require_backend(case)
    backend_name = case.name
    compiled = function(
        [VectorSpace("x", 0)],
        lambda x: x,
        backend=backend_name,
    )
    lowered = compiled.lower()

    (output,) = lowered.execute([np.empty(0)])

    assert lowered.signature == compiled.signature
    assert lowered.signature.outputs[0].shape.dim == (0,)
    assert np.asarray(output).size == 0
