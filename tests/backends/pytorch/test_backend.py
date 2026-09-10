import gc
import weakref

import numpy as np
import pytest
import torch

import coker
from coker import Scalar, VectorSpace
from coker.algebra import Dimension, OP
from coker.algebra.ops import Noop
from coker.backends import get_backend_by_name
from coker.backends.pytorch import PytorchModule
from coker.toolkits.codesign import Minimise, ProblemBuilder


@pytest.fixture
def pytorch_backend():
    return get_backend_by_name("pytorch", set_current=False)


def test_backend_selection(pytorch_backend):
    assert pytorch_backend.name == "pytorch"
    assert get_backend_by_name("pytorch", set_current=False) is not None


def test_to_backend_array_preserves_tensor_dtype_device_and_identity(
    pytorch_backend,
):
    value = torch.tensor([1, 2], dtype=torch.float64)
    converted = pytorch_backend.to_backend_array(value)
    assert converted is value
    assert converted.dtype == value.dtype
    assert converted.device == value.device


def test_function_returns_tensor_and_preserves_autograd():
    fn = coker.function(
        [VectorSpace("x", 2)],
        lambda x: x[0] * x[0] + x[1] * x[1],
        backend="pytorch",
    )
    x = torch.tensor([2.0, -3.0], dtype=torch.float64, requires_grad=True)
    result = fn(x)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == x.dtype
    result.backward()
    assert torch.equal(x.grad, torch.tensor([4.0, -6.0], dtype=x.dtype))


def test_exponential_is_supported_by_pytorch_backend():
    fn = coker.function(
        [Scalar("x")],
        lambda x: np.exp(x),
        backend="pytorch",
    )
    x = torch.tensor(2.0, requires_grad=True)

    result = fn(x)

    assert torch.allclose(result, torch.exp(x))
    result.backward()
    assert torch.allclose(x.grad, torch.exp(x))


def test_as_module_returns_eager_pytorch_module(pytorch_backend):
    fn = coker.function(
        [VectorSpace("x", 2)],
        lambda x: x[0] * x[0] + x[1],
        backend="pytorch",
    )
    module = pytorch_backend.as_module(fn)
    x = torch.tensor([3.0, 2.0], requires_grad=True)

    assert isinstance(module, torch.nn.Module)
    assert isinstance(module, PytorchModule)
    assert list(module.parameters()) == []
    result = module(x)

    assert torch.equal(result, torch.tensor(11.0))
    result.backward()
    assert torch.equal(x.grad, torch.tensor([6.0, 1.0]))


def test_as_module_returns_tuple_for_multiple_outputs(pytorch_backend):
    fn = coker.function(
        [Scalar("x")],
        lambda x: (x, x * x),
        backend="pytorch",
    )
    module = pytorch_backend.as_module(fn)

    outputs = module(torch.tensor(3.0))

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert torch.equal(outputs[0], torch.tensor(3.0))
    assert torch.equal(outputs[1], torch.tensor(9.0))


def test_to_numpy_array_detaches_and_returns_scalars(pytorch_backend):
    scalar = torch.tensor(3.5, dtype=torch.float64, requires_grad=True)
    result = pytorch_backend.to_numpy_array(scalar)
    assert isinstance(result, float)
    assert result == pytest.approx(3.5)

    vector = torch.tensor([1.0, 2.0], requires_grad=True)
    converted = pytorch_backend.to_numpy_array(vector)
    assert isinstance(converted, np.ndarray)
    assert np.array_equal(converted, [1.0, 2.0])

    assert pytorch_backend.to_backend_array([1, 2]).shape == (2,)
    assert pytorch_backend.to_backend_array(4).ndim == 0


def test_import_module_composes_with_pytorch_function_and_preserves_autograd(
    pytorch_backend,
):
    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(2.0, dtype=torch.float64)
            )
            self.register_buffer(
                "bias", torch.tensor(0.5, dtype=torch.float64)
            )

        def forward(self, x):
            return self.weight * x + self.bias

    signature_source = coker.function(
        [Scalar("x")], lambda x: (x,), backend="pytorch"
    )
    module = Affine()
    module_ref = weakref.ref(module)
    imported = pytorch_backend.import_module(
        module, signature_source.signature
    )
    correction = coker.function(
        [Scalar("x")], lambda x: x * 3.0, backend="pytorch"
    )
    composed = coker.function(
        [Scalar("x")],
        lambda x: imported(x) + correction(x),
        backend="pytorch",
    )

    x = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    result = composed(x)

    assert torch.equal(result, torch.tensor(20.5, dtype=torch.float64))
    result.backward()
    assert torch.equal(
        module.weight.grad, torch.tensor(4.0, dtype=torch.float64)
    )
    assert result.dtype == x.dtype
    assert result.device == x.device
    assert module.weight.grad is not None
    del module
    gc.collect()
    assert module_ref() is not None


def test_import_module_supports_multi_output_calls(pytorch_backend):
    class Split(torch.nn.Module):
        def forward(self, x):
            return x + 1.0, x * x

    signature_source = coker.function(
        [Scalar("x")], lambda x: (x + 1.0, x * x), backend="pytorch"
    )
    imported = pytorch_backend.import_module(
        Split(), signature_source.signature
    )

    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    outputs = imported(x)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert torch.equal(outputs[0], torch.tensor(4.0, dtype=torch.float64))
    assert torch.equal(outputs[1], torch.tensor(9.0, dtype=torch.float64))
    torch.stack(outputs).sum().backward()
    assert torch.equal(x.grad, torch.tensor(7.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "value, dimension, expected_shape",
    [
        (torch.tensor([2.0]), Dimension(None), torch.Size([])),
        (torch.arange(6.0), Dimension((2, 3)), torch.Size([2, 3])),
        (torch.arange(6.0).reshape(2, 3), Dimension((6,)), torch.Size([6])),
    ],
)
def test_reshape_scalar_vector_and_matrix(
    pytorch_backend, value, dimension, expected_shape
):
    result = pytorch_backend.reshape(value, dimension)
    assert isinstance(result, torch.Tensor)
    assert result.shape == expected_shape
    assert result.dtype == value.dtype


def test_zero_divided_by_zero_keeps_zero(pytorch_backend):
    numerator = torch.zeros(3, dtype=torch.float64)
    denominator = torch.zeros(3, dtype=torch.float64)
    result = pytorch_backend.call(OP.DIV, numerator, denominator)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, numerator)


def test_if_then_else_uses_tensor_where():
    fn = coker.function(
        [Scalar("x")],
        lambda x: coker.if_then_else(
            x > 0, np.ones(3, dtype=float), np.zeros(3, dtype=float)
        ),
        backend="pytorch",
    )
    x = torch.tensor(-1.0, requires_grad=True)
    result = fn(x)
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([0.0, 0.0, 0.0]))


def test_mathematical_program_construction_is_unsupported():
    with ProblemBuilder() as builder:
        x = builder.new_variable("x")
        builder.objective = Minimise(x * x)
        builder.outputs = [x]
        with pytest.raises(NotImplementedError, match="optimisation"):
            builder.build("pytorch")


def test_variational_solver_creation_is_unsupported():
    with pytest.raises(NotImplementedError, match="variational"):
        get_backend_by_name(
            "pytorch", set_current=False
        ).create_variational_solver(object())


def test_quadrature_dynamics_preserve_autograd(pytorch_backend):
    pytest.importorskip("torchdiffeq")
    x0 = torch.tensor([2.0], requires_grad=True)
    q0 = torch.tensor([0.0], requires_grad=True)

    x_final, z_final, q_final = pytorch_backend.evaluate_integrals(
        [
            lambda _t, x, _z, _u, _p: x * 0,
            Noop(),
            lambda _t, x, _z, _u, _p: x,
        ],
        [x0, None, q0],
        0.5,
        [None, None],
    )

    assert z_final is None
    assert torch.allclose(x_final, x0)
    assert torch.allclose(q_final, torch.tensor([1.0]))
    q_final.sum().backward()
    assert torch.allclose(x0.grad, torch.tensor([0.5]), rtol=1e-5)
    assert torch.allclose(q0.grad, torch.ones_like(q0), rtol=1e-5)
