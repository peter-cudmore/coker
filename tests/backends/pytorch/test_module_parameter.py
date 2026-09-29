import pytest
import torch

from coker import FunctionSpace, Scalar
from coker.backends import get_backend_by_name
from coker.parameters import DenseTensorVariable
from coker.parameters.function_parameters import FunctionParameter


def test_import_module_parameter_uses_stateless_dense_decisions():
    backend = get_backend_by_name("pytorch", set_current=False)
    module = torch.nn.Linear(1, 1).double()
    with torch.no_grad():
        module.weight.fill_(2.0)
        module.bias.fill_(0.5)
    original_state = {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }

    declaration = backend.import_module_parameter(module, name="response")

    assert isinstance(declaration, FunctionParameter)
    assert len(declaration.signature.inputs) == 1
    assert len(declaration.signature.outputs) == 1
    assert isinstance(declaration.signature.inputs[0].space, Scalar)
    assert isinstance(declaration.signature.outputs[0].shape, Scalar)
    blocks = declaration.list_concrete_parameters()
    assert [block.name for block in blocks] == [
        "response_weight",
        "response_bias",
    ]
    assert all(isinstance(block, DenseTensorVariable) for block in blocks)
    assert [block.shape for block in blocks] == [(1, 1), (1,)]
    assert [block.guess.tolist() for block in blocks] == [[[2.0]], [0.5]]

    target = FunctionSpace(
        "response",
        arguments=[Scalar("argument")],
        output=[Scalar("value")],
    )
    parameterization = declaration.build_function(target, "pytorch")
    argument = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[3.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    result = parameterization(argument, weight, bias)

    torch.testing.assert_close(result, torch.tensor(7.0, dtype=torch.float64))
    result.backward()
    torch.testing.assert_close(
        argument.grad, torch.tensor(3.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        weight.grad, torch.tensor([[2.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        bias.grad, torch.tensor([1.0], dtype=torch.float64)
    )
    assert module.weight.grad is None
    assert module.bias.grad is None
    for name, original in original_state.items():
        torch.testing.assert_close(module.state_dict()[name], original)


def test_import_module_parameter_rejects_ambiguous_module_signature():
    class Offset(torch.nn.Module):
        def forward(self, value):
            return value + 1.0

    backend = get_backend_by_name("pytorch", set_current=False)

    with pytest.raises(ValueError, match="deterministically derivable"):
        backend.import_module_parameter(Offset())
