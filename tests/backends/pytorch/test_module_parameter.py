import torch

from coker import FunctionSpace, VectorSpace
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
    input_space = VectorSpace("input", 1)
    output_space = VectorSpace("output", 1)
    declaration = backend.import_module_parameter(
        module,
        input_space,
        output_space,
        name="response",
    )

    assert isinstance(declaration, FunctionParameter)
    assert declaration.signature.inputs[0].space is input_space
    assert declaration.signature.outputs[0].shape is output_space
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
        arguments=[input_space],
        output=[output_space],
    )
    parameterization = declaration.build_function(target, "pytorch")
    argument = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[3.0]], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    result = parameterization(argument, weight, bias)

    torch.testing.assert_close(
        result, torch.tensor([7.0], dtype=torch.float64)
    )
    result.sum().backward()
    torch.testing.assert_close(
        argument.grad, torch.tensor([3.0], dtype=torch.float64)
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


def test_import_module_parameter_supports_explicit_spaces_for_sequential():
    backend = get_backend_by_name("pytorch", set_current=False)
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 1),
    ).double()
    input_space = VectorSpace("input", 2)
    output_space = VectorSpace("output", 1)
    declaration = backend.import_module_parameter(
        module,
        input_space,
        output_space,
        name="response",
    )
    blocks = declaration.list_concrete_parameters()

    assert [block.name for block in blocks] == [
        "response_0_weight",
        "response_0_bias",
        "response_2_weight",
        "response_2_bias",
    ]
    target = FunctionSpace(
        "response",
        arguments=[input_space],
        output=[output_space],
    )
    parameterization = declaration.build_function(target, "pytorch")
    argument = torch.tensor([0.5, -1.0], dtype=torch.float64)
    decisions = tuple(
        torch.tensor(block.guess, dtype=torch.float64, requires_grad=True)
        for block in blocks
    )

    result = parameterization(argument, *decisions)

    torch.testing.assert_close(result, module(argument))
    result.sum().backward()
    assert all(decision.grad is not None for decision in decisions)


def test_import_module_parameter_freezes_module_buffers():
    class BufferedScale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.register_buffer("offset", torch.tensor([0.5]))

        def forward(self, value):
            self.offset.add_(1.0)
            return self.weight * value + self.offset

    backend = get_backend_by_name("pytorch", set_current=False)
    module = BufferedScale().double()
    original_state = {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }
    input_space = VectorSpace("input", 1)
    output_space = VectorSpace("output", 1)
    declaration = backend.import_module_parameter(
        module,
        input_space,
        output_space,
        name="response",
    )
    target = FunctionSpace(
        "response",
        arguments=[input_space],
        output=[output_space],
    )
    parameterization = declaration.build_function(target, "pytorch")
    argument = torch.tensor([2.0], dtype=torch.float64)
    weight = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)

    first = parameterization(argument, weight)
    second = parameterization(argument, weight)

    torch.testing.assert_close(first, torch.tensor([7.5], dtype=torch.float64))
    torch.testing.assert_close(second, first)
    first.sum().backward()
    torch.testing.assert_close(
        weight.grad, torch.tensor([2.0], dtype=torch.float64)
    )
    for name, original in original_state.items():
        torch.testing.assert_close(module.state_dict()[name], original)


def test_import_module_parameter_defaults_to_a_specializable_name():
    backend = get_backend_by_name("pytorch", set_current=False)
    declaration = backend.import_module_parameter(
        torch.nn.Linear(1, 1),
        VectorSpace("input", 1),
        VectorSpace("output", 1),
    )

    assert declaration.name == "pytorch_module"
    assert [
        parameter.name for parameter in declaration.list_concrete_parameters()
    ] == [
        "pytorch_module_weight",
        "pytorch_module_bias",
    ]
