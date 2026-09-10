import torch

import coker
from coker import VectorSpace


def test_pytorch_lowering_preserves_dtype_device_and_autograd():
    compiled = coker.function(
        [VectorSpace("x", 2)],
        lambda x: (x * x, x + 1),
        backend="pytorch",
    )
    lowered = compiled.lower()
    # Compiler capture is not a declared lowering capability for this backend.
    assert getattr(lowered.capabilities, "compiler_capture", False) is False

    x = torch.tensor([2.0, -3.0], dtype=torch.float64, requires_grad=True)
    outputs = lowered.execute([x])
    assert len(outputs) == 2
    assert all(isinstance(value, torch.Tensor) for value in outputs)
    assert outputs[0].dtype == x.dtype
    assert outputs[0].device == x.device
    assert outputs[1].dtype == x.dtype
    assert outputs[1].device == x.device
    (outputs[0].sum() + outputs[1].sum()).backward()
    torch.testing.assert_close(
        x.grad, torch.tensor([5.0, -5.0], dtype=x.dtype)
    )


def test_pytorch_lowered_plan_matches_public_function_for_multiple_outputs():
    compiled = coker.function(
        [VectorSpace("x", 2)],
        lambda x: (x * 3, x * x),
        backend="pytorch",
    )
    lowered = compiled.lower()
    x = torch.tensor([2.0, 4.0], dtype=torch.float32)

    lowered_outputs = lowered(x)
    public_outputs = compiled(x)
    assert isinstance(lowered_outputs, tuple)
    assert isinstance(public_outputs, tuple)
    assert len(lowered_outputs) == len(public_outputs) == 2
    for lowered_value, public_value in zip(lowered_outputs, public_outputs):
        torch.testing.assert_close(lowered_value, public_value)
