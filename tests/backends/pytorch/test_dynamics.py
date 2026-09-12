import numpy as np
import pytest
import torch

from coker.algebra.ops import Noop
from coker.backends import get_backend_by_name
from coker.backends.pytorch import PytorchODESolverParameters


def test_pytorch_ode_preserves_autograd():
    pytest.importorskip("torchdiffeq")
    backend = get_backend_by_name("pytorch", set_current=False)
    x0 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    x_final, z_final, q_final = backend.evaluate_integrals(
        [lambda _t, x, _z, _u, _p: x, Noop(), Noop()],
        [x0, None, None],
        1.0,
        [None, None],
        PytorchODESolverParameters(rtol=1e-8, atol=1e-10),
    )

    assert z_final is None
    assert q_final is None
    assert torch.allclose(x_final, torch.exp(x0), rtol=1e-7, atol=1e-9)
    x_final.sum().backward()
    assert torch.allclose(x0.grad, torch.exp(x0), rtol=1e-7, atol=1e-9)


def test_pytorch_ode_restores_time_grid_layout():
    pytest.importorskip("torchdiffeq")
    backend = get_backend_by_name("pytorch", set_current=False)

    x_values, z_final, q_final = backend.evaluate_integrals(
        [lambda _t, x, _z, _u, _p: x, Noop(), Noop()],
        [torch.tensor([1.0]), None, None],
        np.array([0.5, 1.0]),
        [None, None],
    )

    assert z_final is None
    assert q_final is None
    assert x_values.shape == (1, 2)
    assert torch.allclose(
        x_values,
        torch.tensor([[np.exp(0.5), np.e]], dtype=x_values.dtype),
        rtol=1e-5,
        atol=1e-6,
    )


def test_pytorch_quadrature_restores_time_grid_layout():
    pytest.importorskip("torchdiffeq")
    backend = get_backend_by_name("pytorch", set_current=False)

    x_values, z_final, q_values = backend.evaluate_integrals(
        [
            lambda _t, x, _z, _u, _p: x * 0,
            Noop(),
            lambda _t, x, _z, _u, _p: x,
        ],
        [torch.tensor([2.0]), None, torch.tensor([0.0])],
        np.array([0.5, 1.0]),
        [None, None],
    )

    assert z_final is None
    assert x_values.shape == (1, 2)
    assert torch.equal(x_values, torch.tensor([[2.0, 2.0]]))
    assert q_values.shape == (1, 2)
    assert torch.allclose(q_values, torch.tensor([[1.0, 2.0]]), rtol=1e-5)


def test_pytorch_ode_restores_zero_time_grid_layout():
    pytest.importorskip("torchdiffeq")
    backend = get_backend_by_name("pytorch", set_current=False)

    x_values, z_final, q_final = backend.evaluate_integrals(
        [lambda _t, x, _z, _u, _p: x, Noop(), Noop()],
        [torch.tensor([1.0]), None, None],
        np.array([0.0]),
        [None, None],
    )

    assert z_final is None
    assert q_final is None
    assert torch.equal(x_values, torch.tensor([[1.0]]))


def test_pytorch_ode_rejects_algebraic_constraints():
    pytest.importorskip("torchdiffeq")
    backend = get_backend_by_name("pytorch", set_current=False)

    with pytest.raises(NotImplementedError, match="Algebraic constraints"):
        backend.evaluate_integrals(
            [
                lambda _t, x, _z, _u, _p: x,
                lambda _t, _x, z, _u, _p: z,
                Noop(),
            ],
            [torch.tensor([1.0]), None, None],
            1.0,
            [None, None],
        )
