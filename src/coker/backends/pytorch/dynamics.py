"""Pure ODE integration for the PyTorch backend."""

from dataclasses import dataclass
from typing import Mapping

import torch

from coker.algebra.ops import Noop
from coker.backends.backend import SolverParameters


@dataclass(frozen=True)
class PytorchSolverParameters(SolverParameters):
    """Configuration for a PyTorch ODE initial-value solve.

    ``method``, tolerances, and ``options`` are passed directly to
    :func:`torchdiffeq.odeint`.
    """

    method: str = "dopri5"
    rtol: float = 1e-6
    atol: float = 1e-8
    options: Mapping[str, object] | None = None


def evaluate_integrals(
    backend,
    functions,
    initial_conditions,
    end_point,
    inputs,
    solver_parameters=None,
):
    """Integrate an ODE and optional quadrature with native PyTorch tensors.

    Algebraic constraints are not supported. ``torchdiffeq`` is imported lazily
    so basic PyTorch evaluation does not require the optional ODE dependency.
    """
    dxdt, constraint, dqdt = functions
    x0, z0, q0 = initial_conditions
    u, p = inputs
    has_quadrature = dqdt is not Noop()

    if constraint is not Noop():
        raise NotImplementedError(
            "Algebraic constraints are not implemented for the pytorch backend"
        )
    if z0 is not None:
        raise ValueError(
            "PyTorch ODE integration requires a None algebraic "
            "initial condition"
        )
    if has_quadrature != (q0 is not None):
        raise ValueError(
            "Quadrature dynamics and the quadrature initial condition must "
            "either both be present or both be absent"
        )

    try:
        from torchdiffeq import odeint
    except ImportError as ex:
        raise RuntimeError(
            "PyTorch ODE support requires `pip install coker[pytorch]`"
        ) from ex

    x0 = backend.to_backend_array(x0)
    if x0.ndim == 0:
        x0 = x0.reshape(1)
    if has_quadrature:
        q0 = backend.to_backend_array(q0).to(dtype=x0.dtype, device=x0.device)
        if q0.ndim == 0:
            q0 = q0.reshape(1)

    u = None if u is None else backend.to_backend_array(u)
    p = None if p is None else backend.to_backend_array(p)

    parameters = (
        solver_parameters
        if isinstance(solver_parameters, PytorchSolverParameters)
        else PytorchSolverParameters()
    )
    times, is_scalar_endpoint, drop_initial = _build_time_grid(end_point, x0)
    if times is None:
        q_initial = (
            q0
            if is_scalar_endpoint
            else q0.unsqueeze(-1) if has_quadrature else None
        )
        return x0 if is_scalar_endpoint else x0.unsqueeze(-1), None, q_initial

    x_size = x0.numel()
    initial_state = (
        torch.cat((x0.reshape(-1), q0.reshape(-1)))
        if has_quadrature
        else x0.reshape(-1)
    )

    def rhs(time, state):
        x = state[:x_size].reshape_as(x0)
        dx = dxdt(time, x, None, u, p).reshape(-1)
        if not has_quadrature:
            return dx
        dq = dqdt(time, x, None, u, p).reshape(-1)
        return torch.cat((dx, dq))

    solution = odeint(
        rhs,
        initial_state,
        times,
        method=parameters.method,
        rtol=parameters.rtol,
        atol=parameters.atol,
        options=(
            dict(parameters.options)
            if parameters.options is not None
            else None
        ),
    )
    x_solution = solution[..., :x_size].reshape(-1, *x0.shape)
    q_solution = (
        solution[..., x_size:].reshape(-1, *q0.shape)
        if has_quadrature
        else None
    )
    if is_scalar_endpoint:
        return x_solution[-1], None, q_solution[-1] if has_quadrature else None

    if drop_initial:
        x_solution = x_solution[1:]
        if has_quadrature:
            q_solution = q_solution[1:]
    return (
        x_solution.movedim(0, -1),
        None,
        q_solution.movedim(0, -1) if has_quadrature else None,
    )


def _build_time_grid(end_point, state):
    if isinstance(end_point, (float, int)):
        if end_point == 0:
            return None, True, False
        return (
            torch.tensor(
                [0, end_point], dtype=state.dtype, device=state.device
            ),
            True,
            False,
        )

    times = torch.as_tensor(end_point, dtype=state.dtype, device=state.device)
    if times.ndim != 1:
        raise ValueError(
            "ODE evaluation times must be a one-dimensional tensor"
        )
    if times.numel() == 0:
        raise ValueError("ODE evaluation times must not be empty")
    if times.numel() == 1 and bool(times[0] == 0):
        return None, False, False

    starts_at_zero = bool(times[0] == 0)
    if starts_at_zero:
        integration_times = times
    else:
        integration_times = torch.cat((torch.zeros_like(times[:1]), times))

    deltas = integration_times[1:] - integration_times[:-1]
    if not bool(torch.all(deltas > 0) or torch.all(deltas < 0)):
        raise ValueError("ODE evaluation times must be strictly monotonic")
    return integration_times, False, not starts_at_zero
