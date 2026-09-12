from __future__ import annotations

from dataclasses import dataclass

from coker.backends import get_backend_by_name

import numpy as np
import torch

from coker.backends.backend import VariationalSolver
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
)
from coker.dynamics.controls import BoundedVariable
from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    generate_discritisation_operators,
)
from coker.dynamics.variational.polynomials import InterpolatingPolyCollection
from coker.dynamics.variational.solution import VariationalSolution
from coker.toolkits.codesign.optimisation import SolveFailure, SolveInfo


@dataclass
class _Parameter:
    name: str
    lower: float
    upper: float
    guess: float


class PytorchVariationalSolver(VariationalSolver):
    """Fixed-horizon, bound-only neural-ODE parameter fitter."""

    def __init__(self, problem):
        self.problem = problem
        self._backend = get_backend_by_name("pytorch", set_current=False)
        self._loss = self._backend.lower(problem.loss)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch variational solving requires CUDA availability"
            )
        if problem.horizon_decision is not None:
            raise NotImplementedError(
                "Optimized horizons are not supported by PyTorch variational solving"
            )
        if problem.control:
            raise NotImplementedError(
                "Control declarations are not supported by PyTorch variational solving"
            )
        if problem.quadratures:
            raise NotImplementedError(
                "Quadratures are not supported by PyTorch variational solving"
            )
        if (
            problem.path_constraints
            or problem.terminal_constraints
            or problem.initial_constraints
        ):
            raise NotImplementedError(
                "Variational constraints are not supported by PyTorch variational solving"
            )
        _, z_dim, q_dim = problem.system.get_state_dimensions()
        if z_dim is not None:
            raise NotImplementedError(
                "Algebraic states are not supported by PyTorch variational solving"
            )
        if q_dim is not None:
            raise NotImplementedError(
                "Quadrature states are not supported by PyTorch variational solving"
            )
        self._parameters = []
        seen = set()
        for declaration in problem.parameters or []:
            if not isinstance(declaration, BoundedVariable):
                raise NotImplementedError(
                    "Only BoundedVariable parameters are supported by PyTorch variational solving"
                )
            if declaration.name in seen:
                continue
            seen.add(declaration.name)
            self._parameters.append(
                _Parameter(
                    declaration.name,
                    declaration.lower_bound,
                    declaration.upper_bound,
                    declaration.guess,
                )
            )
        self._names = [p.name for p in self._parameters]
        self._device = torch.device("cuda")
        self._dtype = torch.float32

    @property
    def parameters(self):
        return list(self._names)

    def _check_fixed(self, fixed_parameters):
        unknown = set(fixed_parameters) - set(self._names)
        if unknown:
            raise ValueError(
                f"Unknown variational parameter(s): {sorted(unknown)}"
            )
        for p in self._parameters:
            if p.name in fixed_parameters:
                value = float(fixed_parameters[p.name])
                if (
                    not np.isfinite(value)
                    or value < p.lower
                    or value > p.upper
                ):
                    raise ValueError(
                        f"Fixed parameter {p.name!r} is outside its bounds"
                    )

    def _system_parameters(self, values):
        if values.numel() == 0 or self.problem.system_parameter_map is None:
            return values
        matrix = torch.as_tensor(
            self.problem.system_parameter_map,
            device=self._device,
            dtype=self._dtype,
        )
        return (matrix @ values.reshape(-1, 1)).reshape(-1)

    def _trajectory_times(self):
        tau, map_time, *_ = generate_discritisation_operators(
            (0.0, float(self.problem.t_final)), 31
        )
        return torch.as_tensor(
            [map_time(value) for value in tau],
            device=self._device,
            dtype=self._dtype,
        )

    def _integrate(self, values, times):
        system = self.problem.system
        parameters = self._system_parameters(values)
        if values.numel() == 0 and system.parameters is None:
            parameters = None
        x0, z0 = system.x0(0.0, None, parameters)
        if z0 is not None:
            raise NotImplementedError(
                "Algebraic states are not supported by PyTorch variational solving"
            )
        x0 = torch.as_tensor(
            x0, device=self._device, dtype=self._dtype
        ).reshape(-1)
        try:
            from torchdiffeq import odeint
        except ImportError as ex:
            raise RuntimeError(
                "PyTorch variational solving requires `pip install coker[pytorch]`"
            ) from ex

        def rhs(time, state):
            return system.dxdt(time, state, None, None, parameters).reshape_as(
                state
            )

        return odeint(rhs, x0, times, method="dopri5", rtol=1e-6, atol=1e-8)

    def _evaluate(self, values, *, trajectory=False):
        system = self.problem.system

        def output_native(*args):
            time = torch.as_tensor(
                args[0], device=self._device, dtype=self._dtype
            )
            parameter_values = (args[-1] if len(args) > 1 else values).reshape(
                -1
            )
            parameters = self._system_parameters(parameter_values)
            if parameter_values.numel() == 0 and system.parameters is None:
                parameters = None
            if time.ndim:
                grid = (
                    time
                    if time[0] == 0
                    else torch.cat((torch.zeros_like(time[:1]), time))
                )
                state = self._integrate(parameter_values, grid)[-1]
            elif time == 0:
                state = self._integrate(parameter_values, time.reshape(1))[0]
            else:
                state = self._integrate(
                    parameter_values,
                    torch.stack((torch.zeros_like(time), time)),
                )[-1]
            return system.y(time, state, None, None, parameters, None)

        if trajectory:
            times = self._trajectory_times()
            return times, self._integrate(values, times), output_native
        return output_native, self.problem.loss.input_spaces()[0]

    def _raw_guess(self, parameter):
        guess = float(parameter.guess)
        if not np.isfinite(guess):
            raise ValueError(
                f"Initial guess for parameter {parameter.name!r} must be finite"
            )
        if (np.isfinite(parameter.lower) and guess < parameter.lower) or (
            np.isfinite(parameter.upper) and guess > parameter.upper
        ):
            raise ValueError(
                f"Initial guess for parameter {parameter.name!r} is outside its bounds"
            )
        epsilon = torch.finfo(self._dtype).eps
        lower, upper = parameter.lower, parameter.upper
        if np.isfinite(lower) and np.isfinite(upper):
            if lower >= upper:
                raise ValueError(
                    f"Parameter {parameter.name!r} must have lower bound below upper bound"
                )
            ratio = min(
                max((guess - lower) / (upper - lower), epsilon), 1 - epsilon
            )
            return torch.logit(
                torch.tensor(ratio, device=self._device, dtype=self._dtype)
            )
        if np.isfinite(lower):
            delta = max(guess - lower, epsilon)
            return torch.log(
                torch.expm1(
                    torch.tensor(delta, device=self._device, dtype=self._dtype)
                )
            )
        if np.isfinite(upper):
            delta = max(upper - guess, epsilon)
            return torch.log(
                torch.expm1(
                    torch.tensor(delta, device=self._device, dtype=self._dtype)
                )
            )
        return torch.tensor(guess, device=self._device, dtype=self._dtype)

    def _parameter_value(self, raw_value, parameter):
        lower, upper = parameter.lower, parameter.upper
        if np.isfinite(lower) and np.isfinite(upper):
            return lower + (upper - lower) * torch.sigmoid(raw_value)
        if np.isfinite(lower):
            return lower + torch.nn.functional.softplus(raw_value)
        if np.isfinite(upper):
            return upper - torch.nn.functional.softplus(raw_value)
        return raw_value

    def solve(self, **fixed_parameters):
        self._check_fixed(fixed_parameters)
        free = [p for p in self._parameters if p.name not in fixed_parameters]
        fixed = {
            p.name: torch.tensor(
                float(v), device=self._device, dtype=self._dtype
            )
            for p, v in (
                (p, fixed_parameters[p.name])
                for p in self._parameters
                if p.name in fixed_parameters
            )
        }
        raw = torch.nn.Parameter(
            torch.stack([self._raw_guess(parameter) for parameter in free])
            if free
            else torch.zeros(0, device=self._device, dtype=self._dtype)
        )

        def values_from_raw():
            out = []
            j = 0
            for p in self._parameters:
                if p.name in fixed:
                    out.append(fixed[p.name])
                else:
                    out.append(self._parameter_value(raw[j], p))
                    j += 1
            return (
                torch.stack(out)
                if out
                else torch.zeros(0, device=self._device, dtype=self._dtype)
            )

        def objective():
            values = values_from_raw()
            output_native, solution_space = self._evaluate(values)
            signature = FunctionSignature(
                tuple(
                    FunctionInputSpec(f"arg_{i}", s)
                    for i, s in enumerate(solution_space.arguments)
                ),
                (FunctionOutputSpec("output", solution_space.output[0]),),
            )
            solution = self._backend.import_function(output_native, signature)
            args = [solution]
            if len(self.problem.loss.input_spaces()) > 1:
                args.append(values)
            return self._loss.execute(args)[0]

        try:
            if free:
                optimizer = torch.optim.LBFGS(
                    [raw],
                    max_iter=100,
                    tolerance_grad=1e-6,
                    tolerance_change=1e-9,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    optimizer.zero_grad()
                    cost_value = objective()
                    if not torch.isfinite(cost_value):
                        raise FloatingPointError(
                            "non-finite variational objective"
                        )
                    cost_value.backward()
                    return cost_value

                optimizer.step(closure)
            cost = objective()
            if not torch.isfinite(cost):
                raise FloatingPointError("non-finite variational objective")
        except Exception as ex:
            info = SolveInfo(
                "pytorch", "LBFGS", False, str(ex), iteration_count=None
            )
            raise SolveFailure(
                "PyTorch variational solve failed", info
            ) from ex
        values = values_from_raw().detach()
        times, states, _ = self._evaluate(values, trajectory=True)
        state_values = states.detach().cpu().numpy()
        poly = InterpolatingPoly(
            state_values.shape[1],
            (0.0, float(self.problem.t_final)),
            len(times) - 1,
            state_values.reshape(-1, 1),
        )
        info = SolveInfo(
            "pytorch", "LBFGS", True, "converged", iteration_count=None
        )
        parameter_solutions = {
            p.name: float(values[i].cpu())
            for i, p in enumerate(self._parameters)
        }
        return VariationalSolution(
            cost=float(cost.detach().cpu()),
            path=InterpolatingPolyCollection([poly]),
            projectors=(np.eye(state_values.shape[1]), None, None),
            control_solutions=[],
            parameter_solutions=parameter_solutions,
            parameters=values.cpu().numpy(),
            output=self.problem.system.y,
            t_final=float(self.problem.t_final),
            solve_info=info,
        )


def create_variational_solver(problem):
    from coker.dynamics.variational.problem import VariationalProblem

    if not isinstance(problem, VariationalProblem):
        raise NotImplementedError(
            "PyTorch variational solver requires a VariationalProblem"
        )
    return PytorchVariationalSolver(problem)
