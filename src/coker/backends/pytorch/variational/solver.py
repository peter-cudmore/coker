from __future__ import annotations

from dataclasses import dataclass

from coker.backends import get_backend_by_name

import numpy as np
import torch
from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    generate_discritisation_operators,
)

from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
)
from coker.dynamics.controls import BoundedVariable
from coker.dynamics.variational.polynomials import InterpolatingPolyCollection
from coker.dynamics.variational.solution import VariationalSolution
from coker.toolkits.codesign.optimisation import SolveFailure, SolveInfo


@dataclass
class _Parameter:
    name: str
    lower: float
    upper: float
    guess: float


class PytorchVariationalSolver:
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
        x_dim, z_dim, q_dim = problem.system.get_state_dimensions()
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

    def _evaluate(self, values, *, trajectory=False):
        system = self.problem.system
        p_system = self._system_parameters(values)
        if values.numel() == 0 and self.problem.system.parameters is None:
            p_system = None
        x0, z0 = system.x0(0.0, None, p_system)
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
        if trajectory:
            tau, map_time, *_ = generate_discritisation_operators(
                (0.0, float(self.problem.t_final)), 31
            )
            times = torch.as_tensor(
                [map_time(value) for value in tau],
                device=self._device,
                dtype=self._dtype,
            )
        else:
            times = torch.linspace(
                0.0,
                float(self.problem.t_final),
                32,
                device=self._device,
                dtype=self._dtype,
            )

        def rhs(t, x):
            return system.dxdt(t, x, None, None, p_system).reshape_as(x)

        states = odeint(rhs, x0, times, method="dopri5", rtol=1e-6, atol=1e-8)

        def output_native(*args):
            t = args[0]
            p_value = (args[-1] if len(args) > 1 else values).reshape(-1)
            ps = self._system_parameters(p_value)
            if p_value.numel() == 0 and system.parameters is None:
                ps = None
            initial, initial_z = system.x0(0.0, None, ps)
            if initial_z is not None:
                raise NotImplementedError(
                    "Algebraic states are not supported by PyTorch variational solving"
                )
            initial = torch.as_tensor(
                initial, device=self._device, dtype=self._dtype
            ).reshape(-1)

            def rhs_local(time, state):
                return system.dxdt(time, state, None, None, ps).reshape_as(
                    state
                )

            tt = torch.as_tensor(t, device=self._device, dtype=self._dtype)
            if tt.ndim:
                grid = (
                    tt
                    if tt[0] == 0
                    else torch.cat((torch.zeros_like(tt[:1]), tt))
                )
                state = odeint(rhs_local, initial, grid)[-1]
            elif tt == 0:
                state = initial
            else:
                state = odeint(
                    rhs_local, initial, torch.stack((torch.zeros_like(tt), tt))
                )[-1]
            return system.y(tt, state, None, None, ps, None)

        if trajectory:
            return times, states, output_native
        solution_space = self.problem.loss.input_spaces()[0]
        return output_native, solution_space

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
            torch.zeros(len(free), device=self._device, dtype=self._dtype)
        )

        def values_from_raw():
            out = []
            j = 0
            for p in self._parameters:
                if p.name in fixed:
                    out.append(fixed[p.name])
                else:
                    lo, hi = p.lower, p.upper
                    if np.isfinite(lo) and np.isfinite(hi):
                        out.append(
                            torch.as_tensor(
                                lo, device=self._device, dtype=self._dtype
                            )
                            + (hi - lo) * torch.sigmoid(raw[j])
                        )
                    elif np.isfinite(lo):
                        out.append(
                            torch.as_tensor(
                                lo, device=self._device, dtype=self._dtype
                            )
                            + torch.nn.functional.softplus(raw[j])
                        )
                    elif np.isfinite(hi):
                        out.append(
                            torch.as_tensor(
                                hi, device=self._device, dtype=self._dtype
                            )
                            - torch.nn.functional.softplus(raw[j])
                        )
                    else:
                        out.append(raw[j])
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
