from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from coker.algebra.function import (
    Function,
    Tracer,
    create_function_from_native,
)
from coker.backends import get_backend_by_name
from coker.backends.backend import VariationalSolver
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
)
from coker.backends.pytorch.dynamics import PytorchODESolverParameters
from coker.dynamics.controls import BoundedVariable
from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    generate_discritisation_operators,
)
from coker.dynamics.variational.polynomials import InterpolatingPolyCollection
from coker.dynamics.variational.solution import VariationalSolution
from coker.toolkits.codesign import SolveFailure, SolveInfo, SolverOptions


_TRAJECTORY_GRID_NODES = 31
_OPTIMISER_TYPES = {
    "Adam": torch.optim.Adam,
    "LBFGS": torch.optim.LBFGS,
}


@dataclass(frozen=True)
class PytorchVariationalSolverOptions(SolverOptions):
    """CUDA direct-shooting configuration for a PyTorch variational solve."""

    ode: PytorchODESolverParameters = PytorchODESolverParameters()
    optimiser_method: str = "LBFGS"
    optimiser_options: Mapping[str, object] = field(
        default_factory=lambda: {
            "max_iter": 100,
            "tolerance_grad": 1e-6,
            "tolerance_change": 1e-9,
            "line_search_fn": "strong_wolfe",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.ode, PytorchODESolverParameters):
            raise TypeError("ode must be PytorchODESolverParameters")
        if self.optimiser_method not in _OPTIMISER_TYPES:
            raise ValueError(
                f"Unsupported PyTorch optimiser {self.optimiser_method!r}"
            )
        if not isinstance(self.optimiser_options, Mapping):
            raise TypeError("optimiser_options must be a mapping")


class PytorchVariationalSolver(VariationalSolver):
    """Fixed-horizon, bound-only neural-ODE parameter fitter."""

    def __init__(
        self, problem, options: PytorchVariationalSolverOptions | None = None
    ):
        self.problem = problem
        self._options = options or PytorchVariationalSolverOptions()
        backend = get_backend_by_name("pytorch", set_current=False)
        self._loss_is_trace = isinstance(problem.loss, Tracer)
        loss = (
            Function(problem.loss.tape, problem.loss, backend="pytorch")
            if self._loss_is_trace
            else problem.loss
        )
        self._loss = backend.lower(loss)
        self._quadratures = tuple(
            backend.lower(
                Function(
                    spec.integrand.tape, spec.integrand, backend="pytorch"
                )
            )
            for spec in problem.quadratures
        )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch variational solving requires CUDA availability"
            )
        if problem.horizon_decision is not None:
            raise NotImplementedError(
                "Optimized horizons are not supported by "
                "PyTorch variational solving"
            )
        if problem.control:
            raise NotImplementedError(
                "Control declarations are not supported by "
                "PyTorch variational solving"
            )
        if (
            problem.path_constraints
            or problem.terminal_constraints
            or problem.initial_constraints
        ):
            raise NotImplementedError(
                "Variational constraints are not supported by "
                "PyTorch variational solving"
            )
        _, z_dim, q_dim = problem.system.get_state_dimensions()
        if z_dim is not None:
            raise NotImplementedError(
                "Algebraic states are not supported by "
                "PyTorch variational solving"
            )
        if q_dim is not None:
            raise NotImplementedError(
                "System quadrature states are not supported by "
                "PyTorch variational solving"
            )
        self._parameters = []
        seen = set()
        for declaration in problem.parameters or []:
            if not isinstance(declaration, BoundedVariable):
                raise NotImplementedError(
                    "Only BoundedVariable parameters are supported by "
                    "PyTorch variational solving"
                )
            if declaration.name in seen:
                continue
            seen.add(declaration.name)
            self._parameters.append(declaration)
        self._names = [p.name for p in self._parameters]
        self._device = torch.device("cuda")
        self._dtype = torch.float32
        self._t_initial = torch.zeros(
            (), device=self._device, dtype=self._dtype
        )
        self._t_final = torch.tensor(
            float(problem.t_final), device=self._device, dtype=self._dtype
        )

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
                    or value < p.lower_bound
                    or value > p.upper_bound
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
            (0.0, float(self.problem.t_final)), _TRAJECTORY_GRID_NODES
        )
        return torch.as_tensor(
            [map_time(value) for value in tau],
            device=self._device,
            dtype=self._dtype,
        )

    def _trace_arguments(
        self, signature, time, state, parameters, quadratures
    ):
        system = self.problem.system

        def state_trajectory(_time):
            return state

        def output_trajectory(output_time):
            return system.y(
                output_time, state, None, None, parameters, None
            ).reshape(-1)

        arguments = {
            "t": time,
            "t_final": self._t_final,
            "t_0": self._t_initial,
            "_state": state_trajectory,
            "p": parameters,
            "_output": output_trajectory,
        }
        arguments.update(
            {
                f"q_{spec.channel}": quadratures[index]
                for index, spec in enumerate(self.problem.quadratures)
            }
        )
        try:
            return [
                arguments[input_spec.name] for input_spec in signature.inputs
            ]
        except KeyError as ex:
            raise NotImplementedError(
                f"Unsupported quadrature input {ex.args[0]!r}"
            ) from ex

    def _integrate(self, values, times):
        system = self.problem.system
        parameters = self._system_parameters(values)
        if values.numel() == 0 and system.parameters is None:
            parameters = None
        x0, z0 = system.x0(0.0, None, parameters)
        if z0 is not None:
            raise NotImplementedError(
                "Algebraic states are not supported by "
                "PyTorch variational solving"
            )
        x0 = torch.as_tensor(
            x0, device=self._device, dtype=self._dtype
        ).reshape(-1)
        q0 = torch.as_tensor(
            [spec.initial_state for spec in self.problem.quadratures],
            device=self._device,
            dtype=self._dtype,
        )
        try:
            from torchdiffeq import odeint
        except ImportError as ex:
            raise RuntimeError(
                "PyTorch variational solving requires "
                "`pip install coker[pytorch]`"
            ) from ex

        def rhs(time, integrated):
            state = integrated[: x0.numel()]
            quadratures = integrated[x0.numel() :]
            dx = system.dxdt(time, state, None, None, parameters).reshape(-1)
            if not self._quadratures:
                return dx
            dq = torch.stack(
                [
                    quadrature.execute(
                        self._trace_arguments(
                            quadrature.signature,
                            time,
                            state,
                            parameters,
                            quadratures,
                        )
                    )[0].reshape(())
                    for quadrature in self._quadratures
                ]
            )
            return torch.cat((dx, dq))

        initial_state = torch.cat((x0, q0)) if self._quadratures else x0
        integrated = odeint(
            rhs,
            initial_state,
            times,
            method=self._options.ode.method,
            rtol=self._options.ode.rtol,
            atol=self._options.ode.atol,
            options=self._options.ode.options,
        )
        return (
            integrated[..., : x0.numel()],
            integrated[..., x0.numel() :] if self._quadratures else None,
        )

    def _evaluate(self, values):
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
                    if bool(time[0] == 0)
                    else torch.cat((torch.zeros_like(time[:1]), time))
                )
                state, quadratures = self._integrate(parameter_values, grid)
                state = state[-1]
                quadratures = (
                    quadratures[-1] if quadratures is not None else None
                )
            elif bool(time == 0):
                state, quadratures = self._integrate(
                    parameter_values, time.reshape(1)
                )
                state = state[0]
                quadratures = (
                    quadratures[0] if quadratures is not None else None
                )
            else:
                state, quadratures = self._integrate(
                    parameter_values,
                    torch.stack((torch.zeros_like(time), time)),
                )
                state = state[-1]
                quadratures = (
                    quadratures[-1] if quadratures is not None else None
                )
            return system.y(time, state, None, None, parameters, quadratures)

        return output_native, self.problem.loss.input_spaces()[0]

    def _evaluate_trace_loss(self, values):
        states, quadratures = self._integrate(
            values, torch.stack((self._t_initial, self._t_final))
        )
        parameters = self._system_parameters(values)
        if values.numel() == 0 and self.problem.system.parameters is None:
            parameters = None
        return self._loss.execute(
            self._trace_arguments(
                self._loss.signature,
                self._t_final,
                states[-1],
                parameters,
                quadratures[-1] if quadratures is not None else (),
            )
        )[0]

    def _raw_guess(self, parameter):
        guess = float(parameter.guess)
        if not np.isfinite(guess):
            raise ValueError(
                f"Initial guess for parameter {parameter.name!r} "
                "must be finite"
            )
        if (
            np.isfinite(parameter.lower_bound)
            and guess < parameter.lower_bound
        ) or (
            np.isfinite(parameter.upper_bound)
            and guess > parameter.upper_bound
        ):
            raise ValueError(
                f"Initial guess for parameter {parameter.name!r} is outside "
                "its bounds"
            )
        epsilon = torch.finfo(self._dtype).eps
        lower, upper = parameter.lower_bound, parameter.upper_bound
        if np.isfinite(lower) and np.isfinite(upper):
            if lower >= upper:
                raise ValueError(
                    f"Parameter {parameter.name!r} must have lower bound "
                    "below upper bound"
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
        lower, upper = parameter.lower_bound, parameter.upper_bound
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
            if self._loss_is_trace:
                return self._evaluate_trace_loss(values)
            output_native, solution_space = self._evaluate(values)
            signature = FunctionSignature(
                tuple(
                    FunctionInputSpec(f"arg_{i}", space)
                    for i, space in enumerate(solution_space.arguments)
                ),
                (FunctionOutputSpec("output", solution_space.output[0]),),
            )
            solution = create_function_from_native(
                output_native, signature, backend="pytorch"
            )
            args = [solution]
            if len(self.problem.loss.input_spaces()) > 1:
                args.append(values)
            return self._loss.execute(args)[0]

        try:
            if free:
                optimizer = _OPTIMISER_TYPES[self._options.optimiser_method](
                    [raw], **dict(self._options.optimiser_options)
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
                "pytorch",
                self._options.optimiser_method,
                False,
                str(ex),
                iteration_count=None,
            )
            raise SolveFailure(
                "PyTorch variational solve failed", info
            ) from ex
        values = values_from_raw().detach()
        times = self._trajectory_times()
        states, quadratures = self._integrate(values, times)
        state_values = states.detach().cpu().numpy()
        quadrature_values = (
            quadratures.detach().cpu().numpy()
            if quadratures is not None
            else None
        )
        path_values = (
            np.concatenate((state_values, quadrature_values), axis=1)
            if quadrature_values is not None
            else state_values
        )
        poly = InterpolatingPoly(
            path_values.shape[1],
            (0.0, float(self.problem.t_final)),
            len(times) - 1,
            path_values.reshape(-1, 1),
        )
        state_projector = np.zeros(
            (state_values.shape[1], path_values.shape[1])
        )
        state_projector[:, : state_values.shape[1]] = np.eye(
            state_values.shape[1]
        )
        quadrature_projector = None
        if quadrature_values is not None:
            quadrature_projector = np.zeros(
                (quadrature_values.shape[1], path_values.shape[1])
            )
            quadrature_projector[:, state_values.shape[1] :] = np.eye(
                quadrature_values.shape[1]
            )
        info = SolveInfo(
            "pytorch",
            self._options.optimiser_method,
            True,
            "converged",
            iteration_count=None,
        )
        parameter_solutions = {
            p.name: float(values[i].cpu())
            for i, p in enumerate(self._parameters)
        }
        return VariationalSolution(
            cost=float(cost.detach().cpu()),
            path=InterpolatingPolyCollection([poly]),
            projectors=(state_projector, None, quadrature_projector),
            control_solutions=[],
            parameter_solutions=parameter_solutions,
            parameters=values.cpu().numpy(),
            output=self.problem.system.y,
            t_final=float(self.problem.t_final),
            solve_info=info,
        )


def create_variational_solver(
    problem, options: PytorchVariationalSolverOptions | None = None
):
    from coker.dynamics.variational.problem import VariationalProblem

    if not isinstance(problem, VariationalProblem):
        raise NotImplementedError(
            "PyTorch variational solver requires a VariationalProblem"
        )
    return PytorchVariationalSolver(problem, options)
