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
from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    InterpolatingPolyCollection,
    generate_discritisation_operators,
)
from coker.dynamics.variational.solution import VariationalSolution
from coker.parameters import (
    BoundVector,
    BoundedVariable,
    DenseTensorVariable,
    UnboundedVariable,
)
from coker.parameters.function_parameters import FunctionParameter
from coker.toolkits.codesign import SolveFailure, SolveInfo, SolverOptions


_TRAJECTORY_GRID_NODES = 31
_OPTIMISER_TYPES = {
    "Adam": torch.optim.Adam,
    "LBFGS": torch.optim.LBFGS,
}


@dataclass(frozen=True)
class PytorchVariationalSolverOptions(SolverOptions):
    """Direct-shooting configuration for a PyTorch variational solve.

    ``ode`` is forwarded to :func:`torchdiffeq.odeint`. ``optimiser_method``
    selects a supported :mod:`torch.optim` optimiser, while
    ``optimiser_options`` are forwarded to its constructor.
    """

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


@dataclass(frozen=True)
class _ParameterBlock:
    """One original declaration represented in the flat solver vector."""

    declaration: (
        BoundedVariable | UnboundedVariable | BoundVector | DenseTensorVariable
    )
    offset: int
    shape: tuple[int, ...]
    names: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.names)


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
        self._system_quadrature_size = q_dim.flat() if q_dim else 0
        self._device = backend.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = backend.dtype or torch.get_default_dtype()
        self._blocks = self._collect_parameter_blocks()
        self._names = tuple(
            name for block in self._blocks for name in block.names
        )
        if len(set(self._names)) != len(self._names):
            raise ValueError(
                "PyTorch variational parameter names must be unique"
            )
        self._parameter_indices = {
            name: index for index, name in enumerate(self._names)
        }
        (
            self._lower_bounds,
            self._upper_bounds,
            guesses,
            self._bound_fixed_coordinates,
        ) = self._parameter_vectors()
        self._two_sided_bounds = (
            torch.isfinite(self._lower_bounds)
            & torch.isfinite(self._upper_bounds)
            & ~self._bound_fixed_coordinates
        )
        self._lower_only_bounds = torch.isfinite(
            self._lower_bounds
        ) & ~torch.isfinite(self._upper_bounds)
        self._upper_only_bounds = ~torch.isfinite(
            self._lower_bounds
        ) & torch.isfinite(self._upper_bounds)
        self._has_two_sided_bounds = bool(self._two_sided_bounds.any())
        self._has_lower_only_bounds = bool(self._lower_only_bounds.any())
        self._has_upper_only_bounds = bool(self._upper_only_bounds.any())
        self._raw_initial = self._raw_guess(guesses)
        self._system_parameter_map = (
            torch.as_tensor(
                problem.system_parameter_map,
                device=self._device,
                dtype=self._dtype,
            )
            if problem.system_parameter_map is not None
            else None
        )
        self._t_initial = torch.zeros(
            (), device=self._device, dtype=self._dtype
        )
        self._t_final = torch.tensor(
            float(problem.t_final), device=self._device, dtype=self._dtype
        )

    @property
    def parameters(self):
        return list(self._names)

    def _collect_parameter_blocks(self) -> tuple[_ParameterBlock, ...]:
        """Retain original declaration blocks behind scalar fixing names."""
        layout = self.problem.parameter_layout
        entries: list[
            tuple[
                BoundedVariable
                | UnboundedVariable
                | BoundVector
                | DenseTensorVariable,
                int,
                int,
            ]
        ] = []
        offset = 0
        if layout is None:
            declarations = (
                (declaration, None)
                for declaration in self.problem.parameters or []
            )
            concrete_offsets = None
        else:
            concrete_offsets = layout.concrete_offsets
            declarations = zip(
                layout.declarations,
                (
                    layout.offsets
                    if concrete_offsets is None
                    else concrete_offsets
                ),
            )

        for declaration, layout_offsets in declarations:
            concrete = (
                declaration.list_concrete_parameters()
                if isinstance(declaration, FunctionParameter)
                else (declaration,)
            )
            if concrete_offsets is None:
                block_end = (
                    offset if layout_offsets is None else layout_offsets[0]
                )
            elif len(layout_offsets) != len(concrete):
                raise ValueError(
                    "Function parameter layout does not match its declarations"
                )
            for concrete_index, concrete_declaration in enumerate(concrete):
                if not isinstance(
                    concrete_declaration,
                    (
                        BoundedVariable,
                        UnboundedVariable,
                        BoundVector,
                        DenseTensorVariable,
                    ),
                ):
                    raise NotImplementedError(
                        "PyTorch variational solving requires scalar, vector, "
                        "or dense tensor parameter declarations"
                    )
                size = (
                    1
                    if isinstance(
                        concrete_declaration,
                        (BoundedVariable, UnboundedVariable),
                    )
                    else concrete_declaration.size
                )
                if concrete_offsets is None:
                    block_start = block_end
                    block_end += size
                else:
                    block_start, block_end = layout_offsets[concrete_index]
                    if block_end - block_start != size:
                        raise ValueError(
                            "Function parameter layout does not match its "
                            "declarations"
                        )
                entries.append((concrete_declaration, block_start, block_end))
            if (
                concrete_offsets is None
                and layout_offsets is not None
                and block_end != layout_offsets[1]
            ):
                raise ValueError(
                    "Function parameter layout does not match its declarations"
                )
            offset = block_end

        if concrete_offsets is not None:
            unique_entries = {
                (start, end): (declaration, start, end)
                for declaration, start, end in entries
            }
            entries = [unique_entries[key] for key in sorted(unique_entries)]
            offset = 0
            for _, block_start, block_end in entries:
                if block_start != offset:
                    raise ValueError(
                        "Function parameter layout does not match its "
                        "declarations"
                    )
                offset = block_end

        flat_parameters = self.problem.parameters or []
        width = offset
        if len(flat_parameters) == width and all(
            isinstance(declaration, (BoundedVariable, UnboundedVariable))
            for declaration in flat_parameters
        ):
            names = tuple(declaration.name for declaration in flat_parameters)
        else:
            names = tuple(
                (
                    declaration.name
                    if isinstance(
                        declaration, (BoundedVariable, UnboundedVariable)
                    )
                    else f"{declaration.name}_{index}"
                )
                for declaration, start, end in entries
                for index in range(end - start)
            )
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(
                "PyTorch variational parameter names must be non-empty strings"
            )
        return tuple(
            _ParameterBlock(
                declaration,
                start,
                (
                    ()
                    if isinstance(
                        declaration, (BoundedVariable, UnboundedVariable)
                    )
                    else declaration.shape
                ),
                names[start:end],
            )
            for declaration, start, end in entries
        )

    def _parameter_vectors(self):
        lower_blocks = []
        upper_blocks = []
        guess_blocks = []
        for block in self._blocks:
            declaration = block.declaration
            if isinstance(declaration, (BoundedVariable, UnboundedVariable)):
                lower = np.asarray([declaration.lower_bound], dtype=float)
                upper = np.asarray([declaration.upper_bound], dtype=float)
                guess = np.asarray([declaration.guess], dtype=float)
            elif isinstance(declaration, BoundVector):
                lower = declaration.lower_bound.reshape(-1)
                upper = declaration.upper_bound.reshape(-1)
                guess = declaration.guess.reshape(-1)
            else:
                assert isinstance(declaration, DenseTensorVariable)
                lower = np.full(declaration.size, -np.inf)
                upper = np.full(declaration.size, np.inf)
                guess = declaration.guess.reshape(-1)
            lower_blocks.append(lower)
            upper_blocks.append(upper)
            guess_blocks.append(guess)

        lower_values = (
            np.concatenate(lower_blocks) if lower_blocks else np.zeros(0)
        )
        upper_values = (
            np.concatenate(upper_blocks) if upper_blocks else np.zeros(0)
        )
        guesses = np.concatenate(guess_blocks) if guess_blocks else np.zeros(0)
        if not np.all(np.isfinite(guesses)):
            raise ValueError(
                "PyTorch variational parameter guesses must be finite"
            )
        if (
            np.any(np.isnan(lower_values))
            or np.any(np.isnan(upper_values))
            or np.any(np.isposinf(lower_values))
            or np.any(np.isneginf(upper_values))
        ):
            raise ValueError(
                "PyTorch variational parameter bounds are invalid"
            )
        finite_lower = np.isfinite(lower_values)
        finite_upper = np.isfinite(upper_values)
        if np.any(finite_lower & finite_upper & (lower_values > upper_values)):
            raise ValueError(
                "PyTorch variational lower bounds must not exceed upper bounds"
            )
        if np.any(
            (finite_lower & (guesses < lower_values))
            | (finite_upper & (guesses > upper_values))
        ):
            raise ValueError(
                "PyTorch variational parameter guesses are outside their "
                "bounds"
            )
        return (
            torch.as_tensor(
                lower_values, device=self._device, dtype=self._dtype
            ),
            torch.as_tensor(
                upper_values, device=self._device, dtype=self._dtype
            ),
            torch.as_tensor(guesses, device=self._device, dtype=self._dtype),
            torch.as_tensor(
                finite_lower & finite_upper & (lower_values == upper_values),
                device=self._device,
                dtype=torch.bool,
            ),
        )

    def _check_fixed(self, fixed_parameters):
        unknown = set(fixed_parameters) - set(self._names)
        if unknown:
            raise ValueError(
                f"Unknown variational parameter(s): {sorted(unknown)}"
            )
        for name, fixed_value in fixed_parameters.items():
            value = float(fixed_value)
            index = self._parameter_indices[name]
            lower = self._lower_bounds[index]
            upper = self._upper_bounds[index]
            if not np.isfinite(value) or value < lower or value > upper:
                raise ValueError(
                    f"Fixed parameter {name!r} is outside its bounds"
                )

    def _system_parameters(self, values):
        if values.numel() == 0 or self._system_parameter_map is None:
            return values
        return (self._system_parameter_map @ values.reshape(-1, 1)).reshape(-1)

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
        system_q0 = torch.zeros(
            self._system_quadrature_size,
            device=self._device,
            dtype=self._dtype,
        )
        registered_q0 = torch.as_tensor(
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
            state_end = x0.numel()
            system_q_end = state_end + self._system_quadrature_size
            state = integrated[:state_end]
            registered_quadratures = integrated[system_q_end:]
            dx = system.dxdt(time, state, None, None, parameters).reshape(-1)
            derivatives = [dx]
            if self._system_quadrature_size:
                derivatives.append(
                    system.dqdt(time, state, None, None, parameters).reshape(
                        -1
                    )
                )
            if self._quadratures:
                derivatives.append(
                    torch.stack(
                        [
                            quadrature.execute(
                                self._trace_arguments(
                                    quadrature.signature,
                                    time,
                                    state,
                                    parameters,
                                    registered_quadratures,
                                )
                            )[0].reshape(())
                            for quadrature in self._quadratures
                        ]
                    )
                )
            return torch.cat(derivatives)

        initial_state = torch.cat((x0, system_q0, registered_q0))
        integrated = odeint(
            rhs,
            initial_state,
            times,
            method=self._options.ode.method,
            rtol=self._options.ode.rtol,
            atol=self._options.ode.atol,
            options=self._options.ode.options,
        )
        state_end = x0.numel()
        system_q_end = state_end + self._system_quadrature_size
        return (
            integrated[..., :state_end],
            (
                integrated[..., state_end:system_q_end]
                if self._system_quadrature_size
                else None
            ),
            integrated[..., system_q_end:] if self._quadratures else None,
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
                state, system_q, _ = self._integrate(parameter_values, grid)
                state = state[-1]
                system_q = system_q[-1] if system_q is not None else None
            elif bool(time == 0):
                state, system_q, _ = self._integrate(
                    parameter_values, time.reshape(1)
                )
                state = state[0]
                system_q = system_q[0] if system_q is not None else None
            else:
                state, system_q, _ = self._integrate(
                    parameter_values,
                    torch.stack((torch.zeros_like(time), time)),
                )
                state = state[-1]
                system_q = system_q[-1] if system_q is not None else None
            return system.y(time, state, None, None, parameters, system_q)

        return output_native, self.problem.loss.input_spaces()[0]

    def _evaluate_trace_loss(self, values):
        states, _, registered_q = self._integrate(
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
                registered_q[-1] if registered_q is not None else (),
            )
        )[0]

    def _raw_guess(self, guesses):
        raw = guesses.clone()
        epsilon = torch.finfo(self._dtype).eps
        if self._has_two_sided_bounds:
            mask = self._two_sided_bounds
            ratio = (guesses[mask] - self._lower_bounds[mask]) / (
                self._upper_bounds[mask] - self._lower_bounds[mask]
            )
            raw[mask] = torch.logit(
                torch.clamp(ratio, min=epsilon, max=1 - epsilon)
            )
        if self._has_lower_only_bounds:
            mask = self._lower_only_bounds
            delta = torch.clamp(
                guesses[mask] - self._lower_bounds[mask], min=epsilon
            )
            raw[mask] = delta + torch.log(-torch.expm1(-delta))
        if self._has_upper_only_bounds:
            mask = self._upper_only_bounds
            delta = torch.clamp(
                self._upper_bounds[mask] - guesses[mask], min=epsilon
            )
            raw[mask] = delta + torch.log(-torch.expm1(-delta))
        return raw

    def _parameter_values(self, raw_values):
        values = raw_values.clone()
        values[self._bound_fixed_coordinates] = self._lower_bounds[
            self._bound_fixed_coordinates
        ]
        if self._has_two_sided_bounds:
            mask = self._two_sided_bounds
            values[mask] = self._lower_bounds[mask] + (
                self._upper_bounds[mask] - self._lower_bounds[mask]
            ) * torch.sigmoid(raw_values[mask])
        if self._has_lower_only_bounds:
            mask = self._lower_only_bounds
            values[mask] = self._lower_bounds[
                mask
            ] + torch.nn.functional.softplus(raw_values[mask])
        if self._has_upper_only_bounds:
            mask = self._upper_only_bounds
            values[mask] = self._upper_bounds[
                mask
            ] - torch.nn.functional.softplus(raw_values[mask])
        return values

    def solve(self, **fixed_parameters):
        self._check_fixed(fixed_parameters)
        fixed_indices = torch.as_tensor(
            [self._parameter_indices[name] for name in fixed_parameters],
            device=self._device,
            dtype=torch.long,
        )
        fixed_values = torch.as_tensor(
            [float(value) for value in fixed_parameters.values()],
            device=self._device,
            dtype=self._dtype,
        )
        free_coordinates = ~self._bound_fixed_coordinates.clone()
        if fixed_indices.numel():
            free_coordinates[fixed_indices] = False
        free_indices = torch.arange(
            len(self._names),
            device=self._device,
            dtype=torch.long,
        )[free_coordinates]
        raw = torch.nn.Parameter(
            self._raw_initial.index_select(0, free_indices).clone()
        )

        def values_from_raw():
            raw_values = self._raw_initial.index_copy(0, free_indices, raw)
            values = self._parameter_values(raw_values)
            return values.index_copy(0, fixed_indices, fixed_values)

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
            if free_indices.numel():
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
        states, system_q, registered_q = self._integrate(values, times)
        state_values = states.detach().cpu().numpy()
        quadrature_values = [
            quadrature.detach().cpu().numpy()
            for quadrature in (system_q, registered_q)
            if quadrature is not None
        ]
        all_quadrature_values = (
            np.concatenate(quadrature_values, axis=1)
            if quadrature_values
            else None
        )
        path_values = (
            np.concatenate((state_values, all_quadrature_values), axis=1)
            if all_quadrature_values is not None
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
        if all_quadrature_values is not None:
            quadrature_projector = np.zeros(
                (all_quadrature_values.shape[1], path_values.shape[1])
            )
            quadrature_projector[:, state_values.shape[1] :] = np.eye(
                all_quadrature_values.shape[1]
            )
        info = SolveInfo(
            "pytorch",
            self._options.optimiser_method,
            True,
            "converged",
            iteration_count=None,
        )
        system_parameter_values = self._system_parameters(values)
        public_parameters = (
            self.problem.parameter_layout.reconstruct(
                system_parameter_values,
                get_backend_by_name("pytorch", set_current=False),
            )
            if self.problem.parameter_layout is not None
            else {
                name: float(values[index].cpu())
                for index, name in enumerate(self._names)
            }
        )
        return VariationalSolution.from_solver(
            cost=float(cost.detach().cpu()),
            path=InterpolatingPolyCollection([poly]),
            projectors=(state_projector, None, quadrature_projector),
            control_solutions=[],
            parameters=public_parameters,
            parameter_vector=system_parameter_values.cpu().numpy(),
            solver_parameter_vector=values.cpu().numpy(),
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
