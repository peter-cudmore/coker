from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from coker.algebra.dimensions import ResultBundleDimension
from coker.algebra.ops import OP
from coker.algebra.graph import Tracer
from coker.backends.evaluator import _normalize_evaluate_result
from coker.backends.optimisation import (
    build_initial_guess,
    build_problem_bindings,
    normalise_runtime_args,
)
from coker.toolkits.codesign import (
    BoundedConstraint,
    SolveFailure,
    SolveInfo,
    SolverOptions,
)
from .ops import call_parameterised_op, impls, parameterised_impls


def _reshape(value, dimension):
    if dimension.is_scalar():
        return value.reshape(())
    return torch.reshape(value, dimension.dim)


def _to_backend_array(value, device):
    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device)
    return torch.as_tensor(value, device=device)


def _to_numpy_array(value):
    return value.detach().cpu().numpy()


@dataclass(frozen=True)
class PytorchNLPSolverOptions(SolverOptions):
    """Settings for the CUDA float32 PyTorch LBFGS NLP solver.

    ``inner_iterations`` and ``restoration_iterations`` bound the LBFGS
    iterations for the objective solve and feasibility-restoration solve.
    ``barrier_stages`` and ``augmented_lagrangian_stages`` control the number
    of outer penalty/barrier updates. ``barrier_reduction`` decreases the
    barrier coefficient after each barrier stage, while ``penalty_growth``
    increases the equality-constraint penalty. The three tolerance fields
    configure LBFGS stopping and constraint feasibility, and
    ``interior_margin`` keeps inequality iterates away from their bounds.
    ``history_size`` is passed to :class:`torch.optim.LBFGS` as its curvature
    history limit.

    Device and dtype are properties of :class:`PytorchBackend`, rather than
    solver options, and are validated when an optimisation problem is built.
    """

    inner_iterations: int = field(
        default=25,
        metadata={"doc": "Maximum LBFGS iterations for each objective solve."},
    )
    restoration_iterations: int = field(
        default=25,
        metadata={
            "doc": "Maximum LBFGS iterations for feasibility restoration."
        },
    )
    barrier_stages: int = field(
        default=8,
        metadata={"doc": "Number of inequality barrier stages."},
    )
    augmented_lagrangian_stages: int = field(
        default=10,
        metadata={"doc": "Number of equality augmented-Lagrangian stages."},
    )
    barrier_reduction: float = field(
        default=0.2,
        metadata={"doc": "Multiplicative barrier coefficient reduction."},
    )
    penalty_growth: float = field(
        default=10.0,
        metadata={"doc": "Multiplicative equality penalty growth."},
    )
    tolerance_grad: float = field(
        default=1e-4,
        metadata={"doc": "LBFGS gradient convergence tolerance."},
    )
    tolerance_change: float = field(
        default=1e-5,
        metadata={"doc": "LBFGS objective/parameter change tolerance."},
    )
    tolerance_constraint: float = field(
        default=1e-4,
        metadata={"doc": "Maximum accepted constraint violation."},
    )
    interior_margin: float = field(
        default=1e-4,
        metadata={"doc": "Minimum inequality interior slack."},
    )
    history_size: int = field(
        default=10,
        metadata={"doc": "LBFGS curvature history size."},
    )

    def __post_init__(self):
        super().__post_init__()
        if (
            not isinstance(self.inner_iterations, int)
            or self.inner_iterations <= 0
        ):
            raise ValueError("inner_iterations must be a positive integer")
        if (
            not isinstance(self.restoration_iterations, int)
            or self.restoration_iterations <= 0
        ):
            raise ValueError(
                "restoration_iterations must be a positive integer"
            )
        if (
            not isinstance(self.barrier_stages, int)
            or self.barrier_stages <= 0
        ):
            raise ValueError("barrier_stages must be a positive integer")
        if (
            not isinstance(self.augmented_lagrangian_stages, int)
            or self.augmented_lagrangian_stages <= 0
        ):
            raise ValueError(
                "augmented_lagrangian_stages must be a positive integer"
            )
        if not isinstance(self.history_size, int) or self.history_size <= 0:
            raise ValueError("history_size must be a positive integer")
        if (
            not isinstance(self.barrier_reduction, (int, float))
            or self.barrier_reduction <= 0
        ):
            raise ValueError("barrier_reduction must be positive")
        if (
            not isinstance(self.penalty_growth, (int, float))
            or self.penalty_growth <= 0
        ):
            raise ValueError("penalty_growth must be positive")
        if (
            not isinstance(self.tolerance_grad, (int, float))
            or self.tolerance_grad <= 0
        ):
            raise ValueError("tolerance_grad must be positive")
        if (
            not isinstance(self.tolerance_change, (int, float))
            or self.tolerance_change <= 0
        ):
            raise ValueError("tolerance_change must be positive")
        if (
            not isinstance(self.tolerance_constraint, (int, float))
            or self.tolerance_constraint <= 0
        ):
            raise ValueError("tolerance_constraint must be positive")
        if (
            not isinstance(self.interior_margin, (int, float))
            or self.interior_margin <= 0
        ):
            raise ValueError("interior_margin must be positive")
        if self.barrier_reduction >= 1:
            raise ValueError("barrier_reduction must be less than 1")


class _PytorchOptimisationProblem:
    def __init__(
        self,
        *,
        tape,
        decision_bindings,
        parameter_bindings,
        cost,
        constraints,
        outputs,
        initial_guess,
        options,
        device,
        dtype,
    ):
        self.tape = tape
        self.decision_bindings = decision_bindings
        self.parameter_bindings = parameter_bindings
        self.cost = cost
        self.constraints = constraints
        self.outputs = outputs
        self.initial_guess = initial_guess
        self.options = options
        self.device = device
        self.dtype = dtype
        self._warm_start_decision = None
        self._warm_start_multipliers = None
        self.last_solve_info: SolveInfo | None = None

    def __call__(self, *runtime_args):
        runtime_args = self._normalise_runtime_args(runtime_args)
        decision = (
            torch.as_tensor(
                self.initial_guess,
                dtype=self.dtype,
                device=self.device,
            )
            .clone()
            .detach()
        )
        if self.options.warm_start and self._warm_start_decision is not None:
            candidate = self._warm_start_decision
            if (
                candidate.numel() == decision.numel()
                and candidate.device == decision.device
                and candidate.dtype == decision.dtype
            ):
                decision = candidate.clone().detach()
        if not self.decision_bindings:
            if bool(
                self._constraint_violation(decision, runtime_args)
                > self.options.tolerance_constraint
            ):
                return self._fail("infeasible constraints", 0)
            self.last_solve_info = self._success_info(0)
            return self._results(decision, runtime_args)
        decision.requires_grad_(True)
        if not self.constraints:
            iterations = self._run_lbfgs(
                decision, lambda: self._evaluate_cost(decision, runtime_args)
            )
            self._remember_warm_start(decision, None)
            self.last_solve_info = self._success_info(iterations)
            return self._results(decision, runtime_args)

        bounds = self._constraint_bounds(decision, runtime_args)
        decision = self._restore_feasibility(decision, runtime_args, bounds)
        decision.requires_grad_(True)

        iterations = 0
        try:
            equalities = [
                i
                for i, c in enumerate(self.constraints)
                if self._bounds_equal(c.lower_bound, c.upper_bound)
            ]
            inequalities = [
                i for i in range(len(self.constraints)) if i not in equalities
            ]
            multipliers = [torch.zeros_like(bounds[i][0]) for i in equalities]
            if (
                self.options.warm_start
                and self._warm_start_multipliers is not None
            ):
                cached = self._warm_start_multipliers
                if len(cached) == len(multipliers) and all(
                    v.device == self.device and v.dtype == self.dtype
                    for v in cached
                ):
                    multipliers = [v.clone().detach() for v in cached]
            penalty = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            barrier = torch.tensor(1.0, dtype=self.dtype, device=self.device)

            # Equality constraints are handled by an augmented Lagrangian.  The
            # barrier is kept separate so equality residuals never enter a log.
            for _ in range(
                self.options.augmented_lagrangian_stages if equalities else 0
            ):

                def equality_objective():
                    values = self._constraint_values(decision, runtime_args)
                    return self._with_equality_penalty(
                        self._evaluate_cost(decision, runtime_args),
                        values,
                        bounds,
                        equalities,
                        multipliers,
                        penalty,
                    )

                n = self._run_lbfgs(decision, equality_objective)
                iterations += n
                decision = self._restore_feasibility(
                    decision, runtime_args, bounds
                )
                decision.requires_grad_(True)
                bounds = self._constraint_bounds(decision, runtime_args)
                values = self._constraint_values(decision, runtime_args)
                for j, i in enumerate(equalities):
                    error = values[i] - bounds[i][0]
                    multipliers[j] = multipliers[j] + penalty * error.detach()
                max_error = torch.stack(
                    [
                        torch.max(torch.abs(values[i] - bounds[i][0]))
                        for i in equalities
                    ]
                ).max()
                if bool(max_error <= self.options.tolerance_constraint):
                    break
                penalty = penalty * self.options.penalty_growth
            for _ in range(self.options.barrier_stages if inequalities else 0):

                def barrier_objective():
                    vals = self._constraint_values(decision, runtime_args)
                    value = self._with_equality_penalty(
                        self._evaluate_cost(decision, runtime_args),
                        vals,
                        bounds,
                        equalities,
                        multipliers,
                        penalty,
                    )
                    for index in inequalities:
                        lower, upper = bounds[index]
                        if lower is not None:
                            slack = vals[index] - lower
                            safe_slack = torch.clamp(
                                slack, min=self.options.interior_margin
                            )
                            value = (
                                value - barrier * torch.log(safe_slack).sum()
                            )
                            value = (
                                value
                                + 1e4
                                * torch.relu(
                                    self.options.interior_margin - slack
                                )
                                .square()
                                .sum()
                            )
                        if upper is not None:
                            slack = upper - vals[index]
                            safe_slack = torch.clamp(
                                slack, min=self.options.interior_margin
                            )
                            value = (
                                value - barrier * torch.log(safe_slack).sum()
                            )
                            value = (
                                value
                                + 1e4
                                * torch.relu(
                                    self.options.interior_margin - slack
                                )
                                .square()
                                .sum()
                            )
                    return (
                        torch.nan_to_num(
                            value, nan=1e20, posinf=1e20, neginf=-1e20
                        )
                        + decision.sum() * 0
                    )

                n = self._run_lbfgs(decision, barrier_objective)
                iterations += n
                barrier = barrier * self.options.barrier_reduction

            final = self._evaluate_cost(decision, runtime_args)
            if (
                not torch.isfinite(final)
                or self._constraint_violation(decision, runtime_args)
                > self.options.tolerance_constraint
            ):
                raise FloatingPointError(
                    "non-finite objective or constraint violation"
                )
            if final.requires_grad:
                final.backward()
                if decision.grad is None or not bool(
                    torch.isfinite(decision.grad).all()
                ):
                    raise FloatingPointError("non-finite final gradient")
            self._remember_warm_start(decision, multipliers)
            self.last_solve_info = self._success_info(iterations)
            return self._results(decision, runtime_args)
        except Exception as ex:
            return self._fail(str(ex), iterations, ex)

    def _run_lbfgs(self, decision, objective, *, max_iter=None):
        max_iter = (
            self.options.inner_iterations if max_iter is None else max_iter
        )
        optimizer = torch.optim.LBFGS(
            [decision],
            max_iter=max_iter,
            tolerance_grad=self.options.tolerance_grad,
            tolerance_change=self.options.tolerance_change,
            history_size=self.options.history_size,
            line_search_fn="strong_wolfe",
        )
        state = {"iterations": 0}

        def closure():
            optimizer.zero_grad()
            loss = objective()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite objective")
            if not loss.requires_grad:
                loss = loss + decision.sum() * 0.0
            loss.backward()
            if (
                decision.grad is None
                or not torch.isfinite(decision.grad).all()
            ):
                raise FloatingPointError("non-finite gradient")
            state["iterations"] += 1
            return loss

        optimizer.step(closure)
        return state["iterations"]

    def _restore_feasibility(self, decision, runtime_args, bounds):
        def restoration():
            values = self._constraint_values(decision, runtime_args)
            loss = torch.zeros((), dtype=self.dtype, device=self.device)
            for i, value in enumerate(values):
                lower, upper = bounds[i]
                if self._bounds_equal(
                    self.constraints[i].lower_bound,
                    self.constraints[i].upper_bound,
                ):
                    loss = loss + (value - lower).square().sum()
                    continue
                if lower is not None:
                    loss = (
                        loss
                        + torch.relu(
                            lower + 2 * self.options.interior_margin - value
                        )
                        .square()
                        .sum()
                    )
                if upper is not None:
                    loss = (
                        loss
                        + torch.relu(
                            value - upper + 2 * self.options.interior_margin
                        )
                        .square()
                        .sum()
                    )
            return loss

        self._run_lbfgs(
            decision, restoration, max_iter=self.options.restoration_iterations
        )
        return decision.detach()

    def _constraint_values(self, decision, runtime_args):
        return [
            torch.as_tensor(
                value, dtype=self.dtype, device=self.device
            ).reshape(-1)
            for value in self._evaluate_tracers(
                [constraint.residual for constraint in self.constraints],
                decision,
                runtime_args,
            )
        ]

    def _constraint_bounds(self, decision, runtime_args):
        result = []
        for c in self.constraints:
            result.append(
                (
                    self._bound(c.lower_bound, decision, runtime_args),
                    self._bound(c.upper_bound, decision, runtime_args),
                )
            )
        return result

    def _bound(self, bound, decision, runtime_args):
        if bound is None:
            return None
        if isinstance(bound, Tracer):
            value = self._evaluate_tracers([bound], decision, runtime_args)[0]
        else:
            value = bound
        result = torch.as_tensor(
            value, dtype=self.dtype, device=self.device
        ).reshape(-1)
        if bool(torch.all(torch.isinf(result))):
            return None
        return result

    @staticmethod
    def _with_equality_penalty(
        value, values, bounds, equalities, multipliers, penalty
    ):
        for multiplier, index in zip(multipliers, equalities):
            error = values[index] - bounds[index][0]
            value = value + (multiplier * error).sum()
            value = value + 0.5 * penalty * error.square().sum()
        return value

    def _constraint_violation(self, decision, runtime_args):
        vals = self._constraint_values(decision, runtime_args)
        bounds = self._constraint_bounds(decision, runtime_args)
        result = torch.zeros((), dtype=self.dtype, device=self.device)
        for i, (value, (lower, upper)) in enumerate(zip(vals, bounds)):
            if self._bounds_equal(
                self.constraints[i].lower_bound,
                self.constraints[i].upper_bound,
            ):
                result = torch.maximum(
                    result, torch.max(torch.abs(value - lower))
                )
                continue
            if lower is not None:
                result = torch.maximum(
                    result, torch.max(torch.relu(lower - value))
                )
            if upper is not None:
                result = torch.maximum(
                    result, torch.max(torch.relu(value - upper))
                )
        return result

    @staticmethod
    def _bounds_equal(lower, upper):
        if lower is upper:
            return True
        if isinstance(lower, Tracer) or isinstance(upper, Tracer):
            return False
        try:
            difference = torch.as_tensor(lower) - torch.as_tensor(upper)
            return bool(torch.all(torch.abs(difference) <= 1e-6))
        except (TypeError, ValueError):
            return False

    def _normalise_runtime_args(self, runtime_args: Sequence[object]):
        tensors = [v for v in runtime_args if isinstance(v, torch.Tensor)]
        if tensors:
            if any(v.device.type != "cuda" for v in tensors):
                raise ValueError(
                    "PyTorch optimisation runtime tensors must be on CUDA"
                )
            if len({v.device for v in tensors}) != 1:
                raise ValueError(
                    "PyTorch optimisation runtime tensors must use one CUDA "
                    "device"
                )
            if len(runtime_args) != len(self.parameter_bindings):
                raise ValueError(
                    "Expected "
                    f"{len(self.parameter_bindings)} runtime arguments, "
                    f"got {len(runtime_args)}"
                )
            values = tuple(
                _reshape(
                    _to_backend_array(value, self.device),
                    binding.dim,
                )
                for value, binding in zip(
                    runtime_args, self.parameter_bindings
                )
            )
        else:
            values = normalise_runtime_args(
                runtime_args, self.parameter_bindings
            )
        return tuple(
            torch.as_tensor(value, dtype=self.dtype, device=self.device)
            for value in values
        )

    def _materialise_inputs(self, decision, runtime_args):
        decisions = {
            b.index: _reshape(decision[b.start : b.stop], b.dim)
            for b in self.decision_bindings
        }
        parameters = {
            b.index: v for b, v in zip(self.parameter_bindings, runtime_args)
        }
        return [
            decisions[i] if i in decisions else parameters[i]
            for i in self.tape.input_indicies
        ]

    def _evaluate_tracers(self, tracers, decision, runtime_args):
        inputs = self._materialise_inputs(decision, runtime_args)
        workspace = {-1: None}
        for index, value in zip(self.tape.input_indicies, inputs):
            workspace[index] = _to_backend_array(value, self.device)
        for index in range(len(self.tape.nodes)):
            if index in workspace:
                continue
            op, *nodes = self.tape.nodes[index]
            args = [
                (
                    workspace[node.index]
                    if isinstance(node, Tracer) and node.tape == self.tape
                    else _to_backend_array(node, self.device)
                )
                for node in nodes
            ]
            if op in {OP.VALUE, OP.FUNCTION_VALUE}:
                value = args[0]
            elif op in impls:
                value = impls[op](*args)
            elif op in parameterised_impls:
                value = call_parameterised_op(op, *args)
            else:
                raise NotImplementedError(f"{op} is not implemented")
            value = _normalize_evaluate_result(
                op, args, value, self.tape.dim[index]
            )
            workspace[index] = (
                value
                if isinstance(value, Tracer)
                or isinstance(self.tape.dim[index], ResultBundleDimension)
                else _reshape(value, self.tape.dim[index])
            )
        return [
            None if tracer is None else workspace[tracer.index]
            for tracer in tracers
        ]

    def _evaluate_cost(self, decision, runtime_args):
        return torch.as_tensor(
            self._evaluate_tracers([self.cost], decision, runtime_args)[0],
            dtype=self.dtype,
            device=self.device,
        ).reshape(())

    def _results(self, decision, runtime_args):
        return [
            _to_numpy_array(value)
            for value in self._evaluate_tracers(
                self.outputs, decision, runtime_args
            )
        ]

    def _remember_warm_start(self, decision, multipliers):
        if not self.options.warm_start:
            return
        self._warm_start_decision = decision.detach().clone()
        self._warm_start_multipliers = (
            None
            if multipliers is None
            else [value.detach().clone() for value in multipliers]
        )

    @staticmethod
    def _success_info(iterations):
        return SolveInfo("pytorch", "LBFGS", True, "success", None, iterations)

    def _fail(self, status, iterations, cause=None):
        self.last_solve_info = SolveInfo(
            "pytorch", "LBFGS", False, status, None, iterations
        )
        failure = SolveFailure(
            f"PyTorch optimisation solve failed: {status}",
            self.last_solve_info,
        )
        if cause is not None:
            raise failure from cause
        raise failure


def build_optimisation_problem(
    backend,
    cost,
    constraints,
    parameters,
    outputs,
    initial_conditions,
    *,
    options=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch optimisation requires CUDA availability")
    device = backend.device or torch.device("cuda")
    dtype = backend.dtype or torch.float32
    if device.type != "cuda":
        raise ValueError("PyTorch NLP solving requires a CUDA backend device")
    if dtype is not torch.float32:
        raise ValueError(
            "PyTorch NLP solving requires a float32 backend dtype"
        )
    tape = cost.tape
    if any(
        item.tape != tape for item in (*constraints, *parameters, *outputs)
    ):
        raise ValueError(
            "All optimisation expressions must belong to the cost tape"
        )
    if options is None:
        selected = PytorchNLPSolverOptions()
    elif isinstance(options, PytorchNLPSolverOptions):
        selected = options
    else:
        raise TypeError(
            "PyTorch NLP solving requires PytorchNLPSolverOptions; "
            f"got {type(options).__name__}"
        )
    bindings = build_problem_bindings(tape, [p.index for p in parameters])
    normalised = []
    for constraint in constraints:
        residual, lower, upper = constraint.as_halfplane_bound()
        normalised.append(BoundedConstraint(residual, lower, upper))
    return _PytorchOptimisationProblem(
        tape=tape,
        decision_bindings=bindings.decision_bindings,
        parameter_bindings=bindings.parameter_bindings,
        cost=cost,
        constraints=normalised,
        outputs=outputs,
        initial_guess=build_initial_guess(
            bindings.decision_bindings, initial_conditions
        ),
        options=selected,
        device=device,
        dtype=dtype,
    )
