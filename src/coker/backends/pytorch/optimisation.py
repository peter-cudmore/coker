from __future__ import annotations

from dataclasses import dataclass
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
from coker.toolkits.codesign.optimisation import SolveFailure, SolveInfo


@dataclass(frozen=True)
class _PytorchOptimisationOptions:
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cuda")
    inner_iterations: int = 25
    restoration_iterations: int = 25
    barrier_stages: int = 8
    augmented_lagrangian_stages: int = 10
    barrier_reduction: float = 0.2
    penalty_growth: float = 10.0
    tolerance_grad: float = 1e-4
    tolerance_change: float = 1e-5
    tolerance_constraint: float = 1e-4
    interior_margin: float = 1e-4
    history_size: int = 10
    warm_start: bool = False


@dataclass(frozen=True)
class _Constraint:
    residual: Tracer
    lower_bound: object
    upper_bound: object


class _PytorchOptimisationProblem:
    def __init__(self, *, backend, tape, decision_bindings, parameter_bindings,
                 cost, constraints, outputs, initial_guess, options):
        self.backend = backend
        self.tape = tape
        self.decision_bindings = decision_bindings
        self.parameter_bindings = parameter_bindings
        self.cost = cost
        self.constraints = constraints
        self.outputs = outputs
        self.initial_guess = initial_guess
        self.options = options
        self._warm_start_decision = None
        self._warm_start_multipliers = None
        self.last_solve_info: SolveInfo | None = None

    def __call__(self, *runtime_args):
        runtime_args = self._normalise_runtime_args(runtime_args)
        decision = torch.as_tensor(self.initial_guess, dtype=self.options.dtype,
                                   device=self.options.device).clone().detach()
        if self.options.warm_start and self._warm_start_decision is not None:
            candidate = self._warm_start_decision
            if (candidate.numel() == decision.numel()
                    and candidate.device == decision.device
                    and candidate.dtype == decision.dtype):
                decision = candidate.clone().detach()
        if not self.decision_bindings:
            if bool(self._constraint_violation(decision, runtime_args) > self.options.tolerance_constraint):
                return self._fail("infeasible constraints", 0)
            self.last_solve_info = self._success_info(0)
            return self._results(decision, runtime_args)
        decision.requires_grad_(True)
        if not self.constraints:
            iterations = self._run_lbfgs(decision, lambda: self._evaluate_cost(decision, runtime_args))
            self._remember_warm_start(decision, None)
            self.last_solve_info = self._success_info(iterations)
            return self._results(decision, runtime_args)

        bounds = self._constraint_bounds(decision, runtime_args)
        decision = self._restore_feasibility(decision, runtime_args, bounds)
        decision.requires_grad_(True)

        iterations = 0
        try:
            equalities = [i for i, c in enumerate(self.constraints)
                          if self._bounds_equal(c.lower_bound, c.upper_bound)]
            inequalities = [i for i in range(len(self.constraints)) if i not in equalities]
            multipliers = [torch.zeros_like(bounds[i][0]) for i in equalities]
            if self.options.warm_start and self._warm_start_multipliers is not None:
                cached = self._warm_start_multipliers
                if len(cached) == len(multipliers) and all(
                    v.device == self.options.device and v.dtype == self.options.dtype
                    for v in cached
                ):
                    multipliers = [v.clone().detach() for v in cached]
            penalty = torch.tensor(1.0, dtype=self.options.dtype, device=self.options.device)
            barrier = torch.tensor(1.0, dtype=self.options.dtype, device=self.options.device)
            objective_start = self._evaluate_cost(decision, runtime_args)
            if not torch.isfinite(objective_start):
                raise FloatingPointError("non-finite objective")

            # Equality constraints are handled by an augmented Lagrangian.  The
            # barrier is kept separate so equality residuals never enter a log.
            for _ in range(self.options.augmented_lagrangian_stages if equalities else 0):
                def equality_objective():
                    value = self._evaluate_cost(decision, runtime_args)
                    for j, i in enumerate(equalities):
                        residual = self._constraint_values(decision, runtime_args)[i]
                        target = bounds[i][0]
                        error = residual - target
                        value = value + (multipliers[j] * error).sum() + 0.5 * penalty * (error * error).sum()
                    return value
                n = self._run_lbfgs(decision, equality_objective)
                iterations += n
                decision = self._restore_feasibility(decision, runtime_args, bounds)
                decision.requires_grad_(True)
                bounds = self._constraint_bounds(decision, runtime_args)
                values = self._constraint_values(decision, runtime_args)
                for j, i in enumerate(equalities):
                    error = values[i] - bounds[i][0]
                    multipliers[j] = multipliers[j] + penalty * error.detach()
                max_error = torch.stack(
                    [torch.max(torch.abs(values[i] - bounds[i][0])) for i in equalities]
                ).max()
                if bool(max_error <= self.options.tolerance_constraint):
                    break
                penalty = penalty * self.options.penalty_growth
            for _ in range(self.options.barrier_stages if inequalities else 0):
                def barrier_objective():
                    value = self._evaluate_cost(decision, runtime_args)
                    vals = self._constraint_values(decision, runtime_args)
                    for j, i in enumerate(equalities):
                        error = vals[i] - bounds[i][0]
                        value = value + (multipliers[j] * error).sum() + 0.5 * penalty * (error * error).sum()
                    for i in inequalities:
                        lower, upper = bounds[i]
                        if lower is not None:
                            slack = vals[i] - lower
                            safe_slack = torch.clamp(
                                slack, min=self.options.interior_margin
                            )
                            value = value - barrier * torch.log(safe_slack).sum()
                            value = value + 1e4 * torch.relu(
                                self.options.interior_margin - slack
                            ).square().sum()
                        if upper is not None:
                            slack = upper - vals[i]
                            safe_slack = torch.clamp(
                                slack, min=self.options.interior_margin
                            )
                            value = value - barrier * torch.log(safe_slack).sum()
                            value = value + 1e4 * torch.relu(
                                self.options.interior_margin - slack
                            ).square().sum()
                    return torch.nan_to_num(
                        value, nan=1e20, posinf=1e20, neginf=-1e20
                    ) + decision.sum() * 0
                n = self._run_lbfgs(decision, barrier_objective)
                iterations += n
                if self._constraint_violation(decision, runtime_args) <= self.options.tolerance_constraint:
                    pass
                barrier = barrier * self.options.barrier_reduction

            final = self._evaluate_cost(decision, runtime_args)
            if not torch.isfinite(final) or self._constraint_violation(decision, runtime_args) > self.options.tolerance_constraint:
                raise FloatingPointError("non-finite objective or constraint violation")
            relative = torch.abs(final - objective_start) / torch.clamp(torch.abs(objective_start), min=1.0)
            if not torch.isfinite(relative):
                raise FloatingPointError("non-finite objective change")
            if final.requires_grad:
                final.backward()
                if decision.grad is None or not bool(torch.isfinite(decision.grad).all()):
                    raise FloatingPointError("non-finite final gradient")
            self._remember_warm_start(decision, multipliers)
            self.last_solve_info = self._success_info(iterations)
            return self._results(decision, runtime_args)
        except Exception as ex:
            return self._fail(str(ex), iterations, ex)
    def _run_lbfgs(self, decision, objective, *, max_iter=None):
        max_iter = self.options.inner_iterations if max_iter is None else max_iter
        optimizer = torch.optim.LBFGS(
            [decision], max_iter=max_iter,
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
            if decision.grad is None or not torch.isfinite(decision.grad).all():
                raise FloatingPointError("non-finite gradient")
            state["iterations"] += 1
            return loss
        optimizer.step(closure)
        return state["iterations"]
    def _restore_feasibility(self, decision, runtime_args, bounds):
        def restoration():
            values = self._constraint_values(decision, runtime_args)
            loss = torch.zeros((), dtype=self.options.dtype, device=self.options.device)
            for i, value in enumerate(values):
                lower, upper = bounds[i]
                if self._bounds_equal(self.constraints[i].lower_bound, self.constraints[i].upper_bound):
                    loss = loss + (value - lower).square().sum()
                    continue
                if lower is not None:
                    loss = loss + torch.relu(
                        lower + 2 * self.options.interior_margin - value
                    ).square().sum()
                if upper is not None:
                    loss = loss + torch.relu(
                        value - upper + 2 * self.options.interior_margin
                    ).square().sum()
            return loss
        self._run_lbfgs(decision, restoration, max_iter=self.options.restoration_iterations)
        return decision.detach()


    def _constraint_values(self, decision, runtime_args):
        return [
            torch.as_tensor(value, dtype=self.options.dtype, device=self.options.device).reshape(-1)
            for value in self._evaluate_tracers(
                [constraint.residual for constraint in self.constraints],
                decision,
                runtime_args,
            )
        ]

    def _constraint_bounds(self, decision, runtime_args):
        result = []
        for c in self.constraints:
            result.append((self._bound(c.lower_bound, decision, runtime_args),
                           self._bound(c.upper_bound, decision, runtime_args)))
        return result

    def _bound(self, bound, decision, runtime_args):
        if bound is None:
            return None
        if isinstance(bound, Tracer):
            value = self._evaluate_tracers([bound], decision, runtime_args)[0]
        else:
            value = bound
        result = torch.as_tensor(
            value, dtype=self.options.dtype, device=self.options.device
        ).reshape(-1)
        if bool(torch.all(torch.isinf(result))):
            return None
        return result

    def _strictly_feasible(self, decision, runtime_args, bounds):
        for i, (value, (lower, upper)) in enumerate(zip(self._constraint_values(decision, runtime_args), bounds)):
            if self._bounds_equal(self.constraints[i].lower_bound, self.constraints[i].upper_bound):
                continue
            if lower is not None and not bool(torch.all(value > lower + self.options.interior_margin)):
                return False
            if upper is not None and not bool(torch.all(value < upper - self.options.interior_margin)):
                return False
        return True
    def _constraint_violation(self, decision, runtime_args):
        vals = self._constraint_values(decision, runtime_args)
        bounds = self._constraint_bounds(decision, runtime_args)
        result = torch.zeros((), dtype=self.options.dtype, device=self.options.device)
        for i, (value, (lower, upper)) in enumerate(zip(vals, bounds)):
            if self._bounds_equal(self.constraints[i].lower_bound, self.constraints[i].upper_bound):
                result = torch.maximum(result, torch.max(torch.abs(value - lower)))
                continue
            if lower is not None:
                result = torch.maximum(result, torch.max(torch.relu(lower - value)))
            if upper is not None:
                result = torch.maximum(result, torch.max(torch.relu(value - upper)))
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
                raise ValueError("PyTorch optimisation runtime tensors must be on CUDA")
            if len({v.device for v in tensors}) != 1:
                raise ValueError("PyTorch optimisation runtime tensors must use one CUDA device")
            if len(runtime_args) != len(self.parameter_bindings):
                raise ValueError(
                    f"Expected {len(self.parameter_bindings)} runtime arguments, "
                    f"got {len(runtime_args)}"
                )
            values = tuple(
                self.backend.reshape(value, binding.dim)
                if isinstance(value, torch.Tensor)
                else self.backend.reshape(
                    torch.as_tensor(value, device=self.options.device), binding.dim
                )
                for value, binding in zip(runtime_args, self.parameter_bindings)
            )
        else:
            values = normalise_runtime_args(runtime_args, self.parameter_bindings)
        return tuple(
            torch.as_tensor(value, dtype=self.options.dtype, device=self.options.device)
            for value in values
        )

    def _materialise_inputs(self, decision, runtime_args):
        decisions = {b.index: self.backend.reshape(decision[b.start:b.stop], b.dim) for b in self.decision_bindings}
        parameters = {b.index: v for b, v in zip(self.parameter_bindings, runtime_args)}
        return [
            decisions[i] if i in decisions else parameters[i]
            for i in self.tape.input_indicies
        ]

    def _evaluate_tracers(self, tracers, decision, runtime_args):
        inputs = self._materialise_inputs(decision, runtime_args)
        workspace = {-1: None}
        for index, value in zip(self.tape.input_indicies, inputs):
            workspace[index] = self.backend.to_backend_array(value)
        for index in range(len(self.tape.nodes)):
            if index in workspace:
                continue
            op, *nodes = self.tape.nodes[index]
            args = [
                workspace[node.index]
                if isinstance(node, Tracer) and node.tape == self.tape
                else self.backend.to_backend_array(node)
                for node in nodes
            ]
            value = (
                args[0]
                if op in {OP.VALUE, OP.FUNCTION_VALUE}
                else self.backend.call(op, *args)
            )
            value = _normalize_evaluate_result(op, args, value, self.tape.dim[index])
            workspace[index] = (
                value
                if isinstance(value, Tracer)
                or isinstance(self.tape.dim[index], ResultBundleDimension)
                else self.backend.reshape(value, self.tape.dim[index])
            )
        return [
            None if tracer is None else workspace[tracer.index]
            for tracer in tracers
        ]

    def _evaluate_cost(self, decision, runtime_args):
        return torch.as_tensor(self._evaluate_tracers([self.cost], decision, runtime_args)[0], dtype=self.options.dtype, device=self.options.device).reshape(())

    def _results(self, decision, runtime_args):
        return [
            self.backend.to_numpy_array(v)
            for v in self._evaluate_tracers(self.outputs, decision, runtime_args)
        ]

    def _remember_warm_start(self, decision, multipliers):
        if not self.options.warm_start:
            return
        self._warm_start_decision = decision.detach().clone()
        self._warm_start_multipliers = (
            None if multipliers is None
            else [value.detach().clone() for value in multipliers]
        )


    @staticmethod
    def _success_info(iterations):
        return SolveInfo("pytorch", "LBFGS", True, "success", None, iterations)

    def _fail(self, status, iterations, cause=None):
        self.last_solve_info = SolveInfo("pytorch", "LBFGS", False, status, None, iterations)
        failure = SolveFailure(f"PyTorch optimisation solve failed: {status}", self.last_solve_info)
        if cause is not None: raise failure from cause
        raise failure


def build_optimisation_problem(
    backend, cost, constraints, parameters, outputs, initial_conditions,
    *, options=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch optimisation requires CUDA availability")
    tape = cost.tape
    if any(item.tape != tape for item in (*constraints, *parameters, *outputs)):
        raise ValueError("All optimisation expressions must belong to the cost tape")
    source = options or _PytorchOptimisationOptions()
    selected = _PytorchOptimisationOptions(
        dtype=source.dtype,
        device=torch.device(source.device),
        inner_iterations=source.inner_iterations,
        restoration_iterations=source.restoration_iterations,
        barrier_stages=source.barrier_stages,
        augmented_lagrangian_stages=source.augmented_lagrangian_stages,
        barrier_reduction=source.barrier_reduction,
        penalty_growth=source.penalty_growth,
        tolerance_grad=source.tolerance_grad,
        tolerance_change=source.tolerance_change,
        tolerance_constraint=source.tolerance_constraint,
        interior_margin=source.interior_margin,
        history_size=source.history_size,
        warm_start=source.warm_start,
    )
    if selected.device.type != "cuda" or selected.dtype is not torch.float32:
        raise ValueError("PyTorch optimisation requires CUDA float32 options")
    bindings = build_problem_bindings(tape, [p.index for p in parameters])
    normalised = []
    for constraint in constraints:
        residual, lower, upper = constraint.as_halfplane_bound()
        normalised.append(_Constraint(residual, lower, upper))
    return _PytorchOptimisationProblem(
        backend=backend, tape=tape, decision_bindings=bindings.decision_bindings,
        parameter_bindings=bindings.parameter_bindings, cost=cost, constraints=normalised,
        outputs=outputs, initial_guess=build_initial_guess(bindings.decision_bindings, initial_conditions),
        options=selected,
    )
