from typing import List
from enum import Enum

import numpy as np
import scipy.sparse.csc
import scipy as scp

from coker.algebra import Dimension
from coker.algebra.function import Function, create_function_from_native
from coker.algebra.graph import Tracer
from coker.algebra.ops import Noop

from coker.backends.evaluator import GenericEvaluator
from coker.backends.backend import (
    ArrayLike,
    Backend,
    SolverParameters,
    register_backend,
)
from coker.backends.lowered import FunctionSignature, LoweringOptions

from coker.backends.numpy.lowered import NumpyLoweredFunction
from coker.backends.numpy.evaluator import (
    call_parameterised_op,
    impls,
    parameterised_impls,
)
from coker.backends.numpy.optimisation import build_optimisation_problem


class Solver(Enum):
    RK45 = "RK45"
    LSODA = "LSODA"
    Radau = "Radau"
    BDF = "BDF"


class NumpySolverParameters(SolverParameters):
    def __init__(self, solver: Solver = Solver.RK45):
        self.solver = solver


scalar_types = (
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    float,
    complex,
    int,
    bool,
    np.bool_,
)


class NumpyBackend(Backend):
    def to_numpy_array(self, array) -> ArrayLike:
        return array

    def to_backend_array(self, array: ArrayLike):
        return array

    def reshape(self, arg, dim: Dimension):
        if dim.is_scalar():
            if isinstance(arg, scalar_types):
                return arg
            try:
                (inner,) = arg
            except (ValueError, TypeError) as ex:
                raise TypeError(f"Expecting a scalar, got {arg}") from ex
            return self.reshape(inner, dim)
        if isinstance(arg, np.ndarray):
            return np.reshape(arg, dim.dim)
        if scp.sparse.issparse(arg):
            return np.reshape(arg.toarray(), dim.dim)
        if isinstance(arg, (float, int)):
            return np.array([arg]).reshape(dim.dim)
        if arg is None:
            return arg
        raise NotImplementedError(f"Dont know how to reshape {arg}")

    def lower(
        self, function: Function, options: LoweringOptions | None = None
    ) -> NumpyLoweredFunction:
        return NumpyLoweredFunction(
            self,
            function,
            self.get_evaluator().build_plan(function.tape),
        )

    def get_evaluator(self) -> GenericEvaluator:
        return GenericEvaluator(
            self,
            operations=impls,
            parameterised_operations=parameterised_impls,
        )

    def import_function(
        self,
        implementation,
        signature: FunctionSignature,
        *,
        name: str | None = None,
    ) -> Function:
        """Import a NumPy-compatible callable as a Coker function."""
        return create_function_from_native(
            implementation, signature, backend=self.name, name=name
        )

    def call(self, op, *args) -> ArrayLike:

        if op in impls:
            return impls[op](*args)

        if isinstance(op, tuple(parameterised_impls.keys())):
            return call_parameterised_op(op, *args)

        raise NotImplementedError(f"{op} is not implemented")

    def evaluate_integrals(
        self,
        functions,
        initial_conditions,
        end_point: float,
        inputs,
        solver_parameters=None,
    ):

        dxdt, constraint, dqdt = functions
        x0, z0, q0 = initial_conditions
        u, p = inputs

        if constraint is not Noop():
            raise NotImplementedError(
                "Integrators with constraints are not implemented"
            )

        if not isinstance(x0, np.ndarray):
            x0 = np.array([x0])

        if isinstance(end_point, (float, int)):
            if end_point == 0.0:
                return x0, z0, q0
            else:
                t_eval = [end_point]
                t_span = (0, end_point)
        else:
            t_eval = end_point
            t_span = (0, end_point[-1])

        if dqdt is Noop():
            y0 = x0

            def f(t, x):
                return dxdt(t, x, None, u, p)

        else:
            y0 = (np.concatenate([x0, q0]),)

            def f(t, x):
                return np.concatenate(
                    [dxdt(t, x, None, u, p), dqdt(t, x, None, u, p)]
                )

        if isinstance(solver_parameters, NumpySolverParameters):
            method = solver_parameters.solver.value
        else:
            method = Solver.RK45.value

        sol = scp.integrate.solve_ivp(
            f, t_span, y0, method=method, t_eval=t_eval
        )

        x_out = (
            sol.y[: x0.shape[0], -1]
            if not isinstance(end_point, np.ndarray)
            else sol.y[: x0.shape[0], :]
        )

        if dqdt is None:
            q_out = None
        else:
            q_out = (
                sol.y[x0.shape[0] :, -1]
                if not isinstance(end_point, np.ndarray)
                else sol.y[x0.shape[0] :, :]
            )

        return x_out, None, q_out

    def build_optimisation_problem(
        self,
        cost: Tracer,
        constraints: List[Tracer],
        arguments: List[Tracer],
        outputs: List[Tracer],
        initial_conditions,
    ):
        return build_optimisation_problem(
            self, cost, constraints, arguments, outputs, initial_conditions
        )


register_backend("numpy", NumpyBackend)


#
# OP v_1, v_2, v_3
#
# v is either
# a) a symbol
# b) a constant
# c) another node in the graph
#
# if OP is linear
# -
