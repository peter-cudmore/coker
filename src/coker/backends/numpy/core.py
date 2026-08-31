from typing import List, Any
from enum import Enum

import numpy as np
import scipy.sparse.csc
import scipy as scp

from coker.algebra import Dimension, OP
from coker.algebra.kernel import Tracer, Noop
from coker.algebra.ops import ConcatenateOP, ReshapeOP, NormOP

from coker.backends.backend import Backend, ArrayLike, SolverParameters
from coker.backends.numpy.optimisation import build_optimisation_problem


class Solver(Enum):
    RK45 = "RK45"
    LSODA = "LSODA"
    Radau = "Radau"
    BDF = "BDF"


class NumpySolverParameters(SolverParameters):
    def __init__(self, solver: Solver = Solver.RK45):
        self.solver = solver


def to_array(value, shape):

    if isinstance(value, np.ndarray) and value.shape == shape:
        return value

    raise NotImplementedError


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


def div(num, den):
    if isinstance(den, Tracer):
        return num / den
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(num, den)
    try:
        if np.isscalar(den) and den == 0:
            if np.isscalar(num):
                return float("nan")
            return np.full_like(result, np.nan, dtype=float)
        zero_mask = den == 0
    except ValueError:
        return result
    if np.isscalar(zero_mask):
        return result
    if np.any(zero_mask):
        result = np.asarray(result, dtype=float)
        result[zero_mask] = np.nan
    return result


impls = {
    OP.ADD: np.add,
    OP.SUB: np.subtract,
    OP.MUL: np.multiply,
    OP.DIV: div,
    OP.MATMUL: np.matmul,
    OP.SIN: np.sin,
    OP.COS: np.cos,
    OP.TAN: np.tan,
    OP.EXP: np.exp,
    OP.PWR: np.power,
    OP.INT_PWR: np.power,
    OP.ARCCOS: np.arccos,
    OP.ARCSIN: np.arcsin,
    OP.DOT: np.dot,
    OP.CROSS: np.cross,
    OP.TRANSPOSE: np.transpose,
    OP.NEG: np.negative,
    OP.SQRT: np.sqrt,
    OP.ABS: np.abs,
    OP.ARCTAN2: np.arctan2,
    OP.EQUAL: np.equal,
    OP.LESS_EQUAL: np.less_equal,
    OP.LESS_THAN: np.less,
    OP.CASE: lambda cond, t, f: t if cond else f,
    OP.EVALUATE: lambda op, *args: op(*args),
    OP.LOG: np.log,
}

parameterised_impls = {
    ConcatenateOP: lambda op, *x: np.concatenate(x, axis=op.axis),
    ReshapeOP: lambda op, x: np.reshape(x, shape=op.newshape),
    NormOP: lambda op, x: np.linalg.norm(x, ord=op.ord),
}


def call_parameterised_op(op, *args):
    kls = op.__class__

    result = parameterised_impls[kls](op, *args)

    return result


def reshape(arg, dim):
    if dim.is_scalar():
        if isinstance(arg, scalar_types):
            return arg
        try:
            (inner,) = arg
        except ValueError as ex:
            raise TypeError(f"Expecting a scalar, got {arg}") from ex
        except TypeError as ex:
            raise TypeError(f"Expecting a scalar, got {arg}") from ex
        return reshape(inner, dim)
    if isinstance(arg, np.ndarray):
        return np.reshape(arg, dim.dim)
    if scp.sparse.issparse(arg):
        return np.reshape(arg.toarray(), dim.dim)
    if isinstance(arg, (float, int)):
        return np.array([arg]).reshape(dim.dim)
    if arg is None:
        return arg
    raise NotImplementedError(f"Dont know how to reshape {arg}")


class NumpyBackend(Backend):
    def __init__(self, *args, **kwargs):
        super(NumpyBackend, self).__init__(*args, **kwargs)

    def native_types(self) -> List[Any]:
        return [
            np.ndarray,
            np.int32,
            np.int64,
            np.float64,
            np.float32,
            float,
            complex,
            int,
        ]

    def to_numpy_array(self, array) -> ArrayLike:
        return array

    def to_backend_array(self, array: ArrayLike):
        return array

    def reshape(self, arg, dim: Dimension):
        return reshape(arg, dim)
        raise NotImplementedError(
            f"Don't know how to resize {arg.__class__.__name__}"
        )

    def lower(self, function):
        from coker.backends.evaluator import _build_plan, _cast_outputs

        plan = _build_plan(function.tape, self)
        tape = function.tape
        outputs = function.output
        backend = self

        def compiled(inputs):
            workspace = plan.execute(inputs, backend)
            return _cast_outputs(outputs, tape, workspace, backend)

        return compiled

    def resolve_fn(self, op):
        if op in impls:
            return impls[op]
        if isinstance(op, tuple(parameterised_impls.keys())):
            kls = op.__class__
            _op = op
            return lambda *args: parameterised_impls[kls](_op, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def resolve_post_fn(self, dim):
        # For scalar outputs, reshape extracts the Python scalar from the
        # array. Tracers must pass through unchanged during function
        # composition tracing. Non-scalar outputs already have the
        # correct shape.
        if dim.is_scalar():
            _dim = dim

            def scalar_post(v):
                if isinstance(v, Tracer):
                    return v
                return reshape(v, _dim)

            return scalar_post
        return lambda v: v

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
