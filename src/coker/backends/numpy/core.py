from typing import Type, Tuple, List
from functools import reduce
from operator import mul
import numpy as np
import sympy as sp
import scipy as sy

from coker.algebra import Dimension, OP
from coker.algebra.kernel import Expression, Tracer, VectorSpace
from coker.algebra.ops import ConcatenateOP, ReshapeOP, NormOP

from coker.backends.backend import Backend, ArrayLike
from coker.backends.evaluator import evaluate_inner


def to_array(value, shape):

    if isinstance(value, np.ndarray) and value.shape == shape:
        return value

    raise NotImplementedError


scalar_types = (np.float32, np.float64, np.int32, np.int64, float, complex, int)


def is_scalar_symbol(v):
    if isinstance(v, sp.MatrixSymbol):
        return v.shape == (1, 1)
    return isinstance(v, (sp.Symbol, sp.Expr))


def div(num, den):
    if (num == 0).all() and (den == 0):
        return num
    else:
        return np.divide(num, den)


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
}

parameterised_impls = {
    ConcatenateOP: lambda op, x, y: np.concatenate((x, y), axis=op.axis),
    ReshapeOP: lambda op, x: np.reshape(x, newshape=op.newshape),
    NormOP: lambda op, x: np.linalg.norm(x, ord=op.ord),
}


def call_parameterised_op(op, *args):
    kls = op.__class__
    result = parameterised_impls[kls](op, *args)

    return result


def jacobian(f: sp.Expr, x):
    result = sp.Matrix([f]).jacobian(x)
    return result


def hessian(f: sp.Expr, x):
    h = sp.hessian(f, x)
    return h


def reshape_sympy_matrix(arg, shape):
    if len(shape) == 1:
        out_shape = (shape[0], 1)
    else:
        assert len(shape) == 2
        out_shape = shape

    old_cols = 1 if len(arg.shape) == 1 else arg.shape[1]

    def lookup(i, j):
        offset = i * out_shape[1] + j
        old_row = offset // old_cols
        old_col = offset % old_cols
        value = arg[old_row, old_col]
        return value

    return sp.Matrix(*out_shape, lookup)


class NumpyBackend(Backend):
    def __init__(self, *args, **kwargs):
        super(NumpyBackend, self).__init__(*args, **kwargs)

    def native_types(self) -> Tuple[Type]:
        return (
            np.ndarray,
            np.int32,
            np.int64,
            np.float64,
            np.float32,
            float,
            complex,
            int,
        )

    def to_numpy_array(self, array) -> ArrayLike:
        return array

    def to_backend_array(self, array: ArrayLike):
        return array

    def reshape(self, arg, dim: Dimension):
        if dim.is_scalar():
            if isinstance(arg, scalar_types) or is_scalar_symbol(arg):
                return arg
            else:
                try:
                    (inner,) = arg
                except ValueError as ex:
                    raise TypeError(f"Expecting a scalar, got {arg}") from ex
                return self.reshape(inner, dim)
        elif isinstance(arg, (sp.Matrix, sp.Array, sp.MatrixSlice)):
            if arg.shape == dim.dim:
                return arg
            return reshape_sympy_matrix(arg, dim.dim)
        elif isinstance(arg, np.ndarray):
            return np.reshape(arg, dim.dim)

        raise NotImplementedError(f"Don't know how to resize {arg.__class__.__name__}")

    def call(self, op, *args) -> ArrayLike:

        try:
            result = impls[op](*args)
            return result
        except KeyError:
            pass

        if isinstance(op, tuple(parameterised_impls.keys())):
            return call_parameterised_op(op, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def build_optimisation_problem(
        self,
        cost: Tracer,  # cost
        constraints: List[Expression],
        arguments: List[Tracer],
        outputs: List[Tracer],
    ):

        impl = None
        tape = cost.tape

        assert all(c.tape == tape for c in constraints)
        assert all(a.tape == tape for a in arguments)
        assert all(o.tape == tape for o in outputs)

        # For each variable
        #
        # 2. replace each 'variable' with a projection P, such that
        # V_i = P_iX + \sum_j R_jA_j
        # where A_j is the jth argument

        # construct Q_0, P_0, R_0 and N(X)
        # c_i = X^T Q_i X + P_i X + R_i + N_i(Z)

        arg_indexes = {a.index for a in arguments}
        decision_variables = [i for i in tape.input_indicies if i not in arg_indexes]

        arg_symbols = []
        for i, a in enumerate(arguments):
            shape = (a.shape[0], 1) if len(a.shape) == 1 else a.shape
            assert len(shape) == 2
            arg_symbols.append(sp.MatrixSymbol(f"p_{i}", *shape))

        n = 0
        mappings = []
        for index in decision_variables:
            dim: Dimension = tape.dim[index]
            if dim.is_scalar():
                flat_dim = 1
            else:
                flat_dim = reduce(mul, dim)

            mappings.append((index, dim, (n, flat_dim + n)))
            n += flat_dim

        x = sp.Array([sp.symbols(f"x_{i}") for i in range(n)])
        for idx, dim, (start, stop) in mappings:
            tape.substitute(idx, (OP.VALUE, x[start:stop]))

        workspace = {}
        (cost,) = evaluate_inner(tape, arguments, [cost], self, workspace)

        problem_args = [x, *arg_symbols]
        cost_f = sp.lambdify(problem_args, cost)

        cost_jac = lambda a: sp.lambdify(problem_args, jacobian(cost, x))(a).reshape(
            x.shape
        )
        cost_hess = sp.lambdify(problem_args, hessian(cost, x))

        out_constriants = []
        for i, constraint in enumerate(constraints):
            (c,) = evaluate_inner(
                tape, arguments, [constraint.as_halfplane_bound()], self, workspace
            )
            c_func = sp.lambdify(problem_args, c)
            c_jac = jacobian(c, problem_args)

            if not c_jac.free_symbols:
                c_0 = c_func(*[np.zeros_like(x_i) for x_i in problem_args])
                this_constraint = sy.optimize.LinearConstraint(
                    c_jac.evalf(), -c_0, np.inf
                )
            else:
                this_constraint = sy.optimize.NonlinearConstraint(
                    sp.lambdify(problem_args, c),
                    0,
                    np.inf,
                    jac=sp.lambdify(problem_args, c_jac),
                    hess="cs",
                )
            out_constriants.append(this_constraint)

        out_symbols = evaluate_inner(tape, arguments, outputs, self, workspace)

        out_map = [sp.lambdify(problem_args, o) for o in out_symbols]

        def solver(*solver_args):
            x0 = np.zeros(n)

            soln = sy.optimize.minimize(
                cost_f,
                x0,
                method="trust-constr",
                jac=cost_jac,
                hess=cost_hess,
                constraints=out_constriants,
            )

            inner_args = [soln.x] + list(*solver_args)
            result = [o(*inner_args)[0] for o in out_map]
            return [self.to_backend_array(r) for r in result]

        return solver


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
