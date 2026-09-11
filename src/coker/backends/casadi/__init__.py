from __future__ import annotations

from typing import Any, Sequence, Union

import casadi as ca
import numpy as np


from coker import Dimension, Function, Scalar, VectorSpace
from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Tracer
from coker.algebra.ops import Noop, ReshapeOP
from coker.backends.backend import ArrayLike, Backend, register_backend
from coker.backends.lowered import (
    FunctionInputSpec,
    FunctionOutputSpec,
    FunctionSignature,
    LoweringOptions,
)
from coker.backends.casadi.lower import (
    call_parameterised_op,
    impls,
    lower as _lower_to_casadi,
    parameterised_impls,
    substitute,
)
from coker.backends.casadi.optimiser import build_optimisation_problem
from coker.backends.casadi.lowered import CasadiLoweredFunction
from coker.backends.casadi.variational.solver import (
    create_variational_solver,
)
from coker.dynamics import VariationalProblem

__all__ = ["CasadiBackend"]

scalar_types = (float, int)


def _space_from_casadi_shape(
    name: str, rows: int, columns: int
) -> Scalar | VectorSpace:
    if (rows, columns) == (1, 1):
        return Scalar(name)
    return VectorSpace(name, rows if columns == 1 else (rows, columns))


def _signature_from_casadi_function(
    ca_function: ca.Function,
) -> "FunctionSignature":

    inputs = tuple(
        FunctionInputSpec(
            ca_function.name_in(index),
            _space_from_casadi_shape(
                ca_function.name_in(index),
                ca_function.size1_in(index),
                ca_function.size2_in(index),
            ),
        )
        for index in range(ca_function.n_in())
    )
    outputs = tuple(
        FunctionOutputSpec(
            ca_function.name_out(index),
            Dimension(
                None
                if (ca_function.size1_out(index), ca_function.size2_out(index))
                == (1, 1)
                else (
                    (ca_function.size1_out(index),)
                    if ca_function.size2_out(index) == 1
                    else (
                        ca_function.size1_out(index),
                        ca_function.size2_out(index),
                    )
                )
            ),
        )
        for index in range(ca_function.n_out())
    )
    return FunctionSignature(inputs=inputs, outputs=outputs)


class CasadiBackend(Backend):

    def import_function(
        self,
        ca_function: ca.Function,
        signature: FunctionSignature | None = None,
    ) -> Function:
        """Import a native CasADi function as an ``OP.EVALUATE`` callable.

        CasADi functions carry ordered names and matrix dimensions, so callers
        may omit ``signature``. Pass one only to impose a deliberate Coker
        declaration instead of the native function's interface.
        """
        if not isinstance(ca_function, ca.Function):
            raise TypeError(
                "CasadiBackend.import_function expects a casadi.Function"
            )
        signature = (
            _signature_from_casadi_function(ca_function)
            if signature is None
            else signature
        )
        return Function.from_native(
            ca_function, signature, backend=getattr(self, "name", "casadi")
        )

    def to_numpy_array(self, array: Union[ca.MX, ca.DM]) -> ArrayLike:
        if isinstance(array, ca.MX):
            try:
                return array.to_DM().toarray()
            except RuntimeError:
                pass
        elif isinstance(array, ca.DM):
            return array.toarray()
        try:
            return ca.evalf(array).toarray()
        except RuntimeError:
            pass

        raise ValueError(f"Cannot convert {array} to a numpy array")

    def to_backend_array(self, array):
        import scipy.sparse

        if array is None:
            return ca.DM()
        if isinstance(array, np.ndarray) and array.size == 0:
            return ca.DM.zeros(0, 1)
        if scipy.sparse.issparse(array):
            return ca.DM(scipy.sparse.csc_matrix(array))
        if isinstance(array, scalar_types):
            return ca.DM(array)
        if array.shape == (1, 1):
            return ca.DM(array[0, 0])
        elif array.shape == (1,):
            return ca.DM(array[0])
        elif len(array.shape) >= 2:
            result = ca.DM(array)

        elif len(array.shape) == 1 and array.shape[0] > 1:
            result = ca.DM(array.reshape(-1, 1))

        else:
            raise NotImplementedError(
                f"Don't know how to convert {array} to a casadi array"
            )

        assert not isinstance(result, Tracer)

        return result

    def call(self, op, *args) -> ArrayLike:
        try:
            result: ca.DM = impls[op](*args)
            assert result.is_regular(), f"{op}({args}) =  {result}"

            return result
        except KeyError:
            pass

        if isinstance(op, tuple(parameterised_impls.keys())):
            result = call_parameterised_op(op, *args)
            assert result.is_regular(), f"{op}({args}) =  {result}"
            return result

        if isinstance(op, ReshapeOP):

            (arg,) = args
            shape = op.newshape
            if len(shape) == 1:
                shape = (1, *shape)

            return ca.reshape(arg, shape)

        raise NotImplementedError(f"{op} is not implemented")

    def reshape(self, array: ArrayLike, dim: Dimension) -> ArrayLike:
        if dim.is_scalar():
            return array

        if dim.is_vector():
            shape = (*dim, 1)
        else:
            shape = tuple(dim)

        if isinstance(array, (ca.MX, ca.DM)):
            return ca.reshape(array, *shape)
        if isinstance(array, np.ndarray):
            return ca.reshape(array, *shape)
        raise NotImplementedError

    def _lower_with_evaluate(self, function):
        return CasadiLoweredFunction(self, function)

    def lower(
        self,
        function: Function,
        options: LoweringOptions | None = None,
    ) -> CasadiLoweredFunction:

        if any(
            isinstance(shape, FunctionSpace)
            for shape in function.input_shape()
        ) or any(output is None for output in function.output):
            return self._lower_with_evaluate(function)

        ca_inputs, ca_outputs = _lower_to_casadi(
            function.tape, function.output
        )
        return CasadiLoweredFunction(
            self, function, ca.Function("f", ca_inputs, ca_outputs)
        )

    def restore_public_outputs(
        self, function: Function, outputs: Sequence[Any | None]
    ) -> tuple[Any | None, ...]:
        """Convert CasADi lowered values at the public Coker boundary."""
        restored: list[Any | None] = []
        for value, output in zip(outputs, function.output):
            if value is None or output is None:
                restored.append(None)
                continue
            try:
                numpy_value = self.to_numpy_array(value)
            except ValueError:
                restored.append(value)
                continue
            if output.dim.is_scalar():
                restored.append(float(np.asarray(numpy_value).reshape(-1)[0]))
            else:
                restored.append(np.asarray(numpy_value).reshape(output.shape))
        return tuple(restored)

    def evaluate(
        self, function: Function, inputs: Sequence[Any]
    ) -> list[Any | None]:
        workspace: dict[int, Any] = {}

        for idx, (space, arg) in enumerate(
            zip(function.input_shape(), inputs)
        ):
            assert not isinstance(arg, Tracer)
            index = function.tape.input_indicies[idx]
            if isinstance(arg, np.ndarray):
                workspace[index] = self.to_backend_array(arg)

            else:
                workspace[index] = arg

        y = substitute(function.output, workspace)
        outs: list[Any | None] = []
        for y_i, output_tracer in zip(y, function.output):
            if output_tracer is None:
                outs.append(None)
                continue
            try:
                y_result = self.to_numpy_array(y_i)
                if output_tracer.dim.is_scalar():
                    if y_result.shape == (1, 1):
                        outs.append(float(y_result[0, 0]))
                    elif y_result.shape == (1,):
                        outs.append(float(y_result[0]))
                    else:
                        raise ValueError("Expected a scalar", y_result)
                else:
                    outs.append(y_result.reshape(output_tracer.shape))

            except ValueError:
                outs.append(y_i)

        return outs

    def to_array(self, arg: Union[ca.MX, ca.DM]):

        return self.to_numpy_array(arg)

    def build_optimisation_problem(
        self, cost, constraints, parameters, outputs, initial_conditions
    ):
        return build_optimisation_problem(
            cost, constraints, parameters, outputs, initial_conditions
        )

    def create_variational_solver(self, problem: VariationalProblem):
        return create_variational_solver(problem)

    def evaluate_integrals(
        self,
        functions,
        initial_conditions,
        end_point: float,
        inputs,
        solver_parameters=None,
    ):
        dxdt, g, dqdt = functions

        is_dae = g is not Noop()
        has_quadrature = dqdt is not Noop()
        x0, z0, q0 = (self.to_backend_array(a) for a in initial_conditions)
        if isinstance(end_point, (int, float)):
            if end_point == 0:
                return x0, z0, q0

            t_eval = [end_point]
        else:
            t_eval = end_point

        u, p = inputs
        p = self.to_backend_array(p)
        t = ca.MX.sym("t")
        x = ca.MX.sym("x", x0.shape)
        z = ca.MX.sym("z", z0.shape)

        dx_sym = dxdt(t, x, z, u, p)

        if has_quadrature:
            q = ca.MX.sym("q", q0.shape)
            dq_sym = dqdt(t, x, z, u, p)
            txq = ca.vertcat(t, x, q)
            txq0 = ca.vertcat(ca.DM(0), x0, q0)
            xq_to_x_q = ca.Function("txq_to_x_q", [txq], [x, q])
            dtxq = ca.vertcat(ca.MX(1), dx_sym, dq_sym)
        else:
            txq = ca.vertcat(t, x)
            txq0 = ca.vertcat(ca.DM(0), x0)
            xq_to_x_q = ca.Function("txq_to_x_q", [txq], [x])
            dtxq = ca.vertcat(ca.MX(1), dx_sym)

        initial_conditions = {
            "x0": txq0,
        }
        dae = {
            "x": txq,
            "ode": dtxq,
        }
        if is_dae:
            dae["z"] = z
            dae["alg"] = g(t, x, z, u, p)
            initial_conditions["z0"] = z0

        solver = ca.integrator("solver", "idas", dae, 0, t_eval, {})
        xq_final = solver(**initial_conditions)

        if has_quadrature:
            x_final, q_final = xq_to_x_q(xq_final["xf"])
        else:
            x_final = xq_to_x_q(xq_final["xf"])
            q_final = None
        if is_dae:
            z_final = xq_final["zf"]
        else:
            z_final = None

        return x_final, z_final, q_final


register_backend("casadi", CasadiBackend)
