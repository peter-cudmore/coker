import casadi as ca
import numpy as np
from typing import Tuple, Type, Union

from sympy.core import symbol

from coker import Dimension, Function

from coker.backends.backend import Backend, ArrayLike
from coker.backends.evaluator import evaluate_inner
from coker.backends.casadi.casadi import *
from coker.backends.casadi.optimiser import build_optimisation_problem

scalar_types = (float, int)


class CasadiBackend(Backend):
    def to_numpy_array(self, array: Union[ca.MX, ca.DM]) -> ArrayLike:
        if isinstance(array, ca.MX):

            return array.to_DM().toarray(simplify=True)

        elif isinstance(array, ca.DM):
            return array.toarray(simplify=True)

        raise NotImplementedError(f"Cannot convert {array} to a numpy array")

    def to_backend_array(self, array):
        if isinstance(array, scalar_types):
            return ca.DM(array)
        if array.shape == (1, 1):
            return ca.DM(array[0, 0])
        elif array.shape == (1,):
            return ca.DM(array[0])
        elif len(array.shape) >= 2:
            result = ca.DM.zeros(*array.shape)
            with np.nditer(array, flags=["multi_index"], op_flags=["readonly"]) as it:
                for v in it:
                    if v != 0:
                        key = tuple(it.multi_index)
                        result[key] = v

        elif len(array.shape) == 1 and array.shape[0] > 1:
            (n,) = array.shape
            result = ca.DM(n, 1)
            for i, v in enumerate(array):
                if v != 0:
                    result[i] = v

        else:
            raise NotImplementedError
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

    def native_types(self) -> Tuple[Type]:
        pass

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

    def lower(self, function: Function):
        assert not any(
            isinstance(shape, FunctionSpace) for shape in function.input_shape()
        ), "Cannot lower a partially evaluated function."
        return lower(function.tape, function.output)

    def evaluate(self, function: Function, inputs: ArrayLike):
        workspace = {}
        values = []
        ins = []
        for idx, (space, arg) in enumerate(zip(function.input_shape(), inputs)):
            if isinstance(space, FunctionSpace):
                f_arg_ins, f_arg_outs = arg.lower()
                workspace[idx] = ca.Function(f"fx_{idx}", f_arg_ins, f_arg_outs)

            else:
                x_symbol = ca.MX.sym(f"x_{idx}", *space.shape)
                workspace[idx] = x_symbol
                ins.append(x_symbol)
                if isinstance(arg, (ca.MX, ca.SX)):
                    values.append(arg)
                else:
                    values.append(self.to_backend_array(arg))

        outs = substitute(function.output, workspace)

        f = ca.Function("f", ins, outs)
        y = f(*values)
        if len(function.output) > 1:
            return [self.to_array(y_i) for y_i in y]
        else:
            return (self.to_array(y),)

    def to_array(self, arg: Union[ca.MX, ca.DM]):
        try:
            return self.to_numpy_array(arg)
        except RuntimeError:
            return arg

    def build_optimisation_problem(
        self, cost, constraints, parameters, outputs, initial_conditions
    ):
        return build_optimisation_problem(
            cost, constraints, parameters, outputs, initial_conditions
        )
