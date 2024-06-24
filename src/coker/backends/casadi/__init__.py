import casadi as ca
import numpy as np
from typing import Tuple, Type, Union

from coker import Dimension, Kernel

from coker.backends.backend import Backend, ArrayLike
from coker.backends.evaluator import evaluate_inner
from coker.backends.casadi.casadi import *


scalar_types = (float, int)


class CasadiBackend(Backend):
    def to_numpy_array(self, array: Union[ca.MX, ca.DM]) -> ArrayLike:
        if isinstance(array, ca.MX):

            return array.to_DM().toarray(simplify=True)

        elif isinstance(array, ca.DM):
            return array.toarray(simplify=True)

        raise NotImplementedError

    def to_backend_array(self, array) -> ca.MX:
        if isinstance(array, scalar_types):
            return ca.DM(array)
        if array.shape == (1, 1):
            return ca.DM(array[0, 0])
        elif array.shape == (1,):
            return ca.DM(array[0])
        elif len(array.shape) >= 2:
            result = ca.DM(*array.shape)
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
            return call_parameterised_op(op, *args)

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

    def lower(self, kernel: Kernel):
        return lower(kernel.tape, kernel.output)

    def evaluate(self, kernel: Kernel, inputs: ArrayLike):
        f = lower(kernel.tape, kernel.output)
        y = f(*inputs)

        return [self.to_numpy_array(y)]

    def build_optimisation_problem(self, cost, constraints, inputs, outputs):

        tape = cost.tape
        cost_fn = lower(tape, [cost])
