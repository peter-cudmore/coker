from typing import List

import numpy as np
import scipy.sparse.csc
import scipy as scp

from coker.algebra.dimensions import (
    Dimension,
    FunctionSpace,
    Scalar,
    VectorSpace,
)
from coker.algebra.function import Function
from coker.algebra.graph import Tracer

from coker.backends.evaluator import GenericEvaluator
from coker.backends.backend import (
    ArrayLike,
    Backend,
    import_native_function,
    register_backend,
    split_function_parameter_values,
)
from coker.backends.lowered import FunctionSignature, LoweringOptions

from coker.backends.numpy.lowered import NumpyLoweredFunction
from coker.backends.numpy.evaluator import (
    call_parameterised_op,
    impls,
    parameterised_impls,
)
from coker.backends.numpy.optimisation import build_optimisation_problem
from coker.backends.numpy.dynamics import evaluate_integrals


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
    def _materialize_parameter(self, target, declaration, blocks):
        flat_values = self._concatenate_parameter_blocks(blocks)
        if isinstance(target, FunctionSpace):
            return self._fit_function_parameter(declaration, target, flat_values)
        if isinstance(target, VectorSpace):
            return flat_values.reshape(target.dimension)
        if isinstance(target, Scalar):
            if flat_values.size != 1:
                raise ValueError("scalar parameter must have one solver value")
            return flat_values[0]
        raise TypeError("parameter target must be a scalar, vector, or function space")

    @staticmethod
    def _concatenate_parameter_blocks(blocks):
        if not blocks:
            raise ValueError("parameter blocks must not be empty")
        return np.concatenate(
            tuple(np.asarray(block, dtype=float).reshape(-1) for block in blocks)
        )

    def _fit_function_parameter(self, declaration, target, flat_values):
        from coker.parameters.function_parameters import FittedFunction

        parameters = split_function_parameter_values(declaration, flat_values)
        target = declaration.validate_target(target)
        native = self.lower(declaration.build_function(target, self.name))
        return FittedFunction(
            declaration,
            target,
            lambda *arguments: native(*arguments, *parameters),
            parameters,
        )

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
        return import_native_function(
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
        return evaluate_integrals(
            functions,
            initial_conditions,
            end_point,
            inputs,
            solver_parameters,
        )

    def build_optimisation_problem(
        self,
        cost: Tracer,
        constraints: List[Tracer],
        arguments: List[Tracer],
        outputs: List[Tracer],
        initial_conditions,
        *,
        options=None,
    ):
        if options is not None:
            raise TypeError("NumPy optimisation does not support solver options")
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
