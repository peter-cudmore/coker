"""PyTorch numerical backend."""

import numpy as np
import torch

from coker.algebra import Dimension
from coker.algebra.function import Function, create_function_from_native
from coker.backends.evaluator import GenericEvaluator
from coker.backends.backend import ArrayLike, Backend, register_backend
from coker.backends.lowered import FunctionSignature, LoweringOptions

from .dynamics import PytorchSolverParameters, evaluate_integrals
from .lower import PytorchLoweredFunction, PytorchModule
from .ops import (
    call_parameterised_op,
    impls,
    parameterised_impls,
    scalar_types,
)


class PytorchBackend(Backend):
    """Evaluate Coker expression graphs using PyTorch tensors."""

    def to_numpy_array(self, array) -> ArrayLike:
        if array is None:
            return None
        if isinstance(array, torch.Tensor):
            result = array.detach().cpu().numpy()
        else:
            result = np.asarray(array)
        return result.item() if result.shape == () else result

    def to_backend_array(self, array):
        if isinstance(array, torch.Tensor):
            return array
        import scipy.sparse

        if scipy.sparse.issparse(array):
            array = array.toarray()
        return torch.as_tensor(array)

    def reshape(self, arg, dim: Dimension):
        if arg is None:
            return arg
        if dim.is_scalar():
            if isinstance(arg, torch.Tensor):
                if arg.ndim == 0:
                    return arg
                if arg.numel() != 1:
                    raise TypeError(f"Expecting a scalar, got {arg}")
                return arg.reshape(())
            if isinstance(arg, scalar_types):
                return arg
            try:
                (inner,) = arg
            except (ValueError, TypeError) as ex:
                raise TypeError(f"Expecting a scalar, got {arg}") from ex
            return self.reshape(inner, dim)
        if isinstance(arg, torch.Tensor):
            return torch.reshape(arg, dim.dim)
        if isinstance(arg, np.ndarray):
            return np.reshape(arg, dim.dim)
        if isinstance(arg, (float, int, complex)):
            return torch.as_tensor([arg]).reshape(dim.dim)
        raise NotImplementedError(
            f"Don't know how to resize {arg.__class__.__name__}"
        )

    def call(self, op, *args) -> ArrayLike:
        if op in impls:
            return impls[op](*args)
        if isinstance(op, tuple(parameterised_impls.keys())):
            return call_parameterised_op(op, *args)
        raise NotImplementedError(f"{op} is not implemented")

    def get_evaluator(self) -> GenericEvaluator:
        return GenericEvaluator(
            self,
            operations=impls,
            parameterised_operations=parameterised_impls,
        )

    def evaluate_integrals(
        self,
        functions,
        initial_conditions,
        end_point: float,
        inputs,
        solver_parameters=None,
    ):
        return evaluate_integrals(
            self,
            functions,
            initial_conditions,
            end_point,
            inputs,
            solver_parameters,
        )

    def lower(
        self, function: Function, options: LoweringOptions | None = None
    ) -> PytorchLoweredFunction:
        return PytorchLoweredFunction(
            self,
            function,
            self.get_evaluator().build_plan(function.tape),
        )

    def as_module(self, function):
        """Lower a function to an eager ``torch.nn.Module``."""
        return self.lower(function).as_module()

    def build_optimisation_problem(self, *args, **kwargs):
        raise NotImplementedError(
            "optimisation problem construction is not implemented for the "
            "pytorch backend"
        )

    def import_function(
        self,
        implementation,
        signature: FunctionSignature,
        *,
        name: str | None = None,
    ) -> Function:
        """Import a PyTorch-compatible callable as a Coker function."""
        return create_function_from_native(
            implementation, signature, backend=self.name, name=name
        )

    def import_module(
        self, module: torch.nn.Module, signature: FunctionSignature
    ) -> Function:
        """Import a parameterised PyTorch module as a Coker function.

        The module remains the native callable, so its parameters and buffers
        stay authoritative and connected to PyTorch autograd.
        """
        if not isinstance(module, torch.nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        return self.import_function(module, signature)

    def create_variational_solver(self, problem):
        raise NotImplementedError(
            "variational solving is not implemented for the pytorch backend"
        )


__all__ = ["PytorchBackend", "PytorchModule", "PytorchSolverParameters"]
register_backend("pytorch", PytorchBackend)
