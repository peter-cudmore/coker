"""PyTorch numerical backend."""

from collections.abc import Sequence

import numpy as np
import torch

from coker.algebra import Dimension
from coker.algebra.function import Function, create_function_from_native
from coker.algebra.graph import CallableReference, Tape, Tracer
from coker.algebra.ops import Noop
from coker.backends.evaluator import GenericEvaluator
from coker.backends.backend import (
    ArrayLike,
    Backend,
    split_function_parameter_values,
    register_backend,
)
from coker.backends.lowered import FunctionSignature, LoweringOptions

from .dynamics import PytorchODESolverParameters, evaluate_integrals
from .lower import PytorchLoweredFunction, PytorchModule
from .optimisation import PytorchNLPSolverOptions, build_optimisation_problem
from .ops import (
    call_parameterised_op,
    impls,
    parameterised_impls,
    scalar_types,
)


class _FittedModule(torch.nn.Module):
    def __init__(
        self,
        native: torch.nn.Module,
        concrete_values: tuple[torch.Tensor, ...],
    ) -> None:
        super().__init__()
        self.native = native
        self._concrete_value_names = tuple(
            f"_concrete_value_{index}" for index in range(len(concrete_values))
        )
        for name, value in zip(self._concrete_value_names, concrete_values):
            self.register_buffer(name, value)

    def forward(self, argument):
        return self.native(
            argument,
            *(self.get_buffer(name) for name in self._concrete_value_names),
        )


class PytorchBackend(Backend):
    """Evaluate Coker expression graphs using PyTorch tensors."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype

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
            device_matches = self.device is None or array.device == self.device
            if array.dtype == torch.bool:
                return (
                    array if device_matches else array.to(device=self.device)
                )
            dtype_matches = self.dtype is None or array.dtype == self.dtype
            if device_matches and dtype_matches:
                return array
            return array.to(device=self.device, dtype=self.dtype)
        import scipy.sparse

        if scipy.sparse.issparse(array):
            array = array.toarray()
        return torch.as_tensor(array, device=self.device, dtype=self.dtype)

    def reshape(self, arg, dim: Dimension):
        if arg is None:
            return arg
        if dim.is_scalar():
            if isinstance(arg, torch.Tensor):
                if arg.ndim == 0:
                    return self.to_backend_array(arg)
                if arg.numel() != 1:
                    raise TypeError(f"Expecting a scalar, got {arg}")
                return self.to_backend_array(arg.reshape(()))
            if isinstance(arg, scalar_types):
                if self.device is None and self.dtype is None:
                    return arg
                return self.to_backend_array(arg)
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
            return self.to_backend_array([arg]).reshape(dim.dim)
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
            function,
            self.get_evaluator().build_plan(function.tape),
        )

    def compose(
        self, function: Function, inputs: Sequence[object], outer_tape: Tape
    ) -> list[Tracer | None]:
        if not self._contains_native_callable(function):
            return super().compose(function, inputs, outer_tape)

        native_arguments = tuple(
            None if value is None or isinstance(value, Noop) else value
            for value in inputs
        )
        input_spaces = tuple(
            spec.space
            for spec, value in zip(function.signature.inputs, native_arguments)
            if value is not None
        )
        native_modules = {}
        fallback_module = None

        def module_for(present_inputs):
            nonlocal fallback_module

            tensor = next(
                (
                    value
                    for value in present_inputs
                    if isinstance(value, torch.Tensor)
                ),
                None,
            )
            if tensor is None:
                if fallback_module is None:
                    fallback_module = self.as_module(function)
                return fallback_module
            key = tensor.device, tensor.dtype
            try:
                return native_modules[key]
            except KeyError:
                native = PytorchBackend(
                    device=tensor.device, dtype=tensor.dtype
                ).as_module(function)
                native_modules[key] = native
                return native

        def execute(*present_inputs):
            native = module_for(present_inputs)
            values = iter(present_inputs)
            return native(
                *(
                    None if value is None else next(values)
                    for value in native_arguments
                )
            )

        return Function._append_native_outputs(
            outer_tape,
            execute,
            self.name,
            input_spaces,
            function.signature.outputs,
            tuple(value for value in native_arguments if value is not None),
            name=function.name,
        )

    @staticmethod
    def _contains_native_callable(function: Function) -> bool:
        return any(
            not isinstance(node, Tracer)
            and any(
                isinstance(argument, CallableReference)
                for argument in node[1:]
            )
            for node in function.tape.nodes
        )

    def as_module(self, function):
        """Lower a function to an eager ``torch.nn.Module``."""
        return self.lower(function).as_module()

    def fit_function_parameter(self, declaration, target, values):
        """Build a PyTorch-native fitted function from solver decisions."""
        from coker.parameters.function_parameters import FittedFunction

        flat_values = self.to_backend_array(values).reshape(-1)
        parameters = split_function_parameter_values(declaration, flat_values)
        native = self.as_module(declaration.build_function(target, self.name))
        function = _FittedModule(native, parameters)
        return FittedFunction(declaration, target, function, parameters)

    def build_optimisation_problem(
        self,
        cost,
        constraints,
        parameters,
        outputs,
        initial_conditions,
        *,
        options=None,
    ):
        return build_optimisation_problem(
            cost,
            constraints,
            parameters,
            outputs,
            initial_conditions,
            options=options,
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
        from .variational import create_variational_solver

        return create_variational_solver(problem)


__all__ = [
    "PytorchBackend",
    "PytorchModule",
    "PytorchNLPSolverOptions",
    "PytorchODESolverParameters",
]
register_backend("pytorch", PytorchBackend)
