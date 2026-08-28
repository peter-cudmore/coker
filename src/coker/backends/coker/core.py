"""Coker backend facade and Rust-compiled function handle."""

from typing import Dict, List, Tuple, Type

import numpy as np

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ModuleCallOP
from coker.backends.backend import ArrayLike, Backend
import coker.backends.coker.conversion as _conversion
from coker.backends.coker.module import CokerModule
from coker.backends.coker.optimisation import (
    build_optimisation_problem as build_qp_optimisation_problem,
)


class CokerFunction:
    """A Python tape compiled and executed by the mapped Rust runtime."""

    def __init__(self, function: Function, *, output_labels=None):
        self.function = function
        self._output_labels = output_labels
        self._artifact = None
        self._bound_functions = {}
        self._bound_key = None

    def _compile(self):
        try:
            import coker_compiler
        except ImportError as error:
            raise RuntimeError(
                "coker_compiler extension is required"
            ) from error
        dags = _conversion.function_to_typed_dag(
            self.function,
            output_labels=self._output_labels,
            bound_functions=self._bound_functions,
            return_all=True,
        )
        self._artifact = (
            dags[0].compile_artifact()
            if len(dags) == 1
            else coker_compiler.compile_module(dags)
        )

    @property
    def artifact(self):
        if self._artifact is None:
            self._compile()
        return self._artifact.to_bytes()

    def __call__(self, inputs):
        arrays = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
        if len(arrays) != len(self.function.tape.input_indicies):
            raise ValueError("input count does not match function signature")
        bound_functions = {
            node: value
            for node, value in zip(
                self.function.tape.input_indicies, arrays, strict=True
            )
            if isinstance(self.function.tape.dim[node], FunctionSpace)
            and isinstance(value, Function)
        }
        expected_bound = sum(
            isinstance(self.function.tape.dim[node], FunctionSpace)
            for node in self.function.tape.input_indicies
        )
        if len(bound_functions) != expected_bound:
            raise TypeError("FunctionSpace inputs require concrete Functions")
        bound_key = tuple(
            (node, id(function))
            for node, function in sorted(bound_functions.items())
        )
        if self._bound_key != bound_key:
            self._bound_functions = bound_functions
            self._bound_key = bound_key
            self._artifact = None
        if all(
            output is None
            or (
                isinstance(output, Tracer)
                and not output.dim.is_scalar()
                and int(np.prod(output.dim.dim)) == 0
            )
            for output in self.function.output
        ):
            return [None] * len(self.function.output)
        if any(
            isinstance(raw, tuple)
            and isinstance(raw[0], ModuleCallOP)
            and not hasattr(raw[0].module, "tape")
            for raw in self.function.tape.nodes
        ):
            # QP modules remain host-side callables until their bytecode path
            # is migrated. Evaluate the enclosing ordinary tape through the
            # established NumPy interpreter rather than treating the module
            # as a TypedDag callee.
            return Function(
                self.function.tape, self.function.output, backend="numpy"
            )(inputs)
        if self._artifact is None:
            self._compile()
        arrays = tuple(
            value
            for node, value in zip(
                self.function.tape.input_indicies, arrays, strict=True
            )
            if not isinstance(self.function.tape.dim[node], FunctionSpace)
        )
        flat = [
            np.asarray(value, dtype=np.float32).reshape(-1).tolist()
            for value in arrays
        ]
        return self._restore_outputs(self._artifact.execute(flat))

    def push_forward(self, *tangent_spaces):
        """Execute the mapped primal and forward tangent programs."""
        input_count = len(self.function.tape.input_indicies)
        inputs = tangent_spaces[:input_count]
        tangents = tangent_spaces[input_count:]
        if len(inputs) != input_count or len(tangents) != input_count:
            raise ValueError(
                "push-forward requires one primal and tangent per input"
            )
        self(inputs)
        flat_inputs = [
            np.asarray(value, dtype=np.float32).reshape(-1).tolist()
            for value in inputs
        ]
        flat_tangents = [
            np.asarray(value, dtype=np.float32).reshape(-1).tolist()
            for value in tangents
        ]
        values, tangent_values = self._artifact.push_forward(
            flat_inputs, flat_tangents
        )
        return (
            self._restore_outputs(values),
            self._restore_outputs(tangent_values),
        )

    def _restore_outputs(self, result):
        outputs = []
        cursor = 0
        for output in self.function.output:
            if output is None or (
                not output.dim.is_scalar()
                and int(np.prod(output.dim.dim)) == 0
            ):
                outputs.append(None)
                continue
            size = (
                int(np.prod(output.dim.dim))
                if not output.dim.is_scalar()
                else 1
            )
            values = np.asarray(result[cursor : cursor + size], dtype=float)
            outputs.append(
                values.reshape(output.dim.dim)
                if not output.dim.is_scalar()
                else float(values[0])
            )
            cursor += size
        return outputs


class CokerBackend(Backend):
    def __init__(self):
        pass

    def to_backend_array(self, array: ArrayLike):
        pass

    def to_numpy_array(self, array) -> ArrayLike:
        pass

    def native_types(self) -> Tuple[Type]:
        pass

    def call(self, op, *args) -> ArrayLike:
        pass

    def evaluate(self, function, inputs: ArrayLike):
        if all(output is None for output in function.output):
            return function.output
        return CokerFunction(function)(inputs)

    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        raise NotImplementedError

    def lower(self, function: Function):
        return CokerFunction(function)

    def build_optimisation_problem(
        self,
        cost: Tracer,
        constraints: List[Tracer],
        parameters: List[Tracer],
        outputs: List[Tracer],
        initial_conditions: Dict[int, ArrayLike],
    ):
        return build_qp_optimisation_problem(
            self,
            cost,
            constraints,
            parameters,
            outputs,
            initial_conditions,
        )

    def make_optimisation_module(self, implementation):
        """Wrap a prebuilt QP solver for numerical module composition."""
        return CokerModule(implementation)
