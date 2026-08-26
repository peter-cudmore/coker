"""Coker backend facade and lowered-function handle."""

from typing import Dict, List, Tuple, Type

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ModuleCallOP
from coker.backends.backend import ArrayLike, Backend, get_backend_by_name
from coker.backends.coker.lowering import create_opgraph
from coker.backends.coker.module import CokerModule
from coker.backends.coker.optimisation import (
    build_optimisation_problem as build_qp_optimisation_problem,
)


class CokerFunction:
    """A Coker-lowered function graph and its backend-specific operations."""

    def __init__(self, function: Function):
        self.function = function
        self._graph = None

    @property
    def graph(self):
        """Build the Coker graph only when compilation needs it."""
        if self._graph is None:
            self._graph = create_opgraph(self.function)
        return self._graph

    @property
    def function_id(self) -> int:
        """Return the graph's stable function-table identifier."""
        return self.graph.function_id

    def __call__(self, inputs):
        """Evaluate through the reference interpreter on the Python host."""
        numpy_backend = get_backend_by_name("numpy", set_current=False)
        return numpy_backend.evaluate(self.function, inputs)

    def compile_bytecode(self) -> bytes:
        """Compile this lowered graph into a mapped Coker bytecode module."""
        from coker.backends.coker.runtime import CompiledGraph

        return bytes(CompiledGraph.compile(self.graph).program)

    def compile_artifact(
        self, *, name: str = "coker_function", version: str = "1"
    ):
        """Compile this lowered Coker function into a mapped artifact."""
        from coker.backends.coker.artifacts import _compile_artifact

        return _compile_artifact(self, name=name, version=version)

    def export_payload(self) -> dict[str, object]:
        """Return the deterministic graph payload consumed by the compiler."""
        return self.graph.export_payload()


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

        if any(
            isinstance(function.tape.dim[input_index], FunctionSpace)
            for input_index in function.tape.input_indicies
        ):
            numpy_backend = get_backend_by_name("numpy", set_current=False)
            return numpy_backend.evaluate(function, inputs)

        graph = create_opgraph(function)
        return [graph(*inputs)]

    def reshape(self, array: ArrayLike, shape: Tuple[int, ...]) -> ArrayLike:
        raise NotImplementedError

    def lower(self, function: Function):
        if (
            any(
                isinstance(function.tape.nodes[index][0], ModuleCallOP)
                for index in range(len(function.tape.nodes))
                if index not in function.tape.input_indicies
            )
            or any(
                isinstance(function.tape.dim[input_index], FunctionSpace)
                for input_index in function.tape.input_indicies
            )
            or any(output is None for output in function.output)
        ):
            numpy_backend = get_backend_by_name("numpy", set_current=False)
            return numpy_backend.lower(function)

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
