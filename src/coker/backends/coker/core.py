"""Coker backend facade and Rust-compiled function handle."""

from typing import Dict, List, Tuple, Type

import numpy as np

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Function, Tracer, TapeInner
from coker.algebra.ops import ModuleCallOP, OP
from coker.backends.backend import ArrayLike, Backend, get_backend_by_name
from coker.backends.coker.module import CokerModule
from coker.backends.coker.optimisation import (
    build_optimisation_problem as build_qp_optimisation_problem,
)


def _shape(dim) -> list[int]:
    return [] if dim.is_scalar() else [int(v) for v in dim.dim]


def _flatten_constant(value):
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

def _is_value(raw) -> bool:
    return isinstance(raw, tuple) and len(raw) == 2 and raw[0] is OP.VALUE


class CokerFunction:
    """A Python tape compiled and executed by the mapped Rust runtime."""

    def __init__(self, function: Function):
        self.function = function
        self._artifact = None
        self._program = None

    def _compile(self):
        try:
            import coker_compiler
        except ImportError as error:
            raise RuntimeError("coker_compiler extension is required") from error
        # Reserve each function's module ID before descending into callees.
        # This keeps the entry function at index zero even when its tape
        # contains nested calls, while allowing call nodes to refer to stable
        # IDs during recursive discovery.
        dag_ids, dags, building = {}, [], set()

        def build(function):
            key = id(function)
            if key in dag_ids:
                if key in building:
                    raise ValueError("recursive ordinary calls are unsupported")
                return dags[dag_ids[key]]
            function_id = len(dags)
            dag_ids[key] = function_id
            dags.append(None)
            building.add(key)
            tape = function.tape
            constants, refs = [], {}
            for index in range(len(tape.nodes)):
                raw = tape.nodes[index]
                if _is_value(raw):
                    refs[index] = len(constants)
                    constants.append((raw[1], _shape(tape.dim[index])))
            callees = []
            for index in range(len(tape.nodes)):
                raw = tape.nodes[index]
                if index not in tape.input_indicies and isinstance(raw[0], ModuleCallOP):
                    callees.append(raw[0].module)
            for callee in callees:
                build(callee)
            operands = sum(
                0 if i in tape.input_indicies or _is_value(tape.nodes[i])
                else len(tape.nodes[i]) - 1 for i in range(len(tape.nodes))
            )
            scalars = sum(len(_flatten_constant(v)) + len(s) for v, s in constants)
            scalars += sum(len(_shape(dim)) for dim in tape.dim)
            builder = coker_compiler.Builder(
                len(tape.nodes), operands, len(constants), scalars,
                len(tape.input_indicies), len(function.output), len(callees),
            )
            for value, shape in constants:
                builder.push_constant(_flatten_constant(value), shape)
            for index in range(len(tape.nodes)):
                raw = tape.nodes[index]
                if index in tape.input_indicies:
                    builder.push_node(index, OP.VALUE.value, [], _shape(tape.dim[index]))
                elif _is_value(raw):
                    builder.push_node(index, OP.VALUE.value, [], _shape(tape.dim[index]), refs[index])
                else:
                    operation, *args = raw
                    if isinstance(operation, ModuleCallOP):
                        tag = OP.EVALUATE.value
                        function_reference = dag_ids[id(operation.module)]
                    elif isinstance(operation, OP):
                        tag, function_reference = operation.value, None
                    else:
                        raise NotImplementedError(f"unsupported ordinary node {operation!r}")
                    operands_for_node = [a.index if isinstance(a, Tracer) else int(a) for a in args]
                    builder.push_node(index, tag, operands_for_node, _shape(tape.dim[index]), None, function_reference)
            for position, index in enumerate(tape.input_indicies):
                builder.push_input(tape.input_names[position], index)
            for position, output in enumerate(function.output):
                if not isinstance(output, Tracer):
                    raise NotImplementedError("ordinary outputs must be traced values")
                builder.push_output(str(position), output.index)
            dag = builder.finish_tape()
            dags[function_id] = dag
            building.remove(key)
            return dag
        build(self.function)
        self._artifact = (
            dags[0].compile_artifact() if len(dags) == 1
            else coker_compiler.compile_module(dags)
        )
        self._program = self._artifact

    @property
    def artifact(self):
        if self._program is None:
            self._compile()
        return self._artifact.to_bytes()

    def __call__(self, inputs):
        if self._program is None:
            self._compile()
        arrays = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
        flat = [np.asarray(value, dtype=np.float32).reshape(-1).tolist() for value in arrays]
        result = self._program.execute(flat)
        outputs = []
        cursor = 0
        for output in self.function.output:
            size = int(np.prod(output.dim.dim)) if not output.dim.is_scalar() else 1
            values = np.asarray(result[cursor:cursor + size], dtype=float)
            outputs.append(values.reshape(output.dim.dim) if not output.dim.is_scalar() else float(values[0]))
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
