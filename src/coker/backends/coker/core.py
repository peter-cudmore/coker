"""Coker backend facade and Rust-compiled function handle."""

from typing import Dict, List, Tuple, Type

import numpy as np

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Function, Tracer
from coker.algebra.ops import ConcatenateOP, ModuleCallOP, OP, ReshapeOP
from coker.backends.backend import ArrayLike, Backend
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


def function_to_typed_dag(
    function: Function,
    *,
    output_labels=None,
    bound_functions=None,
    return_all=False,
):
    """Convert a traced function tape into the Rust typed-DAG representation."""
    import coker_compiler

    if bound_functions is None:
        bound_functions = {}
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
        function_input_nodes = {
            index
            for index in tape.input_indicies
            if isinstance(tape.dim[index], FunctionSpace)
        }
        for index in range(len(tape.nodes)):
            raw = tape.nodes[index]
            if _is_value(raw):
                refs[index] = len(constants)
                constants.append((raw[1], _shape(tape.dim[index])))
            elif index in function_input_nodes:
                refs[index] = len(constants)
                constants.append((0.0, []))
        callees = []
        for index in range(len(tape.nodes)):
            raw = tape.nodes[index]
            if index not in tape.input_indicies and isinstance(
                raw[0], ModuleCallOP
            ):
                callees.append(raw[0].module)
        callees.extend(
            bound_functions[index] for index in function_input_nodes
        )
        for callee in callees:
            build(callee)
        operands = sum(
            (
                0
                if i in tape.input_indicies or _is_value(tape.nodes[i])
                else sum(
                    not (
                        isinstance(argument, Tracer)
                        and argument.index in function_input_nodes
                    )
                    for argument in tape.nodes[i][1:]
                )
            )
            for i in range(len(tape.nodes))
        )
        scalars = sum(len(_flatten_constant(v)) + len(s) for v, s in constants)
        scalars += sum(
            len(_shape(dim))
            for dim in tape.dim
            if not isinstance(dim, FunctionSpace)
        )
        output_nodes = [
            output for output in function.output if isinstance(output, Tracer)
        ]
        builder = coker_compiler.Builder(
            len(tape.nodes),
            operands,
            len(constants),
            scalars,
            len(tape.input_indicies) - len(function_input_nodes),
            len(output_nodes),
            len(callees),
        )
        for value, shape in constants:
            builder.push_constant(_flatten_constant(value), shape)
        for index in range(len(tape.nodes)):
            raw = tape.nodes[index]
            if index in tape.input_indicies:
                if index in function_input_nodes:
                    builder.push_node(
                        index, OP.VALUE.value, [], [], refs[index]
                    )
                else:
                    builder.push_node(
                        index, OP.VALUE.value, [], _shape(tape.dim[index])
                    )
            elif _is_value(raw):
                builder.push_node(
                    index,
                    OP.VALUE.value,
                    [],
                    _shape(tape.dim[index]),
                    refs[index],
                )
            else:
                operation, *args = raw
                mapping = None
                if isinstance(operation, ModuleCallOP):
                    tag = OP.EVALUATE.value
                    function_reference = dag_ids[id(operation.module)]
                elif isinstance(operation, ReshapeOP):
                    tag = OP.NEG.value
                    function_reference = None
                    source_shape = _shape(tape.dim[args[0].index])
                    target_shape = _shape(tape.dim[index])
                    mapping = (
                        "reshape",
                        [
                            len(source_shape),
                            *source_shape,
                            len(target_shape),
                            *target_shape,
                        ],
                    )
                elif isinstance(operation, ConcatenateOP):
                    tag = OP.NEG.value
                    function_reference = None
                    input_shapes = [
                        _shape(tape.dim[argument.index]) for argument in args
                    ]
                    rank = len(_shape(tape.dim[index]))
                    parameters = [operation.axis, rank, len(input_shapes)]
                    for input_shape in input_shapes:
                        parameters.extend([len(input_shape), *input_shape])
                    mapping = ("concatenate", parameters)
                elif isinstance(operation, OP):
                    tag = operation.value
                    function_reference = None
                    if operation is OP.EVALUATE:
                        function_input = next(
                            (
                                argument.index
                                for argument in args
                                if isinstance(argument, Tracer)
                                and argument.index in function_input_nodes
                            ),
                            None,
                        )
                        if function_input is None:
                            raise NotImplementedError(
                                "evaluate node is missing callee function reference"
                            )
                        function_reference = dag_ids[
                            id(bound_functions[function_input])
                        ]
                else:
                    raise NotImplementedError(
                        f"unsupported ordinary node {operation!r}"
                    )
                operands_for_node = [
                    a.index
                    for a in args
                    if isinstance(a, Tracer)
                    and a.index not in function_input_nodes
                ]
                builder.push_node(
                    index,
                    tag,
                    operands_for_node,
                    _shape(tape.dim[index]),
                    None,
                    function_reference,
                )
                if mapping is not None:
                    builder.push_mapping(index, *mapping)
        for position, index in enumerate(tape.input_indicies):
            if index not in function_input_nodes:
                builder.push_input(tape.input_names[position], index)
        for position, output in enumerate(function.output):
            if isinstance(output, Tracer) and (
                output.dim.is_scalar() or int(np.prod(output.dim.dim)) != 0
            ):
                label = (
                    str(position)
                    if output_labels is None
                    else output_labels[position]
                )
                builder.push_output(label, output.index)
        dag = builder.finish_tape()
        dags[function_id] = dag
        building.remove(key)
        return dag

    build(function)
    return tuple(dags) if return_all else dags[0]


class CokerFunction:
    """A Python tape compiled and executed by the mapped Rust runtime."""

    def __init__(self, function: Function, *, output_labels=None):
        self.function = function
        self._output_labels = output_labels
        self._artifact = None
        self._program = None
        self._bound_functions = {}
        self._bound_key = None
        self._dags = None

    def _compile(self):
        try:
            import coker_compiler
        except ImportError as error:
            raise RuntimeError(
                "coker_compiler extension is required"
            ) from error
        self._dags = function_to_typed_dag(
            self.function,
            output_labels=self._output_labels,
            bound_functions=self._bound_functions,
            return_all=True,
        )
        self._artifact = (
            self._dags[0].compile_artifact()
            if len(self._dags) == 1
            else coker_compiler.compile_module(self._dags)
        )
        self._program = self._artifact

    @property
    def artifact(self):
        if self._program is None:
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
            self._program = None
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
        if self._program is None:
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
        return self._restore_outputs(self._program.execute(flat))

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
        values, tangent_values = self._program.push_forward(
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
