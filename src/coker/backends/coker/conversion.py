"""Conversion from traced Coker tapes to Rust typed DAGs."""

import numpy as np

from coker.algebra.dimensions import FunctionSpace
from coker.algebra.kernel import Function, Tracer

from coker.algebra.ops import ConcatenateOP, ModuleCallOP, OP, ReshapeOP

__all__ = ["function_to_typed_dag"]


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
    """Convert a traced function tape into the Rust
    typed-DAG representation."""
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
                    input_shapes = [_shape(tape.dim[a.index]) for a in args]
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
                                "evaluate node is missing callee "
                                "function reference"
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
