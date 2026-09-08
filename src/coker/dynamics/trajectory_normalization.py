"""Backend-neutral lowering of builder trajectory expressions."""
from dataclasses import dataclass
from typing import Dict, Type

from coker.algebra.kernel import Function, FunctionSpace, Noop, Scalar, Tape, Tracer
from coker.algebra.ops import OP
from .types import TemporalBinding


class _StateSignal:
    pass


class _InputSignal:
    pass


class _OutputSignal:
    pass


@dataclass(frozen=True)
class _Site:
    expression: Function
    temporal_binding: TemporalBinding


@dataclass(frozen=True)
class _PathSite(_Site):
    pass


@dataclass(frozen=True)
class _InitialSite(_Site):
    pass


@dataclass(frozen=True)
class _TerminalSite(_Site):
    pass


def _site(expression, binding):
    return {_PathSite: _PathSite, _InitialSite: _InitialSite,
            _TerminalSite: _TerminalSite}[{
                TemporalBinding.PATH: _PathSite,
                TemporalBinding.INITIAL: _InitialSite,
                TemporalBinding.TERMINAL: _TerminalSite,
            }[binding]](expression, binding)


def normalize_expression(expression: Tracer, receiver_roles: Dict[int, Type],
                          state_space: FunctionSpace, input_space,
                          output_space: FunctionSpace, backend=None):
    """Lower a builder-tape expression to a fresh canonical function.

    Receiver evaluations of the private state/input/output signals become
    ordinary arguments of the resulting function.  The source tape is never
    retained by the returned function or site record.
    """
    if not isinstance(expression, Tracer):
        if isinstance(expression, Function):
            return expression, None
        raise TypeError("expression must be symbolic")
    source = expression.tape
    binding = _infer_binding(source, expression)
    # Keep the public six-channel ordering where channels exist.  Noop inputs
    # are represented by omitted arguments, matching Function's normal API.
    spaces = [Scalar("t"), state_space.output[0]]
    if state_space.arguments and state_space.arguments[0] is not None:
        pass
    zdim = _dimension_from_state(source, "z")
    if zdim is not None:
        spaces.append(zdim)
    if input_space is not None and not isinstance(input_space, Noop):
        spaces.append(input_space)
    spaces.append(Scalar("p"))
    spaces.append(Scalar("q"))
    target = Tape(backend)
    canonical = [target.input(s) for s in spaces]
    # state/input/output evaluations map to canonical channel arguments.
    role_args = {
        _StateSignal: canonical[1],
        _InputSignal: canonical[3 if zdim is not None else 2],
        _OutputSignal: canonical[1],
    }
    input_map = {}
    for idx, node_idx in enumerate(source.input_indicies):
        if node_idx >= 0 and idx < len(canonical):
            input_map[node_idx] = canonical[idx]

    memo = {}
    def copy(tracer):
        if not isinstance(tracer, Tracer):
            return tracer
        if tracer.tape is not source:
            raise ValueError("expression contains a foreign trace")
        if tracer.index in memo:
            return memo[tracer.index]
        if tracer.index in input_map:
            memo[tracer.index] = input_map[tracer.index]
            return memo[tracer.index]
        node = source.nodes[tracer.index]
        if isinstance(node, Tracer):
            result = copy(node)
        else:
            op, *args = node
            if op == OP.VALUE:
                result = target.insert_value(args[0][1] if isinstance(args[0], tuple) else args[0])
            elif op == OP.EVALUATE and isinstance(args[0], Tracer):
                role = receiver_roles.get(args[0].index)
                if role is None:
                    raise NotImplementedError("unsupported function evaluation")
                result = role_args[role]
            else:
                result = Tracer(target, target.append(op, *(copy(a) if isinstance(a, Tracer) else a for a in args)))
        memo[tracer.index] = result
        return result

    result = copy(expression)
    fn = Function(target, result, backend or source.backend)
    return fn, _site(fn, binding)


def _infer_binding(tape, expression):
    # Temporal markers are conventionally the first three scalar inputs.
    for idx in (1, 2):
        if idx < len(tape.input_indicies) and tape.depends_on(expression, Tracer(tape, tape.input_indicies[idx])):
            return TemporalBinding.TERMINAL if idx == 1 else TemporalBinding.INITIAL
    return TemporalBinding.PATH


def _dimension_from_state(tape, name):
    for i, value in enumerate(tape.list_inputs()):
        if getattr(value, "name", None) == name:
            return value
    return None
