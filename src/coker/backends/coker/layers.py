from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from coker.algebra.dimensions import Dimension
from coker.algebra.kernel import Function, scalar_types
from coker.algebra.ops import OP, ConcatenateOP, ReshapeOP
from coker.backends import get_backend_by_name
from coker.backends.coker.memory import MemorySpec
from coker.backends.coker.sparse_tensor import dok_ndarray
from coker.backends.coker.weights import BilinearWeights

IDENTITY_OP = "identity"
CONSTANT_OP = "constant"
OPAQUE_OP = "opaque"
UNUSED_REF = -(1 << 30)


StoredConstant = Union[float, int, bool, dok_ndarray, Function, object]


@dataclass(frozen=True)
class WorkspaceOperand:
    spec: MemorySpec
    shape: Tuple[int, ...]

    def to_export_dict(self):
        return {
            "kind": "workspace",
            "memory": self.spec.to_export_dict(),
            "shape": _export_shape(self.shape),
        }


@dataclass(frozen=True)
class ConstantOperand:
    value: StoredConstant
    shape: Tuple[int, ...] | None

    def to_export_dict(self):
        return {
            "kind": "constant",
            "shape": _export_shape(self.shape),
            "value": _export_constant_value(self.value, self.shape),
        }


@dataclass(frozen=True)
class OpaqueProgram:
    row_start: int
    shape: Tuple[int, ...]
    op: object
    operands: Tuple[Union[WorkspaceOperand, ConstantOperand], ...]

    @property
    def row_count(self) -> int:
        return int(np.prod(self.shape))

    def to_export_dict(self):
        return {
            "row_start": int(self.row_start),
            "shape": _export_shape(self.shape),
            "op": _export_operator(self.op),
            "operands": [
                operand.to_export_dict() for operand in self.operands
            ],
        }


ArrayOperand = Union[WorkspaceOperand, ConstantOperand]


def _export_shape(shape):
    if shape is None:
        return None
    return [int(size) for size in shape]


def _export_operator(op):
    if isinstance(op, OP):
        return {"kind": "enum", "value": op.name}
    if op in {IDENTITY_OP, CONSTANT_OP, OPAQUE_OP}:
        return {"kind": "internal", "value": op}
    if isinstance(op, ConcatenateOP):
        return {"kind": "concatenate", "axis": int(op.axis)}
    if isinstance(op, ReshapeOP):
        return {
            "kind": "reshape",
            "shape": [int(size) for size in op.newshape],
            "order": op.order,
        }
    raise TypeError(f"Unsupported export operator {op!r}")


def _export_constant_value(value, shape):
    if isinstance(value, dok_ndarray):
        return {"kind": "tensor", "value": value.to_export_dict()}
    if isinstance(value, Function):
        from coker.backends.coker.core import create_opgraph

        return {
            "kind": "function",
            "value": create_opgraph(value).export_payload(),
        }
    if isinstance(value, (float, int, bool, np.bool_)):
        return {"kind": "scalar", "value": float(value)}
    raise TypeError(
        f"Unsupported constant operand {type(value)} for shape {shape}"
    )


def vec(item):
    if isinstance(item, list):
        item = np.array(item)
    if isinstance(item, np.ndarray):
        return np.asarray(item).reshape(-1, order="C")
    if isinstance(item, scalar_types):
        return np.array([item])
    raise NotImplementedError(type(item))


class InputLayer:
    def __init__(self):
        self.input_specs: List[Tuple[MemorySpec, Tuple[int, ...]]] = []
        self.dimension = 0

    def add_input(self, dim: Dimension) -> int:
        idx = len(self.input_specs)
        shape = dim.shape
        count = dim.flat()
        spec = MemorySpec(self.dimension, count)
        self.input_specs.append((spec, shape))
        self.dimension += count
        return idx

    def get_projection(self, arg: int):
        spec, shape = self.input_specs[arg]
        m = dok_ndarray((*shape, self.dimension))
        for offset in range(spec.count):
            idx = np.unravel_index(offset, shape, order="C")
            m[(*idx, spec.location + offset)] = 1
        return m

    def __call__(self, *args):
        assert len(args) == len(self.input_specs)
        if not args:
            return np.zeros((0,), dtype=float)
        return np.concatenate([vec(a) for a in args])

    def forwards(self, *tangent_space, y=None):
        n_args = len(self.input_specs)
        assert len(tangent_space) == 2 * n_args
        dx = tangent_space[n_args : 2 * n_args]
        if not dx:
            return np.zeros((0,), dtype=float)
        return np.concatenate([vec(dx_i) for dx_i in dx])

    def to_export_dict(self):
        return {
            "dimension": self.dimension,
            "inputs": [
                {
                    "memory": spec.to_export_dict(),
                }
                for spec, _shape in self.input_specs
            ],
        }


def scalar_divide(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(np.divide(num, den))


def to_float(x):
    if isinstance(x, (float, int, bool, np.bool_)):
        return float(x)

    assert isinstance(x, np.ndarray)
    x_out = x.reshape(-1, order="C")
    assert x_out.shape == (1,)
    value = x_out[0]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return float(value)


class OutputLayer:
    def __init__(self):
        self.outputs: List[Tuple[MemorySpec, Dimension]] = []

    def inputs(self):
        return [memory for memory, _ in self.outputs]

    def add_output(self, memory: MemorySpec, shape: Dimension):
        self.outputs.append((memory, shape))

    def call(self, workspace: np.ndarray):
        result = []
        for memory, shape in self.outputs:
            raw = workspace[memory.location : memory.location + memory.count]
            if shape.dim is None:
                result.append(to_float(raw))
            else:
                result.append(np.reshape(raw, shape.dim, order="C"))
        if len(result) == 1:
            return result[0]
        return result

    def to_export_dict(self):
        return {
            "outputs": [
                {
                    "memory": memory.to_export_dict(),
                }
                for memory, _shape in self.outputs
            ]
        }


class BilinearWorkspaceLayer:
    def __init__(
        self,
        memory_in: MemorySpec,
        memory_out: MemorySpec,
        weights: BilinearWeights,
    ):
        self.memory_in = memory_in
        self.memory_out = memory_out
        self.weights = weights

    def inputs(self) -> List[MemorySpec]:
        return [self.memory_in]

    def outputs(self) -> List[MemorySpec]:
        return [self.memory_out]

    def __call__(self, workspace: np.ndarray) -> np.ndarray:
        return np.asarray(self.weights(workspace)).reshape(-1, order="C")

    def push_forward(self, workspace: np.ndarray, dworkspace: np.ndarray):
        y, dy = self.weights.push_forwards(workspace, dworkspace)
        return (
            np.asarray(y).reshape(-1, order="C"),
            np.asarray(dy).reshape(-1, order="C"),
        )

    def to_export_dict(self):
        return {
            "kind": "bilinear",
            "memory_in": self.memory_in.to_export_dict(),
            "memory_out": self.memory_out.to_export_dict(),
            "weights": self.weights.to_export_dict(),
        }


class FunctionEvaluationLayer:
    def __init__(
        self,
        memory_in: MemorySpec,
        memory_out: MemorySpec,
        input_bindings: Sequence[ArrayOperand],
        output_bindings: Sequence[MemorySpec],
        callee_graph,
        callee_function_id: int | None = None,
    ):
        self.memory_in = memory_in
        self.memory_out = memory_out
        self.input_bindings = list(input_bindings)
        self.output_bindings = list(output_bindings)
        self.callee_graph = callee_graph
        self.callee_function_id = callee_function_id

    def inputs(self) -> List[MemorySpec]:
        return [self.memory_in]

    def outputs(self) -> List[MemorySpec]:
        return [self.memory_out]

    def __call__(self, workspace: np.ndarray) -> np.ndarray:
        workspace = np.asarray(workspace).reshape(-1, order="C")
        next_workspace = np.zeros(self.memory_out.count, dtype=float)
        next_workspace[: self.memory_in.count] = workspace[
            : self.memory_in.count
        ]
        inputs = [
            self._materialize_value(binding, workspace)
            for binding in self.input_bindings
        ]
        outputs = self._normalize_outputs(self.callee_graph(*inputs))
        self._write_outputs(next_workspace, outputs)
        return next_workspace

    def push_forward(self, workspace: np.ndarray, dworkspace: np.ndarray):
        workspace = np.asarray(workspace).reshape(-1, order="C")
        dworkspace = np.asarray(dworkspace).reshape(-1, order="C")
        next_workspace = np.zeros(self.memory_out.count, dtype=float)
        next_dworkspace = np.zeros(self.memory_out.count, dtype=float)
        next_workspace[: self.memory_in.count] = workspace[
            : self.memory_in.count
        ]
        next_dworkspace[: self.memory_in.count] = dworkspace[
            : self.memory_in.count
        ]
        inputs = [
            self._materialize_value(binding, workspace)
            for binding in self.input_bindings
        ]
        tangents = [
            self._materialize_tangent(binding, dworkspace)
            for binding in self.input_bindings
        ]
        outputs, tangent_outputs = self.callee_graph.push_forward(
            *inputs, *tangents
        )
        self._write_outputs(next_workspace, self._normalize_outputs(outputs))
        self._write_outputs(
            next_dworkspace, self._normalize_outputs(tangent_outputs)
        )
        return next_workspace, next_dworkspace

    def to_export_dict(self):
        if self.callee_function_id is None:
            raise ValueError(
                "callee_function_id must be assigned before export"
            )
        return {
            "kind": "evaluate",
            "memory_in": self.memory_in.to_export_dict(),
            "memory_out": self.memory_out.to_export_dict(),
            "callee_function_id": int(self.callee_function_id),
            "inputs": [
                self._export_input_binding(binding)
                for binding in self.input_bindings
            ],
            "outputs": [
                {
                    "destination_offset": int(binding.location),
                    "length": int(binding.count),
                }
                for binding in self.output_bindings
            ],
        }

    def _materialize_value(self, binding: ArrayOperand, workspace: np.ndarray):
        if isinstance(binding, WorkspaceOperand):
            raw = workspace[
                binding.spec.location : binding.spec.location
                + binding.spec.count
            ]
            return np.reshape(raw, binding.shape, order="C")
        if isinstance(binding.value, dok_ndarray):
            return np.reshape(
                binding.value.toarray(), binding.shape, order="C"
            )
        return binding.value

    def _materialize_tangent(
        self, binding: ArrayOperand, dworkspace: np.ndarray
    ):
        if isinstance(binding, WorkspaceOperand):
            raw = dworkspace[
                binding.spec.location : binding.spec.location
                + binding.spec.count
            ]
            return np.reshape(raw, binding.shape, order="C")
        if isinstance(binding.value, dok_ndarray):
            return np.zeros(binding.shape, dtype=float)
        return 0.0

    def _normalize_outputs(self, outputs) -> List[object]:
        if len(self.output_bindings) == 1:
            return [outputs]
        assert isinstance(outputs, list)
        return outputs

    def _write_outputs(
        self, destination_workspace: np.ndarray, outputs: Sequence[object]
    ):
        assert len(outputs) == len(self.output_bindings)
        for binding, output_value in zip(
            self.output_bindings, outputs, strict=False
        ):
            flat_output = vec(output_value).astype(float, copy=False)
            assert flat_output.shape == (binding.count,)
            start = binding.location
            stop = start + binding.count
            destination_workspace[start:stop] = flat_output

    def _export_input_binding(self, binding: ArrayOperand):
        if isinstance(binding, WorkspaceOperand):
            return {
                "kind": "workspace",
                "offset": int(binding.spec.location),
                "length": int(binding.spec.count),
            }
        values = self._flatten_constant_binding(binding)
        return {
            "kind": "constant",
            "length": len(values),
            "values": values,
        }

    def _flatten_constant_binding(
        self, binding: ConstantOperand
    ) -> List[float]:
        value = binding.value
        if isinstance(value, dok_ndarray):
            array = value.toarray()
            return (
                np.asarray(array, dtype=float).reshape(-1, order="C").tolist()
            )
        if isinstance(value, np.ndarray):
            return (
                np.asarray(value, dtype=float).reshape(-1, order="C").tolist()
            )
        if isinstance(value, (float, int, bool, np.bool_)):
            return [float(value)]
        raise TypeError(f"Unsupported evaluate constant operand {type(value)}")


class GenericVectorLayer:
    def __init__(
        self,
        memory_in: MemorySpec,
        memory_out: MemorySpec,
        ops: Sequence[Tuple[object, int, int, int]],
        constants: Dict[int, float] | None = None,
        opaque_programs: Sequence[OpaqueProgram] | None = None,
    ):
        self.memory_in = memory_in
        self.memory_out = memory_out
        self.ops = list(ops)
        self.constants = constants or {}
        self.opaque_programs = list(opaque_programs or [])

    def inputs(self) -> List[MemorySpec]:
        return [self.memory_in]

    def outputs(self) -> List[MemorySpec]:
        return [self.memory_out]

    def __call__(self, workspace: np.ndarray) -> np.ndarray:
        workspace = np.asarray(workspace).reshape(-1, order="C")
        values = np.empty(self.memory_out.count, dtype=float)
        opaque_rows = {
            row
            for program in self.opaque_programs
            for row in range(
                program.row_start, program.row_start + program.row_count
            )
        }
        for row, (op, first, second, third) in enumerate(self.ops):
            if row in opaque_rows:
                continue
            values[row] = self._eval_scalar_value(
                op, first, second, third, workspace
            )
        for program in self.opaque_programs:
            flat = np.asarray(
                self._eval_opaque_value(program, workspace)
            ).reshape(-1, order="C")
            start = program.row_start
            stop = start + program.row_count
            values[start:stop] = flat
        return values

    def push_forward(self, workspace: np.ndarray, dworkspace: np.ndarray):
        workspace = np.asarray(workspace).reshape(-1, order="C")
        dworkspace = np.asarray(dworkspace).reshape(-1, order="C")
        values = np.empty(self.memory_out.count, dtype=float)
        tangents = np.zeros(self.memory_out.count, dtype=float)

        opaque_rows = {
            row
            for program in self.opaque_programs
            for row in range(
                program.row_start, program.row_start + program.row_count
            )
        }

        for row, (op, first, second, third) in enumerate(self.ops):
            if row in opaque_rows:
                continue
            values[row], tangents[row] = self._eval_scalar_row(
                op, first, second, third, workspace, dworkspace
            )

        for program in self.opaque_programs:
            result, dresult = self._eval_opaque_program(
                program, workspace, dworkspace
            )
            flat = np.asarray(result).reshape(-1, order="C")
            dflat = np.asarray(dresult).reshape(-1, order="C")
            start = program.row_start
            stop = start + program.row_count
            values[start:stop] = flat
            tangents[start:stop] = dflat

        return values, tangents

    def to_export_dict(self):
        if self.constants:
            raise NotImplementedError(
                "Generic constants must be folded into bilinear "
                "layers before export"
            )
        if self.opaque_programs:
            raise NotImplementedError(
                "Function evaluation and opaque programs are "
                "not yet exportable"
            )
        return {
            "kind": "generic",
            "memory_in": self.memory_in.to_export_dict(),
            "memory_out": self.memory_out.to_export_dict(),
            "ops": [
                {
                    "op": _export_operator(op),
                    "first": first,
                    "second": second,
                    "third": third,
                }
                for op, first, second, third in self.ops
            ],
        }

    def _resolve_scalar(self, index: int, workspace: np.ndarray) -> float:
        if index >= 0:
            return float(workspace[index])
        return float(self.constants[index])

    def _resolve_tangent(self, index: int, dworkspace: np.ndarray) -> float:
        if index >= 0:
            return float(dworkspace[index])
        return 0.0

    def _eval_scalar_value(self, op, first, second, third, workspace):
        a = (
            self._resolve_scalar(first, workspace)
            if first != UNUSED_REF
            else None
        )
        b = (
            self._resolve_scalar(second, workspace)
            if second != UNUSED_REF
            else None
        )
        c = (
            self._resolve_scalar(third, workspace)
            if third != UNUSED_REF
            else None
        )

        if op == IDENTITY_OP or op == CONSTANT_OP:
            return a
        if op == OP.SIN:
            return np.sin(a)
        if op == OP.COS:
            return np.cos(a)
        if op == OP.TAN:
            return np.tan(a)
        if op == OP.EXP:
            return np.exp(a)
        if op == OP.SQRT:
            return np.sqrt(a)
        if op == OP.LOG:
            return np.log(a)
        if op == OP.NEG:
            return -a
        if op == OP.ABS:
            return np.abs(a)
        if op == OP.ADD:
            return a + b
        if op == OP.SUB:
            return a - b
        if op == OP.MUL:
            return a * b
        if op == OP.DIV:
            return scalar_divide(a, b)
        if op == OP.PWR or op == OP.INT_PWR:
            return np.power(a, b)
        if op == OP.ARCTAN2:
            return np.arctan2(a, b)
        if op == OP.EQUAL:
            return float(a == b)
        if op == OP.LESS_THAN:
            return float(a < b)
        if op == OP.LESS_EQUAL:
            return float(a <= b)
        if op == OP.CASE:
            return b if bool(a) else c
        raise NotImplementedError(f"Unsupported scalar op {op}")

    def _eval_scalar_row(
        self, op, first, second, third, workspace, dworkspace
    ):
        a = (
            self._resolve_scalar(first, workspace)
            if first != UNUSED_REF
            else None
        )
        b = (
            self._resolve_scalar(second, workspace)
            if second != UNUSED_REF
            else None
        )
        c = (
            self._resolve_scalar(third, workspace)
            if third != UNUSED_REF
            else None
        )
        da = (
            self._resolve_tangent(first, dworkspace)
            if first != UNUSED_REF
            else 0.0
        )
        db = (
            self._resolve_tangent(second, dworkspace)
            if second != UNUSED_REF
            else 0.0
        )
        dc = (
            self._resolve_tangent(third, dworkspace)
            if third != UNUSED_REF
            else 0.0
        )

        if op == IDENTITY_OP:
            return a, da
        if op == CONSTANT_OP:
            return a, 0.0
        if op == OP.SIN:
            return np.sin(a), np.cos(a) * da
        if op == OP.COS:
            return np.cos(a), -np.sin(a) * da
        if op == OP.TAN:
            return np.tan(a), da / (np.cos(a) ** 2)
        if op == OP.EXP:
            value = np.exp(a)
            return value, value * da
        if op == OP.SQRT:
            value = np.sqrt(a)
            return value, da / (2.0 * value)
        if op == OP.LOG:
            return np.log(a), da / a
        if op == OP.NEG:
            return -a, -da
        if op == OP.ABS:
            return np.abs(a), np.sign(a) * da
        if op == OP.ADD:
            return a + b, da + db
        if op == OP.SUB:
            return a - b, da - db
        if op == OP.MUL:
            return a * b, b * da + a * db
        if op == OP.DIV:
            if b == 0:
                return float("nan"), float("nan")
            return scalar_divide(a, b), scalar_divide(da * b - a * db, b * b)
        if op == OP.PWR or op == OP.INT_PWR:
            value = np.power(a, b)
            if a == 0:
                return value, 0.0
            return value, value * (db * np.log(a) + (b * da / a))
        if op == OP.ARCTAN2:
            denom = a * a + b * b
            return np.arctan2(a, b), (b * da - a * db) / denom
        if op == OP.EQUAL:
            return float(a == b), 0.0
        if op == OP.LESS_THAN:
            return float(a < b), 0.0
        if op == OP.LESS_EQUAL:
            return float(a <= b), 0.0
        if op == OP.CASE:
            if bool(a):
                return b, db
            return c, dc
        raise NotImplementedError(f"Unsupported scalar op {op}")

    def _materialize_operand(
        self, operand: ArrayOperand, workspace: np.ndarray
    ):
        if isinstance(operand, WorkspaceOperand):
            raw = workspace[
                operand.spec.location : operand.spec.location
                + operand.spec.count
            ]
            return np.reshape(raw, operand.shape, order="C")
        value = operand.value
        if isinstance(value, dok_ndarray):
            return np.reshape(value.toarray(), operand.shape, order="C")
        return value

    def _eval_opaque_value(
        self, program: OpaqueProgram, workspace: np.ndarray
    ):
        backend = get_backend_by_name("numpy", set_current=False)
        values = [
            self._materialize_operand(operand, workspace)
            for operand in program.operands
        ]
        return backend.call(program.op, *values)

    def _materialize_tangent(
        self, operand: ArrayOperand, dworkspace: np.ndarray
    ):
        if isinstance(operand, WorkspaceOperand):
            raw = dworkspace[
                operand.spec.location : operand.spec.location
                + operand.spec.count
            ]
            return np.reshape(raw, operand.shape, order="C")
        value = operand.value
        if isinstance(value, dok_ndarray):
            return np.zeros(operand.shape, dtype=float)
        if isinstance(value, (float, int, bool, np.bool_)):
            return 0.0
        if isinstance(value, Function):
            return None
        return 0.0

    def _eval_opaque_program(
        self,
        program: OpaqueProgram,
        workspace: np.ndarray,
        dworkspace: np.ndarray,
    ):
        backend = get_backend_by_name("numpy", set_current=False)
        values = [
            self._materialize_operand(operand, workspace)
            for operand in program.operands
        ]

        if program.op == OP.EVALUATE and isinstance(values[0], Function):
            from coker.backends.coker.core import create_opgraph

            graph = create_opgraph(values[0])
            tangents = [
                self._materialize_tangent(operand, dworkspace)
                for operand in program.operands[1:]
            ]
            result, dresult = graph.push_forward(*values[1:], *tangents)
            return result, dresult

        result = backend.call(program.op, *values)
        tangents = [
            self._materialize_tangent(operand, dworkspace)
            for operand in program.operands
        ]

        linear_like = program.op in {OP.TRANSPOSE}
        if linear_like:
            tangent = backend.call(program.op, *tangents)
            return result, tangent

        raise NotImplementedError(
            f"push_forward not implemented for opaque op {program.op}"
        )
