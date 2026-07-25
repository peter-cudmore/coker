import json
from typing import List, Sequence

import numpy as np

from coker.backends.coker.ast_preprocessing import SparseNet
import coker._coker_runtime as coker_runtime


def _flatten_input(arg) -> List[float]:
    if isinstance(arg, (int, float, bool, np.bool_)):
        return [float(arg)]
    if isinstance(arg, np.ndarray):
        return np.asarray(arg, dtype=float).reshape(-1, order="C").tolist()
    raise TypeError(f"Unsupported runtime input {type(arg)}")


def _restore_output(flat_output: Sequence[float], shape):
    if shape is None:
        assert len(flat_output) == 1
        return float(flat_output[0])
    return np.asarray(flat_output, dtype=float).reshape(shape, order="C")


def _aligned_u8_buffer(size: int, alignment: int):
    if size < 0:
        raise ValueError(f"buffer size must be nonnegative, got {size}")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(
            "buffer alignment must be a positive power of two, got "
            f"{alignment}"
        )
    if size == 0:
        backing = np.empty(0, dtype=np.uint8)
        return backing, backing
    backing = np.empty(size + alignment - 1, dtype=np.uint8)
    offset = (-int(backing.ctypes.data)) % alignment
    aligned = backing[offset : offset + size]
    assert int(aligned.ctypes.data) % alignment == 0
    return backing, aligned


class CompiledGraph:

    def __init__(
        self,
        program: bytes,
        input_shapes: Sequence[tuple[int, ...] | None] | None = None,
        output_shapes: Sequence[tuple[int, ...] | None] | None = None,
    ):
        self.program = program
        self._runtime = coker_runtime.load_program(self.program)
        self._info = self._runtime.info()
        self._input_lengths = list(self._info["input_specs"])
        self._output_lengths = list(self._info["output_specs"])
        self._input_shapes = list(
            input_shapes or [None] * len(self._input_lengths)
        )
        self._output_shapes = list(
            output_shapes or [None] * len(self._output_lengths)
        )

    @staticmethod
    def compile(graph: SparseNet) -> "CompiledGraph":
        payload = json.dumps(graph.export_payload()).encode("utf-8")
        program = coker_runtime.compile_exported_graph(payload)
        input_shapes = [
            shape for _spec, shape in graph.input_layer.input_specs
        ]
        output_shapes = [
            shape.dim for _memory, shape in graph.output_layer.outputs
        ]
        return CompiledGraph(
            program,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
        )

    def __call__(self, *args):
        assert len(args) == len(self._input_lengths)
        flat_inputs = [_flatten_input(arg) for arg in args]
        outputs = self._runtime.execute(flat_inputs)
        return self._restore_outputs(outputs)

    def push_forward(self, *tangent_spaces):
        n_args = len(self._input_lengths)
        x, dx = tangent_spaces[0:n_args], tangent_spaces[n_args:]
        assert len(x) == n_args
        assert len(dx) == n_args
        flat_inputs = [_flatten_input(arg) for arg in x]
        flat_tangents = [_flatten_input(arg) for arg in dx]
        outputs, tangent_outputs = self._runtime.push_forward(
            flat_inputs, flat_tangents
        )
        return self._restore_outputs(outputs), self._restore_outputs(
            tangent_outputs
        )

    def _restore_outputs(self, flat_outputs):
        restored = []
        offset = 0
        for output_length, shape in zip(
            self._output_lengths, self._output_shapes, strict=False
        ):
            next_offset = offset + output_length
            restored.append(
                _restore_output(flat_outputs[offset:next_offset], shape)
            )
            offset = next_offset
        if len(restored) == 1:
            return restored[0]
        return restored


class RuntimeQpProgram:
    def __init__(self, program: bytes):
        self.program = bytes(program)
        self._runtime = coker_runtime.load_qp_program(self.program)
        self._info = self._runtime.info()
        self._input_lengths = list(self._info["input_specs"])
        requirements = self._runtime.workspace_requirements()
        self._tangent_workspace_size = int(
            requirements.get("tangent_workspace_size", 0)
        )
        self._arena_backing, self._arena = _aligned_u8_buffer(
            int(requirements["arena_bytes"]),
            int(requirements["arena_alignment"]),
        )
        self._evaluator_workspace = np.zeros(
            int(requirements["evaluator_workspace_size"]),
            dtype=np.float32,
        )
        self._coefficient_outputs = np.zeros(
            int(requirements["coefficient_output_size"]),
            dtype=np.float32,
        )
        self._tangent_evaluator_workspace = np.zeros_like(
            self._evaluator_workspace
        )
        self._tangent_coefficient_outputs = np.zeros_like(
            self._coefficient_outputs
        )
        self._solution_tangent_workspace = np.zeros(
            self._tangent_workspace_size,
            dtype=np.float32,
        )

    @classmethod
    def compile(cls, extracted_qp) -> "RuntimeQpProgram":
        payload = json.dumps(extracted_qp.export_payload()).encode("utf-8")
        return cls(coker_runtime.compile_exported_qp(payload))

    def solve(self, runtime_args, *, warm_start):
        inputs = [_flatten_input(arg) for arg in runtime_args]
        initial = (
            None
            if warm_start is None
            else np.asarray(warm_start, dtype=float)
            .reshape(-1, order="C")
            .tolist()
        )
        solution, success, status = self._runtime.solve(
            inputs,
            self._arena,
            self._evaluator_workspace,
            self._coefficient_outputs,
            initial,
        )
        from coker.optimisation import SolveInfo

        info = SolveInfo(
            backend="coker",
            solver="osqp",
            success=bool(success),
            return_status=str(status),
        )
        return np.asarray(solution, dtype=float), info

    def push_forward(self, *tangent_spaces):
        n_args = len(self._input_lengths)
        x, dx = tangent_spaces[0:n_args], tangent_spaces[n_args:]
        assert len(x) == n_args
        assert len(dx) == n_args
        flat_inputs = [_flatten_input(arg) for arg in x]
        flat_tangents = [_flatten_input(arg) for arg in dx]
        outputs, tangent_outputs = self._runtime.push_forward(
            flat_inputs,
            flat_tangents,
            self._arena,
            self._evaluator_workspace,
            self._coefficient_outputs,
            self._tangent_evaluator_workspace,
            self._tangent_coefficient_outputs,
            self._solution_tangent_workspace,
        )
        return np.asarray(outputs, dtype=float), np.asarray(
            tangent_outputs, dtype=float
        )
