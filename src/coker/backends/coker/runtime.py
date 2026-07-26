import json
from typing import Sequence

import numpy as np

from coker.backends.coker.ast_preprocessing import SparseNet
import coker._coker_runtime as coker_runtime


def _runtime_input_buffer(arg) -> np.ndarray:
    if isinstance(arg, (int, float, bool, np.bool_)):
        return np.asarray([arg], dtype=np.float32)
    if isinstance(arg, np.ndarray):
        return np.asarray(arg, dtype=np.float32).reshape(-1, order="C")
    raise TypeError(f"Unsupported runtime input {type(arg)}")


def _restore_output(flat_output: Sequence[float], shape):
    if shape is None:
        assert len(flat_output) == 1
        return float(flat_output[0])
    return np.asarray(flat_output, dtype=float).reshape(shape, order="C")




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
        output_length = sum(self._output_lengths)
        self._outputs = np.empty(output_length, dtype=np.float32)
        self._tangent_outputs = np.empty(output_length, dtype=np.float32)

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
        inputs = [_runtime_input_buffer(arg) for arg in args]
        self._runtime.execute_into(inputs, self._outputs)
        return self._restore_outputs(self._outputs)

    def push_forward(self, *tangent_spaces):
        n_args = len(self._input_lengths)
        x, dx = tangent_spaces[0:n_args], tangent_spaces[n_args:]
        assert len(x) == n_args
        assert len(dx) == n_args
        inputs = [_runtime_input_buffer(arg) for arg in x]
        tangents = [_runtime_input_buffer(arg) for arg in dx]
        self._runtime.push_forward_into(
            inputs,
            tangents,
            self._outputs,
            self._tangent_outputs,
        )
        return self._restore_outputs(self._outputs), self._restore_outputs(
            self._tangent_outputs
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
        self._output_length = int(self._info["output_spec"])
        self._solution = np.empty(self._output_length, dtype=np.float64)
        self._tangent_solution = np.empty(
            self._output_length, dtype=np.float64
        )

    @classmethod
    def compile(cls, extracted_qp) -> "RuntimeQpProgram":
        payload = json.dumps(extracted_qp.export_payload()).encode("utf-8")
        return cls(coker_runtime.compile_exported_qp(payload))

    def solve(self, runtime_args, *, warm_start):
        inputs = [_runtime_input_buffer(arg) for arg in runtime_args]
        initial = (
            None
            if warm_start is None
            else np.asarray(warm_start, dtype=np.float64).reshape(
                -1, order="C"
            )
        )
        success, status = self._runtime.solve_into(
            inputs,
            self._solution,
            initial,
        )
        from coker.optimisation import SolveInfo

        info = SolveInfo(
            backend="coker",
            solver="osqp",
            success=bool(success),
            return_status=str(status),
        )
        return self._solution.copy(), info

    def push_forward(self, *tangent_spaces):
        raise ValueError(
            "QP push-forward is unsupported: differentiated "
            "KKT solve support is not implemented"
        )
