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


class CompiledGraph:

    def __init__(
        self,
        program: bytes,
        input_shapes: Sequence[tuple[int, ...] | None] | None = None,
        output_shapes: Sequence[tuple[int, ...] | None] | None = None,
    ):
        self.program = bytes(program)
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
        program = bytes(coker_runtime.compile_exported_graph(payload))
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
        restored = [
            _restore_output(flat_output, shape)
            for flat_output, shape in zip(
                flat_outputs, self._output_shapes, strict=False
            )
        ]
        if len(restored) == 1:
            return restored[0]
        return restored
