import numpy as np

import coker._coker_runtime as coker_runtime
from coker.backends.coker.conversion import function_to_typed_dag


def _runtime_input_buffer(arg) -> np.ndarray:
    if isinstance(arg, (int, float, bool, np.bool_)):
        return np.asarray([arg], dtype=np.float32)
    if isinstance(arg, np.ndarray):
        return np.asarray(arg, dtype=np.float32).reshape(-1, order="C")
    raise TypeError(f"Unsupported runtime input {type(arg)}")


class RuntimeQpProgram:
    def __init__(self, program):
        self._artifact = (
            program if hasattr(program, "_mapped_qp_capsule") else None
        )
        self._program = None if self._artifact is not None else bytes(program)
        self._runtime = (
            coker_runtime.load_compiled_qp_program(self._artifact)
            if self._artifact is not None
            else coker_runtime.load_qp_program(self._program)
        )
        self._info = self._runtime.info()
        self._input_lengths = list(self._info["input_specs"])
        self._output_length = int(self._info["output_spec"])
        self._solution = np.empty(self._output_length, dtype=np.float64)

    @property
    def program(self) -> bytes:
        """Export a persistence copy; execution retains the mapped artifact."""
        if self._artifact is not None:
            return self._artifact.to_bytes()
        return self._program

    @classmethod
    def compile(cls, extracted_qp) -> "RuntimeQpProgram":
        import coker_compiler

        from coker.algebra.kernel import Function, Tracer

        source_function = Function(
            extracted_qp.source_tape,
            [Tracer(extracted_qp.source_tape, extracted_qp.cost_node)],
        )
        dag = function_to_typed_dag(source_function)
        artifact = coker_compiler.compile_archive_qp_source(
            dag,
            coker_compiler.SymbolicQpDeclaration(
                extracted_qp.n,
                extracted_qp.m,
                (
                    [
                        binding.index
                        for binding in extracted_qp.parameter_bindings
                    ],
                    [
                        binding.index
                        for binding in extracted_qp.decision_bindings
                    ],
                ),
                extracted_qp.cost_node,
                extracted_qp.residual_nodes,
                (extracted_qp.lower_nodes, extracted_qp.upper_nodes),
            ),
        )
        return cls(artifact)

    def solve(self, runtime_args, *, warm_start):
        """Solve and return an owned solution snapshot.

        This host convenience API copies the reusable internal buffer. For
        allocation-free repeated solves, use the extension's ``solve_into``
        method with a caller-owned output buffer.
        """
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
        from coker.toolkits.codesign.optimisation import SolveInfo

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
