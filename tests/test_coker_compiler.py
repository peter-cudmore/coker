import pytest

import coker_compiler


def _build_scalar_fixture(operation_tag=19):
    builder = coker_compiler.Builder(2, 1, 1, 3, 1, 1, 0)
    assert builder.push_constant([2.0], [1]) == 0
    builder.push_node(0, 0, [], [], 0)
    builder.push_node(1, operation_tag, [0], [1])
    builder.push_input("x", 0)
    builder.push_output("y", 1)
    return builder.finish_tape()


def test_builder_constructs_phase_one_scalar_fixture():
    assert _build_scalar_fixture().counts() == (2, 1, 1, 1, 1)


def test_compile_artifact_exposes_complete_success_diagnostics():
    artifact = _build_scalar_fixture().compile_artifact()

    diagnostics = artifact.info()["diagnostics"]
    assert diagnostics["dag_nodes"] == 2
    assert diagnostics["archive_bytes"] > 0
    assert diagnostics["finalization_ns"] >= 0


@pytest.mark.parametrize("operation_tag", [1, 3, 8, 9, 15, 19, 20, 27])
def test_builder_accepts_runtime_scalar_operation_tags(operation_tag):
    assert _build_scalar_fixture(operation_tag).counts() == (2, 1, 1, 1, 1)


def test_builder_reports_first_invalid_node_context():
    builder = coker_compiler.Builder(1, 0, 0, 0, 0, 0, 0)

    with pytest.raises(ValueError, match="node index 1 is out of order"):
        builder.push_node(1, 0, [], [])


def test_builder_cannot_finish_twice():
    builder = coker_compiler.Builder(1, 0, 0, 0, 0, 0, 0)
    builder.push_node(0, 0, [], [])
    dag = builder.finish_tape()

    assert dag.counts() == (1, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="already been called"):
        builder.finish_tape()


def _build_source_qp_fixture():
    builder = coker_compiler.Builder(2, 2, 0, 0, 1, 0, 0)
    builder.push_node(0, 0, [], [])
    builder.push_node(1, 3, [0, 0], [])
    builder.push_input("x", 0)
    return builder.finish_tape()


def test_symbolic_qp_declaration_compiles_source_qp():
    dag = _build_source_qp_fixture()
    declaration = coker_compiler.SymbolicQpDeclaration(
        1, 0, ([], [0]), 1, [], ([], [])
    )
    artifact = coker_compiler.compile_archive_qp_source(dag, declaration)
    assert artifact.to_bytes()


def test_symbolic_qp_declaration_rejects_invalid_bound_types():
    with pytest.raises(
        ValueError, match="QP bound must be a node ID or numeric sequence"
    ):
        coker_compiler.SymbolicQpDeclaration(
            1, 0, ([], [0]), 0, [], (["invalid"], [])
        )


def test_source_qp_binding_requires_symbolic_declaration_object():
    dag = _build_source_qp_fixture()
    with pytest.raises(TypeError):
        coker_compiler.compile_archive_qp_source(
            dag, 1, 0, ([], [0]), 1, [], ([], [])
        )
