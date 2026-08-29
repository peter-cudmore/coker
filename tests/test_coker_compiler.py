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
