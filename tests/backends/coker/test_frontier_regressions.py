import numpy as np
import pytest

from coker import VectorSpace, function
from coker.algebra.ops import OP
from coker.backends.coker.lowering import create_function_table


BYTECODE_PHASE_REASON = (
    "mapped-bytecode checks resume after the Python residual phase"
)


def _payload(implementation, spaces=(VectorSpace("x", 2),)):
    fn = function(list(spaces), implementation=implementation, backend="coker")
    return fn, create_function_table(fn).entry.export_program_payload()


def _layers(payload, kind):
    return [layer for layer in payload["intermediate_layers"] if layer["kind"] == kind]


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_degree_two_expression_is_one_canonical_frontier():
    fn, payload = _payload(lambda x: x * x + 2.0 * x + np.ones(2))
    bilinear = _layers(payload, "scheduled_bilinear")
    assert len(bilinear) == 1
    rows = bilinear[0]["rows"]
    terms = bilinear[0]["terms"]
    assert [row["output"] for row in rows] == sorted(row["output"] for row in rows)
    for row in rows:
        selected = terms[row["term_start"] : row["term_start"] + row["term_count"]]
        assert [(term["left"], term["right"]) for term in selected] == sorted(
            (term["left"], term["right"]) for term in selected
        )
    value = np.array([0.25, -1.5])
    np.testing.assert_allclose(CompiledGraph.compile(create_function_table(fn))(value), value * value + 2 * value + 1)


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_degree_three_closes_frontier_without_changing_value():
    fn, payload = _payload(lambda x: x * x * x)
    assert len(_layers(payload, "scheduled_bilinear")) >= 2
    value = np.array([0.5, -2.0])
    np.testing.assert_allclose(CompiledGraph.compile(create_function_table(fn))(value), value**3)


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_nonlinear_scalar_closes_between_algebraic_frontiers():
    fn, payload = _payload(lambda x: np.sin(x) + x * x)
    assert _layers(payload, "scheduled_generic")
    assert len(_layers(payload, "scheduled_bilinear")) >= 2
    value = np.array([0.5, -2.0])
    np.testing.assert_allclose(
        CompiledGraph.compile(create_function_table(fn))(value),
        np.sin(value) + value * value,
    )


def test_generic_flush_diagnostics_capture_residual_nonlinear_stage():
    fn = function(
        [VectorSpace("x", 2)],
        implementation=lambda x: np.sin(x) + x * x,
        backend="coker",
    )
    graph = create_function_table(fn).entry
    stages = tuple(graph.residual_stages)
    nonlinear = [stage for stage in stages if hasattr(stage, "operations")]
    assert nonlinear
    assert all(stage.operations for stage in nonlinear)
    assert {operation.op for stage in nonlinear for operation in stage.operations} == {
        OP.SIN
    }
    assert any(hasattr(stage, "rows") for stage in stages)
    value = np.array([0.5, -2.0])
    np.testing.assert_allclose(graph(value), np.sin(value) + value * value)




@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_independent_branches_share_one_frontier():
    fn, payload = _payload(lambda x: np.concatenate([x * x, 3.0 * x + 2.0 * np.ones(2)]))
    assert len(_layers(payload, "scheduled_bilinear")) == 1
    value = np.array([0.5, -2.0])
    expected = np.concatenate([value * value, 3.0 * value + 2.0 * np.ones(2)])
    np.testing.assert_allclose(CompiledGraph.compile(create_function_table(fn))(value), expected)


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_reverse_uses_retain_root_for_later_generic_branches():
    fn, payload = _payload(
        lambda x: (np.sin(x), np.cos(x)),
    )
    generic = _layers(payload, "scheduled_generic")
    assert generic
    value = np.array([0.25, -0.5])
    np.testing.assert_allclose(
        CompiledGraph.compile(create_function_table(fn))(value)[0],
        np.sin(value),
    )
    np.testing.assert_allclose(
        CompiledGraph.compile(create_function_table(fn))(value)[1],
        np.cos(value),
    )


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_independent_frontier_rows_pin_all_roots_until_batch_close():
    fn, payload = _payload(
        lambda x, y: np.concatenate([x * x, y * y]),
        (VectorSpace("x", 2), VectorSpace("y", 2)),
    )
    assert len(_layers(payload, "scheduled_bilinear")) == 1
    value_x = np.array([1.0, 2.0])
    value_y = np.array([3.0, 4.0])
    expected = np.concatenate([value_x * value_x, value_y * value_y])
    np.testing.assert_allclose(
        CompiledGraph.compile(create_function_table(fn))(value_x, value_y),
        expected,
    )


def test_residual_stages_preserve_independent_branch_roots():
    fn = function(
        [VectorSpace("x", 2), VectorSpace("y", 2)],
        lambda x, y: (x * x, y * y),
        backend="coker",
    )
    graph = create_function_table(fn).entry
    bilinear = [stage for stage in graph.residual_stages if hasattr(stage, "rows")]
    assert bilinear
    outputs = {row.output for stage in bilinear for row in stage.rows}
    assert len(outputs) >= 2
    actual = graph(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(actual[0], np.array([1.0, 4.0]))
    np.testing.assert_allclose(actual[1], np.array([9.0, 16.0]))


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_f32_cancellation_removes_zero_sparse_terms():
    fn, payload = _payload(lambda x: (x * x + 3.0 * x) - (x * x + 3.0 * x))
    terms = [term for layer in _layers(payload, "scheduled_bilinear") for term in layer["terms"]]
    assert all(float(term["value"]) != 0.0 for term in terms)
    value = np.array([1.25, -0.75])
    np.testing.assert_allclose(CompiledGraph.compile(create_function_table(fn))(value), np.zeros(2))


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_canonical_frontier_bytecode_is_repeatable():
    implementation = lambda x: np.concatenate([x * x + x, x * 2.0 - np.ones(2)])
    first, first_payload = _payload(implementation)
    second, second_payload = _payload(implementation)
    assert first_payload == second_payload
    assert first.lower().compile_bytecode() == second.lower().compile_bytecode()


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_nested_pure_views_do_not_emit_view_rows():
    fn, payload = _payload(
        lambda x: np.concatenate([np.reshape(x, (1, 2)), np.reshape(x, (1, 2))], axis=0),
        (VectorSpace("x", 2),),
    )
    assert not _layers(payload, "scheduled_bilinear")
    assert not _layers(payload, "scheduled_generic")
    value = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        CompiledGraph.compile(create_function_table(fn))(value),
        [[1.0, 2.0], [1.0, 2.0]],
    )


@pytest.mark.skip(reason=BYTECODE_PHASE_REASON)
def test_alias_and_nested_boundaries_preserve_primal_and_tangent():
    inner = function(
        [VectorSpace("x", 2)],
        lambda x: np.concatenate([x * x + np.ones(2), x[:1]]),
        backend="coker",
    )
    fn = function(
        [VectorSpace("x", 2)],
        lambda x: np.cross(
            np.array([x[0], x[1], x[0]]),
            np.array([x[1], x[0], x[1]]),
        )
        + inner(x),
        backend="coker",
    )
    graph = create_function_table(fn).entry
    compiled = CompiledGraph.compile(create_function_table(fn))
    value = np.array([0.25, -1.5])
    tangent = np.array([0.5, -0.25])
    expected = graph(value)
    actual, dactual = compiled.push_forward(value, tangent)
    gvalue, gtangent = graph.push_forward(value, tangent)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(dactual, gtangent)
    np.testing.assert_allclose(gvalue, expected)

def test_hexapod_frontier_artifact_stays_within_layer_and_copy_budget():
    from scripts.inspect_kinematics_artifact import compile_hexapod, payload_metrics

    model, lowered = compile_hexapod()
    metrics = payload_metrics(lowered, require_v6=False)
    assert metrics["layer_count"] < 50
    assert metrics["explicit_copy_rows"] == 0
    assert metrics["identity_relocation_rows"] == 0
    assert metrics["output_clear_rows"] == 0
    assert model.total_joints() == 18
    assert lowered.residual_stages is not None


def test_hexapod_lowered_matches_source_fk_and_jacobian():
    from scripts.inspect_kinematics_artifact import compile_hexapod

    model, lowered = compile_hexapod()
    angles = np.linspace(-0.4, 0.4, model.total_joints())
    expected_positions, expected_jacobian = model.to_function()(angles)
    actual_positions, actual_jacobian = lowered(angles)
    np.testing.assert_allclose(
        actual_positions, expected_positions, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        actual_jacobian, expected_jacobian, rtol=1e-12, atol=1e-12
    )




def test_dependency_diagnostic_identifies_serial_nonlinear_chain():
    from scripts.inspect_kinematics_artifact import tape_dependency_diagnostic

    fn = function(
        [VectorSpace("x", 1)],
        implementation=lambda x: np.sin(np.sin(np.sin(x))),
        backend="coker",
    )
    report = tape_dependency_diagnostic(fn.tape)
    assert report["serial_nonlinear_call_chain"]
    assert report["nonlinear_call_depth"] >= 3
    nodes = {record["node_id"]: record for record in report["nodes"]}
    nonlinear = [
        node_id
        for node_id, record in nodes.items()
        if record["degree"] == "nonlinear"
    ]
    assert len(nonlinear) == 3
    assert all(nodes[node_id]["operation"] == "sin" for node_id in nonlinear)
    assert all(nodes[node_id]["predecessors"] for node_id in nonlinear[1:])


def test_dependency_diagnostic_keeps_independent_ready_branches_separate():
    from scripts.inspect_kinematics_artifact import tape_dependency_diagnostic

    fn = function(
        [VectorSpace("x", 1)],
        implementation=lambda x: (np.sin(x), np.cos(x)),
        backend="coker",
    )
    report = tape_dependency_diagnostic(fn.tape)
    assert not report["serial_nonlinear_call_chain"]
    assert report["nonlinear_call_depth"] == 1
    nonlinear = [
        record for record in report["nodes"] if record["degree"] == "nonlinear"
    ]
    assert {record["operation"] for record in nonlinear} == {"sin", "cos"}

