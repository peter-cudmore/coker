import numpy as np
import pytest

from coker.algebra.ops import OP
from coker.backends.coker.ast_preprocessing import SparseNet
from coker.backends.coker.residual import (
    BilinearRow,
    BilinearStage,
    BilinearTerm,
    CallStage,
    InputBinding,
    InputMap,
    LinearTerm,
    NonlinearOperation,
    NonlinearStage,
    OutputBinding,
    OutputMap,
    QuadraticTerm,
    RetainedExpression,
    SlotOperand,
    push_forward_bilinear_stage,
    canonical_expression,
    push_forward_nonlinear_stage,
    stage_count,
)


def test_retained_expression_requires_canonical_sparse_terms():
    expression = RetainedExpression(
        roots=(2, 7),
        constant=1.0,
        linear=(LinearTerm(0, 2.0),),
        quadratic=(QuadraticTerm(0, 0, 3.0), QuadraticTerm(0, 1, 4.0)),
    )

    assert expression.roots == (2, 7)

    with pytest.raises(ValueError, match="unique and sorted"):
        RetainedExpression(roots=(7, 2))
    with pytest.raises(ValueError, match="strictly canonical"):
        RetainedExpression(
            roots=(2, 7),
            quadratic=(QuadraticTerm(0, 1, 1.0), QuadraticTerm(0, 1, 2.0)),
        )


def test_canonical_expression_combines_and_cancels_terms():
    expression = canonical_expression(
        roots=(2, 7),
        linear=(LinearTerm(0, 2.0), LinearTerm(0, -2.0)),
        quadratic=(
            QuadraticTerm(1, 0, 3.0),
            QuadraticTerm(0, 1, -1.0),
        ),
    )

    assert expression.linear == ()
    assert expression.quadratic == (QuadraticTerm(0, 1, 2.0),)


def test_nonlinear_stage_rejects_intra_stage_dependencies():
    with pytest.raises(ValueError, match="output dependency"):
        NonlinearStage(
            operations=(
                NonlinearOperation(3, OP.SIN, SlotOperand(0)),
                NonlinearOperation(4, OP.COS, SlotOperand(3)),
            )
        )


def test_bilinear_stage_rejects_intra_stage_dependencies():
    with pytest.raises(ValueError, match="output dependency"):
        BilinearStage(
            rows=(
                BilinearRow(3, (BilinearTerm(0, 1, 1.0),)),
                BilinearRow(4, (BilinearTerm(0, 3, 1.0),)),
            )
        )


def test_residual_stages_evaluate_primal_and_push_forward_in_place():
    bilinear = BilinearStage(
        rows=(BilinearRow(2, (BilinearTerm(0, 1, 2.0),)),)
    )
    nonlinear = NonlinearStage(
        operations=(
            NonlinearOperation(
                3,
                OP.SIN,
                RetainedExpression(
                    roots=(0, 1),
                    constant=1.0,
                    linear=(LinearTerm(0, 2.0),),
                    quadratic=(QuadraticTerm(0, 1, 3.0),),
                ),
            ),
        )
    )
    workspace = np.array([2.0, 5.0, 0.0, 0.0])
    tangent = np.array([7.0, 11.0, 0.0, 0.0])

    push_forward_bilinear_stage(bilinear, workspace, tangent)
    push_forward_nonlinear_stage(nonlinear, workspace, tangent)

    assert workspace[2] == 20.0
    assert tangent[2] == 114.0
    assert workspace[3] == pytest.approx(np.sin(35.0))
    assert tangent[3] == pytest.approx(np.cos(35.0) * 185.0)


def test_sparse_net_executes_residual_stages_without_memory_spec():
    graph = SparseNet(
        4,
        InputMap((InputBinding((0, 1)),)),
        OutputMap((OutputBinding((3,), None),)),
        residual_stages=(
            BilinearStage(
                rows=(BilinearRow(2, (BilinearTerm(0, 1, 2.0),)),)
            ),
            NonlinearStage(
                operations=(NonlinearOperation(3, OP.SIN, SlotOperand(2)),)
            ),
        ),
    )

    output, tangent = graph.push_forward(
        np.array([2.0, 5.0]), np.array([7.0, 11.0])
    )

    assert graph(np.array([2.0, 5.0])) == pytest.approx(np.sin(20.0))
    assert output == pytest.approx(np.sin(20.0))
    assert tangent == pytest.approx(np.cos(20.0) * 114.0)


def test_residual_call_stage_binds_direct_stable_slots():
    callee = SparseNet(
        2,
        InputMap((InputBinding((0,)),)),
        OutputMap((OutputBinding((1,), None),)),
        residual_stages=(
            NonlinearStage(
                operations=(NonlinearOperation(1, OP.SIN, SlotOperand(0)),)
            ),
        ),
    )
    caller = SparseNet(
        2,
        InputMap((InputBinding((0,)),)),
        OutputMap((OutputBinding((1,), None),)),
        residual_stages=(CallStage(callee, ((0,),), (1,)),),
    )

    output, tangent = caller.push_forward(np.array([2.0]), np.array([3.0]))

    assert output == pytest.approx(np.sin(2.0))
    assert tangent == pytest.approx(np.cos(2.0) * 3.0)

def test_residual_alias_input_bindings_share_primal_and_tangent_slot():
    graph = SparseNet(
        2,
        InputMap((InputBinding((0,)), InputBinding((0,)))),
        OutputMap((OutputBinding((1,), None),)),
        residual_stages=(
            NonlinearStage(
                operations=(
                    NonlinearOperation(1, OP.MUL, SlotOperand(0), SlotOperand(0)),
                )
            ),
        ),
    )

    output, tangent = graph.push_forward(
        np.array([3.0]), np.array([3.0]), np.array([2.0]), np.array([2.0])
    )

    assert output == pytest.approx(9.0)
    assert tangent == pytest.approx(12.0)


def test_compiler_preserves_aliases_across_nested_call_push_forward():
    from coker import VectorSpace, function
    from coker.backends.coker.lowering import create_function_table

    inner = function(
        [VectorSpace("x", 2)],
        lambda x: np.concatenate([x * x + np.ones(2), x[:1]]),
        backend="coker",
    )
    outer = function(
        [VectorSpace("x", 2)],
        lambda x: np.cross(
            np.array([x[0], x[1], x[0]]),
            np.array([x[1], x[0], x[1]]),
        )
        + inner(x),
        backend="coker",
    )
    graph = create_function_table(outer).entry
    value = np.array([0.25, -1.5])
    tangent = np.array([0.5, -0.25])
    actual, dactual = graph.push_forward(value, tangent)
    expected = outer(value)
    epsilon = 1e-6
    expected_tangent = (
        outer(value + epsilon * tangent) - outer(value - epsilon * tangent)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(dactual, expected_tangent, atol=1e-7)
