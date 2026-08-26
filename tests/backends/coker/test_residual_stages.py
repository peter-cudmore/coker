import numpy as np
import pytest
from coker import Dimension
from coker.backends.coker.ast_preprocessing import SparseNet
from coker.backends.coker.layers import InputLayer, OutputLayer
from coker.backends.coker.memory import MemorySpec

from coker.algebra.ops import OP
from coker.backends.coker.residual import (
    BilinearRow,
    BilinearStage,
    BilinearTerm,
    LinearTerm,
    NonlinearOperation,
    NonlinearStage,
    QuadraticTerm,
    RetainedExpression,
    SlotOperand,
    push_forward_bilinear_stage,
    push_forward_nonlinear_stage,
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


def test_sparse_net_executes_residual_stages():
    input_layer = InputLayer()
    input_layer.add_input(Dimension((2,)))
    output_layer = OutputLayer()
    output_layer.add_output(MemorySpec(3, 1), Dimension(None))
    graph = SparseNet(
        MemorySpec(0, 4),
        input_layer,
        output_layer,
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
