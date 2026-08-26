"""Stable-slot residual program stage records.

These records are the Python execution representation shared by residual
lowering, reference evaluation, and the later typed bytecode builder.  Slots
are absolute workspace indices; stages never describe compacted ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from coker.algebra.ops import OP


@dataclass(frozen=True)
class SlotOperand:
    """A scalar operand read from one absolute workspace slot."""

    slot: int

    def __post_init__(self):
        if self.slot < 0:
            raise ValueError("slot operands require a non-negative slot")


@dataclass(frozen=True)
class LinearTerm:
    """One linear coefficient indexed into a retained expression's roots."""

    root: int
    coefficient: float


@dataclass(frozen=True)
class QuadraticTerm:
    """One canonical quadratic coefficient indexed into retained roots."""

    left: int
    right: int
    coefficient: float


@dataclass(frozen=True)
class RetainedExpression:
    """A canonical sparse degree-at-most-two expression over stable roots."""

    roots: Tuple[int, ...]
    constant: float = 0.0
    linear: Tuple[LinearTerm, ...] = ()
    quadratic: Tuple[QuadraticTerm, ...] = ()

    def __post_init__(self):
        if tuple(sorted(set(self.roots))) != self.roots:
            raise ValueError("retained expression roots must be unique and sorted")
        if any(root < 0 for root in self.roots):
            raise ValueError("retained expression roots must be non-negative")
        root_count = len(self.roots)
        if any(term.root < 0 or term.root >= root_count for term in self.linear):
            raise ValueError("linear term root is outside retained expression roots")
        previous = None
        for term in self.quadratic:
            if not 0 <= term.left <= term.right < root_count:
                raise ValueError("quadratic term is not a canonical root pair")
            pair = (term.left, term.right)
            if previous is not None and pair <= previous:
                raise ValueError("quadratic terms must be strictly canonical")
            previous = pair


NonlinearOperand = SlotOperand | RetainedExpression


@dataclass(frozen=True)
class NonlinearOperation:
    """One ordered scalar operation writing a stable output slot.

    Missing operands are represented by ``None``.  Present operands are either
    workspace slots or retained expressions evaluated locally by the stage.
    """

    output: int
    op: OP | str
    first: NonlinearOperand | None
    second: NonlinearOperand | None = None
    third: NonlinearOperand | None = None

    def __post_init__(self):
        if self.output < 0:
            raise ValueError("nonlinear output requires a non-negative slot")
        if self.first is None:
            raise ValueError("nonlinear operation requires its first operand")


@dataclass(frozen=True)
class NonlinearStage:
    """An unordered-independent batch of scalar nonlinear operations.

    Every operation reads only stage inputs.  A dependence on a prior output is
    represented by a subsequent ``NonlinearStage``.
    """

    operations: Tuple[NonlinearOperation, ...]

    def __post_init__(self):
        outputs = tuple(operation.output for operation in self.operations)
        if tuple(sorted(outputs)) != outputs or len(set(outputs)) != len(outputs):
            raise ValueError("nonlinear stage outputs must be unique and sorted")
        stage_outputs = set(outputs)
        for operation in self.operations:
            for operand in (operation.first, operation.second, operation.third):
                if isinstance(operand, SlotOperand) and operand.slot in stage_outputs:
                    raise ValueError("nonlinear stage has an output dependency")
                if isinstance(operand, RetainedExpression) and any(
                    root in stage_outputs for root in operand.roots
                ):
                    raise ValueError("nonlinear stage has an expression dependency")


@dataclass(frozen=True)
class BilinearTerm:
    """One homogeneous bilinear term over absolute workspace slots.

    ``None`` denotes the homogeneous constant one operand.
    """

    left: int | None
    right: int | None
    coefficient: float

    def __post_init__(self):
        if self.left is not None and self.left < 0:
            raise ValueError("bilinear left operand requires a non-negative slot")
        if self.right is not None and self.right < 0:
            raise ValueError("bilinear right operand requires a non-negative slot")
        left = -1 if self.left is None else self.left
        right = -1 if self.right is None else self.right
        if left > right:
            raise ValueError("bilinear operands must use canonical order")


@dataclass(frozen=True)
class BilinearRow:
    """All sparse homogeneous terms contributing to one stable output slot."""

    output: int
    terms: Tuple[BilinearTerm, ...]

    def __post_init__(self):
        if self.output < 0:
            raise ValueError("bilinear output requires a non-negative slot")
        pairs = tuple(
            (-1 if term.left is None else term.left,
             -1 if term.right is None else term.right)
            for term in self.terms
        )
        if pairs != tuple(sorted(pairs)) or len(set(pairs)) != len(pairs):
            raise ValueError("bilinear terms must be unique and sorted")


@dataclass(frozen=True)
class BilinearStage:
    """An independent row-grouped sparse bilinear stable-slot batch."""

    rows: Tuple[BilinearRow, ...]

    def __post_init__(self):
        outputs = tuple(row.output for row in self.rows)
        if tuple(sorted(outputs)) != outputs or len(set(outputs)) != len(outputs):
            raise ValueError("bilinear stage outputs must be unique and sorted")
        output_set = set(outputs)
        for row in self.rows:
            for term in row.terms:
                if term.left in output_set or term.right in output_set:
                    raise ValueError("bilinear stage has an output dependency")


def _operand_value(operand: NonlinearOperand, workspace: np.ndarray) -> float:
    if isinstance(operand, SlotOperand):
        return float(workspace[operand.slot])
    value = float(operand.constant)
    for term in operand.linear:
        value += term.coefficient * float(workspace[operand.roots[term.root]])
    for term in operand.quadratic:
        value += (
            term.coefficient
            * float(workspace[operand.roots[term.left]])
            * float(workspace[operand.roots[term.right]])
        )
    return value


def _operand_push_forward(
    operand: NonlinearOperand, workspace: np.ndarray, dworkspace: np.ndarray
) -> tuple[float, float]:
    value = _operand_value(operand, workspace)
    if isinstance(operand, SlotOperand):
        return value, float(dworkspace[operand.slot])
    tangent = 0.0
    for term in operand.linear:
        tangent += term.coefficient * float(dworkspace[operand.roots[term.root]])
    for term in operand.quadratic:
        left = operand.roots[term.left]
        right = operand.roots[term.right]
        tangent += term.coefficient * (
            float(dworkspace[left]) * float(workspace[right])
            + float(workspace[left]) * float(dworkspace[right])
        )
    return value, tangent


def _scalar_value(op: OP | str, first: float, second: float | None, third: float | None):
    if op == "identity" or op == "constant":
        return first
    if op == OP.SIN:
        return float(np.sin(first))
    if op == OP.COS:
        return float(np.cos(first))
    if op == OP.TAN:
        return float(np.tan(first))
    if op == OP.EXP:
        return float(np.exp(first))
    if op == OP.SQRT:
        return float(np.sqrt(first))
    if op == OP.LOG:
        return float(np.log(first))
    if op == OP.NEG:
        return -first
    if op == OP.ABS:
        return float(np.abs(first))
    if op == OP.ADD:
        return first + second
    if op == OP.SUB:
        return first - second
    if op == OP.MUL:
        return first * second
    if op == OP.DIV:
        return float(np.divide(first, second))
    if op == OP.PWR or op == OP.INT_PWR:
        return float(np.power(first, second))
    if op == OP.ARCTAN2:
        return float(np.arctan2(first, second))
    if op == OP.EQUAL:
        return float(first == second)
    if op == OP.LESS_THAN:
        return float(first < second)
    if op == OP.LESS_EQUAL:
        return float(first <= second)
    if op == OP.CASE:
        return second if bool(first) else third
    raise NotImplementedError(f"Unsupported residual scalar operation {op}")


def _scalar_push_forward(
    op: OP | str,
    first: float,
    second: float | None,
    third: float | None,
    dfirst: float,
    dsecond: float | None,
    dthird: float | None,
) -> tuple[float, float]:
    value = _scalar_value(op, first, second, third)
    if op == "identity":
        return value, dfirst
    if op == "constant":
        return value, 0.0
    if op == OP.SIN:
        return value, float(np.cos(first)) * dfirst
    if op == OP.COS:
        return value, -float(np.sin(first)) * dfirst
    if op == OP.TAN:
        return value, dfirst / float(np.cos(first) ** 2)
    if op == OP.EXP:
        return value, value * dfirst
    if op == OP.SQRT:
        return value, dfirst / (2.0 * value)
    if op == OP.LOG:
        return value, dfirst / first
    if op == OP.NEG:
        return value, -dfirst
    if op == OP.ABS:
        return value, float(np.sign(first)) * dfirst
    if op == OP.ADD:
        return value, dfirst + dsecond
    if op == OP.SUB:
        return value, dfirst - dsecond
    if op == OP.MUL:
        return value, second * dfirst + first * dsecond
    if op == OP.DIV:
        return value, (dfirst * second - first * dsecond) / (second * second)
    if op == OP.PWR or op == OP.INT_PWR:
        if first == 0.0:
            return value, 0.0
        return value, value * (dsecond * np.log(first) + second * dfirst / first)
    if op == OP.ARCTAN2:
        return value, (second * dfirst - first * dsecond) / (
            first * first + second * second
        )
    if op in {OP.EQUAL, OP.LESS_THAN, OP.LESS_EQUAL}:
        return value, 0.0
    if op == OP.CASE:
        return value, dsecond if bool(first) else dthird
    raise NotImplementedError(f"Unsupported residual scalar operation {op}")


def apply_bilinear_stage(stage: BilinearStage, workspace: np.ndarray) -> None:
    """Evaluate one independent bilinear stage into an existing workspace."""
    for row in stage.rows:
        value = 0.0
        for term in row.terms:
            left = 1.0 if term.left is None else float(workspace[term.left])
            right = 1.0 if term.right is None else float(workspace[term.right])
            value += term.coefficient * left * right
        workspace[row.output] = value


def push_forward_bilinear_stage(
    stage: BilinearStage, workspace: np.ndarray, dworkspace: np.ndarray
) -> None:
    """Evaluate one bilinear stage and its tangent into existing workspaces."""
    for row in stage.rows:
        value = 0.0
        tangent = 0.0
        for term in row.terms:
            left = 1.0 if term.left is None else float(workspace[term.left])
            right = 1.0 if term.right is None else float(workspace[term.right])
            dleft = 0.0 if term.left is None else float(dworkspace[term.left])
            dright = 0.0 if term.right is None else float(dworkspace[term.right])
            value += term.coefficient * left * right
            tangent += term.coefficient * (dleft * right + left * dright)
        workspace[row.output] = value
        dworkspace[row.output] = tangent


def apply_nonlinear_stage(stage: NonlinearStage, workspace: np.ndarray) -> None:
    """Evaluate one independent nonlinear stage into an existing workspace."""
    for operation in stage.operations:
        first = _operand_value(operation.first, workspace)
        second = (
            None
            if operation.second is None
            else _operand_value(operation.second, workspace)
        )
        third = (
            None
            if operation.third is None
            else _operand_value(operation.third, workspace)
        )
        workspace[operation.output] = _scalar_value(
            operation.op, first, second, third
        )


def push_forward_nonlinear_stage(
    stage: NonlinearStage, workspace: np.ndarray, dworkspace: np.ndarray
) -> None:
    """Evaluate one nonlinear stage and its tangent into existing workspaces."""
    for operation in stage.operations:
        first, dfirst = _operand_push_forward(
            operation.first, workspace, dworkspace
        )
        second, dsecond = (
            (None, None)
            if operation.second is None
            else _operand_push_forward(operation.second, workspace, dworkspace)
        )
        third, dthird = (
            (None, None)
            if operation.third is None
            else _operand_push_forward(operation.third, workspace, dworkspace)
        )
        value, tangent = _scalar_push_forward(
            operation.op, first, second, third, dfirst, dsecond, dthird
        )
        workspace[operation.output] = value
        dworkspace[operation.output] = tangent
