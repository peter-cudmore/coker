import pytest

try:
    import casadi as ca

    from coker.backends.casadi.variational.constraint_preprocessing import (
        ConstraintPreprocessingError,
        ConstraintRow,
        ConstraintRowProvenance,
        reduce_affine_equality_rows,
        preprocess_constraint_rows,
    )

    casadi_available = True
except ImportError:
    casadi_available = False


pytestmark = pytest.mark.skipif(
    not casadi_available, reason="CasADi not available"
)


def _row(residual, lower, upper, source):
    return ConstraintRow(
        residual=residual,
        lower=lower,
        upper=upper,
        provenance=(ConstraintRowProvenance(source, None, None, 0),),
    )


def test_preprocessing_removes_exact_duplicate_rows():
    decision = ca.MX.sym("decision")

    rows = preprocess_constraint_rows(
        (
            _row(decision, 0.0, 0.0, "path interval 0 node 0"),
            _row(decision, 0.0, 0.0, "terminal"),
        ),
        tolerance=1e-12,
    )

    assert len(rows) == 1
    assert [source.category for source in rows[0].provenance] == [
        "path interval 0 node 0",
        "terminal",
    ]


def test_preprocessing_keeps_matching_rows_with_distinct_bounds():
    decision = ca.MX.sym("decision")

    rows = preprocess_constraint_rows(
        (
            _row(decision, 0.0, ca.inf, "lower"),
            _row(decision, -ca.inf, 1.0, "upper"),
        ),
        tolerance=1e-12,
    )

    assert len(rows) == 2


def test_preprocessing_reports_conflicting_duplicate_rows():
    decision = ca.MX.sym("decision")

    with pytest.raises(ConstraintPreprocessingError) as error:
        preprocess_constraint_rows(
            (
                _row(decision, 0.0, 0.0, "path interval 0 node 0"),
                _row(decision, 1.0, 1.0, "terminal"),
            ),
            tolerance=1e-12,
        )

    message = str(error.value)
    assert "path interval 0 node 0" in message
    assert "terminal" in message


def test_affine_reduction_drops_dependent_equalities():
    decision = ca.MX.sym("decision", 2)
    rows = (
        _row(decision[0], 0.0, 0.0, "first"),
        _row(2 * decision[0], 0.0, 0.0, "dependent"),
        _row(decision[1], 0.0, 0.0, "second"),
    )

    reduced = reduce_affine_equality_rows(
        rows,
        decision,
        tolerance=1e-10,
    )

    assert len(reduced) == 2
