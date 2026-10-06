from types import SimpleNamespace

import numpy as np
import pytest
from coker import Dimension, Scalar, VectorSpace, function
from coker.algebra.ops import Noop
from coker.dynamics import ResidualDynamicalSystem, VariationalProblemBuilder
from coker.parameters import BoundedVariable
from coker.toolkits.codesign import Minimise
from coker.dynamics.variational.solution import SegmentDefectDiagnostic


def test_segment_defect_diagnostic_exposes_full_residual():
    full_residual = np.array([0.25, -0.5])
    diagnostic = SegmentDefectDiagnostic(
        normalized_interval=(0.0, 1.0),
        physical_interval=(0.0, 2.0),
        degree=3,
        tolerance=1e-6,
        full_residual=full_residual,
    )

    assert diagnostic.full_residual is full_residual


try:

    from coker.backends.casadi.variational.symbolic_path import (
        SymbolicPoly,
        SymbolicPolyCollection,
    )
    from coker.backends.casadi.variational.transcription import (
        CasadiVariationalSolver,
    )
    from coker.dynamics import InterpolatingPolyCollection
    from coker.toolkits.codesign import SolveFailure
    import casadi as ca

    casadi_available = True
except ImportError:
    casadi_available = False


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
def test_symbolic_poly():
    t = ca.MX.sym("t")
    poly = SymbolicPoly("x", 3, (0, 1), 5)
    x = poly.symbols()
    poly_as_func = ca.Function("poly_as_func", [t, x], [poly(t)])

    def f(t):
        return np.array([1 + 2 * t, -3 * t, t])

    line = ca.vertcat(*[ca.DM(f(t_i)) for t_i in poly.knot_times()])
    assert line.shape == x.shape

    for t_i in np.linspace(0, 1, 100):
        result = np.array(poly_as_func(t_i, line)).flatten()

        assert np.allclose(result, f(t_i))


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
def test_poly_collection_scalar():
    intervals = [(0, 1), (1, 2)]
    collocation_degree = [4, 4]

    dimension = 1

    poly_collection = SymbolicPolyCollection(
        "x",
        dimension,
        intervals,
        collocation_degree,
    )
    x = poly_collection.symbols()
    assert x.shape == (9, 1)

    t_start, x_starts = zip(*list(poly_collection.interval_starts()))
    assert t_start == (0, 1)
    assert x_starts == (x[0], x[4])

    t_end, x_ends = zip(*list(poly_collection.interval_ends()))
    assert t_end == (1, 2)
    assert x_ends == (x[4], x[8])
    assert ca.is_equal(x_ends[0], x_starts[1], 2)

    t, x, dx = zip(*list(poly_collection.knot_points())[:5])

    assert (t[0] == 0) and t[-1] == 1
    for i, t_i in enumerate(t[:-1]):
        x_i = poly_collection(t_i)
        assert x_i == x[i]


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
def test_poly_collection_shares_state_and_quadrature_boundaries():
    collection = SymbolicPolyCollection(
        "path",
        dimension=3,
        intervals=[(0, 1), (1, 2)],
        degrees=[2, 2],
        state_size=1,
        algebraic_size=1,
    )

    _, first_end = collection.polys[0].end_point()
    _, second_start = collection.polys[1].start_point()

    assert ca.is_equal(first_end[0], second_start[0], 2)
    assert not ca.is_equal(first_end[1], second_start[1], 2)
    assert ca.is_equal(first_end[2], second_start[2], 2)


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
def test_poly_collection_vector():
    intervals = [(0, 1), (1, 2)]
    collocation_degree = [4, 4]

    dimension = 3

    poly_collection = SymbolicPolyCollection(
        "x",
        dimension,
        intervals,
        collocation_degree,
    )
    x = poly_collection.symbols()
    assert x.shape == (27, 1)
    assert poly_collection.size() == 27
    t_start, x_starts = zip(*list(poly_collection.interval_starts()))
    assert t_start == (0, 1)

    assert all(x_starts[0][i] == x[i] for i in range(3))
    assert all(x_starts[1][i] == x[12 + i] for i in range(3))

    t_end, x_ends = zip(*list(poly_collection.interval_ends()))
    assert t_end == (1, 2)
    assert all(x_ends[0][i] == x[12 + i] for i in range(3))
    assert all(x_ends[1][i] == x[24 + i] for i in range(3))
    assert all(
        ca.is_equal(x_ends[0][index], x_starts[1][index], 2)
        for index in range(dimension)
    )

    t, x, dx = zip(*list(poly_collection.knot_points())[:5])

    assert (t[0] == 0) and t[-1] == 1
    for i, t_i in enumerate(t[:-1]):
        x_i = poly_collection(t_i)
        assert all(x_i[j] == x[i][j] for j in range(dimension))

    t = ca.MX.sym("t")
    x = poly_collection.symbols()
    poly_as_func = ca.Function("poly_as_func", [t, x], [poly_collection(t)])

    line = ca.DM.ones(x.shape)
    fixed_poly: InterpolatingPolyCollection = poly_collection.to_fixed(line)

    for poly in fixed_poly.polys:
        assert (poly.values == 1).all()

    fixed_result = fixed_poly(0.51)

    assert fixed_result.shape == (3,)
    assert np.isclose(fixed_result, np.ones((3,))).all()
    result = poly_as_func(ca.DM(0.51), line)

    assert result.shape == (3, 1)
    np.testing.assert_allclose(result.full().ravel(), fixed_result)


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
@pytest.mark.parametrize(
    ("objective", "constraint_value", "accepted"),
    [
        (1.0, 5e-7, True),
        (1.0, 2e-6, False),
        (np.nan, 0.0, False),
    ],
)
def test_search_direction_too_small_requires_feasible_finite_result(
    objective, constraint_value, accepted
):
    class StubSolver:
        def __call__(self, **_kwargs):
            return {
                "x": ca.DM([0.0]),
                "f": ca.DM([objective]),
                "g": ca.DM([constraint_value]),
            }

        def stats(self):
            return {
                "success": False,
                "return_status": "Search_Direction_Becomes_Too_Small",
            }

    def map_arguments(_fixed_parameters, _previous_solution):
        return {
            "x0": ca.DM([0.0]),
            "lbx": ca.DM([-ca.inf]),
            "ubx": ca.DM([ca.inf]),
            "lbg": ca.DM([0.0]),
            "ubg": ca.DM([0.0]),
        }

    solver = CasadiVariationalSolver(
        problem=SimpleNamespace(
            transcription_options=SimpleNamespace(absolute_tolerance=1e-6)
        ),
        parameters=[],
        map_arguments=map_arguments,
        solver=StubSolver(),
        assemble_solution=lambda _x, cost, solve_info: (cost, solve_info),
    )

    if not accepted:
        with pytest.raises(SolveFailure) as error:
            solver._solve_once()
        assert not error.value.solve_info.success
        return

    cost, solve_info = solver._solve_once()

    assert cost == 1.0
    assert solve_info.success
    assert solve_info.return_status == "Search_Direction_Becomes_Too_Small"


@pytest.mark.skipif(not casadi_available, reason="CasAdi not available")
def test_independent_coupled_residual_is_specialized_and_solved():
    gain = 0.75
    residual = ResidualDynamicalSystem(
        inputs=Noop(),
        parameters=(Scalar("gain"),),
        x0=function(
            [VectorSpace("initial_algebraic", 1), Noop(), Scalar("gain")],
            lambda _z, _u, parameter: (
                np.array([0.0]),
                np.array([2.0 * parameter / 3.0]),
            ),
            backend="casadi",
        ),
        F=function(
            [
                Scalar("t"),
                VectorSpace("w", 1),
                VectorSpace("wdot", 1),
                VectorSpace("z", 1),
                Noop(),
                Scalar("gain"),
            ],
            lambda _t, w, wdot, z, _u, parameter: np.array(
                [
                    wdot[0] + z[0] - parameter,
                    w[0] + 2.0 * wdot[0] - z[0],
                ]
            ),
            backend="casadi",
        ),
        y=function(
            [
                Scalar("t"),
                VectorSpace("x", 1),
                VectorSpace("z", 1),
                Noop(),
                Scalar("gain"),
                None,
            ],
            lambda _t, x, _z, _u, _parameter, _q: x,
            backend="casadi",
        ),
        differential=Dimension(1),
        algebraic=Dimension(1),
        quadrature=None,
    )
    declaration = BoundedVariable(
        "gain", lower_bound=gain, upper_bound=gain, guess=gain
    )
    with VariationalProblemBuilder(
        residual,
        t_final=1.0,
        parameters=[declaration],
        backend="casadi",
    ) as builder:
        problem = builder.build(
            Minimise(
                (
                    builder.output(builder.t_final)[0]
                    - gain * (1.0 - np.exp(-1.0 / 3.0))
                )
                ** 2
            )
        )

    assert isinstance(problem.system, ResidualDynamicalSystem)
    np.testing.assert_allclose(
        problem.system.F(
            0.0,
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.3]),
            None,
            np.array([gain]),
        ),
        residual.F(
            0.0,
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.3]),
            None,
            gain,
        ),
    )

    solution = problem.get_solver("casadi").solve(gain=gain)

    assert solution.solve_info.success
    assert solution.parameters["gain"] == pytest.approx(gain)
    assert solution.state(1.0)[0] == pytest.approx(
        gain * (1.0 - np.exp(-1.0 / 3.0)), abs=2e-3
    )

    assert solution.segment_defects
    assert all(
        diagnostic.full_residual.shape == (2,)
        for diagnostic in solution.segment_defects
    )
