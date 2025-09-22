import numpy as np
import pytest
try:

    from coker.backends.casadi.variational_solver import (
        InterpolatingPolyCollection,
        SymbolicPolyCollection,
        SymbolicPoly,
    )
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
    assert x.shape == (10, 1)

    t_start, x_starts = zip(*list(poly_collection.interval_starts()))
    assert t_start == (0, 1)
    assert x_starts == (x[0], x[5])

    t_end, x_ends = zip(*list(poly_collection.interval_ends()))
    assert t_end == (1, 2)
    assert x_ends == (x[4], x[9])

    t, x, dx = zip(*list(poly_collection.knot_points())[:5])

    assert (t[0] == 0) and t[-1] == 1
    for i, t_i in enumerate(t[:-1]):
        x_i = poly_collection(t_i)
        assert x_i == x[i]

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
    assert x.shape == (30, 1)
    assert poly_collection.size() == 30
    t_start, x_starts = zip(*list(poly_collection.interval_starts()))
    assert t_start == (0, 1)

    assert all(x_starts[0][i] == x[i] for i in range(3))
    assert all(x_starts[1][i] == x[i + 15] for i in range(3))

    t_end, x_ends = zip(*list(poly_collection.interval_ends()))
    assert t_end == (1, 2)
    assert all(x_ends[0][i] == x[12 + i] for i in range(3))
    assert all(x_ends[1][i] == x[27 + i] for i in range(3))

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
    assert np.isclose(result, fixed_result).all()
