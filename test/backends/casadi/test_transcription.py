from functools import reduce
from operator import mul
from coker.dynamics import SpikeVariable, PiecewiseConstantVariable, ConstantControlVariable, TranscriptionOptions, \
    split_at_non_differentiable_points
from coker.dynamics.transcription_helpers import (generate_discritisation_operators)

from coker.backends.casadi.variational_solver import (InterpolatingPoly, InterpolatingPolyCollection)


def test_poly_collection_scalar():
    intervals = [(0, 1), (1, 2)]
    collocation_degree = [4, 4]

    dimension = 1

    poly_collection = InterpolatingPolyCollection(
        'x', dimension, intervals, collocation_degree,
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


def test_poly_collection_vector():
    intervals = [(0, 1), (1, 2)]
    collocation_degree = [4, 4]

    dimension = 3

    poly_collection = InterpolatingPolyCollection(
        'x', dimension, intervals, collocation_degree,
    )
    x = poly_collection.symbols()
    assert x.shape == (30, 1)

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
