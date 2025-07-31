from coker.dynamics.transcription_helpers import *
from coker.dynamics import ConstantControlVariable
import numpy as np
from functools import reduce
from operator import mul


def test_legendre_coeffs():
    # increasing in powers of x
    truth = [
        [1],
        [0, 1],
        [-1/2, 0, 3/2],
        [0, -3/2, 0, 5/2]
    ]

    for n, coeffs in enumerate(truth):
        test_values = [
            (c,  legendre_coefficient(n, k))
            for k, c in enumerate(coeffs)
        ]
        assert all(
            abs(true_value - test_value) < 1e-4
            for true_value, test_value in test_values)


def test_lgr_points():
    expected = np.array([
        -1,
        (1- np.sqrt(6))/5,
        (1+ np.sqrt(6))/5,
        1
    ])
    result = lgr_points(3)
    assert np.allclose(result, expected)


def test_expand_coeffs():
    test_pairs = [
        ([1, -1], [-1, 0, 1]) # (x + 1)(x - 1) -> (-1 +0x + x^2)

    ]
    for roots, coeffs in test_pairs:
        result = expand_coefficients(roots)

        assert np.allclose(result, coeffs)


def lagrange_polynomial(i, value, points):
    return reduce(
        mul,
        [(value - p_j)/(points[i] - p_j)
         for j, p_j in enumerate(points) if j != i
         ],
       1
    )




def test_generate_discritisation_operators():
    # if we take
    # f(x) = x
    # then we should have
    # int_-1^1 f(x) = 0
    #
    # Evaluate y_i = f(x_i) for the knot points
    # int_0^1 f  = sum(w_i, y_i)  = 0
    # and df = 1

    interval = [-1, 1]
    n = 2

    def f(arg):
        return arg

    x, t, bases, deriv_op, integral_op = generate_discritisation_operators(interval, n)
    assert len(x) == n + 1
    assert deriv_op is not None
    y_i = np.array([f(t(x_i)) for x_i in x])
    int_f = float(integral_op @ y_i)
    assert abs(int_f) <  1e-4
    y_i = np.array(y_i)

    df_at_y = [
        d_i @ y_i for d_i in zip(deriv_op)
    ]
    assert all(
        abs(val - 1) < 1e-4 for val in df_at_y
    ), f"{df_at_y}"




def test_generate_discritisation_operators_with_scaling():
    # if we take
    # f(x) = x^2
    # then we should have
    # int_0^1 f(x) = 1/3
    #
    # Evaluate y_i = f(x_i) for the knot points
    # int_0^1 f  = sum(w_i, y_i)  = 1/3
    # and d

    interval = [0, 1]
    n = 5

    def f(arg):
        return arg**2

    def df(arg):
        return 2 * arg

    x, t, bases, deriv_op, integral_op = generate_discritisation_operators(interval, n)
    assert len(x) == n + 1
    assert deriv_op is not None
    y_i = np.array([f(t(x_i)) for x_i in x])
    int_f = float(integral_op @ y_i)
    assert abs(int_f - 1/3) <  1e-4
    y_i = np.array(y_i)
    df_at_y = [
        (d_i @ y_i, df(t(x_i))) for d_i, x_i in zip(deriv_op, x)
    ]
    assert all(
        abs(val - true_val) < 1e-4 for val, true_val in df_at_y
    ), f"{df_at_y}"

    basis_matrix = np.array(bases)

    # x(tau_j) = sum(x_i * L(i, tau_j,x))
    #          = ([1, tau_j, tauj_^2, ...] @ basis) @ x

    for j, tau_j in enumerate(x):
        t_vector = np.array([
            tau_j **i for i in range(n + 1)
        ]).reshape((1,6))

        expected_bases = np.array([
            lagrange_polynomial(i, tau_j, x)
            for i in range(n + 1)
        ])
        row = t_vector @ basis_matrix
        assert np.allclose(row, expected_bases)



def test_interpolating_poly_collection():

    control_space = [
        # (0 -> 2) should be split into 4
        PiecewiseConstantVariable('u_step', sample_rate=2),

        # (0 -> 0.5) should be split into 2,
        SpikeVariable('u_spike', 0.25),

        # No effect
        ConstantControlVariable('u_constant')
    ]
    expected_intervals = 5
    t_final = 2
    options = TranscriptionOptions(minimum_degree=3)

    intervals = split_at_non_differentiable_points(
        control_space, t_final, options
    )
    assert len(intervals) == expected_intervals
    assert intervals[0][0] == 0
    assert intervals[-1][1] == t_final

    assert all(
        first_end == second_start
        for (_, first_end), (second_start, _)
        in zip(intervals[:-1], intervals[1:])
    )




