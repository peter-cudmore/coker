from functools import reduce
from operator import mul
from typing import List, Tuple, Callable, Optional

import numpy as np

from coker.dynamics.types import (
    TranscriptionOptions,
    ControlVariable,
    PiecewiseConstantVariable,
    SpikeVariable,
    LossFunction
)


def legendre_coefficient(n, k):
    if k > n:
        return 0
    if (n, k) == (0, 0):
        return 1
    if (n, k) == (1, 1):
        return 1
    if k <= 1:
        if (k == 1 and n % 2 == 0) or (k == 0 and n % 2 == 1):
            return 0
        coeff = -(n + k - 1) / (n - k)
        return coeff * legendre_coefficient(n - 2, k)

    coeff = -((n - k + 2) * (n + k - 1) / (k * (k - 1)))
    return coeff * legendre_coefficient(n, k - 2)


def expand_coefficients(roots: List[float]):
    # p(x) = (x - x_0) (x - x_1) ...  (x  - x_n)
    #
    # final case:
    #   p(x) = [] , r(x) = [a_0, a_1, ... a_n, 1]
    #   where r(x) = sum_i(x^i a_i)
    #   -> return r(x)

    # (x - r_0)(x - r_1) (\sum c_i x^i)
    #  -> (x - r_0)( \sum( (r_1 c_i + c_{i-1}) x^i)

    # initial case: p(x) = coeffs, r(x) = [1]
    coeffs = [1]
    while roots:
        root = roots.pop()
        coeffs = [root * coeffs[0]] + [
            root * c + c_last for c, c_last in zip(coeffs[1:] + [0], coeffs)
        ]
    return coeffs


def lgr_points(n: int) -> List[float]:
    assert n > 0
    # Points are roots of P_n(x) + P_{n-1}(x)
    # Where P_n(x) is the nth Legendre Polynomial

    companion_matrix = np.diag(np.ones(n - 1), 1)
    leading_term = legendre_coefficient(n, n)
    for i in range(n):
        companion_matrix[-1, i] = (
            -(legendre_coefficient(n - 1, i) + legendre_coefficient(n, i))
            / leading_term
        )
    roots = np.linalg.eigvals(companion_matrix).tolist()
    roots.append(1)
    roots.sort()

    return [-1] + roots[1:]


def evaluate_legendre_polynomial(x, n):
    return np.polynomial.legendre.legval(x, [0] * n + [1])


def generate_discritisation_operators(
        interval: Tuple[float, float],
        n: int
) -> Tuple[
    List[float],
    Callable[[float], float],
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
]:
    """
    Generates discretisation operators over an interval using Legendre-Gauss-Radau (LGR)
    collocation points. This function computes knot points, Legendre polynomial bases,
    as well as derivative and integral operators for numerical methods.

    Args:
        interval: A tuple representing the interval [a, b] over which the discretisation
            is computed.
        n: The number of discretisation points.

    Returns:
        Tuple containing:
            List[float]: Knot points of length (n + 1).
            Callable[[float], float]: Transformed time variable.
            List[np.ndarray]: Legendre polynomial bases at each point.
            List[np.ndarray]: Derivative operators at each knot point.
            np.ndarray: Integral operator for the interval.
    """
    colocation_times = np.array(lgr_points(n))

    # using n LRG collocation points covering [-1, 1)
    # plus an additional non-collocated point a +1
    #

    bases = np.empty((n + 1, n + 1))
    time_scaling_factor = (interval[1] - interval[0]) / 2
    colocation_coeff = np.zeros((n + 1, n + 1))
    continuity_coeff = np.zeros(n + 1)
    quad_coeff = np.zeros_like(continuity_coeff)

    for i in range(n + 1):
        tau_i = colocation_times[i]
        factors = [
            np.poly1d([1, -tau_j]) / (tau_i - tau_j)
            for tau_j in colocation_times
            if tau_i != tau_j
        ]
        basis_i = reduce(mul, factors)
        bases[:, i] = basis_i.c[::-1]
        dbasis_i = np.polyder(basis_i)

        continuity_coeff[i] = basis_i(1)
        colocation_coeff[i, :] = [
            dbasis_i(tau_j) / time_scaling_factor  for tau_j in colocation_times
        ]
        quad_coeff[i] = np.polyint(basis_i)(1.0) * time_scaling_factor
    # see https://mathworld.wolfram.com/RadauQuadrature.html
    weights = (
        np.array(
            [2 / n**2]
            + [
                (1 - x_i) / (n * evaluate_legendre_polynomial(x_i, n - 1)) ** 2
                for x_i in colocation_times[1:-1]
            ]
            + [0]
        )
        * time_scaling_factor
    )

    t = lambda tau: (interval[0] + interval[1]) / 2 + tau * time_scaling_factor

    derivative = [row for row in colocation_coeff.T]
    bases = [row for row in bases]
    return colocation_times, t, bases, derivative, weights.reshape(1, n + 1)


def split_at_non_differentiable_points(
    control_variables: List[ControlVariable],
    t_final: float,
    transcription_options: TranscriptionOptions,
    additional_points: Optional[List[float]] = None
) -> List[Tuple[float, float]]:

    interval_boundaries = set(additional_points) if additional_points else set()
    for d in control_variables:
        if isinstance(d, PiecewiseConstantVariable):
            assert d.sample_rate > 0, "Sample rate must be positive"
            steps = t_final * d.sample_rate
            interval_boundaries |= {i* t_final / steps for i in range(int(steps))}

        if isinstance(d, SpikeVariable):
            assert (
                0 <= d.time < t_final
            ), "Spike time must be within integration window"

            interval_boundaries.add(d.time)

    if 0 not in interval_boundaries:
        interval_boundaries.add(0)

    if t_final not in interval_boundaries:
        interval_boundaries.add(t_final)

    sorted_boundaries = list(interval_boundaries)

    sorted_boundaries.sort()

    while len(sorted_boundaries) < transcription_options.minimum_n_intervals:
        intervals = [
            (stop - start, (stop + start) / 2)
            for start, stop in zip(
                sorted_boundaries[:-1], sorted_boundaries[1:]
            )
        ]
        max_length = max(l for l, _ in intervals)

        if all(length == max_length for length, _ in intervals):
            sorted_boundaries += [mid for _, mid in intervals]
        else:
            sorted_boundaries += [
                point for length, point in intervals if point == max_length
            ]
        sorted_boundaries.sort()

    return [
        (start, stop)
        for start, stop in zip(
            sorted_boundaries[:-1], sorted_boundaries[1:]
        )
    ]
