from dataclasses import replace

from coker.dynamics.transcription.collocation import (
    InterpolatingPoly,
    InterpolatingPolyCollection,
    expand_coefficients,
    generate_discritisation_operators,
    legendre_coefficient,
    lgr_points,
)
from coker.dynamics.variational.mesh import (
    split_at_non_differentiable_points,
)
from coker.dynamics.variables import (
    PiecewiseConstantVariable,
    SpikeVariable,
)
from coker.dynamics.variational.problem import TranscriptionOptions
from coker.dynamics import ConstantControlVariable
import numpy as np
from functools import reduce
from operator import mul
import pytest

from coker import FunctionSpace, Scalar, VectorSpace
from coker.dynamics import (
    DynamicsSpec,
    create_autonomous_ode,
    create_control_system,
    create_dynamics_from_spec,
)


@pytest.fixture
def public_contract_dynamics_spec():
    return DynamicsSpec(
        inputs=None,
        parameters=None,
        algebraic=None,
        initial_conditions=lambda z, u, p: (np.ones(2), None),
        dynamics=lambda t, x, z, u, p: -x,
        constraints=None,
        outputs=lambda t, x, z, u, p, q: x,
        quadratures=None,
    )


@pytest.mark.parametrize(
    "initial_conditions",
    [
        lambda z, u, p: np.ones(2),
        lambda z, u, p: (np.ones(2), None, None),
    ],
)
def test_public_contract_spec_initial_conditions_require_pair(
    public_contract_dynamics_spec, initial_conditions
):
    public_contract_dynamics_spec.initial_conditions = initial_conditions

    with pytest.raises(ValueError):
        create_dynamics_from_spec(public_contract_dynamics_spec)


def test_public_contract_spec_initial_algebraic_dimension(
    public_contract_dynamics_spec,
):
    public_contract_dynamics_spec.algebraic = VectorSpace("z", 2)
    public_contract_dynamics_spec.initial_conditions = lambda z, u, p: (
        np.ones(2),
        np.zeros(3),
    )
    public_contract_dynamics_spec.constraints = lambda t, x, z, u, p: z

    with pytest.raises(ValueError):
        create_dynamics_from_spec(public_contract_dynamics_spec)


@pytest.mark.parametrize(
    "dynamics",
    [
        lambda t, x, z, u, p: (x, x),
        lambda t, x, z, u, p: np.zeros(3),
    ],
)
def test_public_contract_spec_dynamics_output_shape(
    public_contract_dynamics_spec, dynamics
):
    public_contract_dynamics_spec.dynamics = dynamics

    with pytest.raises(ValueError):
        create_dynamics_from_spec(public_contract_dynamics_spec)


@pytest.mark.parametrize("has_algebraic", [False, True])
def test_public_contract_spec_constraints_require_algebraic_companion(
    public_contract_dynamics_spec, has_algebraic
):
    if has_algebraic:
        public_contract_dynamics_spec.algebraic = VectorSpace("z", 2)
        public_contract_dynamics_spec.initial_conditions = lambda z, u, p: (
            np.ones(2),
            np.zeros(2),
        )
    else:
        public_contract_dynamics_spec.constraints = lambda t, x, z, u, p: x

    with pytest.raises(ValueError):
        create_dynamics_from_spec(public_contract_dynamics_spec)


def test_public_contract_spec_quadrature_requires_single_output(
    public_contract_dynamics_spec,
):
    public_contract_dynamics_spec.quadratures = lambda t, x, z, u, p: (x, x)

    with pytest.raises(ValueError):
        create_dynamics_from_spec(public_contract_dynamics_spec)


@pytest.mark.parametrize(
    "configure",
    [
        lambda spec: replace(spec, initial_conditions=42),
        lambda spec: replace(spec, dynamics=42),
        lambda spec: replace(
            spec,
            algebraic=VectorSpace("z", 2),
            initial_conditions=lambda z, u, p: (np.ones(2), np.zeros(2)),
            constraints=42,
        ),
        lambda spec: replace(spec, quadratures=42),
        lambda spec: replace(spec, outputs=42),
    ],
)
def test_public_contract_spec_callback_type(public_contract_dynamics_spec, configure):
    spec = configure(public_contract_dynamics_spec)

    with pytest.raises(TypeError):
        create_dynamics_from_spec(spec)


@pytest.fixture(params=[create_autonomous_ode, create_control_system])
def public_contract_ode_constructor(request):
    if request.param is create_autonomous_ode:
        return create_autonomous_ode, {"xdot": lambda x, p: -x}
    return create_control_system, {
        "xdot": lambda t, x, u, p: -x,
        "control": FunctionSpace("u", [Scalar("t")], [VectorSpace("u", 2)]),
    }


@pytest.mark.parametrize(
    "parameters,p_init",
    [
        (None, np.ones(2)),
        (Scalar("p"), np.ones(2)),
        (VectorSpace("p", 2), np.ones(3)),
    ],
)
def test_public_contract_ode_initial_parameter_configuration(
    public_contract_ode_constructor, parameters, p_init
):
    constructor, kwargs = public_contract_ode_constructor

    with pytest.raises(ValueError):
        constructor(np.ones(2), parameters=parameters, p_init=p_init, **kwargs)


def test_public_contract_ode_dynamics_shape(public_contract_ode_constructor):
    constructor, kwargs = public_contract_ode_constructor
    kwargs["xdot"] = (
        (lambda x, p: np.zeros(3))
        if constructor is create_autonomous_ode
        else (lambda t, x, u, p: np.zeros(3))
    )

    with pytest.raises(ValueError):
        constructor(np.ones(2), **kwargs)


@pytest.mark.parametrize(
    "u_init,error", [(np.zeros(2), TypeError), (lambda t: np.zeros(3), ValueError)]
)
def test_public_contract_control_initial_input(u_init, error):
    control = FunctionSpace("u", [Scalar("t")], [VectorSpace("u", 2)])

    with pytest.raises(error):
        create_control_system(np.ones(2), lambda t, x, u, p: -x, control, u_init=u_init)


def test_public_contract_control_requires_single_output():
    control = FunctionSpace(
        "u", [Scalar("t")], [VectorSpace("left", 2), VectorSpace("right", 2)]
    )

    with pytest.raises(ValueError):
        create_control_system(np.ones(2), lambda t, x, u, p: -x, control)


def test_public_contract_ode_output_requires_value(public_contract_ode_constructor):
    constructor, kwargs = public_contract_ode_constructor
    kwargs["output"] = (
        (lambda x, p: None)
        if constructor is create_autonomous_ode
        else (lambda t, x, u, p: None)
    )

    with pytest.raises(ValueError):
        constructor(np.ones(2), **kwargs)


def to_float(x: np.ndarray) -> float:
    assert x.ndim == 1
    assert x.shape[0] == 1
    return float(x[0])


def test_legendre_coeffs():
    # increasing in powers of x
    truth = [[1], [0, 1], [-1 / 2, 0, 3 / 2], [0, -3 / 2, 0, 5 / 2]]

    for n, coeffs in enumerate(truth):
        test_values = [(c, legendre_coefficient(n, k)) for k, c in enumerate(coeffs)]
        assert all(
            abs(true_value - test_value) < 1e-4
            for true_value, test_value in test_values
        )


def test_lgr_points():
    expected = np.array([-1, (1 - np.sqrt(6)) / 5, (1 + np.sqrt(6)) / 5, 1])
    result = lgr_points(3)
    assert np.allclose(result, expected)


def test_expand_coeffs():
    test_pairs = [([1, -1], [-1, 0, 1])]  # (x + 1)(x - 1) -> (-1 +0x + x^2)
    for roots, coeffs in test_pairs:
        result = expand_coefficients(roots)

        assert np.allclose(result, coeffs)


def lagrange_polynomial(i, value, points):
    return reduce(
        mul,
        [(value - p_j) / (points[i] - p_j) for j, p_j in enumerate(points) if j != i],
        1,
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
    int_f = to_float((integral_op @ y_i))
    assert abs(int_f) < 1e-4
    y_i = np.array(y_i)

    df_at_y = [d_i @ y_i for d_i in zip(deriv_op)]
    assert all(abs(val - 1) < 1e-4 for val in df_at_y), f"{df_at_y}"


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
    int_f = to_float(integral_op @ y_i)
    assert abs(int_f - 1 / 3) < 1e-4
    y_i = np.array(y_i)
    df_at_y = [(d_i @ y_i, df(t(x_i))) for d_i, x_i in zip(deriv_op, x)]
    assert all(abs(val - true_val) < 1e-4 for val, true_val in df_at_y), f"{df_at_y}"

    basis_matrix = np.array(bases)

    # x(tau_j) = sum(x_i * L(i, tau_j,x))
    #          = ([1, tau_j, tauj_^2, ...] @ basis) @ x

    for j, tau_j in enumerate(x):
        t_vector = np.array([tau_j**i for i in range(n + 1)]).reshape((1, 6))

        expected_bases = np.array(
            [lagrange_polynomial(i, tau_j, x) for i in range(n + 1)]
        )
        row = t_vector @ basis_matrix
        assert np.allclose(row, expected_bases)


def test_interpolating_poly_collection():

    control_space = [
        # (0 -> 2) should be split into 4
        PiecewiseConstantVariable("u_step", sample_rate=2),
        # (0 -> 0.5) should be split into 2,
        SpikeVariable("u_spike", 0.25),
        # No effect
        ConstantControlVariable("u_constant"),
    ]
    expected_intervals = 5
    t_final = 2
    options = TranscriptionOptions(minimum_degree=3)

    intervals = split_at_non_differentiable_points(control_space, t_final, options)
    assert len(intervals) == expected_intervals
    assert intervals[0][0] == 0
    assert intervals[-1][1] == t_final

    assert all(
        first_end == second_start
        for (_, first_end), (second_start, _) in zip(intervals[:-1], intervals[1:])
    )


def test_poly_collection_vector_constant():
    intervals = [(0, 1), (1, 2), (2, 3)]
    collocation_degree = [3, 4, 5]

    polys = [
        InterpolatingPoly(1, interval, degree, np.ones((degree + 1,)))
        for interval, degree in zip(intervals, collocation_degree)
    ]

    poly_collection = InterpolatingPolyCollection(polys)
    points = []
    for n_poly, poly in enumerate(poly_collection.polys):
        for n_point, (t, x, dx) in enumerate(poly.knot_points()):
            assert x == 1, f"Failed on poly {n_poly}, point {n_point}"
            assert abs(dx) < 1e-8
            points.append((t, x, dx))

    assert list(poly_collection.knot_points()) == points

    for t in np.linspace(0, 3, 10):
        value = poly_collection(t)
        assert abs(value - 1) < 1e-6


def test_poly_collection_vector_quadratic():
    def f(x):
        return [x, 2 + x**2]

    intervals = [(0, 1), (1, 2), (2, 3)]
    collocation_degree = [3, 4, 5]

    polys = [
        InterpolatingPoly(2, interval, degree, np.ones((2 * (degree + 1),)))
        for interval, degree in zip(intervals, collocation_degree)
    ]

    for poly in polys:
        for i, t in enumerate(poly.knot_times()):
            poly.values[2 * i : 2 * (i + 1)] = f(t)
    collection = InterpolatingPolyCollection(polys)
    t_eval = np.linspace(0, 3, 100)

    for t in t_eval:
        expected = np.array(f(t))
        actual = collection(t)
        error = np.linalg.norm(expected - actual)
        assert error < 1e-4, f"Error at t={t}: {error}"

    for poly in polys:
        for time, value, derivative in poly.knot_points():
            np.testing.assert_allclose(np.ravel(value), f(time), atol=1e-8)
            np.testing.assert_allclose(
                np.ravel(derivative), [1.0, 2 * time], atol=1e-10
            )
