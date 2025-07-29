import dataclasses
from typing import Optional, Callable
from enum import Enum
import numpy as np

from coker.toolkits.dynamical_systems import DynamicalSystem


# Given
# \dot{x} = f(t, x, z, u; p)
#       0 = g(t, x, z, u; p)
# \dot(q) = h(t, x, z, u; p)

# [x(0), q(0)]  = x0(u,p)

# Specify time points [t_0, ... t_n]
#

# Transcribe
# - reformulate dynamics so that time goes from -1, 1
# - Construct knot points at t_0...t_n
# - for each interval
# [T, X, Q]
#


@dataclasses.dataclass
class Data:
    """A structured format for dynamical system data.

    This class is primarily used to encapsulate system data points for analysis,
    simulation, or processing. It includes the time points, the corresponding
    input signals, and output signals of the dynamical system.

    Attributes:
        t_points (np.ndarray): An array representing the time points at which data
            is sampled.
        u_points (np.ndarray): An array containing the input signals to the system
            at the corresponding time points defined in `t_points`, with each row
            being the input signal for a single time point.
        y_points (np.ndarray): An array containing the output signals of the system
            at the corresponding time points defined in `t_points`, with each row
            being the output signal for a single time point.
    """

    t_points: np.ndarray
    u_points: np.ndarray
    y_points: np.ndarray

    def __post_init__(self):
        assert self.t_points.ndim == 1, "Time points must be a 1D array."
        assert self.u_points.ndim == 2, "Input signals must be a 2D array."
        assert self.y_points.ndim == 2, "Output signals must be a 2D array."
        assert (
            self.t_points.shape[0] == self.u_points.shape[0]
        ), "Number of time points and number of input points must match."
        assert (
            self.t_points.shape[0] == self.y_points.shape[0]
        ), "Number of time points and number of output points must match."


@dataclasses.dataclass
class ParameterBounds:
    """Container for parameter bounds for numerical optimization.

    This class is used to store and manage guessing values (initial estimates),
    lower bounds, and upper bounds for parameters typically involved in numerical
    optimization tasks or related computations.

    Attributes:
        guess (np.ndarray): Array containing initial guess values for the
            parameters.
        lower (np.ndarray): Array containing lower bounds for the parameters.
        upper (np.ndarray): Array containing upper bounds for the parameters.
    """

    guess: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @staticmethod
    def new_unbounded(guess: np.ndarray) -> "ParameterBounds":
        return ParameterBounds(
            guess=guess,
            lower=-np.inf * np.ones_like(guess, float),
            upper=np.inf * np.ones_like(guess, float),
        )

    def __post_init__(self):
        assert (
            self.guess.shape == self.lower.shape
        ), "Guess and lower bounds must have the same shape."
        assert (
            self.guess.shape == self.upper.shape
        ), "Guess and upper bounds must have the same shape."


InterpolationMethod = Enum("InterpolationMethod", ["linear", "hold"])


@dataclasses.dataclass
class FitOptions:
    absolute_tolerance: float = 1e-4
    input_interpolation_method: InterpolationMethod = InterpolationMethod.hold


def loss_mse(y_true: np.ndarray[float], y_pred: np.ndarray[float]):
    assert (
        y_true.shape == y_pred.shape
    ), "Shape of true and predicted values must match."

    y_error = y_true.flatten() - y_pred.flatten()
    (n,) = y_error.shape

    return (y_error.T @ y_error) / n


def fit_dynamical_system(
    system: DynamicalSystem,
    data: Data,
    p_guess: Optional[np.ndarray, ParameterBounds] = None,
    loss: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    options: Optional[FitOptions] = None,
):
    """
    Fits a given dynamical system to provided input and output data points by optimizing
    its parameters to match the observed system behavior.

    We assume the data corresponds to a single experiment.
    That is, if there are multiple paths $u_i(t), y_i(t)$, indexed by $i$, then for any given $t$,
    $u_j(t) = u_i(t)$ and the parameters $p$ (and hence initial conditions) are the same for all $i$.

    We assume that the time axis starts at zero.


    Args:
        system: The dynamical system to be fitted.
        data: The data to be fitted to the system.
        p_guess: Optional initial guess for the parameters of the system. Defaults to zero.
        loss: A loss function to be minimized. Defaults to the mean squared error.

    """
    (n_points,) = t_points.shape
    assert t_points.min() >= 0, "Time must not be negative."
    t_final = data.t_points.max()

    assert (
        n_points == y_points.shape[0]
    ), "Number of time points and number of output points must match."
    assert (
        n_points == u_points.shape[0]
    ), "Number of time points and number of input points must match."
    (y_tracer,) = system.y.output

    assert (
        y_tracer.dim == y_points.shape[1]
    ), "Second dimension of output points must match dimension of output."
    assert (
        system.inputs.dimension == u_points.shape[1]
    ), "Second dimension of input points must match dimension of input."

    if options is None:
        options = FitOptions()

    if loss is None:
        loss = loss_mse
    else:
        assert (
            loss(data.y_points, data.y_points) < options.absolute_tolerance
        ), "Loss function must be below cutoff at the initial guess."

    if p_guess is None:
        p_guess = ParameterBounds.new_unbounded(
            np.zeros((system.parameters.dimension,), float)
        )
    elif isinstance(p_guess, np.ndarray):
        p_guess = ParameterBounds.new_unbounded(p_guess)
    else:
        assert isinstance(
            p_guess, ParameterBounds
        ), f"{p_guess.__class__} not supported. Parameter guess must be a ParameterBounds object or an array."

    # Begin Transcription
    # take

    # t = 0.5 * (t_0 - t_f) s + 0.5 * (t_0 + t_f)
    # s = [-1, 1], dt/ds = T/2, T = (t_0 - t_f)
    #
    #  dxds = (2/T)  * f(t(s), x(s), z(s), u(t(s)); p)
    #     0 = g(t(s), x(s), z(s), u(t(s)); p)
    #  dqds = (2/T) h(t(s), x(s), z(s), u(t(s); p)
    #
    # For each interval, we generate a sequence of knot points {s_k}
    # which we use to construct the interpolation functions $L_i$
    #
    # Construct X = sum L^X_i(s; s_ki) X_i
    #           Z = sum L^Z_i(s; s_ki) Z_i
    #           Q = sum L^Q_i(s; s_ki) Q_i
    #
    # Each L_i induces a differential operator D_i (a matrix)
    # and an Integral operator W_i (a matrix)
    # and an Quatrature integral I_i
    # So that for each interval we have
    #          0 = D_i X_i - f(   )
    #          0 = g(   )
    #          0 = I_i - h(  )
    #       Terminal Constraints
    #          0 = X_i(1) - X_{i-1}(-1)
    #          0 = X_i(1) - X_i(-1) - (T/2) W_i [f, ... ]

    # For each t_i in t_guesss,
    # - find interval that contains t_i
    # - s_i = (2t_i - (t0 + t_f)) / (t_0 - t_f)
    # - evaluate [x(s_i), z(s_i), q_i(s_i)]
    # - set y_i^est = y(t_i, x(s_i), ...; p)
    # - set y^est = sum P_iy_i^est = Y(X, Z, U; p)

    #
