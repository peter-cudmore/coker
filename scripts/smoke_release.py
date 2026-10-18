"""Exercise the installed wheel with python -I outside the checkout."""

import json
import sys
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

import coker
from coker import Scalar, VectorSpace, function
from coker.dynamics import VariationalProblem, create_autonomous_ode
from coker.parameters import BoundedVariable


def check_install() -> None:
    assert sys.flags.isolated, "run this smoke with python -I"
    for location in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (location, *location.parents):
            assert not (
                (parent / "pyproject.toml").is_file()
                and (parent / "src" / "coker").is_dir()
            ), "run the copied smoke outside the source checkout"
    installed = distribution("coker")
    module = Path(coker.__file__).absolute()
    assert module.resolve() == Path(
        installed.locate_file("coker/__init__.py")
    ).resolve(), "coker must import from the installed distribution"
    assert module.is_relative_to(
        Path(sys.prefix).absolute()
    ), "coker must be installed in the smoke environment"
    direct_url = installed.read_text("direct_url.json")
    if direct_url is not None:
        assert not json.loads(direct_url).get("dir_info", {}).get(
            "editable", False
        ), "editable installs do not exercise the release wheel"
    print(f"Installed coker {installed.version}: {module}")


def check_numpy() -> None:
    scalar = function(
        arguments=[Scalar("x")],
        implementation=lambda x: 2 * x + 1,
        backend="numpy",
    )
    matrix = np.array([[1.0, 0.0], [0.0, -1.0]])
    vector = function(
        arguments=[VectorSpace("x", 2)],
        implementation=lambda x: matrix @ x,
        backend="numpy",
    )
    scalar_value = scalar(3)
    vector_value = vector(np.array([1.0, 2.0]))
    np.testing.assert_allclose(scalar_value, 7)
    np.testing.assert_allclose(vector_value, [1.0, -2.0])
    print(f"NumPy: f(3)={scalar_value}, A@[1, 2]={vector_value}")


def check_casadi() -> None:
    horizon = 4.0
    system = create_autonomous_ode(
        x0=np.array([2.0]),
        xdot=lambda x, p: -p[0] * x,
        parameters=VectorSpace("p", 1),
        backend="casadi",
    )
    times = np.linspace(horizon / 8, horizon, 8)
    observations = 2.0 * np.exp(-0.7 * times)

    def loss(trajectory, parameters):
        return sum(
            (trajectory(float(time), parameters) - float(observed)) ** 2
            for time, observed in zip(times, observations)
        )

    fit = VariationalProblem(
        system=system,
        loss=loss,
        parameters=[BoundedVariable("rate", 0.05, 2.0, guess=0.3)],
        t_final=horizon,
        backend="casadi",
    )()
    assert fit.solve_info.success, "CasADi decay-rate fit did not succeed"
    rate = fit.parameters["rate"]
    np.testing.assert_allclose(rate, 0.7, atol=1e-4, rtol=0)
    max_error = 0.0
    # Physical endpoints, mesh boundaries, and off-grid times beyond t=1.
    for fraction in [0.0, 0.125, 0.25, 0.375, 0.5, 0.73, 0.75, 1.0]:
        time = fraction * horizon
        expected = 2.0 * np.exp(-0.7 * time)
        for actual in (fit(time), fit.state(time)):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
            max_error = max(
                max_error, float(np.max(np.abs(actual - expected)))
            )
    print(
        f"CasADi: success={fit.solve_info.success}, rate={rate:.7f}, "
        f"horizon={horizon}, max trajectory error={max_error:.3g}"
    )


def main() -> None:
    check_install()
    check_numpy()
    check_casadi()


if __name__ == "__main__":
    main()
