"""Exercise an installed Coker wheel without test dependencies or source code.

Run from outside the checkout with the target environment's Python:
    /path/to/venv/bin/python -I /checkout/scripts/smoke_installation.py base
"""

from __future__ import annotations

import argparse
from enum import Enum
from importlib import metadata, util
from pathlib import Path
import sys
import sysconfig


class Installation(Enum):
    BASE = "base"
    CASADI = "casadi"
    JAX = "jax"
    PYTORCH = "pytorch"


DEPENDENCIES = {
    Installation.BASE: frozenset(),
    Installation.CASADI: frozenset({"casadi"}),
    Installation.JAX: frozenset({"jax"}),
    Installation.PYTORCH: frozenset({"torch", "torchdiffeq"}),
}
OPTIONAL_MODULES = frozenset({"casadi", "jax", "torch", "torchdiffeq"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_dependencies(installation: Installation) -> None:
    required = DEPENDENCIES[installation]
    for name in sorted(OPTIONAL_MODULES):
        available = util.find_spec(name) is not None
        require(
            available == (name in required),
            f"{installation.value}: {name} must be "
            f"{'installed' if name in required else 'absent'}",
        )
    print(
        f"Dependency isolation: required={sorted(required)}, "
        f"absent={sorted(OPTIONAL_MODULES - required)}"
    )


def check_installation() -> None:
    import coker

    origin = Path(coker.__file__).absolute()
    resolved_origin = origin.resolve()
    checkout = Path(__file__).resolve().parents[1]
    require(
        not resolved_origin.is_relative_to(checkout / "src"),
        f"Source import: {origin}",
    )
    library_paths = {
        Path(sysconfig.get_path(name)).absolute()
        for name in ("purelib", "platlib")
    }
    require(
        any(origin.is_relative_to(path) for path in library_paths),
        f"Coker is outside the selected interpreter's installation: {origin}",
    )
    distribution = metadata.distribution("coker")
    installed_files = distribution.files
    require(
        installed_files is not None, "Coker distribution has no file manifest"
    )
    require(
        any(
            str(file).replace("\\", "/") == "coker/__init__.py"
            and Path(distribution.locate_file(file)).resolve()
            == resolved_origin
            for file in installed_files
        ),
        f"Imported Coker is not the installed distribution artifact: {origin}",
    )
    print(
        f"Installed Coker {distribution.version}: {origin} ({sys.executable})"
    )


def check_base() -> None:
    import coker
    import numpy as np
    import sympy as sp
    from coker.backends import get_backend_by_name

    def polynomial(x, p):
        return x * x + p

    spaces = [coker.Scalar("x"), coker.Scalar("p")]
    numerical = coker.function(spaces, polynomial, backend="numpy")
    np.testing.assert_allclose(numerical(3.0, 2.0), 11.0)

    symbolic = coker.function(spaces, polynomial, backend="sympy")
    backend = get_backend_by_name("sympy", set_current=False)
    arguments, expression = backend.lower_to_symbolic(symbolic)
    require(
        sp.simplify(expression - (arguments[0] ** 2 + arguments[1])) == 0,
        f"Unexpected SymPy polynomial: {expression}",
    )
    np.testing.assert_allclose(
        float(expression.subs(dict(zip(arguments, (3.0, 2.0))))), 11.0
    )
    print("NumPy evaluation and SymPy lowering/substitution: x² + p = 11")


def check_casadi() -> None:
    import casadi as ca
    import coker
    import numpy as np
    from coker.algebra.ops import Noop
    from coker.backends import get_backend_by_name

    polynomial = coker.function(
        [coker.Scalar("x")], lambda x: x * x + 2, backend="casadi"
    )
    np.testing.assert_allclose(np.asarray(polynomial(ca.DM(3))), 11.0)
    backend = get_backend_by_name("casadi", set_current=False)
    x_final, z_final, q_final = backend.evaluate_integrals(
        [
            lambda _t, _x, z, _u, _p: z,
            lambda _t, x, z, _u, _p: z - x,
            Noop(),
        ],
        [np.array([1.0]), np.array([1.0]), None],
        0.25,
        [None, None],
    )
    expected = np.exp(0.25)
    np.testing.assert_allclose(np.asarray(x_final), expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(z_final), expected, rtol=1e-5)
    require(q_final is None, "IDAS returned unexpected quadrature")
    print(
        "CasADi function and Coker residual integration: IDAS, x' = z, z = x"
    )

    # Coker's residual API selects IDAS. Exercise collocation explicitly;
    # an IDAS failure must not silently count as a successful fallback.
    x = ca.MX.sym("x")
    z = ca.MX.sym("z")
    collocation = ca.integrator(
        "smoke_collocation",
        "collocation",
        {"x": x, "z": z, "ode": z, "alg": z - x},
        0.0,
        0.25,
        {"number_of_finite_elements": 20, "interpolation_order": 3},
    )
    result = collocation(x0=1.0, z0=1.0)
    np.testing.assert_allclose(np.asarray(result["xf"]), expected, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(result["zf"]), expected, rtol=1e-5)
    print("CasADi explicit collocation: same residual and exp(0.25) oracle")


def check_jax() -> None:
    import jax

    jax.config.update("jax_platforms", "cpu")
    import jax.numpy as jnp
    import coker
    import numpy as np

    function = coker.function(
        [coker.VectorSpace("x", 2)],
        lambda x: x[0] * x[0] + x[1] * x[1],
        backend="jax",
    )
    argument = jnp.array([2.0, -3.0])
    require(
        all(device.platform == "cpu" for device in argument.devices()),
        "Non-CPU JAX array",
    )
    np.testing.assert_allclose(np.asarray(function(argument)), 13.0)
    print("JAX CPU array through Coker function: squared norm = 13")


def check_pytorch() -> None:
    import coker
    import numpy as np
    import torch
    import torchdiffeq
    from coker.algebra.ops import Noop
    from coker.backends.pytorch import PytorchBackend

    require(callable(torchdiffeq.odeint), "torchdiffeq.odeint is unavailable")
    function = coker.function(
        [coker.VectorSpace("x", 2)],
        lambda x: x[0] * x[0] + x[1] * x[1],
        backend="pytorch",
    )
    argument = torch.tensor(
        [2.0, -3.0], dtype=torch.float64, device="cpu", requires_grad=True
    )
    result = function(argument)
    require(isinstance(result, torch.Tensor), "Coker did not return a tensor")
    require(result.device.type == "cpu", "Non-CPU PyTorch result")
    np.testing.assert_allclose(result.detach().numpy(), 13.0)
    result.backward()
    torch.testing.assert_close(
        argument.grad, torch.tensor([4.0, -6.0], dtype=torch.float64)
    )

    initial = torch.tensor(
        [2.0], dtype=torch.float64, device="cpu", requires_grad=True
    )
    quadrature = torch.zeros_like(initial, requires_grad=True)
    x_final, z_final, q_final = PytorchBackend(
        device="cpu"
    ).evaluate_integrals(
        [
            lambda _t, x, _z, _u, _p: x * 0,
            Noop(),
            lambda _t, x, _z, _u, _p: x,
        ],
        [initial, None, quadrature],
        0.5,
        [None, None],
    )
    require(z_final is None, "Unexpected algebraic ODE result")
    require(
        x_final.device.type == q_final.device.type == "cpu",
        "Non-CPU ODE result",
    )
    torch.testing.assert_close(x_final, initial)
    torch.testing.assert_close(q_final, torch.ones_like(initial))
    q_final.sum().backward()
    torch.testing.assert_close(initial.grad, torch.full_like(initial, 0.5))
    torch.testing.assert_close(quadrature.grad, torch.ones_like(quadrature))
    print("PyTorch CPU tensor/autograd: norm² = 13, gradient = [4, -6]")
    print("torchdiffeq CPU ODE/quadrature/autograd: x=2, q(0.5)=1, dq/dx0=0.5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "installation", choices=[item.value for item in Installation]
    )
    installation = Installation(parser.parse_args().installation)
    require(
        bool(sys.flags.isolated), "Invoke with Python -I for isolated imports"
    )
    check_dependencies(installation)
    check_installation()
    check_base()
    if installation is Installation.CASADI:
        check_casadi()
    elif installation is Installation.JAX:
        check_jax()
    elif installation is Installation.PYTORCH:
        check_pytorch()
    print(f"Installed-wheel smoke passed: {installation.value} (CPU only)")


if __name__ == "__main__":
    main()
