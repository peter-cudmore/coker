import warnings

backends = ["numpy", "sympy"]
variational_backends = [
    "numpy",
]

try:
    import jax

    backends.append("jax")
except ImportError:
    warnings.warn("jax is not installed")

try:
    import casadi

    backends.append("casadi")
    variational_backends.append("casadi")
except ImportError:
    warnings.warn("casadi is not installed")


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", backends)

    if "variational_backend" in metafunc.fixturenames:
        metafunc.parametrize("variational_backend", variational_backends)
