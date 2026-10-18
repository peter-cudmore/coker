"""Exercise optional backend diagnostics in a real base-wheel installation."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


@pytest.fixture(scope="module")
def base_python():
    with tempfile.TemporaryDirectory(
        dir=os.environ.get("TACHI_DEPENDENCY_SCRATCH")
    ) as directory:
        root = Path(directory)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(root / "dist")],
            check=True,
            capture_output=True,
            text=True,
        )
        environment = root / "base"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)],
            check=True,
            capture_output=True,
            text=True,
        )
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        (wheel,) = (root / "dist").glob("*.whl")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)],
            check=True,
            capture_output=True,
            text=True,
        )
        yield python


def _run(python, source):
    result = subprocess.run(
        [str(python), "-I", "-c", source],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "backend_name, dependency",
    [("casadi", "casadi"), ("jax", "jax"), ("pytorch", "torch")],
)
def test_missing_extra_explains_installation(
    base_python, backend_name, dependency
):
    _run(
        base_python,
        f"""
from importlib.util import find_spec
from coker.backends import get_backend_by_name

assert find_spec({dependency!r}) is None
try:
    get_backend_by_name({backend_name!r})
except ImportError as error:
    assert {backend_name!r} in str(error), str(error)
    assert 'pip install "coker[{backend_name}]"' in str(error), str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
    assert error.__cause__.name == {dependency!r}
    if {backend_name!r} == "jax":
        assert "Python >=3.11" in str(error)
else:
    raise AssertionError("missing backend unexpectedly loaded")
""",
    )


def test_base_numpy_and_registration_stay_lazy(base_python):
    _run(
        base_python,
        """
import sys
from importlib.util import find_spec
from coker import Scalar, function
from coker.backends import get_backend_by_name, register_backend
from coker.backends.numpy import NumpyBackend

for name in ("casadi", "jax", "torch"):
    assert find_spec(name) is None

compiled = function([Scalar("x")], lambda x: x * x + 1)
assert compiled(3.0) == 10.0
register_backend("third_party", NumpyBackend)
assert get_backend_by_name("third_party").name == "third_party"
assert function([Scalar("x")], lambda x: x + 2, backend="third_party")(3) == 5
for name in ("casadi", "jax", "torch"):
    assert name not in sys.modules
    assert f"coker.backends.{name}" not in sys.modules
""",
    )


@pytest.mark.parametrize(
    "backend_name, dependency, source, exception, missing_name",
    [
        (
            "casadi",
            "casadi",
            "import missing_backend_transitive",
            "ModuleNotFoundError",
            "missing_backend_transitive",
        ),
        (
            "casadi",
            "casadi",
            "raise ModuleNotFoundError('broken casadi', name='casadi')",
            "ModuleNotFoundError",
            "casadi",
        ),
        (
            "jax",
            "jax",
            "import jaxlib",
            "ModuleNotFoundError",
            "jaxlib",
        ),
        (
            "pytorch",
            "torch",
            "raise ImportError('incompatible native library')",
            "ImportError",
            None,
        ),
        (
            "pytorch",
            "torch",
            "raise OSError('cannot load native library')",
            "OSError",
            None,
        ),
    ],
)
def test_broken_installed_dependency_keeps_diagnostics(
    base_python, backend_name, dependency, source, exception, missing_name
):
    # Install an importable but broken dependency module, not an import hook.
    _run(
        base_python,
        f"""
from pathlib import Path
import site
from coker.backends import get_backend_by_name

module = Path(site.getsitepackages()[0]) / "{dependency}.py"
module.write_text({source!r})
try:
    try:
        get_backend_by_name({backend_name!r})
    except {exception} as error:
        assert error.__cause__ is None
        assert 'pip install' not in str(error)
        if isinstance(error, ModuleNotFoundError):
            assert error.name == {missing_name!r}
        else:
            assert str(error) in {source!r}
    else:
        raise AssertionError("broken backend unexpectedly loaded")
finally:
    module.unlink()
""",
    )


def test_third_party_import_failure_is_not_a_missing_extra(base_python):
    _run(
        base_python,
        """
from pathlib import Path
import site
from coker.backends import get_backend_by_name, register_backend

root = Path(site.getsitepackages()[0])
module = root / "broken_plugin.py"
metadata = root / "broken_plugin-1.0.dist-info"
metadata.mkdir()
(metadata / "METADATA").write_text(
    "Metadata-Version: 2.1\\nName: broken-plugin\\nVersion: 1.0\\n"
)
(metadata / "entry_points.txt").write_text(
    "[coker.backends]\\nthird_party = broken_plugin:Backend\\n"
)
module.write_text("import casadi\\n")
try:
    try:
        get_backend_by_name("third_party")
    except ModuleNotFoundError as error:
        assert error.name == "casadi"
        assert error.__cause__ is None
    else:
        raise AssertionError("broken plugin unexpectedly loaded")

    def broken_factory():
        import casadi

    register_backend("casadi", broken_factory)
    try:
        get_backend_by_name("casadi")
    except ModuleNotFoundError as error:
        assert error.name == "casadi"
        assert error.__cause__ is None
    else:
        raise AssertionError("broken factory unexpectedly loaded")
finally:
    module.unlink()
    for path in metadata.iterdir():
        path.unlink()
    metadata.rmdir()
""",
    )
