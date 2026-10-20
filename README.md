# Coker

Coker is a mathematical programming toolkit and compiler pipeline for technical computing in Python. You define computations as ordinary Python callables, compile them into a symbolic representation, and lower that representation to multiple execution backends. The project is aimed at numerical modelling, optimisation, dynamics, and embedded-oriented execution workflows.

The current package metadata marks Coker as **alpha** software (`Development Status :: 3 - Alpha`).

## What Coker does

Coker combines a few layers that usually live in separate tools:

- **Backend lowering** to `numpy`, `casadi`, `jax`, `pytorch`, and `sympy`.
- **Differentiable execution models** that support evaluation, composition, and conditional expressions.
- **Dynamics and variational problem tooling** for ODE systems, transcription helpers, and solver-backed optimisation workflows.
- **Domain toolkits** for spatial algebra, rigid-body kinematics, and codesign-style mathematical programs.

## Core capabilities

### 1. Compile Python callables into reusable functions

The primary entry point is `coker.function`. You describe argument spaces explicitly, provide a Python implementation, and choose a backend.

Canonical imports are inventoried in [the API reference](docs/api/index.rst);
0.5 migration notes are in [Getting Started](docs/getting_started.rst). NumPy is the base-install default;
backend lookup, evaluation, and lowering do not change ambient selection unless
the application explicitly requests `get_backend_by_name(name, set_current=True)`.

```python
import numpy as np
from coker import function, Scalar, VectorSpace

f = function(
    arguments=[Scalar("x")],
    implementation=lambda x: 2 * x + 1,
    backend="numpy",
)
print(f(3))  # 7

A = np.array([[1.0, 0.0], [0.0, -1.0]])
g = function(
    arguments=[VectorSpace("x", 2)],
    implementation=lambda x: A @ x,
    backend="numpy",
)
print(g(np.array([1.0, 2.0])))  # [ 1. -2.]
```
### 2. Swap execution backends without rewriting the model

The same traced function can be lowered to different execution backends depending on the job:

- `numpy` for direct numerical execution
- `casadi` for optimisation-oriented symbolic workflows
- `jax` for alternate array execution
- `pytorch` for tensor-valued execution, autograd, and optional ODE/quadrature integration
- `sympy` for symbolic inspection and printing

```python
from coker import function, Scalar

f_casadi = function(
    arguments=[Scalar("x")],
    implementation=lambda x: x**2,
    backend="casadi",
)
```

Optional extras declared by the package:

```bash
pip install "coker[casadi]"
pip install "coker[jax]"
pip install "coker[pytorch]"
```

Base installation:

```bash
pip install coker
```

See the [backend guide](docs/backends.rst) for backend limits. Base NumPy/SymPy
installed-wheel CI covers Ubuntu, Windows, and macOS on Python 3.10 and 3.13;
each optional extra is checked separately on Ubuntu on Python 3.11 and 3.13.
JAX requires Python 3.11 or newer. These are CI coverage targets, not a claim
of observed success on every platform or backend feature parity. All accelerator
checks are CPU-only: CUDA tensor execution/autograd, NLP, ODE/quadratures where
implemented, and fitting remain experimental and unverified pending GPU release
evidence. Optional extras on Windows/macOS are outside this isolated-wheel boundary.

## Toolkit areas in this repository

### Symbolic algebra and function composition

`src/coker/algebra/` contains the tracing and function model used throughout the project. The test suite exercises:

- scalar and vector symbolic ops
- higher-order composition with `FunctionSpace`
- conditional expressions via `if_then_else`
- backend-specific lowering paths

### Dynamics and optimisation

`src/coker/dynamics/` exposes:

- `create_autonomous_ode`
- `direct_sum`
- `VariationalProblem`
- transcription helpers such as Legendre/LGR utilities
- backend-specific solver parameters and solve status reporting

The dynamics tests cover variational solvers, callbacks, direct-sum composition, and constrained parameter-fitting style problems.

### Robotics and modelling toolkits

The repository also includes domain-focused toolkits under `src/coker/toolkits/`:

- **spatial**: rotations, isometries, screws, adjoint operators, quaternions
- **kinematics**: rigid-body trees, joints, inertias, forward kinematics, dynamics examples
- **codesign**: a small problem-builder API for optimisation-style programs

The kinematics tests cover joint coordinate contracts, two-link and branched-tree forward kinematics and Jacobians, single-pendulum energy and inverse dynamics, and hexapod-leg models. See `tests/toolkits/kinematics/test_kinematics.py`; `scripts/double_pendulum.py` provides a separate two-link visualisation example.

## Repository layout

```text
src/coker/        Python package source
examples/         Small runnable examples
scripts/          Standalone modelling scripts
tests/            Backend, symbolic, dynamics, and toolkit coverage
docs/             Sphinx documentation
```

## Example files worth reading first

- `docs/getting_started.rst` — minimal symbolic function workflow
- `scripts/double_pendulum.py` — rigid-body visualisation (`uv run scripts/double_pendulum.py` installs its inline dependencies, including the local Coker checkout; set `MPLBACKEND=Agg` for headless startup)
- `tests/benchmarks/benchmark_backends.py` — benchmark scenarios for backend evaluation, ODE integration, and variational problems

## Development

The repository uses `uv` in CI for environment management.

Install a development environment:

```bash
uv sync --group dev --extra casadi --extra jax --extra pytorch
```

Run the test suite:

```bash
uv run pytest
```

Build the documentation:

```bash
uv sync --group docs --extra casadi --extra pytorch
uv run sphinx-build -b doctest docs/ docs/_build/doctest -W --keep-going
uv run sphinx-build docs/ docs/_build/html -W --keep-going
```

See [the documentation authoring guide](docs/AUTHORING.md) for section goals,
writing conventions, and executable examples. Source-only `Author notes`
record page-specific directions and planned additions.

Formatting and linting used in CI:

```bash
uv run black --check --diff src
uv run flake8 src tests docs examples scripts
```

## Project status
The package metadata marks Coker as alpha-stage software. The repository already includes automated coverage for symbolic operations, backend lowering, dynamics, and toolkit examples.

## License

Coker is licensed under the **MPL-2.0**. See `LICENSE.TXT` for the full text.
