# Coker

Coker is a mathematical programming toolkit and compiler pipeline for technical computing in Python. You define computations as ordinary Python callables, compile them into a symbolic representation, and lower that representation to multiple execution backends. The project is aimed at numerical modelling, optimisation, dynamics, and embedded-oriented execution workflows.

The current package metadata marks Coker as **alpha** software (`Development Status :: 3 - Alpha`).

## What Coker does

Coker combines a few layers that usually live in separate tools:

- **Symbolic function tracing** via `coker.function`, `Scalar`, `VectorSpace`, and `FunctionSpace`.
- **Backend lowering** to `numpy`, `casadi`, `sympy`, and the native `coker` backend.
- **Differentiable execution models** that support evaluation, composition, and conditional expressions.
- **Dynamics and variational problem tooling** for ODE systems, transcription helpers, and solver-backed optimisation workflows.
- **Domain toolkits** for spatial algebra, rigid-body kinematics, system modelling, and codesign-style mathematical programs.

## Core capabilities

### 1. Compile Python callables into reusable functions

The primary entry point is `coker.function`. You describe argument spaces explicitly, provide a Python implementation, and choose a backend.

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

The same traced function can be lowered to different backends depending on the job:

- `numpy` for direct numerical execution
- `casadi` for optimisation-oriented symbolic workflows
- `sympy` for symbolic inspection and printing
- `coker` for Coker's native compact execution graph

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
```

Base installation:

```bash
pip install coker
```

## Toolkit areas in this repository

### Symbolic algebra and function composition

`src/coker/algebra/` contains the tracing and function model used throughout the project. The test suite exercises:

- scalar and vector symbolic ops
- higher-order composition with `FunctionSpace`
- conditional expressions via `if_then_else`
- backend-specific lowering paths

### Native Coker backend

The `coker` backend lowers traced functions into a compact workspace-oriented graph. The internal architecture in `docs/backend_architecture.rst` describes:

- contiguous workspace allocation for function values
- sparse bilinear layers for affine/quadratic-compatible ops
- generic vector layers for non-bilinear work
- value and tangent propagation over the same execution graph

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
- **system_modelling**: block/component modelling with a discoverable standard library
- **codesign**: a small problem-builder API for optimisation-style programs

The test suite includes concrete examples such as a single pendulum, double pendulum, SCARA manipulator, and hexapod leg models.

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
- `docs/backend_architecture.rst` — native backend execution model
- `examples/pid_example.py` — block-model style PID/plant composition
- `scripts/double_pendulum.py` — dynamics-oriented script example
- `tests/benchmarks/benchmark_backends.py` — benchmark scenarios for backend evaluation, ODE integration, and variational problems

## Development

The repository uses `uv` in CI for environment management.

Install a development environment:

```bash
uv sync --group dev --extra casadi --extra jax
```

Run the test suite:

```bash
uv run pytest
```

Build the documentation:

```bash
uv sync --group docs
uv run sphinx-build docs/ docs/_build/html -W --keep-going
```

Formatting and linting used in CI:

```bash
uv run black --check --diff src
uv run flake8 src tests docs examples scripts
```

## Project status
The package metadata marks Coker as alpha-stage software. The repository already includes automated coverage for symbolic operations, backend lowering, dynamics, and toolkit examples.

## License

Coker is licensed under the **MPL-2.0**. See `LICENSE.TXT` for the full text.
