# Coker

**Build mathematical models in Python. Use them for simulation, parameter fitting,
and optimisation.**

Moving from a numerical prototype to an optimisation problem often means
rewriting the same equations for a different tool. Coker lets you express
supported computations as Python functions and reuse them through numerical,
symbolic, and tensor backends.

Use NumPy for numerical evaluation, CasADi for solver-backed optimisation and
dynamics, SymPy for symbolic inspection, or PyTorch for tensor and autograd
workflows. Coker also provides tools for composing dynamical systems, fitting
model parameters, and building robotics models.

**Coker is alpha software.** Backend capabilities differ; check the supported
workflows and limitations before choosing a backend.

**[Documentation][documentation] · [Get started][getting-started] ·
[Browse practical examples][how-to] · [Backend support][backend-guide]**

## Install and try

The base package requires Python 3.10 or newer and includes NumPy and SymPy:

```bash
pip install coker
```

### Evaluate a model, then fit it to measurements

For a spring, force is stiffness times extension: `F = kx`. Define that
relationship once, evaluate it numerically, then use the same function in a
constrained least-squares fit. Install the CasADi extra for this example:

```bash
pip install "coker[casadi]"
```

```python
import numpy as np
from coker import Scalar, function
from coker.toolkits.codesign import Minimise, ProblemBuilder

force = function(
    [Scalar("extension"), Scalar("stiffness")],
    lambda extension, stiffness: stiffness * extension,
    backend="casadi",
)
print(float(force(0.02, 150.0)))  # 3.0 N

extensions = np.array([0.01, 0.02, 0.03])  # metres
measurements = np.array([1.5, 3.0, 4.5])  # newtons
with ProblemBuilder() as builder:
    k = builder.new_variable("stiffness", initial_value=100.0)
    residuals = [force(x, k) - y for x, y in zip(extensions, measurements)]
    builder.objective = Minimise(sum(r * r for r in residuals))
    builder.constraints = [k >= 0]
    builder.outputs = [k]
    fit = builder.build("casadi")

loss, stiffness = fit()
assert fit.solve_info is not None and fit.solve_info.success
np.testing.assert_allclose(stiffness, 150.0, atol=1e-5)
print(round(np.asarray(stiffness).item(), 1))  # 150.0 N/m
```

The fit recovers a stiffness of **150 N/m**. CasADi uses IPOPT, which also
prints its solver log. See the [solver tuning recipe][solver-tuning] for solver
options and result inspection.

### Inspect symbolic matrix maths, then differentiate with PyTorch

For coupled springs, the elastic energy is `E(x, K) = ½ xᵀKx`: `x` is a
displacement vector and `K` is a symmetric stiffness matrix. Trace the formula
once, inspect its symbolic expression with SymPy, then evaluate and differentiate
it using PyTorch tensors:

```bash
pip install "coker[pytorch]"
```

```python
import numpy as np
import torch
from coker import VectorSpace, function
from coker.backends.sympy import SympyBackend

energy = function(
    [VectorSpace("x", 2), VectorSpace("K", (2, 2))],
    lambda x, K: 0.5 * np.dot(x, K @ x),
    backend="pytorch",
)
_, expression = SympyBackend().lower_to_symbolic(energy)
print(expression)  # The same quadratic form, as a symbolic expression.

x = torch.tensor([1.0, 2.0], requires_grad=True)
K = torch.tensor([[4.0, -1.0], [-1.0, 2.0]])
value = energy(x, K)
value.backward()
print(value.item())  # 4.0
print(x.grad)       # tensor([2., 3.])
torch.testing.assert_close(value, torch.tensor(4.0))
torch.testing.assert_close(x.grad, K @ x.detach())
```

With displacements in metres and stiffness in N/m, the energy is **4 J**.
For symmetric `K`, the gradient is `Kx`; its negative gives the restoring
forces, here **[-2, -3] N**. The NumPy-style formula becomes a Coker graph,
not a conversion of the input tensors to NumPy, so PyTorch retains the gradient
path. See the [backend interoperability recipes][interoperability] to export
functions as PyTorch modules.

## What can I build?

- **Simulate a dynamical system:** define an ODE and evaluate its state and
  outputs over time. [Simulation recipe][simulation].
- **Fit model parameters:** estimate unknown parameters from observations.
  [System identification recipe][identification].
- **Solve a constrained path-planning problem:** optimise waypoints subject to
  geometric constraints. [Path-planning recipe][path-planning].
- **Build a robot-arm model:** compose joints and rigid bodies, evaluate forward
  kinematics, and plot the result. [Robot-arm recipe][robot-arm].
- **Work with PyTorch modules:** export a Coker function as a module or use a
  native module inside a Coker computation.
  [Backend interoperability recipes][interoperability].

For the execution model and design rationale, read the [architecture guide][architecture].

## Choose a backend

| Backend | Intended use | Installation |
| --- | --- | --- |
| NumPy | Numerical evaluation and small-system reference workflows | `pip install coker` |
| CasADi | Solver-backed optimisation, dynamics, and symbolic workflows | `pip install "coker[casadi]"` |
| SymPy | Symbolic inspection and expression manipulation | Included in the base package |
| PyTorch | Tensor execution, autograd, and supported fitting workflows | `pip install "coker[pytorch]"` |
| JAX | Alternate array execution; Python 3.11 or newer | `pip install "coker[jax]"` |

Backends are not interchangeable for every operation or solver. Follow the
[backend tutorial][backend-tutorial] to compare numerical and symbolic execution,
the [backend guide][backend-guide] for capability limits, and the
[CasADi tuning recipe][solver-tuning] for solver options.

## Status and limitations

Coker is in active, alpha-stage development. Public APIs may change, and support
for a workflow on one backend does not imply support on the others.

Installed-wheel CI covers the base NumPy/SymPy package on Ubuntu, Windows, and
macOS with Python 3.10 and 3.13. Individual CasADi, JAX, and PyTorch extras are
checked on Ubuntu with Python 3.11 and 3.13; optional extras on Windows/macOS are
outside this isolated-wheel coverage. These checks do not certify every backend
feature on every platform.

Backend CI checks use CPU execution. CUDA support remains experimental and
unverified; see the [backend support boundaries][backend-guide].
See [CHANGELOG.md](CHANGELOG.md) for changes and the
[issue tracker](https://github.com/pneumatica-bio/coker/issues) for known problems.

## Contributing and credit

We encourage small, ready-to-integrate contributions such as documentation
improvements and bug fixes. Significant contributions will rarely be accepted
unless they come from existing contributors. CI must pass before maintainer
review begins.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and submission rules;
PRs target `develop`. We give credit by name to all contributors and expect
accurate attribution. Add your name to [AUTHORS.md](AUTHORS.md) when contributing
and follow the [credit policy](CREDITS.md).

## License

Copyright (c) 2026 **Pneumatica Bio**. **Peter Cudmore** is the sole author to date.

Coker is licensed under **MPL-2.0**. See [LICENSE.TXT](LICENSE.TXT).
External contributions are accepted under the same licence without copyright
assignment; see [contribution licensing](CONTRIBUTING.md#contribution-licensing).

[documentation]: https://pneumatica-bio.github.io/coker/
[getting-started]: https://pneumatica-bio.github.io/coker/basics/getting_started.html
[how-to]: https://pneumatica-bio.github.io/coker/how_to/index.html
[backend-guide]: https://pneumatica-bio.github.io/coker/backends.html
[solver-tuning]: https://pneumatica-bio.github.io/coker/how_to/casadi_solver_tuning.html
[interoperability]: https://pneumatica-bio.github.io/coker/how_to/backend_interoperability.html
[simulation]: https://pneumatica-bio.github.io/coker/how_to/simulation.html
[identification]: https://pneumatica-bio.github.io/coker/how_to/system_identification.html
[path-planning]: https://pneumatica-bio.github.io/coker/how_to/path_planning.html
[robot-arm]: https://pneumatica-bio.github.io/coker/how_to/robot_arm.html
[architecture]: https://pneumatica-bio.github.io/coker/architecture/index.html
[backend-tutorial]: https://pneumatica-bio.github.io/coker/tutorials/backends.html
