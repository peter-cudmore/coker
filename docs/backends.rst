Backend guide
=============

Coker can lower the same traced function to several execution backends. The
backend name is selected with the ``backend=...`` argument to
:func:`coker.function`, or indirectly through higher-level APIs such as
:class:`coker.dynamics.VariationalProblem` and
:class:`coker.toolkits.codesign.ProblemBuilder`.

The bundled registry declares ``"numpy"``, ``"jax"``, ``"casadi"``,
``"pytorch"``, and ``"sympy"``. Additional plugins can register arbitrary
names.

``function()`` defaults to NumPy. ``get_current_backend()`` also falls back to
NumPy until the application deliberately selects a backend with
``get_backend_by_name(name, set_current=True)``. Lookup without that flag is
nonmutating. An implicit ``ProblemBuilder.build()`` or low-level
``coker.backends.evaluate(..., backend=None)`` consumes the current selection;
ordinary functions use their own ``backend`` name, even after ambient selection.
Explicit evaluation, lowering, and problem construction never select an ambient
backend. Variational problems retain a separate CasADi solver default.

See :doc:`api/index` for canonical imports and :doc:`getting_started` for migration notes.

.. _installation-support:

Installation CI and support boundary
------------------------------------

Installed-wheel CI covers the following CPU-only configurations:

.. list-table::
   :header-rows: 1

   * - Installation
     - Platform
     - Python
   * - Base (NumPy and SymPy)
     - Ubuntu, Windows, macOS
     - 3.10 and 3.13
   * - Each extra separately: CasADi, JAX, or PyTorch
     - Ubuntu
     - 3.11 and 3.13

CI builds a wheel once and installs it into a fresh environment for each
profile. Smoke checks verify installed-wheel provenance, including symlinked
installs, and run outside the checkout with isolated Python. They exercise the
base workflows in every profile, import the required optional backend without
skips, and assert that the other optional backend modules are absent.
The existing all-extra Ubuntu test suite on Python 3.10--3.13 remains separate;
JAX requires Python 3.11 or newer and is not exercised on Python 3.10.

This is the CI coverage boundary, not a report that every matrix job has
already passed or a promise of backend feature parity. Optional extras on
Windows and macOS are outside the isolated-wheel CI boundary. Broader support
boundary alignment is tracked in
`issue #89 <https://github.com/peter-cudmore/coker/issues/89>`_.

CUDA support is **experimental and unverified pending GPU release evidence**.
CPU-only CI and CUDA-conditional tests that skip without a GPU do not validate
CUDA execution. This caveat applies to tensor execution and autograd, NLP
solves, ODE integration and quadratures where implemented, and parameter
fitting. It does not imply that every backend implements those capabilities.
Consult the capability limits below before selecting a backend.

Backend capability matrix
-------------------------

.. list-table::
   :header-rows: 1

   * - Backend
     - Best fit
     - Dependency story
     - Repository test coverage (not platform or CUDA certification)
   * - ``numpy``
     - Reference execution and small-system fallback.
     - Included in the base install.
     - Covered by ``tests/backends/numpy/`` and the benchmark suite.
   * - ``casadi``
     - Optimisation, simulation, and high-performance numerical work,
       including nonlinear transcription workflows.
     - Install with ``pip install "coker[casadi]"``. Independent residual DAEs
       default to native IDAS. ``CasadiResidualSolverOptions`` selects either
       ``CasadiResidualSolver.IDAS`` or the explicit
       ``CasadiResidualSolver.VARIATIONAL`` collocation path; solver failures
       do not silently switch algorithms.
     - Covered by ``tests/backends/casadi/`` and the variational solver tests.
   * - ``sympy``
     - Symbolic inspection and expression printing. Function-space inputs and
       external callable evaluations lower to named SymPy undefined functions,
       preserving the callable identity across calls with distinct arguments.
     - Included in the base install.
     - Covered by ``tests/backends/sympy/test_sympy_printing.py``.
   * - ``jax``
     - Alternate array backend when you want JAX-native values.
     - Install with ``pip install "coker[jax]"``; Python 3.11 or newer.
     - Array evaluation is included in installed-wheel smoke checks and the
       all-extra suite; there is no dedicated ``tests/backends/jax/`` directory.
   * - ``pytorch``
     - Tensor-valued execution, PyTorch autograd, native-module composition,
       and mathematical-program solves (CPU; CUDA experimental/unverified).
     - Install with ``pip install "coker[pytorch]"``. Mathematical-program NLP
       solves use the configured CPU or CUDA device and floating dtype.
       ``PytorchNLPSolverOptions`` selects ``"LBFGS"`` (default) or ``"Adam"``
       and forwards ``optimiser_options`` to the selected PyTorch optimizer.
       Constraint handling uses the backend's barrier and augmented-Lagrangian
       stages. ``PytorchODESolverParameters`` configures ``torchdiffeq``
       initial-value solves only. ``PytorchVariationalSolverOptions`` sets
       trajectory degree and interval count for fixed-horizon parameter fitting.
     - Covered by ``tests/backends/pytorch/`` and dedicated tensor/autograd
       backend tests. Algebraic DAEs, variational controls, quadratures,
       constraints, and optimized horizons are unsupported.

Choosing a backend
------------------

A good default is:

- use ``numpy`` as a reference implementation or for small systems;
- use ``casadi`` for optimisation, simulation, and high-performance numerical
  work;
- use ``pytorch`` when you need tensor-valued execution, autograd, or a
  nonlinear mathematical-program solve on PyTorch;
- use ``sympy`` when you need symbolic forms or printable expressions.

The repository includes numerical execution tests under ``numpy``, ``jax``,
and ``pytorch``, and variational solve tests through ``casadi``. CUDA-conditional
mathematical-program and fixed-horizon variational fitting tests are not GPU
release evidence; see :ref:`installation-support` for the CI boundary.


Lowering handles
----------------

Call :meth:`~coker.Function.lower` to obtain a backend-specific executable
handle. Every handle exposes an ordered ``signature`` and a common packed
``execute(inputs)`` ABI returning a tuple. Positional handle calls unwrap one
output and return a tuple for multiple outputs. ``Function(...)`` also unwraps
a single output and returns a tuple for multiple outputs.

Configuration types have canonical imports in ``coker.backends.lowered``:
``FunctionSignature``, ``FunctionInputSpec``, ``FunctionOutputSpec``, and
``LoweringOptions``. ``f.lower(LoweringOptions(backend="sympy"))`` creates an
alternate handle without changing ``f.backend`` or the current backend.
``MathematicalProgram.lower()`` intentionally differs: it returns a ``Function``
pinned to its solver backend, whose invocation solves the program.

The NumPy and PyTorch handles retain reusable execution plans. CasADi exposes
its native function through ``ca_function`` when the graph can be represented
as a :class:`casadi.Function`; otherwise its handle uses the same evaluate
fallback.


PyTorch module lowering
-----------------------

Use :meth:`~coker.backends.pytorch.PytorchBackend.as_module` to expose a
PyTorch-backed Coker function as an eager :class:`torch.nn.Module`:

.. code-block:: python

   from coker.backends import get_backend_by_name

   backend = get_backend_by_name("pytorch")
   module = backend.as_module(model_function)
   output = module(input_tensor)

The module accepts one positional tensor per Coker function argument. A
single-output function returns a tensor; a multi-output function returns a
tuple. Its constants remain Coker closure state, so it has no trainable
parameters or registered buffers. It supports eager autograd but is not
currently a TorchScript or ``torch.compile`` target.

Importing native backend functions
----------------------------------

Native import interfaces are backend-specific. CasADi derives the ordered
names and matrix dimensions from its native function, while a PyTorch module
needs an explicit :class:`~coker.backends.lowered.FunctionSignature`:

.. code-block:: python

   casadi_function = casadi_backend.import_function(native_function)
   torch_function = pytorch_backend.import_module(module, signature)

Each imported invocation is recorded as an ``OP.EVALUATE`` node while Coker
traces.  It is lowered only by its owning backend: CasADi imports remain
symbolic CasADi calls, and PyTorch imports retain the module, its parameters,
buffers, device, dtype, and autograd graph.  Cross-backend lowering raises an
error rather than passing Coker tracers into a native callable.

Successful CasADi lowered calls return native CasADi values. PyTorch module
import transfers logical ownership to the Coker wrapper, which retains a strong
reference to the module; Python aliases cannot be invalidated automatically.

PyTorch training patterns
-------------------------

There are three distinct PyTorch workflows:

1. **Explicit Coker parameters.** Declare weights and biases as ordinary Coker
   function inputs, pass ``torch.nn.Parameter`` tensors at evaluation time,
   and train them with any normal PyTorch optimizer. This preserves a fully
   visible Coker network expression.
2. **Imported native modules.** Import a ``torch.nn.Module`` with an explicit
   :class:`~coker.backends.lowered.FunctionSignature`. The module retains
   ownership of its parameters and buffers; Coker records the module call as a
   native evaluation node while PyTorch owns training.
3. **Mathematical programs.** Declare weights with
   :meth:`~coker.toolkits.codesign.ProblemBuilder.new_variable`, build a
   :class:`~coker.toolkits.codesign.MathematicalProgram`, and let Coker's
   PyTorch NLP backend solve the fitting problem. The solver owns one packed
   decision vector and returns the declared outputs rather than caller-owned
   PyTorch parameters.

For the third form, configure the PyTorch device and dtype, then select the
solver through ``PytorchNLPSolverOptions``:

.. code-block:: python

   import torch

   from coker.backends import get_backend_by_name
   from coker.backends.pytorch import PytorchNLPSolverOptions

   backend = get_backend_by_name("pytorch")
   backend.device = torch.device(
       "cuda" if torch.cuda.is_available() else "cpu"
   )
   backend.dtype = (
       torch.float32 if backend.device.type == "cuda" else torch.float64
   )
   options = PytorchNLPSolverOptions(
       optimiser_method="Adam",
       optimiser_options={"lr": 0.05},
       inner_iterations=1_000,
   )

``LBFGS`` remains the default and uses the tolerance/history settings on
``PytorchNLPSolverOptions``. ``Adam`` performs ``inner_iterations`` explicit
optimizer steps. See the three ``examples/pytorch_*training.py`` programs for
runnable versions of each workflow.

Optimisation program composition
--------------------------------

``MathematicalProgram`` uses the same call syntax in concrete and traced code.
A concrete call returns the solved objective followed by its declared outputs:

.. code-block:: python

   objective, solution = program(parameters)

During a ``function`` trace, that same call records one ``OP.EVALUATE`` node
for each result. This allows a normal Coker function to use a solved objective
or output:

.. code-block:: python

   closed_loop = function(
       [VectorSpace("parameters", 2)],
       lambda parameters: program(parameters)[1],
       backend="numpy",
   )

``program.lower()`` returns a ``Function`` using the solver backend selected
when the program was built. Lowering does not accept a second backend: this
prevents an outer graph from selecting a backend that cannot execute the
program's solver callable. An unavailable backend or an unsupported
solver/backend combination is rejected while building or lowering, before
evaluation. Derivatives through an argmin or argmax are not defined.

``numpy`` and ``casadi`` support host-side program composition.
PyTorch implements direct CPU or CUDA mathematical-program construction/solving
and fixed-horizon, bound-only ODE parameter fitting; its CUDA paths remain
experimental/unverified as described in :ref:`installation-support`. JAX does
not construct optimisation programs.

Related guides
--------------

- :doc:`getting_started` for the minimal symbolic function workflow.
- :doc:`dynamics` for ODE and variational problem construction.
- :doc:`toolkits` for robotics and codesign helpers.
