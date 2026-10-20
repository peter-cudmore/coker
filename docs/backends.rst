Backend guide
=============

Coker lowers traced functions to execution backends. Select a backend with
``backend=...`` in :func:`coker.function` or higher-level APIs such as
:class:`coker.dynamics.VariationalProblem` and
:class:`coker.toolkits.codesign.ProblemBuilder`.

The bundled registry declares ``"numpy"``, ``"jax"``, ``"casadi"``,
``"pytorch"``, and ``"sympy"``. Plugins can register additional names.

``function()`` defaults to NumPy and retains its own backend selection.
Applications can explicitly select a current backend for implicit consumers,
such as ``ProblemBuilder.build()``. Lookup alone does not change that selection.
See :ref:`backend-lowering-contract` and :doc:`api/backends` for the precise
lookup, selection and lowering contracts.

For a worked introduction, see :doc:`tutorials/backends`; for a first function,
start with :doc:`basics/getting_started`.

Missing optional dependencies
-----------------------------

Looking up ``casadi``, ``jax``, or ``pytorch`` without its dependency raises an
``ImportError`` naming the backend and its ``pip install "coker[...]"`` command.
The JAX extra requires Python >=3.11; on Python 3.10, upgrade Python before
installing ``coker[jax]``. The original ``ModuleNotFoundError`` is preserved as
the exception's cause. Import failures from installed but broken dependencies
or third-party plugins retain their original diagnostics. Backend lookup stays
lazy: base NumPy evaluation and plugin registration do not load optional backends.

The documentation examples use CPU execution. CUDA execution remains
experimental and unverified.

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
       use native IDAS when available. Explicit solver options select IDAS or
       variational collocation; solver failures do not switch algorithms.
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
       and mathematical-program solves.
     - Install with ``pip install "coker[pytorch]"``. Separate option types
       configure mathematical-program fitting, ``torchdiffeq`` initial-value
       integration and fixed-horizon variational parameter fitting.
     - Covered by ``tests/backends/pytorch/`` and dedicated tensor/autograd
       backend tests. Variational fitting supports quadratures but not algebraic
       DAEs, control declarations, variational constraints, or optimized horizons.

Coverage and selection
----------------------

The repository covers numerical execution with ``numpy``, ``jax``, and
``pytorch``, and variational solves through ``casadi``. The all-extra suite runs
on Linux. Installed-wheel smoke checks cover the base install on Linux, Windows,
and macOS with Python 3.10 and 3.13; isolated CasADi, JAX, and PyTorch extras
are checked on Linux with Python 3.11 and 3.13. JAX and PyTorch smoke checks
use CPU execution. These checks do not certify every backend on every platform.
CUDA-conditional mathematical-program and fixed-horizon variational fitting
tests are not GPU release evidence.

See :doc:`tutorials/backends` for a comparison using one function across
backends.


Lowering handles
----------------

Use :meth:`~coker.Function.lower` when another library needs a backend-specific
execution handle. NumPy and PyTorch retain reusable execution plans. CasADi
exposes a native ``ca_function`` when the graph supports it, otherwise it uses
evaluation. An alternate lowering does not change the function's selected
backend.

The :ref:`backend API <backend-lowering-contract>` defines signatures, packed
execution, positional calls and output conversion. For native export, follow
the :ref:`CasADi <backend-export-casadi>` or
:ref:`PyTorch <backend-export-pytorch>` recipe.

PyTorch module lowering
-----------------------

Use :meth:`~coker.backends.pytorch.PytorchBackend.as_module` to compose a Coker
function with eager PyTorch code. This adapter supports autograd but does not
turn Coker closure constants into trainable module parameters or buffers.
It is not a TorchScript or ``torch.compile`` target.

Follow :ref:`backend-export-pytorch` for a CPU example that evaluates the
module and checks its gradients. See :doc:`api/backends` for the module's
calling contract.

Importing native backend functions
----------------------------------

Use a native import when part of a model already exists in another library.
CasADi can derive names and dimensions from its native function. PyTorch
modules and SymPy-compatible callables require an explicit
:class:`~coker.backends.lowered.FunctionSignature`.

Imports remain owned by their execution backend: they are not portable Coker
arithmetic graphs. Keep a surrounding function on that backend; cross-backend
composition and lowering are rejected. Importing a PyTorch module retains
the module and its authoritative parameters and buffers, rather than copying
or transferring ownership.

Follow the :ref:`CasADi <backend-import-casadi>`,
:ref:`SymPy <backend-import-sympy>` or :ref:`PyTorch <backend-import-pytorch>`
import recipe for numerical checks and composition with Coker expressions.

PyTorch training patterns
-------------------------

Choose the workflow that should own the trainable values:

1. **Explicit Coker parameters.** Declare weights and biases as ordinary Coker
   function inputs, pass ``torch.nn.Parameter`` tensors at evaluation time,
   and train them with a PyTorch optimiser. This preserves a fully
   visible Coker network expression.
2. **Imported native modules.** Import a ``torch.nn.Module`` with an explicit
   signature. Its parameters and buffers remain authoritative while Coker
   records native calls and PyTorch owns training.
3. **Mathematical programs.** Declare weights with
   :meth:`~coker.toolkits.codesign.ProblemBuilder.new_variable`, build a
   :class:`~coker.toolkits.codesign.MathematicalProgram`, and let Coker's
   PyTorch NLP backend solve the fitting problem. The solver owns one packed
   decision vector and returns the declared outputs rather than caller-owned
   PyTorch parameters.

For the third workflow, configure device and dtype on the backend, then pass
:class:`~coker.backends.pytorch.PytorchNLPSolverOptions` to the problem builder.
``LBFGS`` is the default optimiser; ``Adam`` is an alternative. See
:doc:`api/backends` for option defaults and constraints, and the three
``examples/pytorch_*training.py`` programs in :doc:`examples` for these
workflows. Use CPU execution unless you are evaluating the experimental CUDA
path yourself.

Optimisation program composition
--------------------------------

A mathematical program can be called directly or inside a Coker function on
the same backend. Each call solves the program and returns its objective
followed by the declared outputs. In a trace, the solver call is recorded as
a native evaluation; it is not expanded into the surrounding arithmetic graph.
Derivatives through an argmin or argmax are not defined.

``MathematicalProgram.lower()`` returns a Coker function pinned to the program's
solver backend, rather than the execution handle returned by
``Function.lower()``. See :ref:`backend-lowering-contract` for both contracts.
Follow :doc:`how_to/casadi_solver_tuning` for a solved mathematical program,
or :doc:`how_to/system_identification` for a parameter-fitting task.

NumPy and CasADi support host-side program composition. PyTorch supports
mathematical-program solves and fixed-horizon, bound-only ODE parameter fitting;
its variational fitting also supports quadratures. JAX does not construct
optimisation programs.

Related guides
--------------

- :doc:`tutorials/backends` for backend selection and native function extraction.
- :doc:`how_to/backend_interoperability` for native import and export.
- :doc:`dynamics` for ODE and variational problem construction.
- :doc:`toolkits` for robotics and codesign helpers.

.. Author notes

   - Audience: users choosing an execution backend or training workflow.
   - Explain capabilities and tradeoffs; link to the API for precise calling and option contracts.
   - Reuse the interoperability and solver-tuning recipes rather than adding partial snippets.
   - Do not imply CUDA validation from CPU coverage or conditional tests.
