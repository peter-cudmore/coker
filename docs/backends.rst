Backend guide
=============

Coker can lower the same traced function to several execution backends. The
backend name is selected with the ``backend=...`` argument to
:func:`coker.function`, or indirectly through higher-level APIs such as
:class:`coker.dynamics.VariationalProblem` and
:class:`coker.toolkits.codesign.ProblemBuilder`.

The current backend registry accepts the names ``"numpy"``, ``"coker"``,
``"jax"``, ``"casadi"``, ``"pytorch"``, and ``"sympy"``. If no backend is
selected, ``get_current_backend()`` defaults to ``"coker"``.

Backend capability matrix
-------------------------

.. list-table::
   :header-rows: 1

   * - Backend
     - Best fit
     - Dependency story
     - Observed coverage in this repository
   * - ``numpy``
     - Direct numerical evaluation, baseline execution, and host-side ODE work.
     - Included in the base install.
     - Covered by ``tests/backends/numpy/`` and the benchmark suite.
   * - ``casadi``
     - Nonlinear optimisation, transcription-heavy workflows, and the default
       :class:`~coker.dynamics.VariationalProblem` solve path.
     - Install with ``pip install "coker[casadi]"``.
     - Covered by ``tests/backends/casadi/`` and the variational solver tests.
   * - ``sympy``
     - Symbolic inspection and expression printing.
     - Included in the base install.
     - Covered by ``tests/backends/sympy/test_sympy_printing.py``.
   * - ``coker``
     - Coker's compact workspace-oriented execution graph.
     - Included in the base install.
     - Covered by ``tests/backends/coker/`` and described in
       :doc:`backend_architecture`.
   * - ``jax``
     - Alternate array backend when you want JAX-native values.
     - Install with ``pip install "coker[jax]"``.
     - Backend code exists in ``src/coker/backends/jax/``. The current test
       suite does not include a dedicated ``tests/backends/jax/`` directory.
   * - ``pytorch``
     - Tensor-valued numerical execution with PyTorch autograd for
       differentiable model evaluation.
     - Install with ``pip install "coker[pytorch]"``.
     - Covered by ``tests/backends/pytorch/`` and dedicated tensor/autograd
       backend tests. ODE/integral evaluation, variational solvers,
       mathematical-program construction, and mathematical-program solving
       are not supported.

Choosing a backend
------------------

A good default is:

- use ``numpy`` while bringing up a model or debugging array shapes;
- use ``casadi`` for solve-heavy optimisation and parameter-fitting problems;
- use ``pytorch`` when you need tensor-valued execution or PyTorch autograd;
- use ``sympy`` when you need symbolic forms or printable expressions;
- use ``coker`` when you want the native compact execution graph documented in
  :doc:`backend_architecture`.

The repository's own tests validate numerical execution under ``numpy``,
``jax``, and ``pytorch`` (including PyTorch tensor/autograd behavior),
variational solves through ``casadi``, and low-level native graph behaviour
under ``coker``. PyTorch deliberately does not support ODE/integral
evaluation, variational solvers, mathematical-program construction, or
mathematical-program solving.


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

``numpy``, ``casadi``, and ``coker`` support host-side program composition.
The JAX and PyTorch backends do not construct optimisation programs.

Embedded mapped QP calls
------------------------

The Rust runtime executes mapped QP calls only through
``MappedModule::execute_with_qp_contexts``.  Bytecode ``QpCall`` layers identify
their target QP executable with ``qp_function_id`` and their embedding-owned
state with ``call_slot``.

Before execution, prepare one ``QpCallContext`` for every call layer.  Each
context holds the prepared solver created from that mapped QP program's
caller-owned arena, plus caller-owned evaluator workspace, coefficient output,
flat parameter, and primal-solution buffers.  Allocate these buffers during
application setup using ``MappedQpProgram::workspace_requirements()``; the
execution path performs no allocation and does not rebuild a solver.

Pass all contexts to ``execute_with_qp_contexts`` with the parent inputs,
workspace, and outputs.  The runtime validates that every call slot has exactly
one context, that its QP id and buffer widths match bytecode, then evaluates
coefficients, solves, and copies the primal solution to the layer's parent
workspace destination.  QP calls are numerical boundaries and are unsupported
by push-forward execution.

Related guides
--------------

- :doc:`getting_started` for the minimal symbolic function workflow.
- :doc:`dynamics` for ODE and variational problem construction.
- :doc:`toolkits` for robotics, system-modelling, and codesign helpers.
