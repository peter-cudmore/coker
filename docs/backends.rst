Backend guide
=============

Coker can lower the same traced function to several execution backends. The
backend name is selected with the ``backend=...`` argument to
:func:`coker.function`, or indirectly through higher-level APIs such as
:class:`coker.dynamics.VariationalProblem` and
:class:`coker.toolkits.codesign.ProblemBuilder`.

The current backend registry accepts the names ``"numpy"``, ``"coker"``,
``"jax"``, ``"casadi"``, and ``"sympy"``. If no backend is selected,
``get_current_backend()`` defaults to ``"coker"``.

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

Choosing a backend
------------------

A good default is:

- use ``numpy`` while bringing up a model or debugging array shapes;
- use ``casadi`` for solve-heavy optimisation and parameter-fitting problems;
- use ``sympy`` when you need symbolic forms or printable expressions;
- use ``coker`` when you want the native compact execution graph documented in
  :doc:`backend_architecture`.

The repository's own tests follow the same pattern: numerical execution is
validated under ``numpy``, variational solves are exercised through ``casadi``,
and low-level native graph behaviour is tested under ``coker``.

Optimisation program composition
--------------------------------

``MathematicalProgram`` is a numerical module.  A concrete call returns the
solved objective followed by its declared outputs:

.. code-block:: python

   objective, solution = program(parameters)

Programs may be called from a traced ``function``.  Coker records the call
statically and executes its prebuilt QP solver when the enclosing function is
evaluated.  Solver calls remain numerical boundaries: derivatives through an
argmin or argmax are not defined.

``numpy``, ``casadi``, and ``coker`` support host-side program composition.
The JAX backend does not construct optimisation programs.

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
