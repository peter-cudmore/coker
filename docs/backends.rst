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

Related guides
--------------

- :doc:`getting_started` for the minimal symbolic function workflow.
- :doc:`dynamics` for ODE and variational problem construction.
- :doc:`toolkits` for robotics, system-modelling, and codesign helpers.
