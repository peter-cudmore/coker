Codesign mathematical programs
==============================

Supported import paths
----------------------

Ordinary user entry points live in ``coker.toolkits.codesign``: import
``ProblemBuilder``, ``Minimise``, ``MathematicalProgram``, ``bounded``,
``BoundedConstraint``, ``norm``, ``weighted_norm``, ``SolverOptions``,
``SolveInfo``, and ``SolveFailure`` explicitly from that module. They are not
root ``coker`` exports. ``MathematicalProgram`` is available by explicit import
but is currently absent from the module's ``__all__``; do not rely on wildcard
imports. This reference records the current exports, not a future export
inventory or backend-default change.

``OptimisationModule`` and ``WeightedNorm`` are advanced extension interfaces,
not ordinary stable solver plumbing. Private capture records, backend solver
classes, and the ``solve_info_from_*`` normalisation functions are implementation
helpers, not recommended application entry points.

Construction and lifecycle
--------------------------

Declare runtime argument spaces in ``ProblemBuilder(arguments=...)`` and obtain
their symbolic values from ``builder.arguments``. Decisions created by
``new_variable`` or ``new_function_parameter`` are distinct from runtime inputs.
Set ``builder.objective`` to ``Minimise(scalar_expression)`` and set a nonempty
sequence of symbolic ``builder.outputs`` before building. A vector residual is
not a scalar objective: reduce it with a dot product, a norm, or
``weighted_norm`` first. All expressions must use the builder's tape; operation
and shape support remain backend-specific.

The builder owns a tape immediately, even outside a ``with`` block. Entering
and exiting its convenience context neither starts a solve nor closes the
builder. ``build()`` constructs the solver without solving or consuming the
builder. Subsequent builds use the current configuration. Changing the builder
is not a way to reconfigure an already-created solver. Omitting the backend
selects the current backend, rather than guaranteeing CasADi or NumPy.

Missing objective or outputs raise ``ValueError``; an objective not wrapped in
``Minimise`` raises ``TypeError``; a builder without a tape raises
``RuntimeError``. These errors are deliberate public exceptions, not assertions.
This does not imply a general builder validation pass: unsupported expressions
or incompatible backend settings can still fail during backend construction.

``initial_conditions`` starts as an index-to-value mapping populated by decision
declarations. A sequence may instead be supplied in decision declaration order,
excluding runtime arguments. Function decisions expand to concrete coefficient
blocks; those blocks participate in the same initial-guess order.

Constraints and solver support
------------------------------

``bounded(residual, lower_bound, upper_bound)`` creates inclusive componentwise
bounds. Equal bounds impose equality; infinite constants leave one side
unbounded. Symbolic bounds must share the residual's tape and may depend on
runtime arguments. CasADi converts symbolic bounds to residual inequalities
rather than freezing them at build time. Symbolic comparisons are also accepted;
strict ``<`` and ``>`` comparisons become non-strict numerical constraints, not
constraints with a guaranteed strict feasibility margin.

Current standalone NLP implementations are:

* ``casadi``: CasADi/IPOPT, with
  ``coker.backends.casadi.CasadiNLPSolverOptions``. Its flat
  ``optimiser_options`` mapping configures the CasADi solver constructor, for
  example ``"ipopt.max_iter"``. It is copied at construction. The current
  codesign implementation always starts from its original initial guess; the
  inherited ``warm_start`` flag does not enable CasADi warm starts here.
* ``numpy``: SciPy ``trust-constr``. ``solver_options`` must be ``None``.
* ``pytorch``: the native PyTorch NLP implementation, configured through
  ``coker.backends.pytorch.PytorchNLPSolverOptions``. Its tensor/device and
  algorithm limitations are backend-specific; this wrapper's public numeric
  results are still NumPy arrays.

JAX and SymPy do not currently implement standalone codesign NLP solves.
``weighted_norm`` retains structured quadratic metadata but does not select a
QP solver. A solver reporting success does not establish global optimality.
Standalone solving and backend support for nested optimiser calls in a symbolic
graph are separate capabilities.

Complete CasADi example
-----------------------

This example requires the optional CasADi dependency with its IPOPT plugin.
Solver printing is disabled explicitly so the doctest checks results, not logs.
The runtime target also appears in an equality bound.

.. doctest::

   >>> import numpy as np
   >>> from coker import VectorSpace
   >>> from coker.backends.casadi import CasadiNLPSolverOptions
   >>> from coker.toolkits.codesign import Minimise, ProblemBuilder, bounded
   >>> options = CasadiNLPSolverOptions(optimiser_options={
   ...     "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
   ...     "ipopt.max_iter": 50,
   ... })
   >>> with ProblemBuilder(arguments=[VectorSpace("target", 2)], solver_options=options) as builder:
   ...     (target,) = builder.arguments
   ...     x = builder.new_variable("x", shape=(2,), initial_value=np.zeros(2))
   ...     delta = x - target
   ...     builder.objective = Minimise(np.dot(delta, delta))
   ...     builder.constraints = [bounded(x[0], target[0], target[0])]
   ...     builder.outputs = [x]
   ...     problem = builder.build("casadi")
   >>> assert problem.parameters == {} and problem.solve_info is None
   >>> result = problem(np.array([3.0, -1.0]))
   >>> objective, x_value = result
   >>> assert len(result) == 2 and isinstance(objective, float)
   >>> assert abs(objective) < 1e-6
   >>> np.testing.assert_allclose(x_value, [3.0, -1.0], atol=1e-6)
   >>> np.testing.assert_allclose(problem.parameters["x"], x_value, atol=1e-6)
   >>> assert problem.solve_info is not None and problem.solve_info.success
   >>> assert problem.solve_info.backend == "casadi"
   >>> assert problem.solve_info.solver == "ipopt"

Results and failure state
-------------------------

A concrete program call returns the tuple ``(objective, *outputs)`` in requested
output order, never hidden decision captures. The objective is a Python float;
outputs are shaped NumPy arrays, including shape ``(1,)`` for scalar outputs
(the current ``Dimension.scalar().shape`` convention). ``output_shape`` excludes
the objective, whereas ``result_shape``
includes its scalar dimension first.

``program.parameters`` starts empty and is replaced after each successful
numeric solve with a dictionary of named decisions. Scalar decisions are floats;
vector/tensor decisions are NumPy arrays. Function decisions are
``coker.parameters.function_parameters.FittedFunction`` objects exposing the
original ``specification``, declared ``space``, native callable ``function``, and
fitted ``parameters``. The returned arrays and mapping have no isolation/copy
guarantee. Copy values if storing a snapshot or mutating them. The program owns
its current metadata, not a history; concurrent calls are not a supported
state-management interface.

``solve_info`` starts as ``None``. A solver-reported failure raises
``SolveFailure`` instead of returning a tuple and updates ``solve_info`` with
that exception's metadata. ``parameters`` keeps the preceding successful values,
or stays empty if there has been no successful solve: it must not be interpreted
as a solution to the failed call. Argument validation and native backend errors
may occur before new metadata is recorded and may leave an older status intact.
Directly constructed callable wrappers do not provide solver metadata.

.. doctest::

   >>> from coker.toolkits.codesign import SolveFailure
   >>> try:
   ...     objective, values = problem(np.array([2.0, 1.0]))
   ... except SolveFailure as error:
   ...     assert problem.solve_info == error.solve_info
   ...     assert not error.solve_info.success
   ... else:
   ...     assert problem.solve_info.success
   ...     np.testing.assert_allclose(values, [2.0, 1.0], atol=1e-6)

Deliberate lowering difference
------------------------------

``MathematicalProgram.lower()`` takes no options and returns a Coker
``Function`` wrapping the program with objective-first outputs. It retains the
solver backend, or uses NumPy for a directly constructed wrapper without a
backend. By contrast, ``Function.lower(options=...)`` returns a backend
``LoweredFunction`` execution handle. These are intentionally different return
contracts, not interchangeable lowering interfaces. Symbolic calls require an
active tracing context and cannot compose a backend-bound program into a graph
using another backend. Execution of nested solves depends on the backend's
native-call support.

User reference
--------------

.. autoclass:: coker.toolkits.codesign.ProblemBuilder
   :members: new_variable, new_function_parameter, input_shape, output_shape, build

.. autoclass:: coker.toolkits.codesign.Minimise

.. autoclass:: coker.toolkits.codesign.MathematicalProgram
   :members: __call__, result_shape, lower

.. autofunction:: coker.toolkits.codesign.bounded

.. autoclass:: coker.toolkits.codesign.BoundedConstraint

.. autofunction:: coker.toolkits.codesign.norm

.. autofunction:: coker.toolkits.codesign.weighted_norm

.. autoclass:: coker.toolkits.codesign.SolverOptions

.. autoclass:: coker.toolkits.codesign.SolveInfo

.. autoexception:: coker.toolkits.codesign.SolveFailure

Advanced extension reference
----------------------------

.. autoclass:: coker.toolkits.codesign.OptimisationModule
   :members: __call__

.. autoclass:: coker.toolkits.codesign.WeightedNorm
