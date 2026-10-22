Mathematical programs
=====================

Import ordinary optimisation builders, objective and constraint helpers, and
solve results from ``coker.toolkits.codesign``. See :doc:`../toolkits` for
worked construction examples and :doc:`dynamics` for variational problems.

.. Author notes

   - Audience: users who can construct symbolic functions and need exact
     ordinary optimisation construction, call, result and failure contracts.
   - Keep solver-specific options in the backend reference and callable
     parameter declarations in the parameter reference.

Problem construction
--------------------

Runtime arguments are supplied by the caller on every solve; decision
variables are chosen by the optimiser. Use ``ProblemBuilder`` to connect them
on one graph, then call the built ``MathematicalProgram``. Callable decisions
use the finite declarations in :doc:`parameters`.

.. autoclass:: coker.toolkits.codesign.ProblemBuilder
   :members: new_variable, new_function_parameter, input_shape, output_shape, build

Initial guesses are recorded by ``new_variable`` and
``new_function_parameter``. To replace them, assign ``initial_conditions`` a
mapping from symbolic decision indices to values, or a sequence in concrete
decision declaration order. A sequence of the wrong length raises
``ValueError``; an unsupported container raises ``TypeError``.

``build()`` uses the explicitly named backend, or the current backend when
omitted, without changing the current selection. NumPy uses SciPy's
``trust-constr`` solver; CasADi uses IPOPT; PyTorch supplies its own optimiser.
The optional backends require their corresponding extras. JAX and SymPy do not
provide ordinary optimisation solvers. Backend-specific restrictions and option
types are described in :doc:`backends`.

Objectives and constraints
--------------------------

.. autoclass:: coker.toolkits.codesign.Minimise

.. autofunction:: coker.toolkits.codesign.bounded

.. autoclass:: coker.toolkits.codesign.BoundedConstraint
   :members: as_halfplane_bound

Comparisons such as ``x == target`` and ``x <= limit`` also form constraints.
Their residual convention uses a tolerance of ``1e-9`` for equality and
non-strict inequality: equality bounds ``rhs - lhs`` within
``[-1e-9, 1e-9]``, and ``lhs <= rhs`` requires ``rhs - lhs >= -1e-9``.
A strict comparison requires ``rhs - lhs >= 0``; it does not enforce a positive
margin. Use ``bounded`` when explicit numerical bounds matter.

.. autofunction:: coker.toolkits.codesign.weighted_norm

.. autofunction:: coker.toolkits.codesign.norm

``norm`` is unsquared, while ``weighted_norm`` is a squared weighted residual.
Only the latter retains quadratic-factor recognition metadata; neither helper
selects a solver.

Program calls and lowering
--------------------------

.. autoclass:: coker.toolkits.codesign.MathematicalProgram
   :members: input_shape, output_shape, result_shape, lower
   :special-members: __call__

Program calls return the objective followed by requested outputs; named
reconstructed decisions and solve metadata are separate attributes. The source
contract above specifies scalar/array representation and validation.

Program lowering is distinct from :meth:`coker.Function.lower`: it returns a
solver-pinned ``Function`` that runs the solve, not a ``LoweredFunction``
execution handle. See :ref:`backend-lowering-contract` for the two stages.

Solver options and status
-------------------------

.. autoclass:: coker.toolkits.codesign.SolverOptions
   :members: warm_start

Pass a backend-specific options subclass through
``ProblemBuilder(solver_options=...)``. The base ``SolverOptions`` is not a
portable configuration: NumPy rejects non-``None`` options, while CasADi and
PyTorch require their own types. ODE integration uses the separate
``SolverParameters`` hierarchy. See :doc:`backends` for option routing and
warm-start support.

.. autoclass:: coker.toolkits.codesign.SolveInfo
   :members:

.. autoexception:: coker.toolkits.codesign.SolveFailure

An unsuccessful solve raises ``SolveFailure`` with the status available as
``exception.solve_info``. For builder-produced programs, ``program.solve_info``
is refreshed from the solver even when its call raises. Invalid inputs, backend
configuration errors and native solver exceptions are not necessarily wrapped
in ``SolveFailure``.

Solver extension interface
--------------------------

.. autoclass:: coker.toolkits.codesign.OptimisationModule
   :special-members: __call__

This is a solver implementation interface, not a model builder. Private
decision captures belong to the solver ABI and are removed from the public
program result.

.. autoclass:: coker.toolkits.codesign.WeightedNorm
