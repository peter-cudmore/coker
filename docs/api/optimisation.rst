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

.. autoclass:: coker.toolkits.codesign.ProblemBuilder
   :members: new_variable, new_function_parameter, input_shape, output_shape, build

``arguments`` declares the scalar or vector runtime inputs in order.
``builder.arguments`` contains their symbolic values; these are not solver
decisions. ``new_variable(name, shape=None, initial_value=None)`` returns a
symbolic decision: a scalar when ``shape`` is omitted, otherwise an array of
that shape. The default initial value is zero.

Set ``builder.objective`` to ``Minimise(scalar_expression)``,
``builder.constraints`` to a list of comparisons or ``bounded`` constraints,
and ``builder.outputs`` to a non-empty list of symbolic expressions. All
expressions must belong to the builder's graph. ``build(backend=None)`` returns
a :class:`~coker.toolkits.codesign.MathematicalProgram`. A missing objective
(``None``) raises ``ValueError``; a present objective that is not ``Minimise``
raises ``TypeError``. An empty output list raises ``ValueError``. The builder's
``input_shape`` and ``output_shape`` are tuple properties, not methods.

The optional ``with ProblemBuilder(...) as builder:`` context returns the
builder without installing a global tracing context. Leaving it does not
build, solve or discard the problem, and exceptions propagate normally.

``new_function_parameter(target, declaration)`` returns a symbolic callable
whose concrete decisions, guesses and bounds come from the declaration.
``target`` must be a :class:`~coker.FunctionSpace` and ``declaration`` must
implement :class:`~coker.parameters.function_parameters.FunctionParameter`;
wrong types raise ``TypeError``. The declaration and its concrete blocks need
non-empty names, otherwise ``ValueError`` is raised. Incompatible target shapes
also raise ``ValueError``. See :doc:`parameters` for declaration contracts.

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

Wrap the scalar objective expression; this wrapper does not perform a solve.

.. autofunction:: coker.toolkits.codesign.bounded

Returns a :class:`~coker.toolkits.codesign.BoundedConstraint` representing
``lower_bound <= residual <= upper_bound`` component-wise. Bounds may be
constants or symbolic runtime parameters from the residual's graph. Use equal
bounds for an explicit equality constraint and infinite bounds for an unbounded
side. This helper does not validate bound ordering or shape; the selected
backend checks and lowers the constraint.

The arguments are ``residual`` (the symbolic expression), ``lower_bound`` and
``upper_bound`` (numeric values or symbolic bounds). The returned record exposes
those three values unchanged. Its ``tape`` property is the residual's tape, and
``as_halfplane_bound()`` returns ``(residual, lower_bound, upper_bound)``.

.. autoclass:: coker.toolkits.codesign.BoundedConstraint
   :members: as_halfplane_bound

Comparisons such as ``x == target`` and ``x <= limit`` also form constraints.
Their residual convention uses a tolerance of ``1e-9`` for equality and
non-strict inequality: equality bounds ``rhs - lhs`` within
``[-1e-9, 1e-9]``, and ``lhs <= rhs`` requires ``rhs - lhs >= -1e-9``.
A strict comparison requires ``rhs - lhs >= 0``; it does not enforce a positive
margin. Use ``bounded`` when explicit numerical bounds matter.

.. autofunction:: coker.toolkits.codesign.weighted_norm

Returns the scalar squared Euclidean norm ``||weight @ residual||²``, not its
square root. Both arguments must support symbolic matrix multiplication and
the resulting vector's dot product. The helper retains factor metadata for
backends that recognise quadratic objectives; it does not itself select a QP
solver.

``weight`` is the symbolic weighting matrix and ``residual`` the symbolic
residual vector. The return value is a scalar symbolic expression; the
``WeightedNorm`` record attached to it retains those two original factors.

.. autofunction:: coker.toolkits.codesign.norm

Returns ``numpy.linalg.norm(arg, ord=order)``. Symbolic vectors support orders
``1`` and ``2``; other orders and matrix norms depend on the backend. This is
an unsquared norm, unlike ``weighted_norm``.

Program calls and lowering
--------------------------

.. autoclass:: coker.toolkits.codesign.MathematicalProgram
   :members: input_shape, output_shape, result_shape, lower
   :special-members: __call__

A numeric call ``program(*arguments)`` solves for the supplied runtime inputs
and returns ``(objective, *outputs)``. The objective is a Python ``float``;
requested outputs are NumPy arrays reshaped using each declared
``Dimension.shape``. A scalar output has logical dimension metadata ``None``
but is returned as an array of shape ``(1,)``, not a zero-dimensional array.
The public result never includes private parameter captures.

``input_shape()`` describes runtime inputs in declaration order.
``output_shape()`` describes only requested outputs. The ``result_shape``
property prepends the scalar objective dimension. The wrong number of arguments
raises ``ValueError``; symbolic arguments with incompatible dimensions also
raise ``ValueError``. Numeric argument validation is backend-specific. An
invalid backend result count raises ``ValueError`` and a non-scalar objective
raises ``TypeError``.

After a successful numeric solve, ``program.parameters`` maps decision names
to reconstructed scalar, array or :class:`~coker.parameters.function_parameters.FittedFunction`
values. Their numeric representation follows the solver backend; the public
output tuple remains NumPy-based. ``program.solve_info`` holds the most recent
backend solve status and is initially ``None``. A failed solve does not replace
the previously reconstructed parameters.

``program.lower()`` returns a solver-pinned :class:`~coker.Function`, not a
``LoweredFunction``. Calling this function runs the solve and returns only the
objective and requested outputs. It does not select another solver backend or
change the current backend. A program called with symbolic arguments can only
be embedded in a graph with the same backend; a mismatch raises ``ValueError``.
For the separate ``Function.lower(options)`` execution-handle contract, see
:ref:`backend-lowering-contract`.

Solver options and status
-------------------------

.. autoclass:: coker.toolkits.codesign.SolverOptions
   :members: warm_start

``SolverOptions`` is the frozen base for solver configuration, separate from
ODE ``SolverParameters``. ``warm_start`` defaults to ``False`` and must be a
``bool``; another type raises ``TypeError``. Pass supported backend-specific
subclasses through ``ProblemBuilder(solver_options=...)``. The base is not a
portable configuration accepted by every solver: NumPy rejects all non-``None``
options, while CasADi and PyTorch require their own option types. Warm-start
behaviour is defined by the selected solver; see :doc:`backends`.

.. autoclass:: coker.toolkits.codesign.SolveInfo
   :members:

The immutable status records ``backend``, ``solver``, ``success`` and the
backend's ``return_status`` string. ``unified_return_status`` and
``iteration_count`` may be ``None`` when unavailable. Inspect these fields
rather than parsing a backend's raw return values.

Construct ``SolveInfo(backend, solver, success, return_status,
unified_return_status=None, iteration_count=None)`` when implementing a backend.
The first two fields identify the backend and solver; ``success`` is the Boolean
outcome. The last two fields represent optional normalised status and iteration
metadata. This record does not run or validate a solve.

.. autoexception:: coker.toolkits.codesign.SolveFailure

An unsuccessful solve raises ``SolveFailure`` with the status available as
``exception.solve_info``. For builder-produced programs, ``program.solve_info``
is refreshed from the solver even when its call raises. Invalid inputs, backend
configuration errors and native solver exceptions are not necessarily wrapped
in ``SolveFailure``.

``SolveFailure(message, solve_info)`` accepts a diagnostic string and the
corresponding :class:`~coker.toolkits.codesign.SolveInfo`; the latter is exposed
unchanged through ``solve_info``.

Solver extension interface
--------------------------

.. autoclass:: coker.toolkits.codesign.OptimisationModule
   :special-members: __call__

Solver implementations provide ``last_solve_info`` before returning or raising
``SolveFailure``. This interface is not a model builder.
The call receives runtime arguments in declaration order and returns backend
values ordered as objective, requested outputs, and any private concrete
decision captures requested during construction. ``MathematicalProgram``
strips private captures from the public tuple. Backend validation and native
errors may propagate; unsuccessful solver status raises ``SolveFailure``.

.. autoclass:: coker.toolkits.codesign.WeightedNorm

``WeightedNorm`` stores the ``weight`` and ``residual`` recognition metadata
created by ``weighted_norm``. Ordinary model code should call the helper
rather than construct this record.
