Dynamics and optimisation
=========================

Coker's dynamics layer sits on top of the same symbolic function machinery used
by :func:`coker.function`. It adds typed descriptions for state, control,
parameter, and transcription data so you can define ODE systems and then solve
parameter-fitting or optimal-control style problems against them.

Core entry points
-----------------

The main public surface is re-exported from :mod:`coker.dynamics`:

- :func:`coker.dynamics.create_autonomous_ode`
- :func:`coker.dynamics.direct_sum`
- :class:`coker.dynamics.VariationalProblem`
- :class:`coker.dynamics.BoundedVariable`
- :class:`coker.dynamics.TranscriptionOptions`
- :class:`coker.toolkits.codesign.SolveInfo` and :class:`coker.toolkits.codesign.SolveFailure`

``create_autonomous_ode()`` builds a :class:`~coker.dynamics.DynamicalSystem`
from an initial-condition function and an ``xdot`` function. If you pass a
parameter space, the system becomes directly usable inside a
:class:`~coker.dynamics.VariationalProblem`.

Functional problem building
---------------------------

New code should use :class:`coker.dynamics.VariationalProblemBuilder`. Declare
controls and parameters at construction time, then pass a
:class:`coker.toolkits.codesign.Minimise` objective and ``subject_to``
comparisons to ``build``.

For example, a fixed-horizon path-planning problem can constrain the input and
terminal state without constructing a loss callback:

.. code-block:: python

   from coker.dynamics import VariationalProblemBuilder
   from coker.toolkits.codesign import Minimise

   with VariationalProblemBuilder(
       system,
       t_final=5.0,
       control=[control],
   ) as problem:
       x = problem.state(problem.t)
       u = problem.input(problem.t)
       x_goal = problem.state(problem.t_final)
       built = problem.build(
           Minimise(problem.integrate(u @ u) + x_goal @ x_goal),
           subject_to=[u <= 1, u >= -1, x_goal == goal],
       )

For data fitting, declare the unknown as a bounded parameter and use the
symbolic observed output directly in the objective:

.. code-block:: python

   with VariationalProblemBuilder(
       system,
       t_final=1.0,
       parameters=[value],
   ) as problem:
       observed = 2.0
       built = problem.build(
           Minimise((problem.output(problem.t_final)[0] - observed) ** 2)
       )

The builder classifies comparisons by their time binding: expressions at
``t`` are path constraints, expressions at ``0`` are initial point
constraints, expressions at ``t_final`` (and parameter-only expressions) are
terminal point constraints.  A numeric ``t_final`` is fixed; a
``BoundedVariable`` declaration makes the horizon a free decision.  The
``integrate`` operation denotes the dynamic quadrature
``\int_0^T expr(t) dt`` rather than numerical sampling in the builder.
``state(time)`` exposes the differential state, while ``output(time)`` is the
observed system output (and may include algebraic or quadrature channels).
The current executable lowering boundary is the CasADi backend; symbolic
construction is backend-independent, but solving a built problem currently
requires CasADi.

Worked parameter-fitting example
--------------------------------

The test suite includes a minimal parameter-identification problem where the
unknown parameter is the constant state value of a one-dimensional system.

.. code-block:: python

   import numpy as np
   from coker import VectorSpace
   from coker.dynamics import (
       BoundedVariable,
       VariationalProblem,
       create_autonomous_ode,
   )

   def x0(p):
       return p[0]

   def xdot(x, p):
       return 0

   measured_value = np.array([2.0])

   system = create_autonomous_ode(
       parameters=VectorSpace("p", 1),
       x0=x0,
       xdot=xdot,
       backend="numpy",
   )

   def loss(solution, p_inner):
       total_error = 0.0
       for t_i in np.arange(0, 1, 0.1):
           truth = measured_value[0]
           estimate = solution(t_i, p_inner)
           total_error += (truth - estimate) ** 2
       return total_error

   problem = VariationalProblem(
       loss=loss,
       system=system,
       parameters=[
           BoundedVariable("value", upper_bound=3, lower_bound=0.5, guess=2)
       ],
       t_final=1,
       backend="casadi",
   )

   solution = problem()
   print(solution.parameter_solutions["value"])
   print(solution.solve_info.success)

This is the same shape used throughout ``tests/dynamical_systems/``: build a
system, define a loss against the system's callable output, and then solve a
:class:`~coker.dynamics.VariationalProblem`.

What the test suite exercises
-----------------------------

The current dynamics coverage includes:

- direct simulation checks for scalar and vector linear systems;
- parameter fitting for constants, lines, and exponential systems;
- path constraints and regularisation terms;
- solver re-entrancy and warm-start behaviour;
- callback support through ``VariationalIterationCallback``.

That makes the tests a useful source of real usage patterns even where the API
surface is broader than this short guide.

Solve status and failures
-------------------------

Both the dynamics layer and the codesign helpers surface solve metadata through
:class:`coker.toolkits.codesign.SolveInfo`. Failed solves raise
:class:`coker.toolkits.codesign.SolveFailure` with the backend status attached.

Next places to read
-------------------

- ``tests/dynamical_systems/test_variational_solver.py`` for fitting and
  constrained-solve examples.
- ``tests/dynamical_systems/test_variational_solver_callback.py`` for iteration
  callback usage.
- :doc:`backends` for backend-selection advice.
