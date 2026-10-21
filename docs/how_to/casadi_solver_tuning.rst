Tune a CasADi solver
====================

Configure the codesign mathematical-program solver, which uses CasADi's IPOPT
``nlpsol`` path. Install ``pip install "coker[casadi]"``; NumPy is used for
numerical checks. This recipe assumes basic Coker function and
constrained-optimisation knowledge. The equality-constrained quadratic has
a known optimum at ``(1, 2)`` with zero objective.

Set options and solve
---------------------

1. Pass a flat CasADi option mapping to ``optimiser_options``. IPOPT settings
   use the ``ipopt.`` prefix; ``print_time`` belongs to CasADi.
2. Set initial guesses on the decision variables, separately from solver
   options.
3. Select options before ``build("casadi")``: settings are copied when the
   solver is constructed. Solve and check the status, residuals and objective.

Tolerances request numerical termination accuracy, not exact arithmetic or
guaranteed convergence. Iteration and CPU-time limits bound solver effort;
the CPU-time limit is not an end-to-end wall-clock deadline.

.. testcode:: casadi_solver_tuning

   import numpy as np

   from coker.backends.casadi import CasadiNLPSolverOptions
   from coker.toolkits.codesign import Minimise, ProblemBuilder

   maximum_iterations = 100
   options = CasadiNLPSolverOptions(
       optimiser_options={
           "ipopt.tol": 1e-9,
           "ipopt.constr_viol_tol": 1e-9,
           "ipopt.max_iter": maximum_iterations,
           "ipopt.max_cpu_time": 30.0,
           "ipopt.print_level": 0,
           "ipopt.sb": "yes",
           "print_time": False,
       }
   )
   with ProblemBuilder(solver_options=options) as builder:
       # Initial guesses are decision-variable data, not IPOPT settings.
       x = builder.new_variable("x", initial_value=0.5)
       y = builder.new_variable("y", initial_value=2.5)
       builder.objective = Minimise((x - 1.0) ** 2 + (y - 2.0) ** 2)
       builder.constraints = [x + y == 3.0, x >= 0.0, y >= 0.0]
       builder.outputs = [x, y]
       problem = builder.build("casadi")

   objective, x_result, y_result = problem()
   x_value = float(np.asarray(x_result).item())
   y_value = float(np.asarray(y_result).item())
   info = problem.solve_info
   assert info is not None
   assert info.success
   assert info.backend == "casadi"
   assert info.return_status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
   assert info.iteration_count is not None
   assert 0 <= info.iteration_count <= maximum_iterations
   np.testing.assert_allclose([x_value, y_value], [1.0, 2.0], atol=1e-6)
   assert abs(x_value + y_value - 3.0) < 1e-7
   assert min(x_value, y_value) >= -1e-7
   assert abs(objective) < 1e-10
   np.testing.assert_allclose(
       objective, (x_value - 1.0) ** 2 + (y_value - 2.0) ** 2, atol=1e-12
   )
   np.testing.assert_allclose(
       [problem.parameters["x"], problem.parameters["y"]],
       [x_value, y_value],
       atol=1e-12,
   )

Inspect the result
------------------

The assertions check ``(x, y) = (1, 2)`` within ``1e-6``, constraint residuals
below ``1e-7`` and an objective below ``1e-10``. They also check the stored
parameters against the returned solution.

Check ``solve_info.success`` and residuals separately. ``return_status``
exposes the solver's status string; ``iteration_count`` is optional in the
general public result interface. A failed solve raises
``coker.toolkits.codesign.SolveFailure`` carrying ``solve_info``. A resource
limit is neither success nor proof of infeasibility. This recipe exercises
only a successful solve.

Adapt the settings
------------------

If convergence is poor, check units, constraint consistency, initial guesses,
and the relative magnitudes of variables and residuals before changing
tolerances or limits. Rescale the formulation explicitly where needed; these
options do not provide automatic Coker scaling, retries or warm starts.

These settings apply to mathematical-program IPOPT solves, not IDAS residual
integration or variational transcription. See :doc:`../dynamics` and
:doc:`../backends` for those interfaces.

.. Author notes

   - Distinguish mathematical-program IPOPT settings from residual IDAS and variational transcription options.
   - Keep stopping tolerances and iteration or time limits separate from formulation choices.
   - Describe statuses exposed by the public interface; distinguish convergence, infeasibility and resource limits without promising unsupported diagnostics.
   - Discuss supported initial guesses separately from solver settings.
   - Treat variable and residual scaling as formulation choices, not an automatic Coker API.
   - Keep failure handling grounded in supported behaviour; do not promise warm starts, retries or convergence.
