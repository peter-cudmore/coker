How to simulate a dynamical system
==================================

Construct an explicit ODE, choose an integrator, and plot its trajectory.
This recipe assumes basic :doc:`Coker function knowledge
<../tutorials/function_evaluation>`. See :doc:`../dynamics` for the modelling API.

1. Set up the model
-------------------

Install ``coker`` and ``matplotlib``. The NumPy backend uses NumPy and SciPy,
both core Coker dependencies.

With time in seconds, the dimensionless state satisfies
:math:`\dot{x}=-0.7x`, :math:`x(0)=2`, with no external input. Its analytic
solution is :math:`x(t)=2\exp(-0.7t)`.

2. Simulate and plot
--------------------

Define the rate and initial state with ``create_autonomous_ode``.
The output callback keeps a one-element vector for each sample.
``NumpySolverParameters`` selects SciPy's adaptive RK45 integrator with its
default error tolerances. The time grid requests output samples, not fixed
integration steps.

.. plot::
   :include-source: true

   import matplotlib.pyplot as plt
   import numpy as np

   from coker.backends.numpy import NumpySolverParameters, Solver
   from coker.dynamics import create_autonomous_ode

   initial_state = 2.0
   decay_rate = 0.7  # inverse seconds

   def rate(state, _parameters):
       return -decay_rate * state

   system = create_autonomous_ode(
       x0=np.array([initial_state]),
       xdot=rate,
       backend="numpy",
       output=lambda state, _parameters: state,
   )
   system.solver_parameters = NumpySolverParameters(solver=Solver.RK45)

   times = np.linspace(0.0, 4.0, 101)
   states = np.asarray(system(times)).reshape(-1)
   exact = initial_state * np.exp(-decay_rate * times)

   assert states.shape == times.shape
   np.testing.assert_allclose(states, exact, rtol=0.0, atol=1e-3)
   assert abs(states[-1] - 0.1216201253) < 1e-3
   assert np.all(states > 0.0)
   assert np.all(np.diff(states) < 0.0)

   fig, ax = plt.subplots()
   ax.plot(times, states, label="Coker / RK45")
   ax.plot(times, exact, "--", label="Analytic solution")
   ax.set(
       xlabel="Time [s]",
       ylabel="State x [dimensionless]",
       title="Exponential decay from x(0) = 2",
   )
   ax.legend()
   ax.grid(True)
   fig.tight_layout()
   plt.show()

3. Check and adapt
------------------

The curves should overlap, decreasing to approximately ``0.12162`` at four
seconds. The assertions check positivity, monotonic decay, and agreement with
the analytic solution to ``1e-3`` absolute error.

For another ODE, replace the initial state and rate function. Choose a matching
output callback and a reference trajectory or physical check. Adjust
``solver_parameters`` to control integration; changing ``times`` changes the
output sampling, not the solver's internal step size.

.. Author notes

   - Audience and prerequisites: describe the required ODE, initial-condition, and Coker function knowledge.
   - Concrete model: choose a small dynamical system and state its equations, parameters, units, initial state, and any inputs.
   - Steps: outline constructing the dynamics, choosing integration settings and a time grid, evaluating the trajectory, and plotting the state.
   - Expected result: describe the numerical endpoint and trajectory plot, including axis labels and the physical behaviour the reader should see.
   - Backend and dependencies: name the explicit integration backend, optional extras, and plotting dependencies; distinguish solver settings from plotting settings.
   - Verification: specify an analytic solution, conservation check, or trusted reference trajectory and the numerical tolerance used to compare it.
