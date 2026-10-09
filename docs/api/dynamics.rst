Dynamics API
============

Dynamics systems
----------------

Use :func:`coker.dynamics.create_autonomous_ode` for simple explicit models.
Use :class:`coker.dynamics.ResidualDynamicalSystem` when the model is naturally
an implicit DAE. The :doc:`../dynamics` guide contains executable trajectory and
variational-solve examples.

.. autoclass:: coker.dynamics.DynamicsSpec

.. autoclass:: coker.dynamics.DynamicalSystem
   :members:
   :special-members: __call__

.. autoclass:: coker.dynamics.ResidualDynamicalSystem
   :members:
   :special-members: __call__

Variational problems
--------------------

.. autoclass:: coker.dynamics.VariationalProblemBuilder
   :members: state,input,output,integrate,build

.. autoclass:: coker.dynamics.TranscriptionOptions

.. autoclass:: coker.dynamics.VariationalProblem
   :members: get_solver

