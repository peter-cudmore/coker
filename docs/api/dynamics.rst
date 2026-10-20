Dynamics API
============

Dynamics systems
----------------

Use :func:`coker.dynamics.create_autonomous_ode` for an explicit ODE,
:func:`coker.dynamics.create_control_system` for a controlled ODE, and
:class:`coker.dynamics.DynamicsSpec` for a general semi-explicit system.
:func:`coker.dynamics.direct_sum` composes systems.
:class:`coker.dynamics.ResidualDynamicalSystem` represents implicit DAEs.
See :doc:`../dynamics` for callback contracts, backend limits, and examples,
and :doc:`../dynamics/system_analysis` for symbolic rank analysis.


.. autoclass:: coker.dynamics.DynamicalSystem
   :members:
   :special-members: __call__

.. autoclass:: coker.dynamics.ResidualDynamicalSystem
   :members:
   :special-members: __call__

Variational problems
--------------------

Use ``VariationalProblemBuilder`` to construct symbolic objectives and
constraints. ``VariationalProblem`` also accepts a loss callback.
``TranscriptionOptions`` configures discretisation; backend-specific options
select solver behaviour. See :doc:`../dynamics` for CasADi solve support and
current lowering limits.


.. autoclass:: coker.dynamics.VariationalProblemBuilder
   :members: state,input,output,integrate,build

.. autoclass:: coker.dynamics.TranscriptionOptions

.. autoclass:: coker.dynamics.VariationalProblem
   :members: get_solver

