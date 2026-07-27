Solve status
============

Coker exposes lightweight solve status objects so callers can inspect backend
results and report failures without parsing backend-specific return payloads.

.. autoclass:: coker.optimisation.SolveInfo
   :members:

.. autoexception:: coker.optimisation.SolveFailure
