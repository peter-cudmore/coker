Solve status
============

Coker's codesign toolkit exposes lightweight solve status objects so callers
can inspect backend results and report failures without parsing backend-specific
return payloads.

.. autoclass:: coker.toolkits.codesign.SolveInfo
   :members:

.. autoexception:: coker.toolkits.codesign.SolveFailure
