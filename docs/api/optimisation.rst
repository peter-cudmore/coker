Mathematical programs
=====================

Import ``ProblemBuilder`` and ``MathematicalProgram`` from
``coker.toolkits.codesign``. Programs expose callable ``input_shape()`` and
``output_shape()`` metadata; ``result_shape`` is the objective-first property.
See :doc:`../public_api` for the complete public boundary and lowering contract.

.. autoclass:: coker.toolkits.codesign.MathematicalProgram
   :members:

Solve status
------------

Coker's codesign toolkit exposes lightweight solve status objects so callers
can inspect backend results and report failures without parsing backend-specific
return payloads.

.. autoclass:: coker.toolkits.codesign.SolveInfo
   :members:

.. autoexception:: coker.toolkits.codesign.SolveFailure
