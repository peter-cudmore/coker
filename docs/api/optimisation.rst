Mathematical programs
=====================

Import ``ProblemBuilder`` and ``MathematicalProgram`` from
``coker.toolkits.codesign``. Programs expose callable ``input_shape()`` and
``output_shape()`` metadata; ``result_shape`` is the objective-first property.
See :doc:`../getting_started` for migration and lowering contracts.

.. autoclass:: coker.toolkits.codesign.MathematicalProgram
   :members:

Solve status
------------

Inspect backend results and report failures with these types, without parsing
backend-specific return values.

.. autoclass:: coker.toolkits.codesign.SolveInfo
   :members:

.. autoexception:: coker.toolkits.codesign.SolveFailure
