Parameter declarations
======================

Parameter declarations describe fixed values and solver decision blocks. Use a
scalar declaration for one fitted value, :class:`BoundVector` for a bounded
vector, and :class:`DenseTensorVariable` when the recovered value has a dense
array shape.

.. autoclass:: coker.parameters.BoundedVariable

.. autoclass:: coker.parameters.UnboundedVariable

.. autoclass:: coker.parameters.BoundVector

.. autoclass:: coker.parameters.DenseTensorVariable

