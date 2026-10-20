Parameter declarations
======================

Parameter declarations define solver decision blocks. Use
:class:`coker.parameters.BoundedVariable` or
:class:`coker.parameters.UnboundedVariable` for a scalar,
:class:`coker.parameters.BoundVector` for a bounded vector, and
:class:`coker.parameters.DenseTensorVariable` for a dense array reconstructed
with the shape of its initial guess. See :doc:`../dynamics` for parameter order,
function-valued parameters, and fitted-value reconstruction.

.. autoclass:: coker.parameters.BoundedVariable

.. autoclass:: coker.parameters.UnboundedVariable

.. autoclass:: coker.parameters.BoundVector

.. autoclass:: coker.parameters.DenseTensorVariable

