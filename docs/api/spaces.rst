Argument Spaces
===============

Use argument spaces to declare the domain of a :func:`coker.function`.

.. autoclass:: coker.Scalar
   :members: size

.. autoclass:: coker.VectorSpace
   :members: size

.. autoclass:: coker.FunctionSpace
   :members: input_dimensions,output_dimensions,matches_signature,validate_argument,bind_argument,evaluation_dimension,is_scalar,create_scalar_function_space
   :special-members: __contains__

.. autoclass:: coker.Dimension
   :members:

Callable extension interfaces
-----------------------------

These interfaces describe positional callable values and their logical shape
metadata. They do not prescribe backend ownership or execution mechanisms;
see :ref:`backend-lowering-contract` for lowered execution.

.. autoclass:: coker.interfaces.SymbolicCallable
   :members: lower
   :special-members: __call__

.. autoclass:: coker.interfaces.FunctionSignatureValue
   :members: input_shape,output_shape
