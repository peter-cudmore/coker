Functions and tensor helpers
============================

Import callable APIs from ``coker``. Construct functions with
``function(arguments, implementation, backend="numpy", name=None)`` rather than
constructing a tape or ``Function`` yourself. See
:doc:`../tutorials/function_evaluation` for a step-by-step introduction.

Tracing calls the implementation once; Python side effects happen during
construction, not on each evaluation. Supported operations must be traceable:
use NumPy algebra rather than Python conditionals on symbolic values, and use
``if_then_else`` for symbolic branching.

Call and metadata contracts
---------------------------

``Function(*args)`` accepts one positional value per declared space. Wrong arity
and unsupported argument declaration types raise ``TypeError``. Numeric shape
conversion is backend-specific: same-size arrays can be reshaped to declared
shapes; this is not a strict input-shape validator. A ``FunctionSpace`` accepts a
Coker callable or a Python callable traceable against its declared arguments.
Implementation errors propagate; unsupported backend operations fail during
construction, lowering or execution depending on the backend.

Returning an array or scalar declares one output. Returning a list or tuple
declares separate outputs: even ``return (x,)`` makes the public call return a
one-element tuple. Use an array expression for a vector-valued single output.
``Function.lower().execute(inputs)`` always returns a tuple; a lowered positional
call unwraps a single signature output, independently of the implementation's
list/tuple distinction.

``arguments`` contains ordered names; ``input_spaces()`` retains named scalar,
vector and callable declarations. ``input_shape()`` and ``output_shape()`` return
tuples of ``Dimension``/``FunctionSpace`` metadata (``None`` for absent slots),
not values or named spaces. ``signature`` pairs names with input spaces and
output shapes. ``MathematicalProgram`` also exposes shape methods, but its
``output_shape()`` excludes the objective. ``Function.backend``,
``MathematicalProgram.backend`` and ``DynamicalSystem.backend`` are backend-name
properties. Concrete evaluation and alternate lowering do not change ambient
backend selection.

The following checks require only the base NumPy dependency:

.. testcode:: function_contract

   import numpy as np
   from coker import Dimension, Scalar, VectorSpace, function

   twice = function([VectorSpace("x", 2)], lambda x: 2 * x)
   np.testing.assert_allclose(twice(np.array([1., 3.])), [2., 6.])
   assert twice.input_spaces() == [VectorSpace("x", (2,))]
   assert twice.input_shape() == (Dimension((2,)),)
   pair = function([Scalar("x")], lambda x: (x + 1, x * x))
   assert pair(3.) == (4., 9.)
   one_slot = function([Scalar("x")], lambda x: (x + 1,))
   assert one_slot(3.) == (4.,)
   assert one_slot.lower()(3.) == 4.
   assert twice.lower().execute([np.array([1., 3.])])[0].shape == (2,)

Values and ownership
--------------------

NumPy produces host scalars/arrays. PyTorch tensor inputs retain tensors and
autograd, while calls without tensors adapt outputs to NumPy. JAX produces JAX
values; SymPy produces symbolic expressions. A CasADi public ``Function`` adapts
numeric outputs to scalar/NumPy shapes, while its lowered handle exposes native
CasADi values. Native imports can only lower or compose on their owning backend.
See :doc:`backends` for dependencies, native imports, registry semantics and
handle lifetimes. The external native ``coker`` plugin is not supplied by the
base installation.

A function retains graph constants and cached lowerings for its lifetime. Do
not assume mutating captured arrays updates compiled backend constants, nor
assume results are copies or handles are thread-safe. Keep model constants
unchanged after tracing and explicitly copy a result when retaining a mutable
snapshot is necessary.

.. autofunction:: coker.function

.. autoclass:: coker.Function
   :members: arguments, input_spaces, input_shape, output_shape, lower, call_inline
   :special-members: __call__

.. autoclass:: coker.SymbolicCallable
   :members: lower

Tensor helpers
--------------

.. autofunction:: coker.zeros

.. autoclass:: coker.SymbolicVector
   :members: collapse

Sparse matrices
---------------

.. autoclass:: coker.SparseMatrixBuilder
   :members: nnz, data_space, matrix

.. autoclass:: coker.SparseMatrixPattern

The numeric path retains CSC storage rather than allocating a dense matrix:

.. testcode:: sparse_contract

   import numpy as np
   from coker import SparseMatrixBuilder

   builder = SparseMatrixBuilder(np.array([[True, False], [False, True]]))
   matrix = builder.matrix(np.array([2.0, 3.0]))
   np.testing.assert_allclose(matrix.toarray(), [[2.0, 0.0], [0.0, 3.0]])
