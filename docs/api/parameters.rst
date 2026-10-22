Parameter declarations
======================

Import scalar and array decisions from ``coker.parameters``. Callable
declarations and fitted results belong to
``coker.parameters.function_parameters``. These declarations describe solver
decisions; they are not themselves fitted numeric values. See
:doc:`optimisation` for ordinary programs and :doc:`dynamics` for dynamics
parameter ordering and variational results.

.. Author notes

   - Audience: model authors selecting finite scalar, array and callable
     decision declarations, and inspecting reconstructed fitted values.
   - State target shapes, concrete block order, validation and backend limits;
     keep task walkthroughs in the cookbook and custom declaration hooks here.
   - Keep basic usage in source docstrings so autodoc and doctest share examples.

Scalar and array decisions
--------------------------

Use ``BoundedVariable`` or ``UnboundedVariable`` for scalars, ``BoundVector``
for component-wise bounded vectors, and ``DenseTensorVariable`` for unbounded
arrays. Scalar declarations defer validation to their consumer; dense
declarations validate their arrays at construction.

.. autoclass:: coker.parameters.BoundedVariable
   :members: degrees_of_freedom

.. autoclass:: coker.parameters.UnboundedVariable
   :members: degrees_of_freedom, lower_bound, upper_bound

.. autoclass:: coker.parameters.BoundVector
   :members: shape, size, degrees_of_freedom

.. autoclass:: coker.parameters.DenseTensorVariable
   :members: shape, size, degrees_of_freedom

Names identify fitted values, not units: use ``program.parameters[name]`` for
ordinary optimisation results. Vector and tensor reconstruction retains the
declared shape. Guesses, bounds and model equations must use consistent units;
these declarations do not attach physical units or convert values.

Callable declarations
---------------------

A callable declaration supplies finite scalar or dense decision blocks for a
:class:`~coker.FunctionSpace`. A builder validates that space and inserts the
blocks into its optimisation graph. Set a non-empty ``name`` when the consuming
builder requires named decisions, including ``ProblemBuilder``. An optional
name does not itself guarantee that the builder accepts an unnamed declaration.

Choose a family by its target signature and the meaning of its bounds:

* ``ClosureParameter`` binds leading decisions to an existing function;
  ``BoundFunctionParameter`` binds selected input positions.
* ``DenseLayer`` supplies an unbounded affine scalar or vector layer followed
  by a Coker activation.
* ``MonotonePiecewiseLinear`` and ``RadialBasisFunction`` are scalar-to-scalar.
  The former bounds the output and enforces monotonicity; the latter optionally
  bounds coefficients, not output values.

Numeric calls and fitted value types follow the materialising backend.
A valid declaration does not imply that the backend can solve its containing
optimisation problem; see :doc:`backends`.

Closures and bound inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.ClosureParameter
   :members: bind_callable, validate_target, list_concrete_parameters, evaluate

.. autoclass:: coker.parameters.function_parameters.BoundFunctionParameter
   :members: validate_target, list_concrete_parameters, evaluate, build_function

For direct ``evaluate(parameters, argument)`` calls, pass a single public value
when one input remains, a sequence of public values for multiple inputs, or
``None`` for a fully bound function. ``build_function`` instead exposes the
public inputs as separate arguments followed by concrete blocks. Reconstruction
and calls of fitted multi-input or fully bound functions also depend on the
backend adapter; the declaration's shape validation alone does not establish
backend support.

Dense layers
~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.DenseLayer
   :members: hidden_size, output_size, validate_target, list_concrete_parameters, evaluate

Monotone functions
~~~~~~~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.MonotonePiecewiseLinear
   :members: size, validate_target, list_concrete_parameters, evaluate

Radial basis functions
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: coker.parameters.function_parameters.RadialBasisFunction
   :members: validate_target, list_concrete_parameters, evaluate

Fitted results
--------------

.. autoclass:: coker.parameters.function_parameters.FittedFunction
   :members: input_shape, output_shape
   :special-members: __call__

Solvers reconstruct callable decisions as ``FittedFunction`` values, available
through the named parameters of ordinary and variational results. Unlike a
compilable Coker ``Function``, this wrapper delegates to a concrete backend
callable without imposing a numeric representation or extra shape validation.

Custom declarations
-------------------

.. autoclass:: coker.parameters.function_parameters.FunctionParameter
   :members: validate_target, list_concrete_parameters, evaluate, build_function

Custom callable declarations implement ``FunctionParameter``. Concrete blocks
must have stable names, order and shapes so builders can reconstruct results.
The default function builder supports one public argument; override it for
other argument-passing conventions, as ``BoundFunctionParameter`` does.

.. autoclass:: coker.parameters.ParameterMixin
   :members: degrees_of_freedom

Scalar and dense declarations implement this separate decision-count interface.
Callable declarations use ``FunctionParameter`` rather than ``ParameterMixin``.
