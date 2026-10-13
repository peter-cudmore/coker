Parameters and fitted functions
===============================

Import numeric decisions from ``coker.parameters`` and function-valued
realizations/results from ``coker.parameters.function_parameters``. A declaration
is construction-time metadata, not a fitted value or a universal solve result.
It specifies named scalar decision blocks, bounds and initial guesses. Numeric
constants bind values without adding solver decisions.

Numeric declarations
--------------------

``BoundedVariable(name, lower_bound, upper_bound, guess=0)`` and
``UnboundedVariable(name, guess=0)`` represent one scalar decision.
``BoundVector`` requires one-dimensional bounds and guess of identical shape,
ordered bounds and a finite feasible guess. ``DenseTensorVariable`` takes a
finite non-scalar initial array; its shape comes from the guess. CasADi supports
one- and two-dimensional parameter tensors. Copy mutable arrays before passing
them when construction must not retain an alias.

.. autoclass:: coker.parameters.BoundedVariable
   :members: degrees_of_freedom

.. autoclass:: coker.parameters.UnboundedVariable
   :members: degrees_of_freedom, lower_bound, upper_bound

.. autoclass:: coker.parameters.BoundVector
   :members: shape, size, degrees_of_freedom

.. autoclass:: coker.parameters.DenseTensorVariable
   :members: shape, size, degrees_of_freedom

Function-valued decisions
-------------------------

Declare a named ``FunctionSpace`` and choose a compatible finite realization.
Most realizations take one scalar argument/output; dense layers also support
vectors. ``validate_target`` rejects incompatible spaces with ``TypeError`` or
``ValueError``. Concrete parameter order from ``list_concrete_parameters()`` is
the solver reconstruction order; changing it changes the model contract.

``BoundFunctionParameter`` binds selected input indices of a Coker function to
decisions, retaining the other inputs as the public domain. ``ClosureParameter``
uses a source function with concrete blocks first, then target inputs.
``DenseLayer`` uses an affine map followed by a Coker activation; neither is an
arbitrary untraceable Python neural network. PyTorch's
``backend.import_as_parameter`` supports native modules as solver-owned
function parameters; see :doc:`backends`.

A ``FittedFunction`` is callable and retains ``specification`` (the declaration),
``space`` (the public domain/codomain), ``function`` (native callable) and
``parameters`` (structured fitted blocks). ``input_shape()``/``output_shape()``
return dimensional metadata. Backend-native fitted blocks may be arrays or
tensors; they are not universally immutable snapshots. Keep the fitted object
and backend dependencies alive when evaluating it.

A complete base-install binding example (solver construction is shown in
:doc:`optimisation`):

.. doctest::

   >>> from coker import FunctionSpace, Scalar, function
   >>> from coker.parameters import UnboundedVariable
   >>> from coker.parameters.function_parameters import BoundFunctionParameter
   >>> affine = function([Scalar("t"), Scalar("slope")], lambda t, slope: slope * t)
   >>> declaration = BoundFunctionParameter(affine, {1: UnboundedVariable("slope", 2.)}, name="response")
   >>> target = FunctionSpace("response", [Scalar("t")], [Scalar("value")])
   >>> validated = declaration.validate_target(target)
   >>> assert declaration.evaluate((3.,), 2.) == 6.

.. autoclass:: coker.parameters.function_parameters.BoundFunctionParameter
   :members: validate_target, list_concrete_parameters, evaluate

.. autoclass:: coker.parameters.function_parameters.ClosureParameter
   :members: bind_callable, validate_target, list_concrete_parameters, evaluate

.. autoclass:: coker.parameters.function_parameters.DenseLayer
   :members: validate_target, list_concrete_parameters, evaluate

.. autoclass:: coker.parameters.function_parameters.MonotonePiecewiseLinear
   :members: validate_target, list_concrete_parameters, evaluate

.. autoclass:: coker.parameters.function_parameters.RadialBasisFunction
   :members: validate_target, list_concrete_parameters, evaluate

.. autoclass:: coker.parameters.function_parameters.FittedFunction
   :members: input_shape, output_shape
   :special-members: __call__

Custom realizations implement the advanced :class:`~coker.parameters.function_parameters.FunctionParameter`
interface described in :doc:`extensions`, rather than depending on tape helpers.
