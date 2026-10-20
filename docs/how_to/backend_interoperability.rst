Backend import and export
=========================

Import native CasADi, SymPy or PyTorch objects into a Coker function, or expose a
Coker function to native tooling. These recipes assume you can
:doc:`create a function <../tutorials/function_evaluation>`. For backend selection
and capabilities, see the :doc:`backend guide <../backends>`.

Dependencies and entry points
-----------------------------

SymPy and NumPy are included in the base installation. For CasADi and PyTorch,
install the optional backends:

.. code-block:: console

   pip install "coker[casadi,pytorch]"

The CasADi export recipe continues its import recipe; the other recipes are
independent. The PyTorch recipes use CPU ``float64`` tensors; CUDA is not
required.

.. list-table:: Supported interfaces
   :header-rows: 1

   * - Backend
     - Native object into Coker
     - Coker into native tooling
   * - CasADi
     - ``backend.import_function(casadi_function, signature=None)``
     - ``f.lower().ca_function`` when native lowering is available
   * - SymPy
     - ``backend.import_function(callable, signature, name=...)``;
       wrap an expression with ``sympy.Lambda``
     - ``backend.lower_to_symbolic(f)`` returns arguments and expressions
   * - PyTorch
     - ``backend.import_module(module, signature)``
     - ``backend.as_module(f)`` returns an eager ``torch.nn.Module``

A ``FunctionSignature`` declares positional inputs and ordered outputs; it is
not inferred from Python type annotations. Imports become native evaluation
nodes during Coker tracing. Their implementations run on their owning backend,
not on Coker tracers; cross-backend composition and lowering are rejected.

For ordinary scalar/vector outputs, Coker calls and positional lowered-handle
calls return one output directly or multiple outputs as a tuple.
``handle.execute([arg0, arg1, ...])`` always returns a tuple. All adapters
preserve argument order and declared shapes; none automatically batches samples.

.. _backend-import-casadi:

Import a CasADi function
------------------------

Call ``backend.import_function`` with a native ``casadi.Function``. Its input
and output names, order and dimensions supply the signature: ``(1, 1)`` becomes
a scalar, ``(n, 1)`` becomes a length-``n`` vector, and other matrices retain
their shape. Import two outputs, then compose the vector output with Coker
arithmetic:

.. testcode:: backend_interop_casadi

   import casadi as ca
   import numpy as np

   from coker import Scalar, VectorSpace, function
   from coker.backends import get_backend_by_name

   backend = get_backend_by_name("casadi")
   native_time = ca.MX.sym("native_time")
   native_state = ca.MX.sym("native_state", 2, 1)
   native = ca.Function(
       "native_response",
       [native_time, native_state],
       [native_time + 1, 2 * native_state],
       ["time", "state"],
       ["offset_time", "scaled_state"],
   )
   imported = backend.import_function(native)
   assert [spec.name for spec in imported.signature.inputs] == ["time", "state"]
   assert [spec.name for spec in imported.signature.outputs] == [
       "offset_time", "scaled_state"
   ]

   time_result, state_result = imported(2.0, np.array([3.0, 4.0]))
   assert float(time_result) == 3.0
   np.testing.assert_allclose(np.asarray(state_result), [6.0, 8.0])

   composed = function(
       [Scalar("time"), VectorSpace("state", 2)],
       lambda time, state: imported(time + 1, state)[1] + time,
       backend="casadi",
   )
   np.testing.assert_allclose(
       np.asarray(composed(2.0, np.array([3.0, 4.0]))), [8.0, 10.0]
   )

To override the derived declaration, pass a ``FunctionSignature`` whose order
and shapes match the native call.

.. _backend-export-casadi:

Export a CasADi function
------------------------

Retrieve ``ca_function`` from the lowered Coker function for native numerical
evaluation, symbolic composition and differentiation:

.. testcode:: backend_interop_casadi

   lowered = composed.lower()
   exported = lowered.ca_function
   assert isinstance(exported, ca.Function)
   np.testing.assert_allclose(
       np.asarray(exported(2.0, ca.DM([3.0, 4.0]))), [[8.0], [10.0]]
   )

   symbolic_time = ca.MX.sym("time")
   symbolic_state = ca.MX.sym("state", 2, 1)
   symbolic_result = exported(symbolic_time, symbolic_state)
   state_jacobian = ca.Function(
       "state_jacobian",
       [symbolic_time, symbolic_state],
       [ca.jacobian(symbolic_result, symbolic_state)],
   )
   np.testing.assert_allclose(
       np.asarray(state_jacobian(2.0, ca.DM([3.0, 4.0]))), 2 * np.eye(2)
   )

``ca_function`` is ``None`` for the evaluate fallback, including functions with
function-space inputs or absent (``None``) outputs. Those handles can execute
supported Coker calls but cannot supply a native CasADi function. Native calls
retain CasADi's column-matrix representation rather than Coker's one-dimensional
vectors. The export is an in-process function, not generated C code.

.. _backend-import-sympy:

Import a SymPy expression
-------------------------

Wrap the expression in ``sympy.Lambda`` to fix its argument order, then pass it
and a ``FunctionSignature`` to ``backend.import_function``. Bare expressions
with implicit free-symbol ordering are not accepted. Other imported callables
must accept SymPy values and return the declared shapes.

.. testcode:: backend_interop_sympy_import

   import sympy as sp

   from coker import Scalar, function
   from coker.backends import get_backend_by_name
   from coker.backends.lowered import (
       FunctionInputSpec,
       FunctionOutputSpec,
       FunctionSignature,
   )

   backend = get_backend_by_name("sympy")
   z = sp.Symbol("z")
   native_expression = z**2 + 1
   native = sp.Lambda(z, native_expression)
   signature = FunctionSignature(
       inputs=(FunctionInputSpec("value", Scalar("value")),),
       outputs=(FunctionOutputSpec("result", Scalar("result")),),
   )
   imported = backend.import_function(native, signature, name="polynomial")
   composed = function(
       [Scalar("x")], lambda x: imported(x) + x, backend="sympy"
   )
   assert float(imported(2.0)) == 5.0
   assert float(composed(2.0)) == 7.0

   # An explicit symbolic evaluation executes the SymPy-compatible callable.
   q = sp.Symbol("q")
   evaluated = composed.lower()(q)
   assert sp.simplify(evaluated - (q**2 + q + 1)) == 0
   assert float(evaluated.subs(q, 2)) == 7.0

   # Inspection preserves the imported callable's identity instead of executing it.
   args, opaque_expression = backend.lower_to_symbolic(composed)
   x = sp.Symbol("x")
   assert args == [x]
   assert opaque_expression == sp.Function("polynomial")(x) + x

Choose the symbolic operation you need:

* Call the lowered handle with symbolic arguments to evaluate the imported
  callable and expand its expression.
* Use ``lower_to_symbolic`` to inspect the graph. Imports remain named,
  undefined functions, even for ``sympy.Lambda`` implementations. Supply a
  definition for ``polynomial(x)`` before evaluating that inspection expression
  numerically.

.. _backend-export-sympy:

Export SymPy expressions
------------------------

Call ``backend.lower_to_symbolic`` on a Coker arithmetic graph to obtain its
ordered arguments and expressions. The second return value is one expression
for a single output or a list for multiple outputs. Scalar inputs become
``Symbol`` objects; vector and matrix inputs become arrays of component symbols.

Use substitution, differentiation or ``sympy.lambdify`` on the result:

.. testcode:: backend_interop_sympy_export

   import numpy as np
   import sympy as sp

   from coker import Scalar, function
   from coker.backends import get_backend_by_name

   backend = get_backend_by_name("sympy")
   polynomial = function(
       [Scalar("x"), Scalar("offset")],
       lambda x, offset: x**2 + offset,
       backend="sympy",
   )
   args, expression = backend.lower_to_symbolic(polynomial)
   x, offset = args
   assert args == [sp.Symbol("x"), sp.Symbol("offset")]
   assert sp.simplify(expression - (x**2 + offset)) == 0
   assert sp.diff(expression, x) == 2 * x
   assert float(expression.subs({x: 2, offset: 3})) == 7.0

   numerical = sp.lambdify(args, expression, modules="numpy")
   points = np.array([-2.0, 0.0, 2.0])
   np.testing.assert_allclose(numerical(points, 3.0), [7.0, 3.0, 7.0])
   assert float(polynomial(2.0, 3.0)) == float(numerical(2.0, 3.0))

``lower_to_symbolic`` returns inspection expressions, unlike the execution
handle returned by ``f.lower()``. It does not expand arbitrary Python callables
or unsupported operations into SymPy algebra. ``lambdify`` is a SymPy API; the
array broadcasting above comes from NumPy, not Coker batching. The SymPy backend
does not provide Coker optimisation solves.

.. _backend-import-pytorch:

Import a PyTorch module
-----------------------

Call ``backend.import_module`` to compose a ``torch.nn.Module`` with Coker
arithmetic while training its original parameters. Declare one input per
positional ``forward`` argument and one output per result, in order. The
signature does not describe keyword-only arguments or nested Python containers.

This recipe uses length-one inputs and outputs, adds a fixed offset to the
model's output, and fits its weight and bias:

.. testcode:: backend_interop_pytorch_import

   import torch

   from coker import VectorSpace, function
   from coker.backends import get_backend_by_name
   from coker.backends.lowered import (
       FunctionInputSpec,
       FunctionOutputSpec,
       FunctionSignature,
   )

   class Affine(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
           self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

       def forward(self, features):
           return self.weight * features + self.bias

   backend = get_backend_by_name("pytorch")
   native = Affine().to(device="cpu", dtype=torch.float64)
   signature = FunctionSignature(
       inputs=(FunctionInputSpec("features", VectorSpace("features", 1)),),
       outputs=(FunctionOutputSpec("prediction", VectorSpace("prediction", 1)),),
   )
   imported = backend.import_module(native, signature)
   prediction = function(
       [VectorSpace("features", 1)],
       lambda features: imported(features) + 1.0,
       backend="pytorch",
   )

   features = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
   targets = torch.tensor([[0.0], [6.0]], dtype=torch.float64)
   optimizer = torch.optim.SGD(native.parameters(), lr=0.25)

   def training_loss():
       outputs = torch.stack([prediction(row) for row in features])
       return ((outputs - targets) ** 2).mean()

   native.train()
   initial_loss = training_loss().detach()
   for step in range(60):
       optimizer.zero_grad()
       loss = training_loss()
       loss.backward()
       if step == 0:
           torch.testing.assert_close(
               native.weight.grad, torch.tensor([-6.0], dtype=torch.float64)
           )
           torch.testing.assert_close(
               native.bias.grad, torch.tensor([-4.0], dtype=torch.float64)
           )
       optimizer.step()

   native.eval()
   with torch.no_grad():
       assert training_loss() < initial_loss * 1e-20
       torch.testing.assert_close(
           native.weight, torch.tensor([3.0], dtype=torch.float64),
           rtol=0, atol=1e-12,
       )
       torch.testing.assert_close(
           native.bias, torch.tensor([2.0], dtype=torch.float64),
           rtol=0, atol=1e-12,
       )
       torch.testing.assert_close(
           imported(torch.tensor([2.0], dtype=torch.float64)),
           torch.tensor([8.0], dtype=torch.float64), rtol=0, atol=1e-12,
       )
       torch.testing.assert_close(
           prediction(torch.tensor([2.0], dtype=torch.float64)),
           torch.tensor([9.0], dtype=torch.float64), rtol=0, atol=1e-12,
       )

   probe = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
   prediction(probe).sum().backward()
   torch.testing.assert_close(
       probe.grad, torch.tensor([3.0], dtype=torch.float64), rtol=0, atol=1e-12
   )

The import retains the original module without copying parameters or buffers.
You control its placement, training mode and optimiser. Keep the module,
buffers and tensor inputs on compatible devices and dtypes; importing does not
move them.

Differentiable PyTorch operations preserve autograd paths to module parameters
and tensor inputs through Coker composition. Detaching values or converting
them to NumPy inside ``forward`` breaks those paths.

This trains the caller's module with PyTorch, not a Coker solver. For a
Coker-owned function parameter, ``import_as_parameter`` instead treats
floating-point module parameters as solver decisions without mutating the
original module.

.. _backend-export-pytorch:

Export a PyTorch module
-----------------------

Call ``backend.as_module`` to expose a Coker function as an eager
``torch.nn.Module``. This model accepts features, coefficients and bias in
declaration order and returns a scalar tensor. Train the caller-owned
``nn.Parameter`` inputs while keeping the features fixed:

.. testcode:: backend_interop_pytorch_export

   import numpy as np
   import torch

   from coker import Scalar, VectorSpace, function
   from coker.backends.pytorch import PytorchBackend

   backend = PytorchBackend(device="cpu", dtype=torch.float64)
   affine = function(
       [VectorSpace("features", 2), VectorSpace("coefficients", 2), Scalar("bias")],
       lambda features, coefficients, bias: np.dot(coefficients, features) + bias,
       backend="pytorch",
   )
   model = backend.as_module(affine)
   assert isinstance(model, torch.nn.Module)
   assert list(model.parameters()) == []

   features = torch.tensor(
       [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
       dtype=torch.float64,
   )
   targets = torch.tensor([3.0, -1.0, -2.0, 4.0], dtype=torch.float64)
   coefficients = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
   bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
   optimizer = torch.optim.SGD([coefficients, bias], lr=0.25)

   def training_loss():
       predictions = torch.stack([model(row, coefficients, bias) for row in features])
       return ((predictions - targets) ** 2).mean()

   initial_loss = training_loss().detach()
   for step in range(100):
       optimizer.zero_grad()
       loss = training_loss()
       loss.backward()
       if step == 0:
           torch.testing.assert_close(
               coefficients.grad, torch.tensor([-2.0, 3.0], dtype=torch.float64)
           )
           torch.testing.assert_close(bias.grad, torch.tensor(-2.0, dtype=torch.float64))
       optimizer.step()

   with torch.no_grad():
       assert training_loss() < initial_loss * 1e-20
       torch.testing.assert_close(
           coefficients, torch.tensor([2.0, -3.0], dtype=torch.float64),
           rtol=0, atol=1e-10,
       )
       torch.testing.assert_close(
           bias, torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-10
       )
       torch.testing.assert_close(
           model(torch.tensor([2.0, 1.0], dtype=torch.float64), coefficients, bias),
           torch.tensor(2.0, dtype=torch.float64), rtol=0, atol=1e-10,
       )

The adapter registers no new trainable parameters. Input tensors remain
caller-owned and connected to autograd; ``optimizer.step()`` updates the
parameters supplied to the optimiser. Constants stay in Coker closure state,
not registered module parameters or buffers. Move input tensors directly:
moving the parameter-free adapter does not move them.

``as_module`` supports eager autograd, not TorchScript or ``torch.compile``
export. For Coker-owned optimisation rather than native PyTorch training, see
the :doc:`toolkit guide <../toolkits>`.
