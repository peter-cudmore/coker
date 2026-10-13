Backend selection and integration
=================================

User configuration
------------------

``function(..., backend="numpy")`` defaults to the base NumPy backend. Always
supply the backend when building mathematical or variational programs. In this
checkout, the unresolved ambient default from ``get_current_backend()`` is
``"coker"`` (an external plugin), and ``get_backend_by_name(name)`` as well as
``Function.lower()`` set the current backend. Lowering to an alternate backend
does not change ``Function.backend``, but can affect later implicit builders.
These current semantics are documented rather than changed here; explicit
selection avoids this dependency on evaluation order.

The registry uses extensible string/plugin names, not a closed enum. Instances
are lazily discovered through the ``coker.backends`` package entry points and
cached process-wide. ``get_backend_by_name(name, set_current=False)`` performs a
lookup without changing ambient selection. A missing name raises
``NotImplementedError``; conflicting registrations raise ``ValueError``;
missing optional dependencies raise their import errors.

.. list-table:: Execution and solve support
   :header-rows: 1

   * - Name / install
     - Function values
     - Solving limits
   * - ``numpy`` / base
     - Host scalars and NumPy arrays
     - SciPy initial-value integration and host NLP; no symbolic autograd
   * - ``sympy`` / base
     - Symbolic expressions and named callable expressions
     - Inspection/rank analysis; no numeric optimisation solver
   * - ``casadi`` / ``pip install 'coker[casadi]'``
     - Public numeric functions adapt to NumPy; lowered/native calls use CasADi
     - NLP and ODE/DAE variational transcription; 3.7.x, not 3.8.x
   * - ``pytorch`` / ``pip install 'coker[pytorch]'``
     - Tensors, device/dtype policy and eager autograd
     - CPU/CUDA NLP; fixed-horizon bound-only ODE fitting with quadratures.
       Fitting does not support algebraic DAEs, declared controls, comparison
       constraints or optimized horizons.
   * - ``jax`` / ``pip install 'coker[jax]'``
     - JAX arrays
     - No optimisation program construction
   * - ``coker`` / separately installed ``coker_backend``
     - Plugin-specific native/workspace execution
     - Not available in the base package; see :doc:`../backend_architecture`

Backend operation coverage is not identical. Do not infer support for an
operation/solver from numeric function support alone. See :doc:`dynamics` for
quadrature and transcription restrictions and :doc:`optimisation` for solve
status/failure contracts.

.. autofunction:: coker.backends.get_backend_by_name

.. autofunction:: coker.backends.get_current_backend

Signatures and lowering handles
-------------------------------

``Function.lower(LoweringOptions(backend=None))`` caches an executable handle
for each backend-instance/options pair. ``execute([*inputs])`` is the packed ABI
and returns a tuple in signature order, including absent ``None`` slots.
``handle(*inputs)`` unwraps one declared output. Input spaces are named domains;
output shapes are dimensional metadata, not runtime values.

Handles retain their execution plans and referenced callables/constants; they
are not portable serialized functions. NumPy and PyTorch reuse compiled plans;
PyTorch tensors preserve autograd. CasADi exposes ``ca_function`` when available
and otherwise uses an evaluator fallback. No common ownership, mutation,
thread-safety or differentiation-through-solves guarantee is implied by
``LoweringCapabilities``. Backend-native values can share storage: make an
explicit copy when a mutable snapshot is required.

.. autoclass:: coker.backends.lowered.FunctionInputSpec

.. autoclass:: coker.backends.lowered.FunctionOutputSpec

.. autoclass:: coker.backends.lowered.FunctionSignature

.. autoclass:: coker.backends.lowered.LoweringOptions

.. autoclass:: coker.backends.lowered.LoweringCapabilities

.. autoclass:: coker.backends.lowered.LoweredFunction
   :members: backend_name, signature, capabilities, execute, restore_public_outputs
   :special-members: __call__

Native CasADi import
--------------------

Install ``coker[casadi]``. The imported signature normally comes from native
ordered names and matrix dimensions: a ``(1, 1)`` is scalar and a column is a
one-dimensional vector. An explicit signature may override that declaration.
Native imports are pinned to their owning backend; lowering or composing them
with another backend raises ``RuntimeError``.

.. doctest::

   >>> import casadi as ca
   >>> import numpy as np
   >>> from coker.backends import get_backend_by_name
   >>> from coker.backends.lowered import LoweringOptions
   >>> x = ca.MX.sym("x", 2)
   >>> native = ca.Function("double", [x], [2 * x], ["x"], ["y"])
   >>> casadi_backend = get_backend_by_name("casadi", set_current=False)
   >>> imported = casadi_backend.import_function(native)
   >>> np.testing.assert_allclose(imported(np.array([1., 2.])), [2., 4.])
   >>> result = imported.lower().execute([np.array([1., 2.])])[0]
   >>> assert isinstance(result, ca.DM)

.. autoclass:: coker.backends.casadi.CasadiBackend
   :members: import_function

.. autoclass:: coker.backends.casadi.optimiser.CasadiNLPSolverOptions

.. autoclass:: coker.backends.casadi.CasadiVariationalOptions

Native PyTorch import and eager module export
---------------------------------------------

Install ``coker[pytorch]``. Provide an explicit ``FunctionSignature`` for native
imports; the module's parameters, buffers, training mode, device and dtype
remain authoritative. The wrapper retains a strong reference (logical ownership
transfer does not invalidate Python aliases). Mutating an alias changes the
imported module too. Configure the shared backend instance deliberately; tensor
conversion may move/cast inputs when device/dtype is set.

``as_module(function)`` creates an eager ``torch.nn.Module`` wrapper, not a
TorchScript or ``torch.compile`` target. For an ordinary traced graph its
constants stay closure state, not trainable parameters/registered buffers. An
imported module retains its native trainable state and gradients.
``import_as_parameter`` is different: it snapshots buffers and exposes floating
parameters as solver decisions without mutating the caller's module.

.. doctest::

   >>> import torch
   >>> from coker import Dimension, VectorSpace, function
   >>> from coker.backends import get_backend_by_name
   >>> from coker.backends.lowered import FunctionInputSpec, FunctionOutputSpec, FunctionSignature
   >>> backend = get_backend_by_name("pytorch", set_current=False)
   >>> signature = FunctionSignature(
   ...     inputs=(FunctionInputSpec("x", VectorSpace("x", 2)),),
   ...     outputs=(FunctionOutputSpec("y", Dimension((2,))),),
   ... )
   >>> native_module = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
   >>> with torch.no_grad():
   ...     _ = native_module.weight.copy_(2 * torch.eye(2, dtype=torch.float64))
   >>> imported = backend.import_module(native_module, signature)
   >>> composed = function([VectorSpace("x", 2)], lambda x: imported(x) + x, backend="pytorch")
   >>> tensor = torch.tensor([1., 3.], dtype=torch.float64, requires_grad=True)
   >>> output = backend.as_module(composed)(tensor)
   >>> torch.testing.assert_close(output, 3 * tensor)
   >>> output.sum().backward()
   >>> torch.testing.assert_close(tensor.grad, torch.tensor([3., 3.], dtype=torch.float64))
   >>> torch.testing.assert_close(native_module.weight.grad, torch.tensor([[1., 3.], [1., 3.]], dtype=torch.float64))

.. autoclass:: coker.backends.pytorch.PytorchBackend
   :members: as_module, import_function, import_module, import_as_parameter

.. autoclass:: coker.backends.pytorch.PytorchModule
   :members: forward

.. autoclass:: coker.backends.pytorch.PytorchNLPSolverOptions

.. autoclass:: coker.backends.pytorch.PytorchODESolverParameters

Advanced PyTorch variational configuration
------------------------------------------

The ordinary ``problem.get_solver()`` path uses backend defaults. For direct
shooting configuration, import ``PytorchVariationalSolverOptions`` and
``create_variational_solver`` from ``coker.backends.pytorch.variational``.
Its ``ode`` configures ``torchdiffeq`` and ``optimiser_options`` are forwarded
to the selected PyTorch optimizer. This is a backend-specific interface, not
an interchangeable CasADi transcription policy.

.. autoclass:: coker.backends.pytorch.variational.PytorchVariationalSolverOptions

.. autofunction:: coker.backends.pytorch.variational.create_variational_solver
