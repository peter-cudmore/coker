Backend API
============

This reference defines backend selection, lowering handles and solver options.
See :doc:`../backends` for capability and selection advice,
:doc:`../tutorials/backends` for an introduction, and
:doc:`../how_to/backend_interoperability` for native import and export recipes.

Lookup and selection
--------------------

The public lookup functions are exported by ``coker.backends``. Names are
resolved through registered factories or the ``coker.backends`` entry-point
group. A backend instance is cached per name.

.. autofunction:: coker.backends.get_backend_by_name

   Return the named backend. The default ``set_current=False`` does not change
   the current selection; ``set_current=True`` selects the returned instance.
   An unknown name raises ``NotImplementedError``. Conflicting plugin
   registrations raise ``ValueError``. Optional backend dependencies must be
   installed before their plugins can be loaded.

.. autofunction:: coker.backends.get_current_backend

   Return the selected backend, or NumPy if none has been selected.

.. autofunction:: coker.backends.register_backend

   The factory takes no arguments and returns a backend instance.

.. py:function:: coker.backends.evaluate(function, args, backend=None)

   Evaluate a Coker function through the interpreter with an ordered sequence
   of arguments. Return a list of outputs, including absent ``None`` slots.
   ``backend=None`` uses the current selection, not ``function.backend``;
   an explicit backend name does not change the selection.

Ordinary :func:`coker.function` construction defaults to ``backend="numpy"``.
Concrete :class:`coker.Function` calls use that function's backend rather than
the current selection. An implicit
:meth:`coker.toolkits.codesign.ProblemBuilder.build` uses the current selection;
explicit construction and lowering do not change it. Variational problems
have a separate CasADi solver default.

.. _backend-lowering-contract:

Lowering and execution
----------------------

:meth:`coker.Function.lower` accepts ``options=None`` or a
:class:`~coker.backends.lowered.LoweringOptions` instance and returns a
:class:`~coker.backends.lowered.LoweredFunction`. ``None`` selects the function's
backend. ``LoweringOptions(backend="sympy")``, for example, requests another
backend without changing the function or current selection. Handles are cached
per backend instance and options value. An object of another options type raises
``TypeError``. A native imported callable cannot be lowered by a different
backend; this raises ``RuntimeError``.

:meth:`coker.toolkits.codesign.MathematicalProgram.lower` instead takes no
options and returns a :class:`coker.Function` pinned to the program's configured
solver backend (NumPy when unset). Calling the returned function solves the
program; it does not return a lowered execution handle. Its outputs are the
objective followed by the declared outputs. To obtain a handle, lower that
returned function. This does not permit moving its solver to another backend.

The common execution ABI is positional and ordered:

* ``handle.execute(inputs)`` takes one sequence containing all arguments in
  ``signature.inputs`` order. It always returns a tuple in
  ``signature.outputs`` order, retaining absent ``None`` outputs.
* ``handle(*inputs)`` takes separate positional arguments and unwraps a
  one-output tuple. Zero or multiple outputs remain tuples.
* A concrete ``Function`` call additionally restores the public output
  representation. A single returned expression is unwrapped; an implementation
  returning a list or tuple retains tuple output, even with one element.

Native handle results need not have the same representation as public function
results. In particular, native CasADi handles retain CasADi values, while
``Function`` restores numerical scalars and NumPy arrays where conversion is
possible. With tensor inputs, PyTorch handles retain tensors and their autograd
graph; without tensor inputs, their numerical outputs are converted to NumPy
values. Shapes and output order come from the signature, not argument names
passed as keywords.

Signature records
~~~~~~~~~~~~~~~~~

The canonical records are imported from ``coker.backends.lowered``. They are
frozen dataclasses. Input and output tuples preserve declaration order; names
describe the interface rather than enabling keyword dispatch. Native imports
need an explicit signature unless their backend can derive one.

.. autoclass:: coker.backends.lowered.FunctionInputSpec

   ``space`` is a ``Scalar``, ``VectorSpace`` or ``FunctionSpace`` declaration.

.. autoclass:: coker.backends.lowered.FunctionOutputSpec

   ``shape`` is a ``Dimension``, ``Scalar``, ``VectorSpace``, ``FunctionSpace``
   or ``None``. ``None`` declares an absent output slot.

.. autoclass:: coker.backends.lowered.FunctionSignature

   ``inputs`` is a tuple of ``FunctionInputSpec`` records; ``outputs`` is a
   tuple of ``FunctionOutputSpec`` records. These are the public record names,
   not ``InputSpec`` or ``OutputSpec``.

.. autoclass:: coker.backends.lowered.LoweringOptions

   ``backend=None`` retains the function's backend; a string selects a
   registered backend for this lowering.

.. autoclass:: coker.backends.lowered.LoweringCapabilities

   These flags advertise execution mechanisms, not numerical correctness,
   ownership, lifetime or thread-safety guarantees. In particular, advertised
   JAX autograd support does not resolve the known derivative-correctness
   limitations tracked in issue 99.

.. autoclass:: coker.backends.lowered.LoweredFunction
   :members: backend_name,signature,capabilities,execute,restore_public_outputs
   :special-members: __call__

Solver configuration interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lowering options select an execution backend; they do not configure integration
algorithms, backend device/dtype settings or mathematical-program optimisers.
Use the concrete options class documented for the selected solver.

.. autoclass:: coker.interfaces.SolverParameters

CasADi
-------

Install ``coker[casadi]`` to use these objects. For mathematical programs, pass
``CasadiNLPSolverOptions`` as ``ProblemBuilder(solver_options=...)`` before
building with ``"casadi"``. For a worked solve, see
:doc:`../how_to/casadi_solver_tuning`.

.. autoclass:: coker.backends.casadi.CasadiBackend
   :members: import_function

   ``import_function`` derives ordered names and matrix dimensions from a
   native ``casadi.Function`` when no signature is supplied. Scalars have
   native shape ``(1, 1)``; column vectors map to one-dimensional Coker vectors.
   See :ref:`backend-import-casadi`.

.. autoclass:: coker.backends.casadi.CasadiLoweredFunction
   :members: ca_function

   ``ca_function`` is a native ``casadi.Function`` when the graph supports native
   lowering, otherwise ``None``. Function-space inputs and absent outputs use
   the evaluation fallback. See :ref:`backend-export-casadi`.

.. autoclass:: coker.backends.casadi.CasadiNLPSolverOptions

   ``optimiser_options`` is a flat mapping passed to the IPOPT ``nlpsol``
   constructor, copied when the solver is built. IPOPT settings use the
   ``ipopt.`` prefix. The inherited ``warm_start=False`` field does not enable
   a Coker-managed warm start on this solver path.

.. autoclass:: coker.backends.casadi.CasadiResidualSolver
   :members:

.. autoclass:: coker.backends.casadi.CasadiResidualSolverOptions

   Use with independent residual-DAE integration, not mathematical programs.
   ``solver`` is required and must be a ``CasadiResidualSolver`` member.
   ``idas_options`` and ``variational_options`` are separate option mappings;
   only the selected solver's mapping is used. Without explicit selection,
   integration uses IDAS when available, otherwise variational collocation.
   A solver failure does not trigger an algorithm switch.

.. autoclass:: coker.backends.casadi.CasadiVariationalOptions

   Supply through ``TranscriptionOptions(backend_options=options)`` on the
   variational problem. These settings configure transcription, IPOPT, scaling
   and optional adaptive refinement, not independent IDAS integration.
   Refinement is disabled by default. The callback keyword is spelt
   ``interation_callback``.

NumPy
------

NumPy is included in the base installation.

.. autoclass:: coker.backends.numpy.NumpyBackend

.. autoclass:: coker.backends.numpy.Solver
   :members:

.. autoclass:: coker.backends.numpy.NumpySolverParameters

   ``solver`` selects SciPy ``solve_ivp`` for explicit ODEs: ``RK45`` (default),
   ``LSODA``, ``Radau`` or ``BDF``. For independent square residual DAEs,
   ``implicit_maximum_time_step`` (default ``1e-3``) bounds the implicit step,
   and ``implicit_nonlinear_tolerance`` (default ``sqrt(machine epsilon)``)
   controls the nonlinear solve. Both implicit settings must be finite and
   positive; the explicit ODE method does not select the residual algorithm.

NumPy mathematical programs use SciPy's ``trust-constr`` path and do not expose
a backend-specific public NLP options class.

PyTorch
--------

Install ``coker[pytorch]`` to use these objects. Device and dtype are backend
properties, separate from solver options. The documentation recipes use CPU
tensors; CUDA execution remains experimental and unverified.

.. autoclass:: coker.backends.pytorch.PytorchBackend
   :members: as_module,import_function,import_module

   ``as_module`` returns an eager module accepting one positional input per
   signature input. Closure constants are not registered parameters or buffers.
   The adapter supports eager autograd, not TorchScript or ``torch.compile``.
   See :ref:`backend-export-pytorch`.

   ``import_module`` retains the supplied module and its authoritative
   parameters and buffers; it does not transfer or copy ownership. Both native
   import methods require a ``FunctionSignature``. The callable runs only on
   its owning backend. See :ref:`backend-import-pytorch`.

.. autoclass:: coker.backends.pytorch.PytorchModule
   :members: forward

.. py:class:: coker.backends.pytorch.PytorchNLPSolverOptions

   Frozen solver-options dataclass accepted by
   ``ProblemBuilder(solver_options=...)`` when building with ``"pytorch"``.
   It inherits the Boolean ``warm_start`` field (default ``False``), which
   enables reuse of compatible decisions and multipliers from a prior solve.

   ``optimiser_method`` selects ``"LBFGS"`` (default) or ``"Adam"``;
   ``optimiser_options`` is a mapping forwarded to that PyTorch optimiser
   (default: empty). Adam performs explicit optimiser steps. For LBFGS, the
   dedicated iteration, tolerance and history fields override corresponding
   mapping entries, and ``line_search_fn`` is fixed to ``"strong_wolfe"``.

   .. list-table:: Numerical settings
      :header-rows: 1

      * - Field
        - Default
        - Purpose
      * - ``inner_iterations``
        - ``25``
        - Objective optimisation iteration limit.
      * - ``restoration_iterations``
        - ``25``
        - Feasibility-restoration iteration limit.
      * - ``barrier_stages``
        - ``8``
        - Maximum barrier stages.
      * - ``augmented_lagrangian_stages``
        - ``10``
        - Maximum augmented-Lagrangian stages.
      * - ``barrier_reduction``
        - ``0.2``
        - Barrier reduction factor, strictly between zero and one.
      * - ``penalty_growth``
        - ``10.0``
        - Positive penalty growth factor.
      * - ``tolerance_grad``, ``tolerance_change``
        - ``1e-4``, ``1e-5``
        - Positive LBFGS stopping tolerances.
      * - ``tolerance_constraint``
        - ``1e-4``
        - Positive constraint tolerance.
      * - ``interior_margin``
        - ``1e-4``
        - Positive interior margin for inequality handling.
      * - ``history_size``
        - ``10``
        - Positive LBFGS history size.

   Iteration counts, stage counts and history size must be positive integers.
   Invalid optimiser names raise ``ValueError``; a non-mapping
   ``optimiser_options`` raises ``TypeError``. Constraint handling combines
   feasibility restoration, barrier and augmented-Lagrangian stages.

.. autoclass:: coker.backends.pytorch.PytorchODESolverParameters

   ``method="dopri5"``, ``rtol=1e-6``, ``atol=1e-8`` and ``options=None``
   map to ``torchdiffeq.odeint``. This initial-value path supports ODEs and
   quadratures, not algebraic constraints.

.. autoclass:: coker.backends.pytorch.variational.PytorchVariationalSolverOptions

   Import from ``coker.backends.pytorch.variational`` and pass as ``options``
   to ``PytorchVariationalSolver(problem, options)``. The ordinary
   ``problem.get_solver("pytorch")`` path constructs default options; it does
   not consume ``TranscriptionOptions.backend_options``.

   ``ode`` holds ``PytorchODESolverParameters``. ``optimiser_method`` accepts
   ``"LBFGS"`` (default) or ``"Adam"``; ``optimiser_options`` is forwarded to
   its constructor. The defaults are ``max_iter=100``, ``tolerance_grad=1e-6``,
   ``tolerance_change=1e-9`` and ``line_search_fn="strong_wolfe"``. Replace these
   LBFGS-specific options when selecting Adam; each solve makes one optimiser
   ``step`` call.

   ``trajectory_degree=8`` and ``trajectory_n_intervals=4`` must be positive
   integers. They configure the returned state and quadrature interpolants,
   not the ODE solver's time steps. The inherited ``warm_start`` field defaults
   to ``False`` but is not used by this solver.

   This solver fits parameters for fixed-horizon, bound-only ODE problems,
   including quadratures. Algebraic DAEs, control declarations, variational
   constraints and optimised horizons are not supported.

.. autoclass:: coker.backends.pytorch.variational.PytorchVariationalSolver
   :members: solve

.. Author notes

   - Audience: users integrating native libraries or configuring backend solvers.
   - Own ordered signatures, lowering return types, execution ABI and solver-option contracts here.
   - Keep native interoperability walkthroughs in the cookbook and selection advice in the guide.
   - Check solver-option routing against each backend; variational configuration is not uniform.
