Advanced extension interfaces
=============================

These interfaces are for backend/realization authors, not prerequisites for
building user models. Ordinary API navigation deliberately excludes tape
representations, operator enums, evaluator plans, solver transcription factories,
collocation implementation helpers and system-modelling registries/lazy handles.
An importable internal helper is not a stability promise.

Backend plugins
---------------

Register a zero-argument ``Backend`` factory through
``coker.backends.register_backend(name, factory)`` or the installed package's
``coker.backends`` entry-point group. Re-registering the same factory is
idempotent; a different factory for the same name raises ``ValueError``. Backend
names remain extensible strings. Factories are instantiated lazily and assigned
the selected registry name; cached instances are shared for the process.

A backend implements conversion/reshape, primitive operation evaluation,
compiled-plan evaluation/lowering and any advertised numerical solve paths.
``lower`` returns a ``LoweredFunction`` with an ordered signature, a packed
``execute(inputs) -> tuple`` ABI and capability flags. Unsupported solve paths
must fail rather than silently claim another backend's execution semantics.
Native imports must preserve owning-backend execution and output slots. The
flags describe execution support, not reentrancy or memory ownership.

.. autoclass:: coker.backends.backend.Backend
   :members: to_numpy_array, to_backend_array, reshape, call, get_evaluator, evaluate, lower, append_native_call, build_optimisation_problem, make_optimisation_module, evaluate_integrals, create_variational_solver

.. autoclass:: coker.backends.backend.VariationalSolver
   :members: parameters, solve

Function realisations
---------------------

Implement a shallow ``FunctionParameter`` subclass. Validate the target domain,
return finite named concrete decision declarations in stable order and evaluate
from structured blocks using traceable operations. See the
:ref:`custom declaration contract <custom-parameter-declarations>` for the
single-argument default builder and the required methods.

The separate :class:`coker.parameters.ParameterMixin` supplies a decision-count
interface for scalar and array declarations; it is not the callable interface.

Numerical program modules
-------------------------

``OptimisationModule`` is the extension boundary used by mathematical programs,
not an ordinary model constructor. A numerical solve returns the scalar
objective, declared outputs and any private parameter captures requested by the
builder. Set ``last_solve_info`` before returning and before raising
``SolveFailure``. The public program removes captures from the objective-first
result tuple and reconstructs ``parameters`` separately.

See :class:`coker.toolkits.codesign.OptimisationModule` in the codesign
extension reference for the callable interface.

Mapped runtime allocation, bytecode alignment and caller-workspace contracts are
separate from Python handle ownership; see
:doc:`../architecture/evaluation_pipeline`.
