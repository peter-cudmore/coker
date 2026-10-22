Dynamics API
============

System construction
-------------------

Choose ``create_autonomous_ode`` for time-independent ODEs,
``create_control_system`` for controlled ODEs, or ``DynamicsSpec`` for
semi-explicit systems. Construction traces callbacks; it does not integrate a
trajectory. ``backend`` defaults to ``"numpy"``. See :doc:`../dynamics` for
worked models and the residual-DAE formulation.

The callback conventions differ: ``DynamicsSpec`` receives the input
**trajectory callable** ``u`` (evaluate it with ``u(t)``), while
``create_control_system`` callbacks receive its current **value**. Absent
channels in ``DynamicsSpec`` are declared with ``None``; keep their callback
positions. The source contracts below specify each signature and validation.

.. autoclass:: coker.dynamics.DynamicsSpec

.. autofunction:: coker.dynamics.create_dynamics_from_spec

.. autofunction:: coker.dynamics.create_control_system

.. autofunction:: coker.dynamics.create_autonomous_ode

Trajectory objects
------------------

A system call evaluates its declared output, not a tuple of internal state
channels or a fitted ``VariationalSolution``. Pass time, an input trajectory
when declared, and parameter values in declaration order. Numerical support
and output shapes are documented on the trajectory classes below.

``DynamicalSystem`` does not directly evaluate quadratures; convert with
:func:`~coker.dynamics.to_residual_dynamical_system` for a residual trajectory,
whose quadratures start at zero.

Numerical support depends on the integration backend, not just whether
callbacks can be traced:

* NumPy integrates explicit ODEs, including converted ODEs, with SciPy.
  Independent square index-one residual models use a fixed-step implicit
  method. Converted semi-explicit DAEs are not supported by its direct
  integration path.
* PyTorch evaluates explicit ODEs, including converted ODEs,
  but rejects algebraic constraints and independent implicit residual models.
  ODE integration requires the PyTorch extra and ``torchdiffeq``.
* CasADi supports semi-explicit ODEs/DAEs and independent residual models.
  For independent residual models, select the IDAS or variational path with
  :class:`~coker.backends.casadi.CasadiResidualSolverOptions`.
* SymPy provides symbolic lowering and analysis, not numerical integration.

.. autoclass:: coker.dynamics.DynamicalSystem
   :members: backend,get_state_dimensions,output_as_function_space
   :special-members: __call__

.. autoclass:: coker.dynamics.ResidualDynamicalSystem
   :members: backend,get_state_dimensions,output_as_function_space
   :special-members: __call__

.. autofunction:: coker.dynamics.to_residual_dynamical_system

Composition
-----------

``direct_sum`` joins independent systems in subsystem order and returns the
combined system alongside its projections. Numeric channels use flat-vector
``CompositionOperator`` projections; function-valued parameters require
``StructuredParameterProjection`` to preserve callable values and declaration
order. Numeric-only tuples of parameter declarations are not supported by the
flat-vector composition path. See :doc:`../dynamics` for joining and splitting
an evaluated output.

.. autofunction:: coker.dynamics.direct_sum

.. autoclass:: coker.dynamics.ProjectionSet

.. autoclass:: coker.dynamics.CompositionOperator
   :members: inverse,as_matrices,offsets,sizes,dim
   :special-members: __call__

.. autoclass:: coker.dynamics.StructuredParameterProjection
   :members: split

Symbolic analysis
-----------------

See :doc:`../dynamics/system_analysis` for the equations, rank-test
interpretation and supported model scope. These entry points accept a
``DynamicalSystem``, ``SymbolicSystem`` or ``SymbolicDAESystem``.
They use SymPy lowering regardless of the model's numerical backend.

The supported domain is smooth systems without explicit time dependence,
with scalar or finite one-dimensional-vector states, controls, parameters
and outputs. Semi-explicit DAEs must be index one, with one constraint per
algebraic variable and a generically nonsingular algebraic Jacobian.
Quadratures, function-valued parameters, implicit symbolic functions,
nonsmooth expressions and control-dependent DAE constraints are unsupported.
An independent ``ResidualDynamicalSystem`` is not an analysis input.

Every analysis accepts keyword-only ``max_order=None``: ``None`` allows
expansion until rank closure; a non-negative integer limits expansion order.
An unsupported model, invalid limit or exhausted limit returns
``INCONCLUSIVE`` with a ``reason`` rather than establishing a negative result.
Unsupported symbolic operations can therefore make analysis inconclusive
even when numerical evaluation succeeds.

.. autofunction:: coker.dynamics.analyse_controllability

.. autofunction:: coker.dynamics.analyse_observability

.. autofunction:: coker.dynamics.analyse_identifiability

.. autoclass:: coker.dynamics.AnalysisStatus

   Compare ``result.status`` explicitly, not through its truth value.

.. autoclass:: coker.dynamics.AnalysisResult

.. autoclass:: coker.dynamics.ControllabilityResult

.. autoclass:: coker.dynamics.ObservabilityResult

.. autoclass:: coker.dynamics.IdentifiabilityResult

Symbolic representations
~~~~~~~~~~~~~~~~~~~~~~~~

These lower-level interfaces expose ordered SymPy data for advanced analysis.
Unlike the analysis entry points, lowering raises
:class:`~coker.dynamics.UnsupportedSystemError` for unsupported models.

.. autofunction:: coker.dynamics.lower_system

.. autofunction:: coker.dynamics.lower_dae_system

.. autoclass:: coker.dynamics.SymbolicSystem

.. autoclass:: coker.dynamics.SymbolicDAESystem

.. autoexception:: coker.dynamics.UnsupportedSystemError

Variational problems
--------------------

Use ``VariationalProblemBuilder`` to construct symbolic objectives and
constraints. ``VariationalProblem`` also accepts a loss callback.
``TranscriptionOptions`` configures discretisation; backend-specific options
select solver behaviour. See :doc:`../dynamics` for CasADi solve support and
current lowering limits.


.. autoclass:: coker.dynamics.VariationalProblemBuilder
   :members: t,t_final,parameters,state,input,output,integrate,build
   :special-members: __enter__,__exit__

.. autoclass:: coker.dynamics.TranscriptionOptions

.. autoclass:: coker.dynamics.VariationalProblem
   :members: horizon_decision,decision_declarations,get_solver
   :special-members: __call__

Controls and callbacks
~~~~~~~~~~~~~~~~~~~~~~

Declare scalar control decisions with the classes below. Their coefficients,
sample rates and spike locations use the normalised transcription coordinate,
not physical time. Inspect a solved control in physical time through
:meth:`~coker.dynamics.VariationalSolution.control_law`.

.. autoclass:: coker.dynamics.ConstantControlVariable
   :members: degrees_of_freedom,to_solution

.. autoclass:: coker.dynamics.PiecewiseConstantVariable
   :members: degrees_of_freedom,to_solution

.. autoclass:: coker.dynamics.SpikeVariable
   :members: degrees_of_freedom,to_solution

The corresponding solution objects are callable at **normalised** time.
They are stored in ``VariationalSolution.control_solutions``; they do not
themselves convert physical time.

.. autoclass:: coker.dynamics.ConstantControlSolution
   :special-members: __call__

.. autoclass:: coker.dynamics.PiecewiseControlSolution
   :special-members: __call__

.. autoclass:: coker.dynamics.SpikeControlSolution
   :special-members: __call__

``ControlVariable`` and ``ControlSolution`` are unions of their three concrete
types above. ``ControlLaw`` is the callable control annotation. ``Solution``
and ``LossFunction`` annotate the trajectory and objective callbacks used by
``VariationalProblem``; they are not additional result containers.

.. autoclass:: coker.dynamics.VariationalIterationCallback
   :special-members: __call__

.. _variational-solution:

Solve results
-------------

Calling a variational problem or solver returns
:class:`~coker.dynamics.VariationalSolution`. An unsuccessful solve raises
:class:`~coker.toolkits.codesign.SolveFailure` with ``exception.solve_info``;
the result interface does not turn failure into a successful trajectory.
Native solver, configuration and evaluation exceptions are not necessarily
wrapped. See :doc:`optimisation` for :class:`~coker.toolkits.codesign.SolveInfo`
and :doc:`../dynamics` for a worked solve and result inspection.

Public result methods use scalar **physical** time on ``[0, t_final]``.
The stored ``path`` and ``control_solutions`` instead use **normalised** time.
Prefer public methods for channel inspection and output evaluation;
``as_raw()`` samples internal channels, whereas ``to_poly()`` constructs an
output interpolant. Their endpoint and interpolation differences are specified
below.

Treat result data as read-only. The result object is mutable and does not
promise defensive copies of stored arrays, mappings or evaluator inputs.
Changing ``parameters`` does not refit the trajectory or synchronise the
numeric parameter vector used by outputs and constraints.

Constraint violations and segment diagnostics are not full feasibility or
global error certificates. Adaptive diagnostics record a maximum sampled
coupled residual; fixed-mesh segment diagnostics retain weighted signed
physical residual vectors.

.. autoclass:: coker.dynamics.VariationalSolution
   :members: from_solver,path_constraints,terminal_constraints,as_raw,state,algebraic,quadratures,control_law,to_poly
   :special-members: __call__

.. autoclass:: coker.dynamics.variational.solution.SegmentDefectDiagnostic
