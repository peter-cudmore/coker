System analysis
===============

:mod:`coker.dynamics` performs symbolic rank tests for supported autonomous ODE
and semi-explicit index-one DAE models.  Use it during model development to
establish local structural properties before selecting a controller, an
observer, or a numerical estimation experiment.

The public entry points are:

- :func:`coker.dynamics.analyse_controllability` for local accessibility;
- :func:`coker.dynamics.analyse_observability` for local observability; and
- :func:`coker.dynamics.analyse_identifiability` for local structural
  identifiability.

All conclusions are **generic and local**.  They apply where the reported
:attr:`coker.dynamics.AnalysisResult.rank_conditions` are nonzero; they do not
assert global reachability, global state reconstruction, or globally unique
parameters.

Model representation
--------------------

For an ODE, the symbolic analysis representation is

.. math::

   \dot{x} = f(x, u, p), \qquad y = h(x, u, p),

where :math:`x \in \mathbb{R}^n` is the differential state, :math:`u` is a
known control, :math:`p` is the parameter vector, and :math:`y` is the declared
output.  :class:`coker.dynamics.SymbolicSystem` exposes the ordered SymPy
expressions for direct use.

Supported DAEs have the semi-explicit index-one form

.. math::

   \dot{x} = f(x, z, u, p), \qquad 0 = g(x, z, p), \qquad y = h(x, z, u, p),

with algebraic state :math:`z` and generically nonsingular
:math:`J_z = \partial g / \partial z`.  Coker uses symbolic linear solves,
not an explicit symbolic inverse, to obtain tangent fields and restricted
output gradients:

.. math::

   \dot z = -J_z^{-1} J_x \dot x,
   \qquad
   d h|_{g=0} = h_x - w J_x,
   \qquad
   J_z^\mathsf{T} w^\mathsf{T} = h_z^\mathsf{T}.

The inverse notation above defines the mathematics only; implementation uses
``LUsolve`` for both equations.

Results and rank conditions
---------------------------

Each entry point returns its own specialization of the common
:class:`coker.dynamics.AnalysisResult` base class:

- :class:`coker.dynamics.ControllabilityResult`;
- :class:`coker.dynamics.ObservabilityResult`; or
- :class:`coker.dynamics.IdentifiabilityResult`.

``status`` is :class:`coker.dynamics.AnalysisStatus.TRUE`, ``FALSE``, or
``INCONCLUSIVE``.  The result type identifies the property that the status
refers to.  ``rank`` is the established rank, ``required_rank`` is the rank
needed to establish the property, ``matrix`` is the final symbolic rank matrix,
and ``generators`` are the vector fields or scalar expressions used to build
it.

``rank_conditions`` is the precise validity domain of the reported generic
rank.  Each expression in the tuple must be nonzero.  Coker includes a nonzero
maximal minor witnessing the reported rank and, for a DAE, prepends
:math:`\det J_z`.  For example,

.. code-block:: python

   result = analyse_observability(system)
   for condition in result.rank_conditions:
       print(condition, "!= 0")

means that the reported rank holds on the intersection of those nonzero sets.
A condition such as ``x_0`` excludes the singular set :math:`x_0 = 0`; it does
not mean the model fails there.  An empty tuple is expected for rank zero,
because no nonzero minor can witness it.

Local accessibility
-------------------

For an input-affine system,

.. math::

   \dot{x} = f_0(x) + \sum_{i=1}^{m} u_i f_i(x),

Coker begins with the control vector fields :math:`f_i` and repeatedly closes
them under Lie brackets with the drift and every control field:

.. math::

   [v, w] = D w\,v - D v\,w.

Let :math:`\mathcal{L}` be the resulting Lie algebra evaluated at a generic
state.  A full rank field matrix,

.. math::

   \operatorname{rank}[v_1\ \cdots\ v_k] = n,

produces ``TRUE``.  A stabilized rank defect produces ``FALSE``.  This is the
Lie-algebra rank test for **local accessibility**.  It does not prove global
controllability, finite-time reachability, constrained-input controllability,
or provide a control law.

Local observability
-------------------

Observability treats inputs as known arbitrary signals.  For the same
affine system, Coker starts with every scalar output component and closes it
under Lie derivatives along :math:`f_0, f_1, \ldots, f_m`:

.. math::

   \mathcal{L}_v q = Dq\,v.

The observability matrix stacks the gradients of the resulting expressions,

.. math::

   \mathcal{O}(x) =
   \begin{bmatrix}
     D h \\
     D\mathcal{L}_{v_1}h \\
     \vdots
   \end{bmatrix}.

If :math:`\operatorname{rank}\mathcal{O}=n`, the result is ``TRUE``.  A
stabilized rank defect produces ``FALSE``.  This establishes generic local
observability only; it does not construct an observer or establish global
state reconstruction.  Non-affine controls are ``INCONCLUSIVE`` because the
input-affine codistribution construction does not apply.

Structural local identifiability
--------------------------------

Identifiability is observability of the augmented autonomous system.  Fitted
parameters are promoted to constant states:

.. math::

   \tilde{x} = (x, p),
   \qquad
   \dot{\tilde{x}} = (f(x, p), 0).

Coker applies the autonomous observability construction to
:math:`\tilde{x}`.  Thus a full rank

.. math::

   \operatorname{rank}\mathcal{O}_{\tilde{x}} = n + \dim p

establishes ``TRUE``: the declared parameters are locally distinguishable from
ideal continuous output data.  A stabilized rank defect produces ``FALSE``.
Known numeric parameters are substituted before analysis and do not contribute
to the required rank.  Controlled systems are currently ``INCONCLUSIVE`` for
identifiability, because this implementation does not model derivatives of
known input trajectories.

A structural result is not a practical estimation guarantee.  Noise, sample
times, excitation, conditioning, parameter scaling, and the estimator still
determine whether a numerical fit is useful.

Bounds and supported scope
--------------------------

Every analysis accepts ``max_order``.  It limits Lie-bracket expansion for
accessibility and Lie-derivative expansion for observability or
identifiability.  If the limit is reached before rank closure, the result is
``INCONCLUSIVE`` rather than ``FALSE``.

The supported path accepts smooth, autonomous ODEs and semi-explicit index-one
DAEs with scalar or finite one-dimensional-vector states, parameters, controls,
and outputs.  Quadratures, function-valued parameters, control-dependent DAE
constraints, and non-affine controls are outside this rank-analysis scope.
Lower-level :func:`coker.dynamics.lower_system` and
:func:`coker.dynamics.lower_dae_system` raise
:class:`coker.dynamics.UnsupportedSystemError`; the public analyses instead
return ``INCONCLUSIVE`` with ``reason``.
