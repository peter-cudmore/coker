System analysis
===============

:mod:`coker.dynamics` uses symbolic rank tests to analyse autonomous ODEs and
semi-explicit index-one DAEs. Use these tests to assess local structural
properties before designing a controller, observer, or parameter-fitting
experiment.

The public entry points are:

- :func:`coker.dynamics.analyse_controllability` for local accessibility;
- :func:`coker.dynamics.analyse_observability` for local observability; and
- :func:`coker.dynamics.analyse_identifiability` for local structural
  identifiability.

Results are **generic and local**. They apply where the reported
:attr:`coker.dynamics.AnalysisResult.rank_conditions` are nonzero. They do not
establish global reachability, state reconstruction, or parameter uniqueness.

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
:math:`J_z = \partial g / \partial z`. Tangent fields and restricted output
gradients satisfy:

.. math::

   \dot z = -J_z^{-1} J_x \dot x,
   \qquad
   d h|_{g=0} = h_x - w J_x,
   \qquad
   J_z^\mathsf{T} w^\mathsf{T} = h_z^\mathsf{T}.

Coker uses symbolic ``LUsolve`` for both equations, not an explicit inverse.

Results and rank conditions
---------------------------

The entry points return subclasses of :class:`coker.dynamics.AnalysisResult`:

- :class:`coker.dynamics.ControllabilityResult`;
- :class:`coker.dynamics.ObservabilityResult`; and
- :class:`coker.dynamics.IdentifiabilityResult`.

``status`` is :class:`coker.dynamics.AnalysisStatus.TRUE`, ``FALSE``, or
``INCONCLUSIVE`` for the property named by the result type. The other fields are:

- ``rank``: the established rank;
- ``required_rank``: the rank needed to establish the property;
- ``matrix``: the final symbolic rank matrix;
- ``generators``: the vector fields or scalar expressions used to build it.

``rank_conditions`` defines the domain of the reported generic rank. Every
expression in the tuple must be nonzero. It includes a nonzero maximal minor
witnessing the rank and, for a DAE, a leading :math:`\det J_z`. For example:

.. code-block:: python

   result = analyse_observability(system)
   for condition in result.rank_conditions:
       print(condition, "!= 0")

The reported rank holds where all printed conditions are nonzero. A condition
such as ``x_0`` excludes :math:`x_0 = 0`; it does not establish what happens
there. An empty tuple is expected for rank zero because no nonzero minor can
witness it.

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

produces ``TRUE``. A stabilised rank defect produces ``FALSE``. This tests
**local accessibility**, not global controllability, finite-time reachability,
or constrained-input controllability. It does not provide a control law.

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

If :math:`\operatorname{rank}\mathcal{O}=n`, the result is ``TRUE``. A
stabilised rank defect produces ``FALSE``. This tests generic local
observability; it does not construct an observer. Non-affine controls produce
``INCONCLUSIVE`` because the input-affine codistribution construction does not
apply.

Structural local identifiability
--------------------------------

Identifiability treats the unknown parameters as constant states in an
augmented autonomous system:

.. math::

   \tilde{x} = (x, p),
   \qquad
   \dot{\tilde{x}} = (f(x, p), 0).

Coker applies the autonomous observability construction to :math:`\tilde{x}`.
A full rank

.. math::

   \operatorname{rank}\mathcal{O}_{\tilde{x}} = n + \dim p

produces ``TRUE``: the declared parameters are locally distinguishable from
ideal continuous output data. A stabilised rank defect produces ``FALSE``.
Known numeric parameters are substituted before analysis and do not contribute
to the required rank. Controlled systems produce ``INCONCLUSIVE`` because
this implementation does not model derivatives of known input trajectories.

This is structural analysis, not numerical fitting. Noise, sample times,
excitation, conditioning, parameter scaling, and the estimator determine
whether a fit is useful. See :doc:`../how_to/system_identification` for a
fitting workflow.

Limits and supported scope
--------------------------

Every analysis accepts ``max_order``.  It limits Lie-bracket expansion for
accessibility and Lie-derivative expansion for observability or
identifiability.  If the limit is reached before rank closure, the result is
``INCONCLUSIVE`` rather than ``FALSE``.

Analysis supports smooth, autonomous ODEs and semi-explicit index-one DAEs with
scalar or finite one-dimensional-vector states, parameters, controls, and
outputs. Quadratures, function-valued parameters, control-dependent DAE
constraints, and non-affine controls are outside its scope.

The lower-level :func:`coker.dynamics.lower_system` and
:func:`coker.dynamics.lower_dae_system` raise
:class:`coker.dynamics.UnsupportedSystemError` for unsupported models. Public
analysis functions instead return ``INCONCLUSIVE`` with a ``reason``.
