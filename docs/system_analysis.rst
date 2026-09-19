System analysis
===============

:mod:`coker.analysis` performs symbolic rank tests on a supported ODE model.
It is intended for structural questions during model development, before a
numerical experiment or a controller is chosen.  The results describe *generic
local* rank properties: they hold away from symbolic singular sets, and do not
make global claims.

The two primary analyses are:

- :func:`coker.analysis.analyse_controllability`, which tests generic local
  accessibility of the state; and
- :func:`coker.analysis.analyse_identifiability`, which tests structural local
  identifiability of model parameters from the declared outputs.

Both functions accept a Coker system that can be lowered to the supported
symbolic representation.  For direct symbolic use,
:class:`coker.analysis.SymbolicSystem` stores the ordered state, parameter,
control, dynamics, and output expressions.  Its fields contain SymPy symbols
and expressions, so their ordering is part of the model definition.
:func:`coker.analysis.lower_system` exposes the same lowering step for callers
who need the symbolic system itself.

Results and generic conditions
------------------------------

Both functions return an :class:`coker.analysis.AnalysisResult`:

.. code-block:: python

   from coker.analysis import analyse_controllability

   result = analyse_controllability(system)
   print(result.status, result.rank, result.required_rank)

``status`` is an :class:`coker.analysis.AnalysisStatus` value.  Accessibility
uses ``ACCESSIBLE`` and ``NOT_ACCESSIBLE``; identifiability uses
``IDENTIFIABLE`` and ``NOT_IDENTIFIABLE``.  Either analysis may instead return
``INCONCLUSIVE``.

``rank`` is the rank established by the symbolic test and ``required_rank`` is
the rank needed for that analysis to conclude positively.  ``matrix`` and
``generators`` retain their SymPy values for inspection.  When a nonzero rank
is established, ``generic_conditions`` contains symbolic expressions that must
not vanish for the reported generic rank.  There is no such witness for rank
zero, so this tuple may be empty.  For an inconclusive result, ``reason``
explains the unsupported model or analysis limit that prevented a conclusion.

For example, a witness such as ``x`` means that the displayed rank conclusion
is generic on the region where ``x != 0``.  It does not assert the same rank at
``x = 0`` or at every parameter value.

Local accessibility
-------------------

``analyse_controllability(system, *, max_order=None)`` forms the symbolic
accessibility rank test for an input-affine ODE.  A full state rank produces
``ACCESSIBLE``; a demonstrated rank defect produces ``NOT_ACCESSIBLE``.  The
name of this API is conventional, but its conclusion is **local
accessibility**, not global controllability.

In particular, an ``ACCESSIBLE`` result does not prove reachability between
arbitrary states, reachability in finite time, controllability subject to input
bounds, or the existence of a control law.  It also does not synthesize a
controller.  Those are separate, global or constrained-control questions.

A double integrator can be described directly with SymPy expressions:

.. code-block:: python

   import sympy as sp
   from coker.analysis import (
       AnalysisStatus,
       SymbolicSystem,
       analyse_controllability,
   )

   position, velocity, force = sp.symbols("position velocity force")
   double_integrator = SymbolicSystem(
       state=(position, velocity),
       parameters=(),
       controls=(force,),
       dynamics=(velocity, force),
       outputs=(position, velocity),
   )

   result = analyse_controllability(double_integrator)
   assert result.status is AnalysisStatus.ACCESSIBLE
   assert (result.rank, result.required_rank) == (2, 2)

Structural local identifiability
--------------------------------

``analyse_identifiability(system, *, max_order=None)`` tests whether the
parameters are structurally locally distinguishable from the model outputs.
It is a symbolic property of the exact model and observations.  An
``IDENTIFIABLE`` result is not a claim that a numerical fit will recover a
parameter accurately: practical numerical estimability also depends on data,
noise, sampling times, excitation, parameter scaling, conditioning, and the
estimation method.

Conversely, a local structural result is not a global uniqueness theorem.  It
may identify a parameter only locally and generically, subject to the reported
nonzero conditions.

For a scalar exponential-growth model with its state observed, the growth rate
is generically structurally locally identifiable:

.. code-block:: python

   import sympy as sp
   from coker.analysis import (
       AnalysisStatus,
       SymbolicSystem,
       analyse_identifiability,
   )

   population, growth_rate = sp.symbols("population growth_rate")
   growth_model = SymbolicSystem(
       state=(population,),
       parameters=(growth_rate,),
       controls=(),
       dynamics=(growth_rate * population,),
       outputs=(population,),
   )

   result = analyse_identifiability(growth_model)
   assert result.status is AnalysisStatus.IDENTIFIABLE
   assert result.rank == result.required_rank

Supported scope and inconclusive results
-----------------------------------------

The initial symbolic path supports explicit, smooth, autonomous ODEs with
finite :class:`coker.Scalar` or one-dimensional finite
:class:`coker.VectorSpace` state, parameters, controls, and outputs.  It
requires dynamics without algebraic variables, constraints, or quadratures.
Neither dynamics nor outputs may depend explicitly on time; time is not
silently promoted to an extra state.  Controls in a system passed to the
symbolic path must enter affinely.

The following are deliberately outside this initial scope:

- differential-algebraic systems (DAEs), algebraic variables, constraints, or
  quadratures;
- function-valued parameters;
- non-affine controls;
- controlled systems for identifiability analysis; and
- state, parameter, control, or output dimensions other than supported scalar
  or one-dimensional finite-vector dimensions.

Rather than guessing or reducing an unsupported system to a different problem,
the analysis functions return ``INCONCLUSIVE`` and populate ``reason``.  The
lower-level :func:`coker.analysis.lower_system` instead raises
:class:`coker.analysis.UnsupportedSystemError` when called directly on such a
model, allowing applications to decide how to present that limitation.

``max_order`` can cap the symbolic generator or output-derivative order used by
an analysis.  A cap that prevents the rank procedure from reaching a conclusion
returns ``INCONCLUSIVE``; it must not be read as evidence for an adverse rank
result.  Leave it as ``None`` unless a bounded symbolic calculation is
required.
