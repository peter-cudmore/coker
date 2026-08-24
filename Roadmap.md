# Coker Roadmap



## DONE: Keep QP coefficient-graph lowering sparse

**Problem.** Lowering the 24-decision, 66-row hexapod motion-fusion QP
coefficient function exceeded 5.6 GiB working set and did not finish within 20
seconds.

**Profile (Windows 11, CPython 3.13).**

- Original specification: 0.016 s, 0.14 MiB peak traced allocation.
- Original `_build_coefficient_function`: 6.64 s, 36.66 MiB peak traced
  allocation.
- Original `create_opgraph`: still running after 20.25 s at 5.65 GiB working
  set.
- 17.68 s of the interrupted graph phase was in 142 calls to
  `numpy.ndarray.nonzero`, reached through dense sparse-tensor round trips.

**Causes and fixes.**

- `BilinearWeights.__rmatmul__` converted sparse coefficient tensors to dense
  arrays and immediately back to DOK. NumPy projections now invoke the sparse
  contraction directly.
- `dot` converted coefficient tensors to dense arrays. It now accumulates DOK
  entries directly.
- Graph lowering extended every earlier workspace value into every later
  layer. A tape use-count pass now retains only live values.
- `BilinearWeights @ ndarray` fell through to generic lowering, splitting a
  fixed matrix-vector projection into many workspace layers. It now contracts
  the last output axis directly in DOK storage.

**Regression coverage.**

- NumPy projection and constant-vector contraction retain DOK constant,
  linear, and quadratic tensors.
- Sparse dot does not call `toarray`.
- A repeated-operation graph test bounds workspace extension by live uses.
- Coker backend, runtime, QP, operation, and sparse-tensor tests pass: 51
  tests.

**Result.** With the fusion objective expressed as 78 caller-owned weighted
residual rows rather than caller-assembled Hessian coefficients, the generated
coefficient tape has 8,393 nodes, five inputs, and 2,041 scalar outputs.
`_build_coefficient_function` completes in 0.829 s with 13.54 MiB peak traced
allocation; `create_opgraph` completes in 15.05 s with 70.09 MiB peak traced
allocation. Full QP bytecode compilation completes in 18.30 s with 83.55 MiB
peak traced allocation. An external sample during compilation observed a
211 MiB process working set, replacing the prior runaway 5.65 GiB growth.
