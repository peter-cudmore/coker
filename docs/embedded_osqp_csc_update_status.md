# Outstanding: direct sparse fusion QP still fails embedded OSQP matrix update

Checked after the latest Coker updates on 2026-08-24.

## Reproduction

From `C:/projects/hexapod/hexapod_py` with `PYTHONPATH=C:/projects/hexapod/hexapod_py`:

```powershell
uv run --directory C:/projects/coker --extra casadi python -m pytest C:/projects/hexapod/hexapod_py/tests/test_phase5_fusion_qp.py -k "coker_fusion_qp_matches_casadi and nominal_support" --maxfail=1 -q
```

## Result

The compact direct fusion QP now constructs promptly, proving the prior post-extraction construction issue is resolved. Runtime execution still fails in `RuntimeQpProgram.solve()`:

```text
ValueError: embedded OSQP ABI update_data_mat failed with status 1
```

The failure occurs at:

```text
coker.backends.coker.runtime.RuntimeQpProgram.solve
  self._runtime.solve_into(inputs, self._solution, initial)
```

## Missing implementation

The direct sparse weighted-norm coefficient stream is still incompatible with the embedded OSQP matrix update ABI. Resolve and test all of the following together:

1. generated `P` CSC value count/order, including OSQP's required upper-triangular representation;
2. generated `A` CSC value count/order;
3. coefficient evaluator stream partitioning into `P`, `q`, `A`, `l`, and `u`;
4. runtime `update_data_mat` lengths and pattern identity checks;
5. nominal 24-variable / 66-row fusion solve equivalence against the CasADi reference.

Do not restore dense assembly, basis probing, or an application-level workaround. The Hexapod artifact, native QP execution, replay, and cutover gates remain blocked until this test passes.
