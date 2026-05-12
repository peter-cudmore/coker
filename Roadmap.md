# Coker Roadmap

## v0.3 - Dynamical Systems Tools

Goals:
 - Add representations for ODE/DAE's and Variational Optimisation problems (VOP)
 - Implement solver API for casadi/numpy/scipy
 - Implemented transcription of ODE/DAE/VOP to nonlinear program via pseudospectral methods
 - Tools for Monte-Carlo simulation of parameterised models.
 - 

## Planner-driven gaps
- Accept runtime parameters in solved optimisation programs instead of raising `NotImplementedError` when `CasadiSolver` has parameter inputs.
- Implement vector and matrix norm support in the codesign/program-builder layer so optimisation models do not need manual `sqrt(dot(x, x))` rewrites.
- Surface backend solve status and infeasibility to callers; the current CasADi/IPOPT path can return a locally infeasible iterate without exposing the failure as an error or structured status.
