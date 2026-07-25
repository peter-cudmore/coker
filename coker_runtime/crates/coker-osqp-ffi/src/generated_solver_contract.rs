#![allow(dead_code)]

//! Build-time contract for externally supplied OSQP 0.6.3 generated artifacts.
//!
//! The build script expects a generated solver output directory with the same
//! layout produced by upstream OSQP 0.6.3 code generation: a public
//! `include/workspace.h`, generated support headers in `include/`, and the
//! generated solver translation unit at `src/osqp/workspace.c`.

pub const GENERATED_SOLVER_ENV: &str = "COKER_OSQP_GENERATED_SOLVER_DIR";
pub const GENERATED_SOLVER_INCLUDE_DIR: &str = "include";
pub const GENERATED_SOLVER_CONFIGURE_DIR: &str = "configure";
pub const GENERATED_SOLVER_SOURCE_DIR: &str = "src/osqp";
pub const GENERATED_SOLVER_WORKSPACE_HEADER: &str = "workspace.h";
pub const GENERATED_SOLVER_TYPES_HEADER: &str = "types.h";
pub const GENERATED_SOLVER_QDLDL_INTERFACE_HEADER: &str = "qdldl_interface.h";
pub const GENERATED_SOLVER_CONFIGURE_HEADER: &str = "osqp_configure.h";
pub const GENERATED_SOLVER_WORKSPACE_SOURCE: &str = "src/osqp/workspace.c";
