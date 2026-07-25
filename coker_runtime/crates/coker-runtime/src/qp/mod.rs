#![cfg_attr(not(any(feature = "std", osqp_embedded)), allow(dead_code))]

#[cfg(all(feature = "std", not(osqp_embedded)))]
use alloc::vec::Vec;
use alloc::{format, string::ToString};
#[cfg(any(osqp_embedded, feature = "std"))]
use core::{
    marker::PhantomData,
    mem::{align_of, size_of, MaybeUninit},
    ptr::NonNull,
    slice,
};

#[cfg(all(feature = "std", not(osqp_embedded)))]
use coker_bytecode::ArchivedQpOutputSlice;
use coker_bytecode::{
    archived_module, ArchivedInputSpec, ArchivedOutputSpec, ArchivedQpCoefficientOutputs,
    ArchivedQpProgram,
};
use coker_osqp_ffi as ffi;
#[cfg(any(osqp_embedded, feature = "std"))]
use rkyv::rend::u32_le;

use crate::{MappedModule, MappedProgram, QpProgramInfo, RuntimeError, SpecInfo};
#[cfg(all(feature = "std", not(osqp_embedded)))]
use core::ptr;
mod embedded;
mod host;
mod support;

#[cfg(all(feature = "std", not(osqp_embedded)))]
pub use self::host::{QpRuntime, QpWorkspaceLayout, QpWorkspaceRegion};

#[allow(unused_imports)]
use self::support::*;

/// Terminal status returned by the QP solve backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpSolveStatus {
    Solved,
    SolvedInaccurate,
    MaxIterReached,
    PrimalInfeasible,
    PrimalInfeasibleInaccurate,
    DualInfeasible,
    DualInfeasibleInaccurate,
    NonCvx,
    SigInt,
    TimeLimitReached,
    Unsolved,
    Other(i32),
}

/// Host-side solve outputs borrowed from the live OSQP workspace.
#[derive(Debug, Clone)]
pub struct QpSolveResult<'a> {
    pub status: QpSolveStatus,
    pub primal: Option<&'a [f64]>,
    pub dual: Option<&'a [f64]>,
    pub iterations: Option<usize>,
    pub primal_residual: Option<f64>,
    pub dual_residual: Option<f64>,
}
impl QpSolveStatus {
    fn from_raw(status: ffi::c_int) -> Self {
        match status {
            x if x == ffi::OSQP_SOLVED as ffi::c_int => Self::Solved,
            x if x == ffi::OSQP_SOLVED_INACCURATE as ffi::c_int => Self::SolvedInaccurate,
            x if x == ffi::OSQP_MAX_ITER_REACHED as ffi::c_int => Self::MaxIterReached,
            x if x == ffi::OSQP_PRIMAL_INFEASIBLE as ffi::c_int => Self::PrimalInfeasible,
            x if x == ffi::OSQP_PRIMAL_INFEASIBLE_INACCURATE as ffi::c_int => {
                Self::PrimalInfeasibleInaccurate
            }
            x if x == ffi::OSQP_DUAL_INFEASIBLE as ffi::c_int => Self::DualInfeasible,
            x if x == ffi::OSQP_DUAL_INFEASIBLE_INACCURATE as ffi::c_int => {
                Self::DualInfeasibleInaccurate
            }
            x if x == ffi::OSQP_NON_CVX as ffi::c_int => Self::NonCvx,
            x if x == ffi::OSQP_SIGINT as ffi::c_int => Self::SigInt,
            x if x == ffi::OSQP_TIME_LIMIT_REACHED as ffi::c_int => Self::TimeLimitReached,
            x if x == ffi::OSQP_UNSOLVED as ffi::c_int => Self::Unsolved,
            other => Self::Other(other as i32),
        }
    }
    #[cfg(osqp_embedded)]
    fn from_embedded_raw(status: ffi::CokerOsqpSolveStatus) -> Self {
        match status {
            ffi::COKER_OSQP_SOLVE_SOLVED => Self::Solved,
            ffi::COKER_OSQP_SOLVE_SOLVED_INACCURATE => Self::SolvedInaccurate,
            ffi::COKER_OSQP_SOLVE_MAX_ITER_REACHED => Self::MaxIterReached,
            ffi::COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE => Self::PrimalInfeasible,
            ffi::COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE_INACCURATE => Self::PrimalInfeasibleInaccurate,
            ffi::COKER_OSQP_SOLVE_DUAL_INFEASIBLE => Self::DualInfeasible,
            ffi::COKER_OSQP_SOLVE_DUAL_INFEASIBLE_INACCURATE => Self::DualInfeasibleInaccurate,
            ffi::COKER_OSQP_SOLVE_NON_CONVEX => Self::NonCvx,
            ffi::COKER_OSQP_SOLVE_INTERRUPTED => Self::SigInt,
            ffi::COKER_OSQP_SOLVE_TIME_LIMIT_REACHED => Self::TimeLimitReached,
            ffi::COKER_OSQP_SOLVE_UNSOLVED => Self::Unsolved,
            other => Self::Other(other),
        }
    }
}

/// Caller-owned workspace sizes required to execute a mapped QP program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QpWorkspaceRequirements {
    pub evaluator_workspace_size: usize,
    pub tangent_workspace_size: usize,
    pub coefficient_output_size: usize,
    pub arena_bytes: usize,
    pub arena_alignment: usize,
}

/// Caller-owned primal workspace for evaluating QP coefficients.
#[derive(Debug)]
pub struct MappedQpWorkspace<'a> {
    pub evaluator_workspace: &'a mut [f32],
    pub coefficient_outputs: &'a mut [f32],
}

impl<'a> MappedQpWorkspace<'a> {
    /// Wraps caller-owned evaluator and coefficient buffers without allocating.
    pub fn new(evaluator_workspace: &'a mut [f32], coefficient_outputs: &'a mut [f32]) -> Self {
        Self {
            evaluator_workspace,
            coefficient_outputs,
        }
    }

    fn validate_for(&self, requirements: QpWorkspaceRequirements) -> Result<(), RuntimeError> {
        if self.evaluator_workspace.len() < requirements.evaluator_workspace_size {
            return Err(RuntimeError::WorkspaceTooSmall {
                expected: requirements.evaluator_workspace_size,
                actual: self.evaluator_workspace.len(),
            });
        }
        if self.coefficient_outputs.len() < requirements.coefficient_output_size {
            return Err(RuntimeError::WorkspaceTooSmall {
                expected: requirements.coefficient_output_size,
                actual: self.coefficient_outputs.len(),
            });
        }
        Ok(())
    }
}

/// Caller-owned primal and tangent workspaces for QP push-forward evaluation.
#[derive(Debug)]
pub struct MappedQpPushForwardWorkspace<'a> {
    pub primal: MappedQpWorkspace<'a>,
    pub tangent: MappedQpWorkspace<'a>,
    pub solution_tangent_workspace: &'a mut [f32],
}

impl<'a> MappedQpPushForwardWorkspace<'a> {
    /// Wraps caller-owned primal, tangent, and solution-tangent buffers.
    pub fn new(
        primal_evaluator_workspace: &'a mut [f32],
        primal_coefficient_outputs: &'a mut [f32],
        tangent_evaluator_workspace: &'a mut [f32],
        tangent_coefficient_outputs: &'a mut [f32],
        solution_tangent_workspace: &'a mut [f32],
    ) -> Self {
        Self {
            primal: MappedQpWorkspace::new(primal_evaluator_workspace, primal_coefficient_outputs),
            tangent: MappedQpWorkspace::new(
                tangent_evaluator_workspace,
                tangent_coefficient_outputs,
            ),
            solution_tangent_workspace,
        }
    }

    fn validate_for(&self, requirements: QpWorkspaceRequirements) -> Result<(), RuntimeError> {
        self.primal.validate_for(requirements)?;
        self.tangent.validate_for(requirements)?;
        if self.solution_tangent_workspace.len() < requirements.tangent_workspace_size {
            return Err(RuntimeError::WorkspaceTooSmall {
                expected: requirements.tangent_workspace_size,
                actual: self.solution_tangent_workspace.len(),
            });
        }
        Ok(())
    }
}

/// Lightweight diagnostics returned after a successful QP solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpSolveDiagnostics {
    pub status: QpSolveStatus,
    pub iterations: i32,
    pub primal_residual: f32,
    pub dual_residual: f32,
}

/// Borrowed view of a validated QP program inside a mapped bytecode module.
#[derive(Clone, Copy)]
pub struct MappedQpProgram<'a> {
    function_id: u16,
    qp_program: &'a ArchivedQpProgram,
    evaluator: MappedProgram<'a>,
    workspace_requirements: QpWorkspaceRequirements,
    n: usize,
    m: usize,
    p_nnz: usize,
    a_nnz: usize,
}

impl<'a> core::fmt::Debug for MappedQpProgram<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MappedQpProgram")
            .field("function_id", &self.function_id())
            .field("n", &self.n)
            .field("m", &self.m)
            .field("p_nnz", &self.p_nnz)
            .field("a_nnz", &self.a_nnz)
            .finish()
    }
}

/// Bound embedded QP program paired with a caller-provided OSQP arena.
#[cfg(osqp_embedded)]
pub struct BoundMappedQpProgram<'module, 'arena> {
    program: MappedQpProgram<'module>,
    arena: BoundQpArena<'arena>,
    instance: Option<ffi::CokerOsqpInstance>,
}

#[cfg(osqp_embedded)]
struct BoundQpArena<'a> {
    base: NonNull<u8>,
    bytes: usize,
    _borrowed: PhantomData<&'a mut [MaybeUninit<u8>]>,
}
/// Bound host QP program paired with a reusable host-side runtime workspace.
#[cfg(all(feature = "std", not(osqp_embedded)))]
pub struct BoundMappedQpProgram<'module, 'arena> {
    program: MappedQpProgram<'module>,
    runtime: QpRuntime<'module, 'arena>,
}

impl<'a> MappedModule<'a> {
    /// Returns the validated QP program for `function_id`.
    pub fn qp_program(&self, function_id: u16) -> Result<MappedQpProgram<'a>, RuntimeError> {
        MappedQpProgram::new(*self, function_id)
    }
}

impl<'a> MappedQpProgram<'a> {
    /// Maps a serialized bytecode module and returns its validated QP entry point.
    pub fn new_from_bytes(bytes: &'a [u8], function_id: u16) -> Result<Self, RuntimeError> {
        let module = MappedModule::from_archived_unchecked(archived_module(bytes)?);
        Self::new(module, function_id)
    }

    /// Binds a validated mapped module plus QP function id into a QP program view.
    pub fn new(module: MappedModule<'a>, function_id: u16) -> Result<Self, RuntimeError> {
        let bytecode_module = module.bytecode_module();
        bytecode_module.validate_semantics()?;
        let qp_program = bytecode_module.qp_program(function_id).ok_or_else(|| {
            RuntimeError::Validation(format!("missing QP function_id {function_id}"))
        })?;
        let evaluator = module.program(qp_program.coefficient_function_id())?;
        let (n, m, p_nnz, a_nnz, workspace_requirements) =
            validate_mapped_qp_program(qp_program, evaluator)?;
        Ok(Self {
            function_id,
            qp_program,
            evaluator,
            workspace_requirements,
            n,
            m,
            p_nnz,
            a_nnz,
        })
    }

    /// Returns the owning bytecode function id.
    pub fn function_id(&self) -> u16 {
        self.function_id
    }

    /// Returns static metadata for the mapped QP entry point.
    pub fn info(&self) -> QpProgramInfo<'_, ArchivedInputSpec, ArchivedOutputSpec> {
        qp_program_info_from_program(self.qp_program)
    }

    /// Returns caller-owned workspace requirements for primal, tangent, and arena storage.
    pub fn workspace_requirements(&self) -> QpWorkspaceRequirements {
        self.workspace_requirements
    }
    #[cfg(all(feature = "std", not(osqp_embedded)))]
    /// Computes the packed host workspace layout for this program.
    pub fn workspace_layout(&self) -> Result<QpWorkspaceLayout, RuntimeError> {
        QpWorkspaceLayout::from_validated_parts(
            self.workspace_requirements.evaluator_workspace_size,
            self.workspace_requirements.coefficient_output_size,
            self.p_nnz,
            self.a_nnz,
            self.n,
            self.m,
        )
    }
    fn validate_tangents(&self, tangents: &[&[f32]]) -> Result<(), RuntimeError> {
        self.validate_parameters(tangents)
    }

    fn validate_push_forward_outputs(
        &self,
        outputs: &[f32],
        tangent_outputs: &[f32],
    ) -> Result<(), RuntimeError> {
        self.validate_output_buffer(outputs)?;
        self.validate_output_buffer(tangent_outputs)
    }

    fn push_forward_unsupported_error() -> RuntimeError {
        RuntimeError::QpPushForwardUnsupported {
            reason: "differentiated KKT solve support is not implemented",
        }
    }

    fn validate_parameters(&self, parameters: &[&[f32]]) -> Result<(), RuntimeError> {
        let info = self.info();
        if parameters.len() != info.input_specs.len() {
            return Err(RuntimeError::InputCountMismatch {
                expected: info.input_specs.len(),
                actual: parameters.len(),
            });
        }
        for (index, (parameter, spec)) in parameters.iter().zip(info.input_specs.iter()).enumerate()
        {
            let expected = spec.length();
            let actual = parameter.len();
            if actual != expected {
                return Err(RuntimeError::InputSizeMismatch {
                    index,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn validate_output_buffer(&self, outputs: &[f32]) -> Result<(), RuntimeError> {
        let expected = self.info().output_spec.length();
        let actual = outputs.len();
        if actual != expected {
            return Err(RuntimeError::OutputBufferSizeMismatch { expected, actual });
        }
        Ok(())
    }
}
