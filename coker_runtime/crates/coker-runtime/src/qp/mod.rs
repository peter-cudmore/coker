#![cfg_attr(not(any(feature = "std", osqp_embedded)), allow(dead_code))]

#[cfg(any(osqp_embedded, feature = "std"))]
use core::{convert::TryFrom, marker::PhantomData, mem::MaybeUninit, ptr::NonNull, slice};

#[cfg(all(feature = "std", not(osqp_embedded)))]
use coker_bytecode::ArchivedQpOutputSlice;
use coker_bytecode::{
    archived_module, ArchivedInputSpec, ArchivedOutputSpec, ArchivedQpCoefficientOutputs,
    ArchivedQpProgram,
};
use coker_osqp_ffi::{self as ffi, raw_embedded as raw};

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

#[cfg(all(feature = "std", not(osqp_embedded)))]
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

#[cfg(osqp_embedded)]
type EmbeddedNumericSlices<'a> = (
    &'a mut [f32],
    &'a mut [f32],
    &'a mut [f32],
    &'a mut [f32],
    &'a mut [f32],
);

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
    instance: Option<EmbeddedOsqpInstance>,
}

#[cfg(osqp_embedded)]
struct BoundQpArena<'a> {
    base: NonNull<u8>,
    bytes: usize,
    _borrowed: PhantomData<&'a mut [MaybeUninit<u8>]>,
}
/// Detached prepared QP solver state whose arena lifetime is managed externally.
///
/// # Safety
///
/// The caller must keep the arena passed to [`MappedQpProgram::prepare_detached`] alive,
/// writable, and at a stable address for the full lifetime of this struct.
#[cfg(osqp_embedded)]
pub struct PreparedQpProgram {
    function_id: u16,
    instance: Option<EmbeddedOsqpInstance>,
    arena_base: NonNull<u8>,
    arena_bytes: usize,
}

#[cfg(osqp_embedded)]
#[derive(Debug)]
struct EmbeddedOsqpInstance {
    solver: raw::OSQPSolver,
    data: raw::OSQPData,
    settings: raw::OSQPSettings,
    solution: raw::OSQPSolution,
    info: raw::OSQPInfo,
    workspace: raw::OSQPWorkspace,
    pdata_csc: raw::OSQPCscMatrix,
    adata_csc: raw::OSQPCscMatrix,
    qdldl_l_csc: raw::OSQPCscMatrix,
    qdldl_kkt_csc: raw::OSQPCscMatrix,
    pdata_matrix: raw::OSQPMatrix,
    adata_matrix: raw::OSQPMatrix,
    q_vector: raw::OSQPVectorf,
    l_vector: raw::OSQPVectorf,
    u_vector: raw::OSQPVectorf,
    rho_vec: raw::OSQPVectorf,
    rho_inv_vec: raw::OSQPVectorf,
    constr_type: raw::OSQPVectori,
    x: raw::OSQPVectorf,
    y: raw::OSQPVectorf,
    z: raw::OSQPVectorf,
    xz_tilde: raw::OSQPVectorf,
    xtilde_view: raw::OSQPVectorf,
    ztilde_view: raw::OSQPVectorf,
    x_prev: raw::OSQPVectorf,
    z_prev: raw::OSQPVectorf,
    ax: raw::OSQPVectorf,
    px: raw::OSQPVectorf,
    aty: raw::OSQPVectorf,
    delta_y: raw::OSQPVectorf,
    atdelta_y: raw::OSQPVectorf,
    delta_x: raw::OSQPVectorf,
    pdelta_x: raw::OSQPVectorf,
    adelta_x: raw::OSQPVectorf,
    qdldl: raw::qdldl,
}

#[cfg(osqp_embedded)]
impl EmbeddedOsqpInstance {
    fn solver(&mut self) -> ffi::embedded_bind::EmbeddedSolver {
        unsafe { ffi::embedded_bind::EmbeddedSolver::from_ptr(&mut self.solver).unwrap() }
    }

    fn numeric_slices_mut(
        &mut self,
        p_nnz: usize,
        a_nnz: usize,
        n: usize,
        m: usize,
    ) -> Result<EmbeddedNumericSlices<'_>, RuntimeError> {
        let p = &mut self.pdata_csc;
        let a = &mut self.adata_csc;
        if (p_nnz != 0 && p.x.is_null())
            || (a_nnz != 0 && a.x.is_null())
            || (n != 0 && (self.q_vector.values.is_null()))
            || (m != 0 && (self.l_vector.values.is_null() || self.u_vector.values.is_null()))
        {
            return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
        }
        Ok(unsafe {
            (
                slice::from_raw_parts_mut(p.x, p_nnz),
                slice::from_raw_parts_mut(a.x, a_nnz),
                slice::from_raw_parts_mut(self.q_vector.values, n),
                slice::from_raw_parts_mut(self.l_vector.values, m),
                slice::from_raw_parts_mut(self.u_vector.values, m),
            )
        })
    }

    fn apply_settings(
        &mut self,
        settings: &coker_bytecode::ArchivedEmbeddedOsqpSettings,
    ) -> Result<(), RuntimeError> {
        unsafe { raw::osqp_set_default_settings(&mut self.settings) };
        self.settings.device = 0;
        self.settings.allocate_solution = 0;
        self.settings.verbose = 0;
        self.settings.profiler_level = 0;
        self.settings.polishing = 0;
        self.settings.rho = checked_f32_setting(settings.rho.into(), "QP rho")?;
        self.settings.sigma = checked_f32_setting(settings.sigma.into(), "QP sigma")?;
        self.settings.alpha = checked_f32_setting(settings.alpha.into(), "QP alpha")?;
        self.settings.adaptive_rho = if settings.adaptive_rho { 1 } else { 0 };
        self.settings.adaptive_rho_interval =
            i32::try_from(settings.adaptive_rho_interval.to_native())
                .map_err(|_| RuntimeError::EmbeddedQpWorkspaceInvalid)?;
        self.settings.adaptive_rho_tolerance = checked_f32_setting(
            settings.adaptive_rho_tolerance.into(),
            "QP adaptive_rho_tolerance",
        )?;
        self.settings.max_iter = i32::try_from(settings.max_iter.to_native())
            .map_err(|_| RuntimeError::EmbeddedQpWorkspaceInvalid)?;
        self.settings.eps_abs = checked_f32_setting(settings.eps_abs.into(), "QP eps_abs")?;
        self.settings.eps_rel = checked_f32_setting(settings.eps_rel.into(), "QP eps_rel")?;
        self.settings.eps_prim_inf =
            checked_f32_setting(settings.eps_prim_inf.into(), "QP eps_prim_inf")?;
        self.settings.eps_dual_inf =
            checked_f32_setting(settings.eps_dual_inf.into(), "QP eps_dual_inf")?;
        self.settings.scaling = i32::try_from(settings.scaling.to_native())
            .map_err(|_| RuntimeError::EmbeddedQpWorkspaceInvalid)?;
        self.settings.scaled_termination = if settings.scaled_termination { 1 } else { 0 };
        self.settings.check_termination = i32::try_from(settings.check_termination.to_native())
            .map_err(|_| RuntimeError::EmbeddedQpWorkspaceInvalid)?;
        self.settings.warm_starting = if settings.warm_start { 1 } else { 0 };
        self.settings.linsys_solver = raw::osqp_linsys_solver_type_OSQP_DIRECT_SOLVER;
        self.settings.rho_is_vec = 1;
        Ok(())
    }
    fn refresh_self_pointers(&mut self) {
        ffi::embedded_bind::bind_matrix(
            &mut self.pdata_matrix,
            &mut self.pdata_csc,
            raw::OSQPMatrix_symmetry_type_TRIU,
        );
        ffi::embedded_bind::bind_matrix(
            &mut self.adata_matrix,
            &mut self.adata_csc,
            raw::OSQPMatrix_symmetry_type_NONE,
        );
        self.data.P = &mut self.pdata_matrix;
        self.data.A = &mut self.adata_matrix;
        self.data.q = &mut self.q_vector;
        self.data.l = &mut self.l_vector;
        self.data.u = &mut self.u_vector;
        self.qdldl.L = &mut self.qdldl_l_csc;
        self.qdldl.KKT = &mut self.qdldl_kkt_csc;
        self.qdldl.sigma = self.settings.sigma;
        self.workspace.data = &mut self.data;
        self.workspace.linsys_solver = (&mut self.qdldl as *mut raw::qdldl).cast();
        self.workspace.rho_vec = &mut self.rho_vec;
        self.workspace.rho_inv_vec = &mut self.rho_inv_vec;
        self.workspace.constr_type = &mut self.constr_type;
        self.workspace.x = &mut self.x;
        self.workspace.y = &mut self.y;
        self.workspace.z = &mut self.z;
        self.workspace.xz_tilde = &mut self.xz_tilde;
        self.workspace.xtilde_view = &mut self.xtilde_view;
        self.workspace.ztilde_view = &mut self.ztilde_view;
        self.workspace.x_prev = &mut self.x_prev;
        self.workspace.z_prev = &mut self.z_prev;
        self.workspace.Ax = &mut self.ax;
        self.workspace.Px = &mut self.px;
        self.workspace.Aty = &mut self.aty;
        self.workspace.delta_y = &mut self.delta_y;
        self.workspace.Atdelta_y = &mut self.atdelta_y;
        self.workspace.delta_x = &mut self.delta_x;
        self.workspace.Pdelta_x = &mut self.pdelta_x;
        self.workspace.Adelta_x = &mut self.adelta_x;
        self.solver.settings = &mut self.settings;
        self.solver.solution = &mut self.solution;
        self.solution.prim_inf_cert = self.delta_y.values;
        self.solution.dual_inf_cert = self.delta_x.values;
        self.solver.info = &mut self.info;
        self.solver.work = &mut self.workspace;
    }

    fn bind(
        qp_program: &ArchivedQpProgram,
        arena: &BoundQpArena<'_>,
    ) -> Result<Self, RuntimeError> {
        let plan = qp_program.embedded_plan();
        let layout = plan.arena_layout();
        let qdldl_plan = plan.qdldl_plan();
        let symbolic_l = qdldl_plan.symbolic_l();
        let n = checked_embedded_ffi_length(qp_program.p_pattern().ncols.to_native() as usize)?;
        let m = checked_embedded_ffi_length(qp_program.a_pattern().nrows.to_native() as usize)?;
        let n_plus_m = n
            .checked_add(m)
            .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?;
        let (p_indptr, p_indices, p_nnz) = mapped_osqp_csc_ptrs(qp_program.p_pattern(), "QP P")?;
        let (a_indptr, a_indices, a_nnz) = mapped_osqp_csc_ptrs(qp_program.a_pattern(), "QP A")?;
        let (kkt_indptr, kkt_indices, kkt_nnz) =
            mapped_osqp_csc_ptrs(qdldl_plan.kkt_pattern(), "QP KKT")?;
        let l_nnz = checked_embedded_ffi_length(symbolic_l.l_pattern().indices.len())?;
        let storage = unsafe {
            slice::from_raw_parts_mut(arena.base.as_ptr().cast::<MaybeUninit<u8>>(), arena.bytes)
        };
        let mut embedded_arena = ffi::embedded_bind::EmbeddedArena::new(storage);
        if !embedded_arena.zero_region(0, arena.bytes) {
            return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
        }
        macro_rules! region_ptr {
            ($ty:ty, $region:expr) => {{
                embedded_arena
                    .region_ptr::<$ty>(
                        checked_embedded_usize(($region).byte_offset(), "QP arena offset")?,
                        checked_embedded_usize(($region).byte_len(), "QP arena length")?,
                        checked_embedded_usize(($region).byte_alignment(), "QP arena alignment")?,
                    )
                    .map(NonNull::as_ptr)
                    .ok_or(RuntimeError::Validation(concat!(
                        "embedded OSQP arena region is invalid for ",
                        stringify!($region)
                    )))?
            }};
        }
        let mut instance: Self = unsafe { core::mem::zeroed() };
        ffi::embedded_bind::bind_csc_matrix(
            &mut instance.pdata_csc,
            n,
            n,
            p_nnz,
            p_indptr,
            p_indices,
            region_ptr!(f32, layout.pdata_x()),
        );
        ffi::embedded_bind::bind_csc_matrix(
            &mut instance.adata_csc,
            m,
            n,
            a_nnz,
            a_indptr,
            a_indices,
            region_ptr!(f32, layout.adata_x()),
        );
        let l_indptr = region_ptr!(i32, layout.qdldl_l_p());
        let l_indices = region_ptr!(i32, layout.qdldl_l_i());
        unsafe {
            let destination = slice::from_raw_parts_mut(l_indptr, n_plus_m as usize + 1);
            for (destination, source) in destination
                .iter_mut()
                .zip(symbolic_l.l_pattern().indptr.iter())
            {
                *destination = checked_embedded_ffi_length(source.to_native() as usize)?;
            }
            let destination = slice::from_raw_parts_mut(l_indices, l_nnz as usize);
            for (destination, source) in destination
                .iter_mut()
                .zip(symbolic_l.l_pattern().indices.iter())
            {
                *destination = checked_embedded_ffi_length(source.to_native() as usize)?;
            }
        }
        ffi::embedded_bind::bind_csc_matrix(
            &mut instance.qdldl_l_csc,
            n_plus_m,
            n_plus_m,
            l_nnz,
            l_indptr,
            l_indices,
            region_ptr!(f32, layout.qdldl_l_x()),
        );
        ffi::embedded_bind::bind_csc_matrix(
            &mut instance.qdldl_kkt_csc,
            n_plus_m,
            n_plus_m,
            kkt_nnz,
            kkt_indptr,
            kkt_indices,
            region_ptr!(f32, layout.qdldl_kkt_x()),
        );
        ffi::embedded_bind::bind_matrix(
            &mut instance.pdata_matrix,
            &mut instance.pdata_csc,
            raw::OSQPMatrix_symmetry_type_TRIU,
        );
        ffi::embedded_bind::bind_matrix(
            &mut instance.adata_matrix,
            &mut instance.adata_csc,
            raw::OSQPMatrix_symmetry_type_NONE,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.q_vector,
            region_ptr!(f32, layout.qdata()),
            n,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.l_vector,
            region_ptr!(f32, layout.ldata()),
            m,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.u_vector,
            region_ptr!(f32, layout.udata()),
            m,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.rho_vec,
            region_ptr!(f32, layout.work_rho_vec()),
            m,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.rho_inv_vec,
            region_ptr!(f32, layout.work_rho_inv_vec()),
            m,
        );
        ffi::embedded_bind::bind_vectori(
            &mut instance.constr_type,
            region_ptr!(i32, layout.work_constr_type()),
            m,
        );
        ffi::embedded_bind::bind_vectorf(&mut instance.x, region_ptr!(f32, layout.work_x()), n);
        ffi::embedded_bind::bind_vectorf(&mut instance.y, region_ptr!(f32, layout.work_y()), m);
        ffi::embedded_bind::bind_vectorf(&mut instance.z, region_ptr!(f32, layout.work_z()), m);
        ffi::embedded_bind::bind_vectorf(
            &mut instance.xz_tilde,
            region_ptr!(f32, layout.work_xz_tilde()),
            n_plus_m,
        );
        ffi::embedded_bind::bind_vectorf(&mut instance.xtilde_view, instance.xz_tilde.values, n);
        ffi::embedded_bind::bind_vectorf(
            &mut instance.ztilde_view,
            unsafe { instance.xz_tilde.values.add(n as usize) },
            m,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.x_prev,
            region_ptr!(f32, layout.work_x_prev()),
            n,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.z_prev,
            region_ptr!(f32, layout.work_z_prev()),
            m,
        );
        ffi::embedded_bind::bind_vectorf(&mut instance.ax, region_ptr!(f32, layout.work_ax()), m);
        ffi::embedded_bind::bind_vectorf(&mut instance.px, region_ptr!(f32, layout.work_px()), n);
        ffi::embedded_bind::bind_vectorf(&mut instance.aty, region_ptr!(f32, layout.work_aty()), n);
        ffi::embedded_bind::bind_vectorf(
            &mut instance.delta_y,
            region_ptr!(f32, layout.work_delta_y()),
            m,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.atdelta_y,
            region_ptr!(f32, layout.work_atdelta_y()),
            n,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.delta_x,
            region_ptr!(f32, layout.work_delta_x()),
            n,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.pdelta_x,
            region_ptr!(f32, layout.work_pdelta_x()),
            n,
        );
        ffi::embedded_bind::bind_vectorf(
            &mut instance.adelta_x,
            region_ptr!(f32, layout.work_adelta_x()),
            m,
        );
        instance.data.n = n;
        instance.data.m = m;
        instance.data.P = &mut instance.pdata_matrix;
        instance.data.A = &mut instance.adata_matrix;
        instance.data.q = &mut instance.q_vector;
        instance.data.l = &mut instance.l_vector;
        instance.data.u = &mut instance.u_vector;
        instance.apply_settings(plan.settings())?;
        instance.solution.x = region_ptr!(f32, layout.xsolution());
        instance.solution.y = region_ptr!(f32, layout.ysolution());
        instance.qdldl.type_ = raw::osqp_linsys_solver_type_OSQP_DIRECT_SOLVER;
        instance.qdldl.name = Some(raw::name_qdldl);
        instance.qdldl.solve = Some(raw::solve_linsys_qdldl);
        instance.qdldl.update_settings = Some(raw::update_settings_linsys_solver_qdldl);
        instance.qdldl.warm_start = Some(raw::warm_start_linsys_solver_qdldl);
        instance.qdldl.update_matrices = Some(raw::update_linsys_solver_matrices_qdldl);
        instance.qdldl.update_rho_vec = Some(raw::update_linsys_solver_rho_vec_qdldl);
        instance.qdldl.nthreads = 1;
        instance.qdldl.L = &mut instance.qdldl_l_csc;
        instance.qdldl.Dinv = region_ptr!(f32, layout.qdldl_dinv());
        instance.qdldl.P = qdldl_plan.kkt_permutation.as_ptr().cast::<i32>().cast_mut();
        instance.qdldl.bp = region_ptr!(f32, layout.qdldl_bp());
        instance.qdldl.sol = region_ptr!(f32, layout.qdldl_sol());
        instance.qdldl.rho_inv_vec = region_ptr!(f32, layout.qdldl_rho_inv_vec());
        instance.qdldl.sigma = instance.settings.sigma;
        instance.qdldl.rho_inv = 0.0;
        instance.qdldl.n = n;
        instance.qdldl.m = m;
        instance.qdldl.KKT = &mut instance.qdldl_kkt_csc;
        instance.qdldl.PtoKKT = qdldl_plan.p_to_kkt.as_ptr().cast::<i32>().cast_mut();
        instance.qdldl.AtoKKT = qdldl_plan.a_to_kkt.as_ptr().cast::<i32>().cast_mut();
        instance.qdldl.rhotoKKT = qdldl_plan.rho_to_kkt.as_ptr().cast::<i32>().cast_mut();
        instance.qdldl.D = region_ptr!(f32, layout.qdldl_d());
        instance.qdldl.etree = symbolic_l.etree().as_ptr().cast::<i32>().cast_mut();
        instance.qdldl.Lnz = symbolic_l.lnz().as_ptr().cast::<i32>().cast_mut();
        instance.qdldl.iwork = region_ptr!(i32, layout.qdldl_iwork());
        instance.qdldl.bwork = region_ptr!(u8, layout.qdldl_bwork());
        instance.qdldl.fwork = region_ptr!(f32, layout.qdldl_fwork());
        instance.qdldl.adj = core::ptr::null_mut();
        instance.workspace.data = &mut instance.data;
        instance.workspace.linsys_solver = (&mut instance.qdldl as *mut raw::qdldl).cast();
        instance.workspace.rho_vec = &mut instance.rho_vec;
        instance.workspace.rho_inv_vec = &mut instance.rho_inv_vec;
        instance.workspace.constr_type = &mut instance.constr_type;
        instance.workspace.x = &mut instance.x;
        instance.workspace.y = &mut instance.y;
        instance.workspace.z = &mut instance.z;
        instance.workspace.xz_tilde = &mut instance.xz_tilde;
        instance.workspace.xtilde_view = &mut instance.xtilde_view;
        instance.workspace.ztilde_view = &mut instance.ztilde_view;
        instance.workspace.x_prev = &mut instance.x_prev;
        instance.workspace.z_prev = &mut instance.z_prev;
        instance.workspace.Ax = &mut instance.ax;
        instance.workspace.Px = &mut instance.px;
        instance.workspace.Aty = &mut instance.aty;
        instance.workspace.delta_y = &mut instance.delta_y;
        instance.workspace.Atdelta_y = &mut instance.atdelta_y;
        instance.workspace.delta_x = &mut instance.delta_x;
        instance.workspace.Pdelta_x = &mut instance.pdelta_x;
        instance.workspace.Adelta_x = &mut instance.adelta_x;
        instance.workspace.D_temp = core::ptr::null_mut();
        instance.workspace.D_temp_A = core::ptr::null_mut();
        instance.workspace.E_temp = core::ptr::null_mut();
        instance.solution.prim_inf_cert = instance.delta_y.values;
        instance.solution.dual_inf_cert = instance.delta_x.values;
        instance.workspace.scaled_dual_res = 0.0;
        instance.workspace.rho_inv = 0.0;
        instance.workspace.rho_updated = 0;
        instance.workspace.last_rel_kkt = 0.0;
        instance.solver.settings = &mut instance.settings;
        instance.solver.solution = &mut instance.solution;
        instance.solver.info = &mut instance.info;
        instance.solver.work = &mut instance.workspace;
        instance.reset_solution_state();
        unsafe {
            raw::set_rho_vec(&mut instance.solver);
            if m != 0 {
                let rho_inv = slice::from_raw_parts(instance.rho_inv_vec.values, m as usize);
                let qdldl_rho_inv =
                    slice::from_raw_parts_mut(instance.qdldl.rho_inv_vec, m as usize);
                qdldl_rho_inv.copy_from_slice(rho_inv);
                raw::update_KKT_param2(
                    instance.qdldl.KKT,
                    instance.qdldl.rho_inv_vec,
                    instance.qdldl.rho_inv,
                    instance.qdldl.rhotoKKT,
                    m,
                );
            }
            let update_status = raw::update_linsys_solver_matrices_qdldl(
                &mut instance.qdldl,
                &instance.pdata_matrix,
                core::ptr::null(),
                p_nnz,
                &instance.adata_matrix,
                core::ptr::null(),
                a_nnz,
            );
            if update_status != 0 {
                return Err(RuntimeError::Validation(
                    "embedded QDLDL factorization failed during Rust binder setup",
                ));
            }
        }
        Ok(instance)
    }

    fn reset_solution_state(&mut self) {
        self.solution.prim_inf_cert = self.delta_y.values;
        self.solution.dual_inf_cert = self.delta_x.values;
        unsafe { raw::reset_info(&mut self.info) };
        self.info.rho_estimate = self.settings.rho;
    }
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
        MappedQpProgram::from_validated_module(*self, function_id)
    }
}

impl<'a> MappedQpProgram<'a> {
    /// Maps a serialized bytecode module and returns its validated QP entry point.
    pub fn new_from_bytes(bytes: &'a [u8], function_id: u16) -> Result<Self, RuntimeError> {
        let module = MappedModule::from_archived_unchecked(archived_module(bytes)?);
        Self::from_validated_module(module, function_id)
    }

    /// Binds a validated mapped module plus QP function id into a QP program view.
    pub fn new(module: MappedModule<'a>, function_id: u16) -> Result<Self, RuntimeError> {
        Self::from_validated_module(module, function_id)
    }

    fn from_validated_module(
        module: MappedModule<'a>,
        function_id: u16,
    ) -> Result<Self, RuntimeError> {
        let bytecode_module = module.bytecode_module();
        let qp_program = bytecode_module
            .qp_program(function_id)
            .ok_or(RuntimeError::MissingQpFunction { function_id })?;
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
