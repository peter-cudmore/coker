use core::ffi::c_char;

use {c_float, c_int, OSQPTimer};

pub const QDLDL_SOLVER: linsys_solver_type = 0;
pub const MKL_PARDISO_SOLVER: linsys_solver_type = 1;
pub const UNKNOWN_SOLVER: linsys_solver_type = 99;
pub type linsys_solver_type = core::ffi::c_uint;

pub const OSQP_DUAL_INFEASIBLE_INACCURATE: ffi_osqp_solve_status = 4;
pub const OSQP_PRIMAL_INFEASIBLE_INACCURATE: ffi_osqp_solve_status = 3;
pub const OSQP_SOLVED_INACCURATE: ffi_osqp_solve_status = 2;
pub const OSQP_SOLVED: ffi_osqp_solve_status = 1;
pub const OSQP_MAX_ITER_REACHED: ffi_osqp_solve_status = -2;
pub const OSQP_PRIMAL_INFEASIBLE: ffi_osqp_solve_status = -3;
pub const OSQP_DUAL_INFEASIBLE: ffi_osqp_solve_status = -4;
pub const OSQP_SIGINT: ffi_osqp_solve_status = -5;
pub const OSQP_TIME_LIMIT_REACHED: ffi_osqp_solve_status = -6;
pub const OSQP_NON_CVX: ffi_osqp_solve_status = -7;
pub const OSQP_UNSOLVED: ffi_osqp_solve_status = -10;
pub type ffi_osqp_solve_status = core::ffi::c_int;

#[repr(C)]
pub struct OSQPVectorf {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OSQPMatrix {
    _private: [u8; 0],
}

#[repr(C)]
pub struct csc {
    pub nzmax: c_int,
    pub m: c_int,
    pub n: c_int,
    pub p: *mut c_int,
    pub i: *mut c_int,
    pub x: *mut c_float,
    pub nz: c_int,
}

#[repr(C)]
pub struct OSQPScaling {
    pub c: c_float,
    pub D: *mut c_float,
    pub E: *mut c_float,
    pub cinv: c_float,
    pub Dinv: *mut c_float,
    pub Einv: *mut c_float,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OSQPSolution {
    pub x: *mut c_float,
    pub y: *mut c_float,
}

#[repr(C)]
pub struct OSQPInfo {
    pub iter: c_int,
    pub status: [c_char; 32],
    pub status_val: c_int,
    pub obj_val: c_float,
    pub pri_res: c_float,
    pub dua_res: c_float,
    pub rho_updates: c_int,
    pub rho_estimate: c_float,
}

#[repr(C)]
pub struct OSQPData {
    pub n: c_int,
    pub m: c_int,
    pub P: *mut csc,
    pub A: *mut csc,
    pub q: *mut c_float,
    pub l: *mut c_float,
    pub u: *mut c_float,
}

#[repr(C)]
pub struct OSQPSettings {
    pub rho: c_float,
    pub sigma: c_float,
    pub scaling: c_int,
    pub adaptive_rho: c_int,
    pub adaptive_rho_interval: c_int,
    pub adaptive_rho_tolerance: c_float,
    pub max_iter: c_int,
    pub eps_abs: c_float,
    pub eps_rel: c_float,
    pub eps_prim_inf: c_float,
    pub eps_dual_inf: c_float,
    pub alpha: c_float,
    pub linsys_solver: linsys_solver_type,
    pub scaled_termination: c_int,
    pub check_termination: c_int,
    pub warm_start: c_int,
}

#[repr(C)]
pub struct OSQPWorkspace {
    pub data: *mut OSQPData,
    pub linsys_solver: *mut LinSysSolver,
    pub rho_vec: *mut c_float,
    pub rho_inv_vec: *mut c_float,
    pub constr_type: *mut c_int,
    pub x: *mut c_float,
    pub y: *mut c_float,
    pub z: *mut c_float,
    pub xz_tilde: *mut c_float,
    pub x_prev: *mut c_float,
    pub z_prev: *mut c_float,
    pub Ax: *mut c_float,
    pub Px: *mut c_float,
    pub Aty: *mut c_float,
    pub delta_y: *mut c_float,
    pub Atdelta_y: *mut c_float,
    pub delta_x: *mut c_float,
    pub Pdelta_x: *mut c_float,
    pub Adelta_x: *mut c_float,
    pub D_temp: *mut c_float,
    pub D_temp_A: *mut c_float,
    pub E_temp: *mut c_float,
    pub settings: *mut OSQPSettings,
    pub scaling: *mut OSQPScaling,
    pub solution: *mut OSQPSolution,
    pub info: *mut OSQPInfo,
}

#[repr(C)]
pub struct linsys_solver {
    pub type_: linsys_solver_type,
    pub solve: Option<unsafe extern "C" fn(*mut linsys_solver, *mut c_float) -> c_int>,
    pub update_matrices:
        Option<unsafe extern "C" fn(*mut linsys_solver, *const csc, *const csc) -> c_int>,
    pub update_rho_vec: Option<unsafe extern "C" fn(*mut linsys_solver, *const c_float) -> c_int>,
    pub nthreads: c_int,
}

pub type LinSysSolver = linsys_solver;

#[repr(C)]
pub struct qdldl {
    pub type_: linsys_solver_type,
    pub name: Option<unsafe extern "C" fn(*mut qdldl_solver) -> *const c_char>,
    pub solve: Option<unsafe extern "C" fn(*mut qdldl_solver, *mut OSQPVectorf, c_int) -> c_int>,
    pub update_settings: Option<unsafe extern "C" fn(*mut qdldl_solver, *const OSQPSettings)>,
    pub warm_start: Option<unsafe extern "C" fn(*mut qdldl_solver, *const OSQPVectorf)>,
    pub update_matrices: Option<
        unsafe extern "C" fn(
            *mut qdldl_solver,
            *const OSQPMatrix,
            *const c_int,
            c_int,
            *const OSQPMatrix,
            *const c_int,
            c_int,
        ) -> c_int,
    >,
    pub update_rho_vec: Option<unsafe extern "C" fn(*mut qdldl_solver, *const OSQPVectorf) -> c_int>,
    pub nthreads: c_int,
    pub L: *mut csc,
    pub Dinv: *mut QDLDL_float,
    pub P: *mut QDLDL_int,
    pub bp: *mut QDLDL_float,
    pub sol: *mut QDLDL_float,
    pub rho_inv_vec: *mut QDLDL_float,
    pub sigma: c_float,
    pub rho_inv: c_float,
    pub n: c_int,
    pub m: c_int,
    pub Pdiag_idx: *mut c_int,
    pub Pdiag_n: c_int,
    pub KKT: *mut csc,
    pub PtoKKT: *mut c_int,
    pub AtoKKT: *mut c_int,
    pub rhotoKKT: *mut c_int,
    pub D: *mut QDLDL_float,
    pub etree: *mut QDLDL_int,
    pub Lnz: *mut QDLDL_int,
    pub iwork: *mut QDLDL_int,
    pub bwork: *mut QDLDL_bool,
    pub fwork: *mut QDLDL_float,
}

pub type qdldl_solver = qdldl;
pub type QDLDL_float = c_float;
pub type QDLDL_int = c_int;
pub type QDLDL_bool = u8;

#[repr(C)]
pub struct CokerOsqpBufferRegion {
    pub ptr: *mut core::ffi::c_void,
    pub bytes: usize,
    pub alignment: usize,
}

#[repr(C)]
pub struct CokerOsqpLayoutRegion {
    pub bytes: usize,
    pub alignment: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpCscView {
    pub col_ptr: *const i32,
    pub row_idx: *const i32,
    pub nnz: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpProblemShape {
    pub n: i32,
    pub m: i32,
    pub p: CokerOsqpCscView,
    pub a: CokerOsqpCscView,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpNumericUpdate {
    pub p_x: *const f32,
    pub p_nnz: i32,
    pub a_x: *const f32,
    pub a_nnz: i32,
    pub q: *const f32,
    pub q_len: i32,
    pub l: *const f32,
    pub l_len: i32,
    pub u: *const f32,
    pub u_len: i32,
}

#[repr(C)]
pub struct CokerOsqpBuffers {
    pub pdata_p: CokerOsqpBufferRegion,
    pub pdata_i: CokerOsqpBufferRegion,
    pub pdata_x: CokerOsqpBufferRegion,
    pub pdata: CokerOsqpBufferRegion,
    pub adata_p: CokerOsqpBufferRegion,
    pub adata_i: CokerOsqpBufferRegion,
    pub adata_x: CokerOsqpBufferRegion,
    pub adata: CokerOsqpBufferRegion,
    pub qdata: CokerOsqpBufferRegion,
    pub ldata: CokerOsqpBufferRegion,
    pub udata: CokerOsqpBufferRegion,
    pub data: CokerOsqpBufferRegion,
    pub settings: CokerOsqpBufferRegion,
    pub scaling: CokerOsqpBufferRegion,
    pub xsolution: CokerOsqpBufferRegion,
    pub ysolution: CokerOsqpBufferRegion,
    pub solution: CokerOsqpBufferRegion,
    pub info: CokerOsqpBufferRegion,
    pub qdldl_L: CokerOsqpBufferRegion,
    pub qdldl_L_p: CokerOsqpBufferRegion,
    pub qdldl_L_i: CokerOsqpBufferRegion,
    pub qdldl_L_x: CokerOsqpBufferRegion,
    pub qdldl_KKT: CokerOsqpBufferRegion,
    pub qdldl_KKT_p: CokerOsqpBufferRegion,
    pub qdldl_KKT_i: CokerOsqpBufferRegion,
    pub qdldl_KKT_x: CokerOsqpBufferRegion,
    pub qdldl: CokerOsqpBufferRegion,
    pub qdldl_Dinv: CokerOsqpBufferRegion,
    pub qdldl_P: CokerOsqpBufferRegion,
    pub qdldl_bp: CokerOsqpBufferRegion,
    pub qdldl_sol: CokerOsqpBufferRegion,
    pub qdldl_rho_inv_vec: CokerOsqpBufferRegion,
    pub qdldl_Pdiag_idx: CokerOsqpBufferRegion,
    pub qdldl_PtoKKT: CokerOsqpBufferRegion,
    pub qdldl_AtoKKT: CokerOsqpBufferRegion,
    pub qdldl_rhotoKKT: CokerOsqpBufferRegion,
    pub qdldl_D: CokerOsqpBufferRegion,
    pub qdldl_etree: CokerOsqpBufferRegion,
    pub qdldl_Lnz: CokerOsqpBufferRegion,
    pub qdldl_iwork: CokerOsqpBufferRegion,
    pub qdldl_bwork: CokerOsqpBufferRegion,
    pub qdldl_fwork: CokerOsqpBufferRegion,
    pub work_rho_vec: CokerOsqpBufferRegion,
    pub work_rho_inv_vec: CokerOsqpBufferRegion,
    pub work_constr_type: CokerOsqpBufferRegion,
    pub work_x: CokerOsqpBufferRegion,
    pub work_y: CokerOsqpBufferRegion,
    pub work_z: CokerOsqpBufferRegion,
    pub work_xz_tilde: CokerOsqpBufferRegion,
    pub work_x_prev: CokerOsqpBufferRegion,
    pub work_z_prev: CokerOsqpBufferRegion,
    pub work_Ax: CokerOsqpBufferRegion,
    pub work_Px: CokerOsqpBufferRegion,
    pub work_Aty: CokerOsqpBufferRegion,
    pub work_delta_y: CokerOsqpBufferRegion,
    pub work_Atdelta_y: CokerOsqpBufferRegion,
    pub work_delta_x: CokerOsqpBufferRegion,
    pub work_Pdelta_x: CokerOsqpBufferRegion,
    pub work_Adelta_x: CokerOsqpBufferRegion,
    pub work_D_temp: CokerOsqpBufferRegion,
    pub work_D_temp_A: CokerOsqpBufferRegion,
    pub work_E_temp: CokerOsqpBufferRegion,
    pub workspace: CokerOsqpBufferRegion,
}

#[repr(C)]
pub struct CokerOsqpLayout {
    pub pdata_p: CokerOsqpLayoutRegion,
    pub pdata_i: CokerOsqpLayoutRegion,
    pub pdata_x: CokerOsqpLayoutRegion,
    pub pdata: CokerOsqpLayoutRegion,
    pub adata_p: CokerOsqpLayoutRegion,
    pub adata_i: CokerOsqpLayoutRegion,
    pub adata_x: CokerOsqpLayoutRegion,
    pub adata: CokerOsqpLayoutRegion,
    pub qdata: CokerOsqpLayoutRegion,
    pub ldata: CokerOsqpLayoutRegion,
    pub udata: CokerOsqpLayoutRegion,
    pub data: CokerOsqpLayoutRegion,
    pub settings: CokerOsqpLayoutRegion,
    pub scaling: CokerOsqpLayoutRegion,
    pub xsolution: CokerOsqpLayoutRegion,
    pub ysolution: CokerOsqpLayoutRegion,
    pub solution: CokerOsqpLayoutRegion,
    pub info: CokerOsqpLayoutRegion,
    pub qdldl_L: CokerOsqpLayoutRegion,
    pub qdldl_L_p: CokerOsqpLayoutRegion,
    pub qdldl_L_i: CokerOsqpLayoutRegion,
    pub qdldl_L_x: CokerOsqpLayoutRegion,
    pub qdldl_KKT: CokerOsqpLayoutRegion,
    pub qdldl_KKT_p: CokerOsqpLayoutRegion,
    pub qdldl_KKT_i: CokerOsqpLayoutRegion,
    pub qdldl_KKT_x: CokerOsqpLayoutRegion,
    pub qdldl: CokerOsqpLayoutRegion,
    pub qdldl_Dinv: CokerOsqpLayoutRegion,
    pub qdldl_P: CokerOsqpLayoutRegion,
    pub qdldl_bp: CokerOsqpLayoutRegion,
    pub qdldl_sol: CokerOsqpLayoutRegion,
    pub qdldl_rho_inv_vec: CokerOsqpLayoutRegion,
    pub qdldl_Pdiag_idx: CokerOsqpLayoutRegion,
    pub qdldl_PtoKKT: CokerOsqpLayoutRegion,
    pub qdldl_AtoKKT: CokerOsqpLayoutRegion,
    pub qdldl_rhotoKKT: CokerOsqpLayoutRegion,
    pub qdldl_D: CokerOsqpLayoutRegion,
    pub qdldl_etree: CokerOsqpLayoutRegion,
    pub qdldl_Lnz: CokerOsqpLayoutRegion,
    pub qdldl_iwork: CokerOsqpLayoutRegion,
    pub qdldl_bwork: CokerOsqpLayoutRegion,
    pub qdldl_fwork: CokerOsqpLayoutRegion,
    pub work_rho_vec: CokerOsqpLayoutRegion,
    pub work_rho_inv_vec: CokerOsqpLayoutRegion,
    pub work_constr_type: CokerOsqpLayoutRegion,
    pub work_x: CokerOsqpLayoutRegion,
    pub work_y: CokerOsqpLayoutRegion,
    pub work_z: CokerOsqpLayoutRegion,
    pub work_xz_tilde: CokerOsqpLayoutRegion,
    pub work_x_prev: CokerOsqpLayoutRegion,
    pub work_z_prev: CokerOsqpLayoutRegion,
    pub work_Ax: CokerOsqpLayoutRegion,
    pub work_Px: CokerOsqpLayoutRegion,
    pub work_Aty: CokerOsqpLayoutRegion,
    pub work_delta_y: CokerOsqpLayoutRegion,
    pub work_Atdelta_y: CokerOsqpLayoutRegion,
    pub work_delta_x: CokerOsqpLayoutRegion,
    pub work_Pdelta_x: CokerOsqpLayoutRegion,
    pub work_Adelta_x: CokerOsqpLayoutRegion,
    pub work_D_temp: CokerOsqpLayoutRegion,
    pub work_D_temp_A: CokerOsqpLayoutRegion,
    pub work_E_temp: CokerOsqpLayoutRegion,
    pub workspace: CokerOsqpLayoutRegion,
}
pub const COKER_OSQP_PLAN_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpIndexView {
    pub ptr: *const i32,
    pub len: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpArenaRegion {
    pub offset: usize,
    pub bytes: usize,
    pub alignment: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpArenaLayout {
    pub bytes: usize,
    pub alignment: usize,
    pub pdata_x: CokerOsqpArenaRegion,
    pub pdata: CokerOsqpArenaRegion,
    pub adata_x: CokerOsqpArenaRegion,
    pub adata: CokerOsqpArenaRegion,
    pub qdata: CokerOsqpArenaRegion,
    pub ldata: CokerOsqpArenaRegion,
    pub udata: CokerOsqpArenaRegion,
    pub data: CokerOsqpArenaRegion,
    pub settings: CokerOsqpArenaRegion,
    pub xsolution: CokerOsqpArenaRegion,
    pub ysolution: CokerOsqpArenaRegion,
    pub solution: CokerOsqpArenaRegion,
    pub info: CokerOsqpArenaRegion,
    pub qdldl_L_x: CokerOsqpArenaRegion,
    pub qdldl_L: CokerOsqpArenaRegion,
    pub qdldl_KKT_x: CokerOsqpArenaRegion,
    pub qdldl_KKT: CokerOsqpArenaRegion,
    pub qdldl: CokerOsqpArenaRegion,
    pub qdldl_Dinv: CokerOsqpArenaRegion,
    pub qdldl_bp: CokerOsqpArenaRegion,
    pub qdldl_sol: CokerOsqpArenaRegion,
    pub qdldl_rho_inv_vec: CokerOsqpArenaRegion,
    pub qdldl_D: CokerOsqpArenaRegion,
    pub qdldl_iwork: CokerOsqpArenaRegion,
    pub qdldl_bwork: CokerOsqpArenaRegion,
    pub qdldl_fwork: CokerOsqpArenaRegion,
    pub work_rho_vec: CokerOsqpArenaRegion,
    pub work_rho_inv_vec: CokerOsqpArenaRegion,
    pub work_constr_type: CokerOsqpArenaRegion,
    pub work_x: CokerOsqpArenaRegion,
    pub work_y: CokerOsqpArenaRegion,
    pub work_z: CokerOsqpArenaRegion,
    pub work_xz_tilde: CokerOsqpArenaRegion,
    pub work_x_prev: CokerOsqpArenaRegion,
    pub work_z_prev: CokerOsqpArenaRegion,
    pub work_Ax: CokerOsqpArenaRegion,
    pub work_Px: CokerOsqpArenaRegion,
    pub work_Aty: CokerOsqpArenaRegion,
    pub work_delta_y: CokerOsqpArenaRegion,
    pub work_Atdelta_y: CokerOsqpArenaRegion,
    pub work_delta_x: CokerOsqpArenaRegion,
    pub work_Pdelta_x: CokerOsqpArenaRegion,
    pub work_Adelta_x: CokerOsqpArenaRegion,
    pub workspace: CokerOsqpArenaRegion,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpArena {
    pub base: *mut core::ffi::c_void,
    pub bytes: usize,
    pub alignment: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpSettings {
    pub rho: f32,
    pub sigma: f32,
    pub scaling: i32,
    pub adaptive_rho: i32,
    pub adaptive_rho_interval: i32,
    pub adaptive_rho_tolerance: f32,
    pub max_iter: i32,
    pub eps_abs: f32,
    pub eps_rel: f32,
    pub eps_prim_inf: f32,
    pub eps_dual_inf: f32,
    pub alpha: f32,
    pub linsys_solver: u32,
    pub scaled_termination: i32,
    pub check_termination: i32,
    pub warm_start: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpPlan {
    pub abi_version: u32,
    pub n: i32,
    pub m: i32,
    pub n_plus_m: i32,
    pub p: CokerOsqpCscView,
    pub a: CokerOsqpCscView,
    pub kkt: CokerOsqpCscView,
    pub qdldl_l: CokerOsqpCscView,
    pub p_to_kkt: CokerOsqpIndexView,
    pub a_to_kkt: CokerOsqpIndexView,
    pub rho_to_kkt: CokerOsqpIndexView,
    pub p_diagonal_idx: CokerOsqpIndexView,
    pub permutation: CokerOsqpIndexView,
    pub qdldl_etree: CokerOsqpIndexView,
    pub qdldl_lnz: CokerOsqpIndexView,
    pub settings: CokerOsqpSettings,
    pub arena_layout: CokerOsqpArenaLayout,
}


#[repr(C)]
pub struct CokerOsqpInstance {
    pub pdata: *mut csc,
    pub adata: *mut csc,
    pub data: *mut OSQPData,
    pub settings: *mut OSQPSettings,
    pub scaling: *mut OSQPScaling,
    pub solution: *mut OSQPSolution,
    pub info: *mut OSQPInfo,
    pub linsys_solver: *mut LinSysSolver,
    pub qdldl: *mut qdldl_solver,
    pub workspace: *mut OSQPWorkspace,
}

pub type CokerOsqpStatus = core::ffi::c_int;
pub const COKER_OSQP_OK: CokerOsqpStatus = 0;
pub const COKER_OSQP_INVALID_ARGUMENT: CokerOsqpStatus = -1;
pub const COKER_OSQP_INVALID_SHAPE: CokerOsqpStatus = -2;
pub const COKER_OSQP_LAYOUT_MISMATCH: CokerOsqpStatus = -3;
pub const COKER_OSQP_INVALID_NUMERIC_UPDATE: CokerOsqpStatus = -4;
pub const COKER_OSQP_NOT_BOUND: CokerOsqpStatus = -5;
pub const COKER_OSQP_UNSUPPORTED: CokerOsqpStatus = -6;

pub type CokerOsqpSolveStatus = core::ffi::c_int;
pub const COKER_OSQP_SOLVE_UNSOLVED: CokerOsqpSolveStatus = 0;
pub const COKER_OSQP_SOLVE_SOLVED: CokerOsqpSolveStatus = 1;
pub const COKER_OSQP_SOLVE_SOLVED_INACCURATE: CokerOsqpSolveStatus = 2;
pub const COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE_INACCURATE: CokerOsqpSolveStatus = 3;
pub const COKER_OSQP_SOLVE_DUAL_INFEASIBLE_INACCURATE: CokerOsqpSolveStatus = 4;
pub const COKER_OSQP_SOLVE_MAX_ITER_REACHED: CokerOsqpSolveStatus = -2;
pub const COKER_OSQP_SOLVE_PRIMAL_INFEASIBLE: CokerOsqpSolveStatus = -3;
pub const COKER_OSQP_SOLVE_DUAL_INFEASIBLE: CokerOsqpSolveStatus = -4;
pub const COKER_OSQP_SOLVE_INTERRUPTED: CokerOsqpSolveStatus = -5;
pub const COKER_OSQP_SOLVE_TIME_LIMIT_REACHED: CokerOsqpSolveStatus = -6;
pub const COKER_OSQP_SOLVE_NON_CONVEX: CokerOsqpSolveStatus = -7;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CokerOsqpSolution {
    pub primal: *const f32,
    pub primal_len: i32,
    pub dual: *const f32,
    pub dual_len: i32,
    pub status: CokerOsqpSolveStatus,
    pub iterations: i32,
    pub primal_residual: f32,
    pub dual_residual: f32,
}

extern "C" {
    pub fn coker_osqp_layout_for_shape(
        shape: *const CokerOsqpProblemShape,
        layout: *mut CokerOsqpLayout,
    ) -> CokerOsqpStatus;

    pub fn coker_osqp_bind(
        shape: *const CokerOsqpProblemShape,
        layout: *const CokerOsqpLayout,
        buffers: *const CokerOsqpBuffers,
        instance: *mut CokerOsqpInstance,
    ) -> CokerOsqpStatus;
    pub fn coker_osqp_bind_plan(
        plan: *const CokerOsqpPlan,
        arena: CokerOsqpArena,
        instance: *mut CokerOsqpInstance,
    ) -> CokerOsqpStatus;


    pub fn coker_osqp_update(
        instance: *mut CokerOsqpInstance,
        update: *const CokerOsqpNumericUpdate,
    ) -> CokerOsqpStatus;
    pub fn coker_osqp_solve(
        instance: *mut CokerOsqpInstance,
        solve_status: *mut CokerOsqpSolveStatus,
    ) -> CokerOsqpStatus;
    pub fn coker_osqp_solution(
        instance: *const CokerOsqpInstance,
        solution: *mut CokerOsqpSolution,
    ) -> CokerOsqpStatus;
}
