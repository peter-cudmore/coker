use coker_bytecode::Program;
use serde::Deserialize;
use serde_json::Value;
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedModule {
    pub(crate) functions: Vec<ExportedFunction>,
    #[serde(default)]
    pub(crate) qp_programs: Vec<ExportedQpProgram>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedFunction {
    pub(crate) function_id: u32,
    pub(crate) program: ExportedProgram,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedProgram {
    pub(crate) workspace: ExportedMemorySpec,
    pub(crate) input_layer: ExportedInputLayer,
    pub(crate) output_layer: ExportedOutputLayer,
    pub(crate) intermediate_layers: Vec<ExportedLayer>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQpProgram {
    pub(crate) function_id: u32,
    pub(crate) coefficient_function_id: u32,
    pub(crate) required_primal_workspace_size: u64,
    pub(crate) required_tangent_workspace_size: u64,
    pub(crate) input_specs: Vec<ExportedInputSpec>,
    pub(crate) output_spec: ExportedOutputSpec,
    pub(crate) p_pattern: ExportedEmbeddedCscPattern,
    pub(crate) a_pattern: ExportedEmbeddedCscPattern,
    pub(crate) coefficient_outputs: ExportedQpCoefficientOutputs,
    pub(crate) embedded_plan: ExportedQpProgramPlan,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQpCoefficientOutputs {
    pub(crate) px: ExportedQpOutput,
    pub(crate) q: ExportedQpOutput,
    pub(crate) ax: ExportedQpOutput,
    pub(crate) l: ExportedQpOutput,
    pub(crate) u: ExportedQpOutput,
    pub(crate) r: ExportedQpOutput,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQpOutput {
    pub(crate) start: usize,
    pub(crate) length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(crate) enum ExportedEmbeddedQpProfile {
    #[serde(rename = "Osqp063Embedded2Qdldl")]
    Osqp063Embedded2Qdldl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(crate) enum ExportedEmbeddedLinsysSolver {
    Qdldl,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQpProgramPlan {
    pub(crate) abi_version: u64,
    pub(crate) profile: ExportedEmbeddedQpProfile,
    pub(crate) version: u64,
    pub(crate) settings: ExportedEmbeddedOsqpSettings,
    pub(crate) arena_layout: ExportedQpProgramArenaLayout,
    pub(crate) qdldl_plan: ExportedQpProgramQdldlPlan,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedEmbeddedOsqpSettings {
    pub(crate) rho: f64,
    pub(crate) sigma: f64,
    pub(crate) alpha: f64,
    pub(crate) adaptive_rho: bool,
    pub(crate) adaptive_rho_interval: u64,
    pub(crate) adaptive_rho_tolerance: f64,
    pub(crate) max_iter: u64,
    pub(crate) eps_abs: f64,
    pub(crate) eps_rel: f64,
    pub(crate) eps_prim_inf: f64,
    pub(crate) eps_dual_inf: f64,
    pub(crate) scaling: u64,
    pub(crate) scaled_termination: bool,
    pub(crate) check_termination: u64,
    pub(crate) warm_start: bool,
    pub(crate) linsys_solver: ExportedEmbeddedLinsysSolver,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQpProgramQdldlPlan {
    pub(crate) p_pattern: ExportedEmbeddedCscPattern,
    pub(crate) a_pattern: ExportedEmbeddedCscPattern,
    pub(crate) kkt_pattern: ExportedEmbeddedCscPattern,
    pub(crate) p_diag_indices: Vec<u64>,
    pub(crate) kkt_permutation: Vec<u64>,
    pub(crate) p_to_kkt: Vec<u64>,
    pub(crate) a_to_kkt: Vec<u64>,
    pub(crate) rho_to_kkt: Vec<u64>,
    pub(crate) symbolic_l: ExportedQdldlSymbolicL,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQdldlSymbolicL {
    pub(crate) l_pattern: ExportedEmbeddedCscPattern,
    pub(crate) etree: Vec<u64>,
    pub(crate) lnz: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedEmbeddedCscPattern {
    pub(crate) nrows: u64,
    pub(crate) ncols: u64,
    pub(crate) nnz: u64,
    pub(crate) indptr: Vec<u64>,
    pub(crate) indices: Vec<u64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQpProgramArenaLayout {
    pub(crate) total_bytes: u64,
    pub(crate) arena_alignment: u64,
    pub(crate) pdata_x: ExportedQpProgramArenaRegion,
    pub(crate) pdata: ExportedQpProgramArenaRegion,
    pub(crate) adata_x: ExportedQpProgramArenaRegion,
    pub(crate) adata: ExportedQpProgramArenaRegion,
    pub(crate) qdata: ExportedQpProgramArenaRegion,
    pub(crate) ldata: ExportedQpProgramArenaRegion,
    pub(crate) udata: ExportedQpProgramArenaRegion,
    pub(crate) data: ExportedQpProgramArenaRegion,
    pub(crate) settings: ExportedQpProgramArenaRegion,
    pub(crate) xsolution: ExportedQpProgramArenaRegion,
    pub(crate) ysolution: ExportedQpProgramArenaRegion,
    pub(crate) solution: ExportedQpProgramArenaRegion,
    pub(crate) info: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_l_x: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_l_p: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_l_i: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_l: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_kkt_x: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_kkt: ExportedQpProgramArenaRegion,
    pub(crate) qdldl: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_dinv: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_bp: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_sol: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_rho_inv_vec: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_d: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_iwork: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_bwork: ExportedQpProgramArenaRegion,
    pub(crate) qdldl_fwork: ExportedQpProgramArenaRegion,
    pub(crate) work_rho_vec: ExportedQpProgramArenaRegion,
    pub(crate) work_rho_inv_vec: ExportedQpProgramArenaRegion,
    pub(crate) work_constr_type: ExportedQpProgramArenaRegion,
    pub(crate) work_x: ExportedQpProgramArenaRegion,
    pub(crate) work_y: ExportedQpProgramArenaRegion,
    pub(crate) work_z: ExportedQpProgramArenaRegion,
    pub(crate) work_xz_tilde: ExportedQpProgramArenaRegion,
    pub(crate) work_x_prev: ExportedQpProgramArenaRegion,
    pub(crate) work_z_prev: ExportedQpProgramArenaRegion,
    pub(crate) work_ax: ExportedQpProgramArenaRegion,
    pub(crate) work_px: ExportedQpProgramArenaRegion,
    pub(crate) work_aty: ExportedQpProgramArenaRegion,
    pub(crate) work_delta_y: ExportedQpProgramArenaRegion,
    pub(crate) work_atdelta_y: ExportedQpProgramArenaRegion,
    pub(crate) work_delta_x: ExportedQpProgramArenaRegion,
    pub(crate) work_pdelta_x: ExportedQpProgramArenaRegion,
    pub(crate) work_adelta_x: ExportedQpProgramArenaRegion,
    pub(crate) workspace: ExportedQpProgramArenaRegion,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExportedQpProgramArenaRegion {
    pub(crate) byte_offset: u64,
    pub(crate) byte_len: u64,
    pub(crate) byte_alignment: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedInputLayer {
    pub(crate) inputs: Vec<ExportedInputSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedInputSpec {
    pub(crate) memory: ExportedMemorySpec,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedOutputLayer {
    pub(crate) outputs: Vec<ExportedOutputSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedOutputSpec {
    pub(crate) memory: ExportedMemorySpec,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedMemorySpec {
    pub(crate) location: u32,
    pub(crate) count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedLayer {
    pub(crate) kind: String,
    pub(crate) memory_in: Option<ExportedMemorySpec>,
    pub(crate) memory_out: Option<ExportedMemorySpec>,
    pub(crate) weights: Option<ExportedWeights>,
    pub(crate) ops: Option<Vec<ExportedRowOp>>,
    pub(crate) constants: Option<Vec<Value>>,
    pub(crate) opaque_programs: Option<Vec<Value>>,
    pub(crate) callee_function_id: Option<u32>,
    pub(crate) inputs: Option<Vec<ExportedEvaluateInputBinding>>,
    pub(crate) outputs: Option<Vec<ExportedEvaluateOutputBinding>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedWeights {
    pub(crate) quadratic: ExportedSparseTensor,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedSparseTensor {
    pub(crate) shape: Vec<u32>,
    pub(crate) entries: Vec<ExportedSparseEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedSparseEntry {
    pub(crate) index: Vec<u32>,
    pub(crate) value: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedRowOp {
    pub(crate) op: Value,
    pub(crate) first: i32,
    pub(crate) second: i32,
    pub(crate) third: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
pub(crate) enum ExportedEvaluateInputBinding {
    #[serde(rename = "workspace")]
    Workspace { offset: u32, length: u32 },
    #[serde(rename = "constant")]
    Constant { length: u32, values: Vec<f32> },
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedEvaluateOutputBinding {
    pub(crate) destination_offset: u32,
    pub(crate) length: u32,
}

pub(crate) struct CompileContext {
    pub(crate) exported_programs: Vec<ExportedProgram>,
    pub(crate) compiled_programs: Vec<Option<Program>>,
    pub(crate) visiting: Vec<bool>,
}
