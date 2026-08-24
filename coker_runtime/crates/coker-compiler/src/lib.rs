mod context;
mod lower;
mod model;
#[cfg(test)]
mod tests;
mod util;

use core::mem::{align_of, size_of};

use crate::util::checked_u16;
use coker_bytecode::{
    encode_module, BytecodeModule, EmbeddedCscPattern, EmbeddedLinsysSolver, EmbeddedOsqpSettings,
    EmbeddedQpProfile, Executable, InputSpec, OutputSpec, Program, QdldlSymbolicL,
    QpCoefficientOutputs, QpOutputSlice, QpProgram, QpProgramArenaLayout, QpProgramArenaRegion,
    QpProgramPlan, QpProgramQdldlPlan,
};
use coker_osqp_ffi::raw_embedded as ffi_raw;
use thiserror::Error;

pub(crate) const UNUSED_OPERAND: u16 = u16::MAX;

/// Errors produced while lowering exported graph JSON into validated bytecode.
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("failed to parse exported graph json: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("missing field {field}")]
    MissingField { field: &'static str },
    #[error("invalid field {field}: {reason}")]
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("bytecode validation or encoding failed: {0}")]
    Bytecode(#[from] coker_bytecode::BytecodeError),
}

/// Compiles exported graph JSON into a serialized bytecode module without requiring QP programs.
pub fn compile_exported_json(exported_graph_json: &[u8]) -> Result<Vec<u8>, CompileError> {
    let exported_module: model::ExportedModule = serde_json::from_slice(exported_graph_json)?;
    compile_exported_module_bytes(exported_module, false)
}

/// Compiles exported graph JSON into a serialized bytecode module that must contain QP programs.
pub fn compile_exported_qp_json(exported_qp_json: &[u8]) -> Result<Vec<u8>, CompileError> {
    let exported_module: model::ExportedModule = serde_json::from_slice(exported_qp_json)?;
    compile_exported_module_bytes(exported_module, true)
}

fn compile_exported_module_bytes(
    exported_module: model::ExportedModule,
    require_qp_programs: bool,
) -> Result<Vec<u8>, CompileError> {
    if require_qp_programs && exported_module.qp_programs.is_empty() {
        return Err(CompileError::InvalidField {
            field: "qp_programs",
            reason: "expected at least one QP program",
        });
    }
    let bytecode_module = build_exported_module(exported_module)?;
    validate_bytecode_module(&bytecode_module)?;
    encode_module(&bytecode_module).map_err(CompileError::from)
}

fn build_exported_module(
    exported_module: model::ExportedModule,
) -> Result<BytecodeModule, CompileError> {
    let model::ExportedModule {
        functions,
        qp_programs,
    } = exported_module;
    let function_module = context::compile_exported_module(model::ExportedModule {
        functions,
        qp_programs: Vec::new(),
    })?;
    let functions: Vec<Program> = function_module
        .functions()
        .map(|(_, program)| program.clone())
        .collect();
    let function_count = functions.len();
    let qp_count = qp_programs.len();
    let mut indexed_qp_programs = vec![None; qp_count];
    for exported_qp in qp_programs {
        let function_id = checked_u16(exported_qp.function_id, "function_id")? as usize;
        if function_id < function_count || function_id >= function_count + qp_count {
            return Err(CompileError::InvalidField {
                field: "function_id",
                reason:
                    "expected QP function ids to occupy the dense range after ordinary functions",
            });
        }
        let qp_index = function_id - function_count;
        if indexed_qp_programs[qp_index].is_some() {
            return Err(CompileError::InvalidField {
                field: "function_id",
                reason: "duplicate QP function id",
            });
        }
        indexed_qp_programs[qp_index] = Some(exported_qp);
    }
    let qp_programs = indexed_qp_programs
        .into_iter()
        .map(|exported_qp| {
            let exported_qp = exported_qp.ok_or(CompileError::InvalidField {
                field: "function_id",
                reason:
                    "expected QP function ids to occupy the dense range after ordinary functions",
            })?;
            build_exported_qp_program(&functions, exported_qp)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut executables = function_module.executables;
    executables.extend(qp_programs.into_iter().map(Executable::QpProgram));
    Ok(BytecodeModule::from_executables(executables))
}

fn validate_bytecode_module(module: &BytecodeModule) -> Result<(), CompileError> {
    module.validate_semantics().map_err(CompileError::from)
}

fn build_exported_qp_program(
    functions: &[Program],
    exported_qp: model::ExportedQpProgram,
) -> Result<QpProgram, CompileError> {
    let coefficient_function_id = checked_u16(
        exported_qp.coefficient_function_id,
        "coefficient_function_id",
    )?;
    let required_primal_workspace_size = checked_embedded_qp_u32(
        exported_qp.required_primal_workspace_size,
        "required_primal_workspace_size",
    )?;
    let required_tangent_workspace_size = checked_embedded_qp_u32(
        exported_qp.required_tangent_workspace_size,
        "required_tangent_workspace_size",
    )?;
    let input_specs = lower_input_specs(exported_qp.input_specs)?;
    let output_spec = lower_output_spec(exported_qp.output_spec)?;
    let p_pattern = lower_embedded_csc_pattern(
        exported_qp.p_pattern,
        "p_pattern.nrows",
        "p_pattern.ncols",
        "p_pattern.indptr",
        "p_pattern.indices",
    )?;
    let a_pattern = lower_embedded_csc_pattern(
        exported_qp.a_pattern,
        "a_pattern.nrows",
        "a_pattern.ncols",
        "a_pattern.indptr",
        "a_pattern.indices",
    )?;
    let coefficient_outputs = lower_qp_coefficient_outputs(&exported_qp.coefficient_outputs)?;
    let embedded_plan = lower_qp_program_plan(exported_qp.embedded_plan, &p_pattern, &a_pattern)?;

    let coefficient_program =
        functions
            .get(coefficient_function_id as usize)
            .ok_or(CompileError::InvalidField {
                field: "coefficient_function_id",
                reason: "must reference an ordinary function in the same module",
            })?;
    validate_qp_program_fields(
        &input_specs,
        &output_spec,
        &p_pattern,
        &a_pattern,
        &coefficient_outputs,
        &embedded_plan,
        required_primal_workspace_size,
        required_tangent_workspace_size,
        coefficient_program,
    )?;

    Ok(QpProgram::new(
        coefficient_function_id,
        required_primal_workspace_size,
        required_tangent_workspace_size,
        input_specs,
        output_spec,
        p_pattern,
        a_pattern,
        coefficient_outputs,
        embedded_plan,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_qp_program_fields(
    input_specs: &[InputSpec],
    output_spec: &OutputSpec,
    p_pattern: &EmbeddedCscPattern,
    a_pattern: &EmbeddedCscPattern,
    coefficient_outputs: &QpCoefficientOutputs,
    embedded_plan: &QpProgramPlan,
    required_primal_workspace_size: u32,
    required_tangent_workspace_size: u32,
    coefficient_program: &Program,
) -> Result<(), CompileError> {
    let n = usize::try_from(p_pattern.ncols).map_err(|_| CompileError::InvalidField {
        field: "p_pattern.ncols",
        reason: "value exceeds usize",
    })?;
    let m = usize::try_from(a_pattern.nrows).map_err(|_| CompileError::InvalidField {
        field: "a_pattern.nrows",
        reason: "value exceeds usize",
    })?;
    if p_pattern.nrows != p_pattern.ncols {
        return Err(CompileError::InvalidField {
            field: "p_pattern",
            reason: "dimensions must be square",
        });
    }
    if a_pattern.ncols != p_pattern.ncols {
        return Err(CompileError::InvalidField {
            field: "a_pattern",
            reason: "column count must match p_pattern",
        });
    }
    validate_csc_structure(
        &p_pattern
            .indptr
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>(),
        &p_pattern
            .indices
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>(),
        n,
        n,
        true,
        "p_pattern",
    )?;
    validate_csc_structure(
        &a_pattern
            .indptr
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>(),
        &a_pattern
            .indices
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>(),
        m,
        n,
        false,
        "a_pattern",
    )?;
    validate_output_slices_from_archive(coefficient_outputs, n, m, p_pattern, a_pattern)?;
    validate_output_spec_range(
        output_spec,
        required_primal_workspace_size,
        "required_primal_workspace_size",
    )?;
    validate_output_spec_range(
        output_spec,
        required_tangent_workspace_size,
        "required_tangent_workspace_size",
    )?;
    if u32::from(output_spec.length) != p_pattern.ncols {
        return Err(CompileError::InvalidField {
            field: "output_spec",
            reason: "length must match the decision-vector dimension",
        });
    }
    if input_specs != coefficient_program.input_specs {
        return Err(CompileError::InvalidField {
            field: "input_specs",
            reason: "must match the referenced coefficient evaluator inputs",
        });
    }
    if coefficient_program.checked_flat_output_size()?
        != expected_qp_output_length(coefficient_outputs)?
    {
        return Err(CompileError::InvalidField {
            field: "coefficient_outputs",
            reason: "must match the referenced coefficient evaluator outputs",
        });
    }
    embedded_plan
        .validate()
        .map_err(|_| CompileError::InvalidField {
            field: "embedded_plan",
            reason: "failed QP embedded plan schema validation",
        })?;
    if embedded_plan.qdldl_plan.p_pattern != *p_pattern {
        return Err(CompileError::InvalidField {
            field: "embedded_plan.qdldl_plan.p_pattern",
            reason: "P sparsity must match QP p_pattern",
        });
    }
    if embedded_plan.qdldl_plan.a_pattern != *a_pattern {
        return Err(CompileError::InvalidField {
            field: "embedded_plan.qdldl_plan.a_pattern",
            reason: "A sparsity must match QP a_pattern",
        });
    }
    Ok(())
}

fn lower_input_specs(
    input_specs: Vec<model::ExportedInputSpec>,
) -> Result<Vec<InputSpec>, CompileError> {
    input_specs
        .into_iter()
        .map(|input_spec| {
            Ok(InputSpec {
                workspace_offset: input_spec.memory.location,
                length: checked_u16(input_spec.memory.count, "input_specs.memory.count")?,
            })
        })
        .collect()
}

fn lower_output_spec(output_spec: model::ExportedOutputSpec) -> Result<OutputSpec, CompileError> {
    Ok(OutputSpec {
        workspace_offset: output_spec.memory.location,
        length: checked_u16(output_spec.memory.count, "output_spec.memory.count")?,
    })
}

fn lower_qp_coefficient_outputs(
    coefficient_outputs: &model::ExportedQpCoefficientOutputs,
) -> Result<QpCoefficientOutputs, CompileError> {
    validate_output_slices(coefficient_outputs, "coefficient_outputs")?;
    Ok(QpCoefficientOutputs {
        px: output_slice_to_archive(&coefficient_outputs.px, "coefficient_outputs.px")?,
        q: output_slice_to_archive(&coefficient_outputs.q, "coefficient_outputs.q")?,
        ax: output_slice_to_archive(&coefficient_outputs.ax, "coefficient_outputs.ax")?,
        l: output_slice_to_archive(&coefficient_outputs.l, "coefficient_outputs.l")?,
        u: output_slice_to_archive(&coefficient_outputs.u, "coefficient_outputs.u")?,
        r: output_slice_to_archive(&coefficient_outputs.r, "coefficient_outputs.r")?,
    })
}

fn lower_qp_program_plan(
    exported_plan: model::ExportedQpProgramPlan,
    p_pattern: &EmbeddedCscPattern,
    a_pattern: &EmbeddedCscPattern,
) -> Result<QpProgramPlan, CompileError> {
    if exported_plan.profile != model::ExportedEmbeddedQpProfile::Osqp063Embedded2Qdldl {
        return Err(CompileError::InvalidField {
            field: "embedded_plan.profile",
            reason: "unsupported embedded QP plan profile",
        });
    }
    let abi_version =
        checked_embedded_qp_u16(exported_plan.abi_version, "embedded_plan.abi_version")?;
    if abi_version != QpProgramPlan::ABI_VERSION {
        return Err(CompileError::InvalidField {
            field: "embedded_plan.abi_version",
            reason: "unsupported embedded QP plan abi version",
        });
    }
    let version = checked_embedded_qp_u16(exported_plan.version, "embedded_plan.version")?;
    if version != QpProgramPlan::VERSION {
        return Err(CompileError::InvalidField {
            field: "embedded_plan.version",
            reason: "unsupported embedded QP plan version",
        });
    }

    let settings = lower_embedded_osqp_settings(exported_plan.settings)?;
    let qdldl_plan = lower_qp_program_qdldl_plan(exported_plan.qdldl_plan)?;
    let arena_layout = compute_qp_program_arena_layout(p_pattern, a_pattern, &qdldl_plan)?;

    Ok(QpProgramPlan {
        abi_version,
        profile: EmbeddedQpProfile::Osqp063Embedded2Qdldl,
        version,
        settings,
        arena_layout,
        qdldl_plan,
    })
}

fn lower_qp_program_qdldl_plan(
    plan: model::ExportedQpProgramQdldlPlan,
) -> Result<QpProgramQdldlPlan, CompileError> {
    Ok(QpProgramQdldlPlan {
        p_pattern: lower_embedded_csc_pattern(
            plan.p_pattern,
            "embedded_plan.qdldl_plan.p_pattern.nrows",
            "embedded_plan.qdldl_plan.p_pattern.ncols",
            "embedded_plan.qdldl_plan.p_pattern.indptr",
            "embedded_plan.qdldl_plan.p_pattern.indices",
        )?,
        a_pattern: lower_embedded_csc_pattern(
            plan.a_pattern,
            "embedded_plan.qdldl_plan.a_pattern.nrows",
            "embedded_plan.qdldl_plan.a_pattern.ncols",
            "embedded_plan.qdldl_plan.a_pattern.indptr",
            "embedded_plan.qdldl_plan.a_pattern.indices",
        )?,
        kkt_pattern: lower_embedded_csc_pattern(
            plan.kkt_pattern,
            "embedded_plan.qdldl_plan.kkt_pattern.nrows",
            "embedded_plan.qdldl_plan.kkt_pattern.ncols",
            "embedded_plan.qdldl_plan.kkt_pattern.indptr",
            "embedded_plan.qdldl_plan.kkt_pattern.indices",
        )?,
        p_diag_indices: lower_embedded_u32_vec(
            plan.p_diag_indices,
            "embedded_plan.qdldl_plan.p_diag_indices",
        )?,
        kkt_permutation: lower_embedded_u32_vec(
            plan.kkt_permutation,
            "embedded_plan.qdldl_plan.kkt_permutation",
        )?,
        p_to_kkt: lower_embedded_u32_vec(plan.p_to_kkt, "embedded_plan.qdldl_plan.p_to_kkt")?,
        a_to_kkt: lower_embedded_u32_vec(plan.a_to_kkt, "embedded_plan.qdldl_plan.a_to_kkt")?,
        rho_to_kkt: lower_embedded_u32_vec(plan.rho_to_kkt, "embedded_plan.qdldl_plan.rho_to_kkt")?,
        symbolic_l: lower_qdldl_symbolic_l(plan.symbolic_l)?,
    })
}

fn lower_qdldl_symbolic_l(
    symbolic_l: model::ExportedQdldlSymbolicL,
) -> Result<QdldlSymbolicL, CompileError> {
    Ok(QdldlSymbolicL {
        l_pattern: lower_embedded_csc_pattern(
            symbolic_l.l_pattern,
            "embedded_plan.qdldl_plan.symbolic_l.l_pattern.nrows",
            "embedded_plan.qdldl_plan.symbolic_l.l_pattern.ncols",
            "embedded_plan.qdldl_plan.symbolic_l.l_pattern.indptr",
            "embedded_plan.qdldl_plan.symbolic_l.l_pattern.indices",
        )?,
        etree: lower_embedded_u32_vec(
            symbolic_l.etree,
            "embedded_plan.qdldl_plan.symbolic_l.etree",
        )?,
        lnz: lower_embedded_u32_vec(symbolic_l.lnz, "embedded_plan.qdldl_plan.symbolic_l.lnz")?,
    })
}

fn compute_qp_program_arena_layout(
    p_pattern: &EmbeddedCscPattern,
    a_pattern: &EmbeddedCscPattern,
    qdldl_plan: &QpProgramQdldlPlan,
) -> Result<QpProgramArenaLayout, CompileError> {
    let n = p_pattern.ncols as usize;
    let m = a_pattern.nrows as usize;
    let p_nnz = p_pattern.indices.len();
    let a_nnz = a_pattern.indices.len();
    let n_plus_m = n.checked_add(m).ok_or(CompileError::InvalidField {
        field: "embedded_plan.arena_layout.total_bytes",
        reason: "arena layout dimensions overflow usize",
    })?;
    let kkt_nnz = qdldl_plan.kkt_pattern.indices.len();
    let l_nnz = qdldl_plan.symbolic_l.l_pattern.indices.len();
    let l_indptr_count = n_plus_m.checked_add(1).ok_or(CompileError::InvalidField {
        field: "embedded_plan.arena_layout.qdldl_l_p",
        reason: "arena layout dimensions overflow usize",
    })?;
    let iwork_count = n_plus_m.checked_mul(3).ok_or(CompileError::InvalidField {
        field: "embedded_plan.arena_layout.qdldl_iwork",
        reason: "arena layout dimensions overflow usize",
    })?;

    let mut offset = 0usize;
    let mut arena_alignment = 1usize;

    let pdata_x = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        p_nnz,
        "embedded_plan.arena_layout.pdata_x",
    )?;
    let pdata = push_qp_arena_region::<ffi_raw::OSQPCscMatrix>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.pdata",
    )?;
    let adata_x = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        a_nnz,
        "embedded_plan.arena_layout.adata_x",
    )?;
    let adata = push_qp_arena_region::<ffi_raw::OSQPCscMatrix>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.adata",
    )?;
    let qdata = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.qdata",
    )?;
    let ldata = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.ldata",
    )?;
    let udata = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.udata",
    )?;
    let data = push_qp_arena_region::<ffi_raw::OSQPData>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.data",
    )?;
    let settings = push_qp_arena_region::<ffi_raw::OSQPSettings>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.settings",
    )?;
    let xsolution = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.xsolution",
    )?;
    let ysolution = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.ysolution",
    )?;
    let solution = push_qp_arena_region::<ffi_raw::OSQPSolution>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.solution",
    )?;
    let info = push_qp_arena_region::<ffi_raw::OSQPInfo>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.info",
    )?;
    let qdldl_l_x = push_qp_arena_region::<ffi_raw::QDLDL_float>(
        &mut offset,
        &mut arena_alignment,
        l_nnz,
        "embedded_plan.arena_layout.qdldl_l_x",
    )?;
    let qdldl_l_p = push_qp_arena_region::<ffi_raw::QDLDL_int>(
        &mut offset,
        &mut arena_alignment,
        l_indptr_count,
        "embedded_plan.arena_layout.qdldl_l_p",
    )?;
    let qdldl_l_i = push_qp_arena_region::<ffi_raw::QDLDL_int>(
        &mut offset,
        &mut arena_alignment,
        l_nnz,
        "embedded_plan.arena_layout.qdldl_l_i",
    )?;
    let qdldl_l = push_qp_arena_region::<ffi_raw::OSQPCscMatrix>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.qdldl_l",
    )?;
    let qdldl_kkt_x = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        kkt_nnz,
        "embedded_plan.arena_layout.qdldl_kkt_x",
    )?;
    let qdldl_kkt = push_qp_arena_region::<ffi_raw::OSQPCscMatrix>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.qdldl_kkt",
    )?;
    let qdldl = push_qp_arena_region::<ffi_raw::qdldl_solver>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.qdldl",
    )?;
    let qdldl_dinv = push_qp_arena_region::<ffi_raw::QDLDL_float>(
        &mut offset,
        &mut arena_alignment,
        n_plus_m,
        "embedded_plan.arena_layout.qdldl_dinv",
    )?;
    let qdldl_bp = push_qp_arena_region::<ffi_raw::QDLDL_float>(
        &mut offset,
        &mut arena_alignment,
        n_plus_m,
        "embedded_plan.arena_layout.qdldl_bp",
    )?;
    let qdldl_sol = push_qp_arena_region::<ffi_raw::QDLDL_float>(
        &mut offset,
        &mut arena_alignment,
        n_plus_m,
        "embedded_plan.arena_layout.qdldl_sol",
    )?;
    let qdldl_rho_inv_vec = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.qdldl_rho_inv_vec",
    )?;
    let qdldl_d = push_qp_arena_region::<ffi_raw::QDLDL_float>(
        &mut offset,
        &mut arena_alignment,
        n_plus_m,
        "embedded_plan.arena_layout.qdldl_d",
    )?;
    let qdldl_iwork = push_qp_arena_region::<ffi_raw::QDLDL_int>(
        &mut offset,
        &mut arena_alignment,
        iwork_count,
        "embedded_plan.arena_layout.qdldl_iwork",
    )?;
    let qdldl_bwork = push_qp_arena_region::<ffi_raw::QDLDL_bool>(
        &mut offset,
        &mut arena_alignment,
        n_plus_m,
        "embedded_plan.arena_layout.qdldl_bwork",
    )?;
    let qdldl_fwork = push_qp_arena_region::<ffi_raw::QDLDL_float>(
        &mut offset,
        &mut arena_alignment,
        n_plus_m,
        "embedded_plan.arena_layout.qdldl_fwork",
    )?;
    let work_rho_vec = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_rho_vec",
    )?;
    let work_rho_inv_vec = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_rho_inv_vec",
    )?;
    let work_constr_type = push_qp_arena_region::<ffi_raw::OSQPInt>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_constr_type",
    )?;
    let work_x = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.work_x",
    )?;
    let work_y = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_y",
    )?;
    let work_z = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_z",
    )?;
    let work_xz_tilde = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n_plus_m,
        "embedded_plan.arena_layout.work_xz_tilde",
    )?;
    let work_x_prev = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.work_x_prev",
    )?;
    let work_z_prev = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_z_prev",
    )?;
    let work_ax = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_ax",
    )?;
    let work_px = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.work_px",
    )?;
    let work_aty = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.work_aty",
    )?;
    let work_delta_y = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_delta_y",
    )?;
    let work_atdelta_y = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.work_atdelta_y",
    )?;
    let work_delta_x = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.work_delta_x",
    )?;
    let work_pdelta_x = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        n,
        "embedded_plan.arena_layout.work_pdelta_x",
    )?;
    let work_adelta_x = push_qp_arena_region::<ffi_raw::OSQPFloat>(
        &mut offset,
        &mut arena_alignment,
        m,
        "embedded_plan.arena_layout.work_adelta_x",
    )?;
    let workspace = push_qp_arena_region::<ffi_raw::OSQPWorkspace>(
        &mut offset,
        &mut arena_alignment,
        1,
        "embedded_plan.arena_layout.workspace",
    )?;

    let total_bytes = align_up_checked(
        offset,
        arena_alignment,
        "embedded_plan.arena_layout.total_bytes",
    )?;

    Ok(QpProgramArenaLayout {
        total_bytes: checked_qp_u32(total_bytes, "embedded_plan.arena_layout.total_bytes")?,
        arena_alignment: checked_qp_u32(
            arena_alignment,
            "embedded_plan.arena_layout.arena_alignment",
        )?,
        pdata_x,
        pdata,
        adata_x,
        adata,
        qdata,
        ldata,
        udata,
        data,
        settings,
        xsolution,
        ysolution,
        solution,
        info,
        qdldl_l_x,
        qdldl_l_p,
        qdldl_l_i,
        qdldl_l,
        qdldl_kkt_x,
        qdldl_kkt,
        qdldl,
        qdldl_dinv,
        qdldl_bp,
        qdldl_sol,
        qdldl_rho_inv_vec,
        qdldl_d,
        qdldl_iwork,
        qdldl_bwork,
        qdldl_fwork,
        work_rho_vec,
        work_rho_inv_vec,
        work_constr_type,
        work_x,
        work_y,
        work_z,
        work_xz_tilde,
        work_x_prev,
        work_z_prev,
        work_ax,
        work_px,
        work_aty,
        work_delta_y,
        work_atdelta_y,
        work_delta_x,
        work_pdelta_x,
        work_adelta_x,
        workspace,
    })
}

fn push_qp_arena_region<T>(
    offset: &mut usize,
    arena_alignment: &mut usize,
    count: usize,
    field: &'static str,
) -> Result<QpProgramArenaRegion, CompileError> {
    let alignment = align_of::<T>();
    let byte_len = count
        .checked_mul(size_of::<T>())
        .ok_or(CompileError::InvalidField {
            field,
            reason: "arena region length overflow",
        })?;
    *arena_alignment = (*arena_alignment).max(alignment);
    let byte_offset = align_up_checked(*offset, alignment, field)?;
    *offset = byte_offset
        .checked_add(byte_len)
        .ok_or(CompileError::InvalidField {
            field,
            reason: "arena region end overflow",
        })?;
    Ok(QpProgramArenaRegion {
        byte_offset: checked_qp_u32(byte_offset, field)?,
        byte_len: checked_qp_u32(byte_len, field)?,
        byte_alignment: checked_qp_u32(alignment, field)?,
    })
}

fn align_up_checked(
    value: usize,
    alignment: usize,
    field: &'static str,
) -> Result<usize, CompileError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(CompileError::InvalidField {
            field,
            reason: "alignment must be a nonzero power of two",
        });
    }
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or(CompileError::InvalidField {
            field,
            reason: "arena layout overflow",
        })
}

fn lower_embedded_osqp_settings(
    settings: model::ExportedEmbeddedOsqpSettings,
) -> Result<EmbeddedOsqpSettings, CompileError> {
    if settings.linsys_solver != model::ExportedEmbeddedLinsysSolver::Qdldl {
        return Err(CompileError::InvalidField {
            field: "embedded_plan.settings.linsys_solver",
            reason: "embedded QP plans must use the QDLDL solver",
        });
    }

    Ok(EmbeddedOsqpSettings {
        rho: settings.rho,
        sigma: settings.sigma,
        alpha: settings.alpha,
        adaptive_rho: settings.adaptive_rho,
        adaptive_rho_interval: checked_embedded_qp_u32(
            settings.adaptive_rho_interval,
            "embedded_plan.settings.adaptive_rho_interval",
        )?,
        adaptive_rho_tolerance: settings.adaptive_rho_tolerance,
        max_iter: checked_embedded_qp_u32(settings.max_iter, "embedded_plan.settings.max_iter")?,
        eps_abs: settings.eps_abs,
        eps_rel: settings.eps_rel,
        eps_prim_inf: settings.eps_prim_inf,
        eps_dual_inf: settings.eps_dual_inf,
        scaling: checked_embedded_qp_u32(settings.scaling, "embedded_plan.settings.scaling")?,
        scaled_termination: settings.scaled_termination,
        check_termination: checked_embedded_qp_u32(
            settings.check_termination,
            "embedded_plan.settings.check_termination",
        )?,
        warm_start: settings.warm_start,
        linsys_solver: EmbeddedLinsysSolver::Qdldl,
    })
}

fn lower_embedded_csc_pattern(
    pattern: model::ExportedEmbeddedCscPattern,
    nrows_field: &'static str,
    ncols_field: &'static str,
    indptr_field: &'static str,
    indices_field: &'static str,
) -> Result<EmbeddedCscPattern, CompileError> {
    Ok(EmbeddedCscPattern {
        nrows: checked_embedded_qp_u32(pattern.nrows, nrows_field)?,
        ncols: checked_embedded_qp_u32(pattern.ncols, ncols_field)?,
        indptr: lower_embedded_u32_vec(pattern.indptr, indptr_field)?,
        indices: lower_embedded_u32_vec(pattern.indices, indices_field)?,
    })
}

fn lower_embedded_u32_vec(values: Vec<u64>, field: &'static str) -> Result<Vec<u32>, CompileError> {
    values
        .into_iter()
        .map(|value| checked_embedded_qp_u32(value, field))
        .collect()
}

fn checked_embedded_qp_u16(value: u64, field: &'static str) -> Result<u16, CompileError> {
    u16::try_from(value).map_err(|_| CompileError::InvalidField {
        field,
        reason: "value exceeds u16",
    })
}

fn checked_embedded_qp_u32(value: u64, field: &'static str) -> Result<u32, CompileError> {
    u32::try_from(value).map_err(|_| CompileError::InvalidField {
        field,
        reason: "value exceeds u32",
    })
}

fn checked_qp_u32(value: usize, field: &'static str) -> Result<u32, CompileError> {
    u32::try_from(value).map_err(|_| CompileError::InvalidField {
        field,
        reason: "value exceeds u32",
    })
}

fn output_slice_to_archive(
    slice: &model::ExportedQpOutput,
    field: &'static str,
) -> Result<QpOutputSlice, CompileError> {
    Ok(QpOutputSlice {
        start: checked_qp_u32(slice.start, field)?,
        length: checked_qp_u32(slice.length, field)?,
    })
}

fn expected_qp_output_length(
    coefficient_outputs: &QpCoefficientOutputs,
) -> Result<u32, CompileError> {
    coefficient_outputs
        .r
        .start
        .checked_add(coefficient_outputs.r.length)
        .ok_or(CompileError::InvalidField {
            field: "coefficient_outputs",
            reason: "output slice lengths overflow",
        })
}

fn validate_output_slices_from_archive(
    coefficient_outputs: &QpCoefficientOutputs,
    n: usize,
    m: usize,
    p_pattern: &EmbeddedCscPattern,
    a_pattern: &EmbeddedCscPattern,
) -> Result<(), CompileError> {
    if coefficient_outputs.q.length != checked_qp_u32(n, "p_pattern.ncols")?
        || coefficient_outputs.l.length != checked_qp_u32(m, "a_pattern.nrows")?
        || coefficient_outputs.u.length != checked_qp_u32(m, "a_pattern.nrows")?
        || coefficient_outputs.px.length
            != checked_qp_u32(p_pattern.indices.len(), "p_pattern.indices")?
        || coefficient_outputs.ax.length
            != checked_qp_u32(a_pattern.indices.len(), "a_pattern.indices")?
        || coefficient_outputs.r.length != 1
    {
        return Err(CompileError::InvalidField {
            field: "coefficient_outputs",
            reason: "lengths must match QP dimensions, sparsity, and reporting output",
        });
    }
    Ok(())
}

fn validate_output_spec_range(
    output_spec: &OutputSpec,
    required_workspace_size: u32,
    field: &'static str,
) -> Result<(), CompileError> {
    let end = output_spec
        .workspace_offset
        .checked_add(u32::from(output_spec.length))
        .ok_or(CompileError::InvalidField {
            field,
            reason: "output_spec range overflows u32",
        })?;
    if end > required_workspace_size {
        return Err(CompileError::InvalidField {
            field,
            reason: "output_spec exceeds the declared workspace capacity",
        });
    }
    Ok(())
}

fn validate_nondecreasing_offsets(
    offsets: &[usize],
    terminal: usize,
    field: &'static str,
) -> Result<(), CompileError> {
    if offsets.first().copied().unwrap_or(0) != 0 {
        return Err(CompileError::InvalidField {
            field,
            reason: "offsets must start at zero",
        });
    }
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(CompileError::InvalidField {
            field,
            reason: "offsets must be nondecreasing",
        });
    }
    if offsets.last().copied().unwrap_or(0) != terminal {
        return Err(CompileError::InvalidField {
            field,
            reason: "terminal offset does not match expected length",
        });
    }
    Ok(())
}

fn validate_csc_structure(
    indptr: &[usize],
    indices: &[usize],
    nrows: usize,
    ncols: usize,
    upper_triangular: bool,
    field: &'static str,
) -> Result<(), CompileError> {
    if indptr.len() != ncols + 1 {
        return Err(CompileError::InvalidField {
            field,
            reason: "indptr length does not match column count",
        });
    }
    validate_nondecreasing_offsets(indptr, indices.len(), field)?;
    for col in 0..ncols {
        let start = indptr[col];
        let end = indptr[col + 1];
        if start > end || end > indices.len() {
            return Err(CompileError::InvalidField {
                field,
                reason: "indptr entries must be monotonic and in bounds",
            });
        }
        let mut previous_row = None;
        for &row in &indices[start..end] {
            if row >= nrows {
                return Err(CompileError::InvalidField {
                    field,
                    reason: "row index out of bounds",
                });
            }
            if upper_triangular && row > col {
                return Err(CompileError::InvalidField {
                    field,
                    reason: "P entries must be upper triangular",
                });
            }
            if let Some(previous_row) = previous_row {
                if row <= previous_row {
                    return Err(CompileError::InvalidField {
                        field,
                        reason: "row indices must be strictly increasing within each column",
                    });
                }
            }
            previous_row = Some(row);
        }
    }
    Ok(())
}

fn validate_output_slices(
    outputs: &model::ExportedQpCoefficientOutputs,
    field: &'static str,
) -> Result<(), CompileError> {
    let mut expected_start = 0usize;
    for (name, slice) in [
        ("px", &outputs.px),
        ("q", &outputs.q),
        ("ax", &outputs.ax),
        ("l", &outputs.l),
        ("u", &outputs.u),
        ("r", &outputs.r),
    ] {
        if slice.start != expected_start {
            return Err(CompileError::InvalidField {
                field,
                reason: match name {
                    "px" => "px output must start at zero",
                    _ => "output slices must be contiguous",
                },
            });
        }
        expected_start =
            expected_start
                .checked_add(slice.length)
                .ok_or(CompileError::InvalidField {
                    field,
                    reason: "output slice lengths overflow",
                })?;
    }
    Ok(())
}
