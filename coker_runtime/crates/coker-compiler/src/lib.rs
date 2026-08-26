mod context;
mod lower;
mod model;
#[cfg(test)]
mod tests;
mod qp;
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
use qp::build_exported_qp_program;

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
