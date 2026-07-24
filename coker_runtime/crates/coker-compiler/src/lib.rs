mod context;
mod lower;
mod model;
#[cfg(test)]
mod tests;
mod util;

use coker_bytecode::{encode_module, encode_qp_program, QpProgramArchive};
use thiserror::Error;

pub(crate) const UNUSED_OPERAND: u16 = u16::MAX;

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
    #[error("failed to encode bytecode: {0}")]
    Encode(#[from] coker_bytecode::BytecodeError),
}

pub fn compile_exported_json(exported_graph_json: &[u8]) -> Result<Vec<u8>, CompileError> {
    let exported_module: model::ExportedModule = serde_json::from_slice(exported_graph_json)?;
    let bytecode_module = context::compile_exported_module(exported_module)?;
    encode_module(&bytecode_module).map_err(CompileError::from)
}

pub fn compile_exported_qp_json(exported_qp_json: &[u8]) -> Result<Vec<u8>, CompileError> {
    let wrapper: serde_json::Value = serde_json::from_slice(exported_qp_json)?;
    let exported_qp: model::ExportedQpProgram = serde_json::from_value(
        wrapper
            .get("program")
            .cloned()
            .ok_or(CompileError::MissingField { field: "program" })?,
    )?;
    validate_exported_qp(&exported_qp)?;
    let evaluator_json = serde_json::to_vec(&exported_qp.coefficient_evaluator)
        .map_err(CompileError::InvalidJson)?;
    let coefficient_program = compile_exported_json(&evaluator_json)?;
    let outputs = &exported_qp.coefficient_outputs;
    let coefficient_lengths = [
        outputs.px.length,
        outputs.q.length,
        outputs.ax.length,
        outputs.l.length,
        outputs.u.length,
        outputs.r.length,
    ]
    .into_iter()
    .map(|value| u32::try_from(value).map_err(|_| CompileError::InvalidField {
        field: "coefficient_outputs",
        reason: "length exceeds u32",
    }))
    .collect::<Result<Vec<_>, _>>()?;
    encode_qp_program(&QpProgramArchive {
        n: checked_qp_u32(exported_qp.n, "n")?,
        m: checked_qp_u32(exported_qp.m, "m")?,
        parameter_lengths: exported_qp.parameter_inputs.into_iter()
            .map(|input| checked_qp_u32(input.length, "parameter_inputs.length"))
            .collect::<Result<Vec<_>, _>>()?,
        p_rows: exported_qp.p_structure.iter().map(|entry| checked_qp_u32(entry.row, "p_structure.row")).collect::<Result<Vec<_>, _>>()?,
        p_cols: exported_qp.p_structure.iter().map(|entry| checked_qp_u32(entry.col, "p_structure.col")).collect::<Result<Vec<_>, _>>()?,
        a_rows: exported_qp.a_structure.iter().map(|entry| checked_qp_u32(entry.row, "a_structure.row")).collect::<Result<Vec<_>, _>>()?,
        a_cols: exported_qp.a_structure.iter().map(|entry| checked_qp_u32(entry.col, "a_structure.col")).collect::<Result<Vec<_>, _>>()?,
        coefficient_program,
        coefficient_lengths,
        warm_start: exported_qp.warm_start,
    }).map_err(CompileError::from)
}

fn checked_qp_u32(value: usize, field: &'static str) -> Result<u32, CompileError> {
    u32::try_from(value).map_err(|_| CompileError::InvalidField {
        field,
        reason: "value exceeds u32",
    })
}

fn validate_exported_qp(qp: &model::ExportedQpProgram) -> Result<(), CompileError> {
    if qp.p_structure.iter().any(|entry| entry.row > entry.col || entry.col >= qp.n) {
        return Err(CompileError::InvalidField {
            field: "p_structure",
            reason: "entries must be upper triangular and in bounds",
        });
    }
    if qp.a_structure.iter().any(|entry| entry.row >= qp.m || entry.col >= qp.n) {
        return Err(CompileError::InvalidField {
            field: "a_structure",
            reason: "entries must be in bounds",
        });
    }
    if qp.coefficient_outputs.q.length != qp.n
        || qp.coefficient_outputs.l.length != qp.m
        || qp.coefficient_outputs.u.length != qp.m
        || qp.coefficient_outputs.px.length != qp.p_structure.len()
        || qp.coefficient_outputs.ax.length != qp.a_structure.len()
    {
        return Err(CompileError::InvalidField {
            field: "coefficient_outputs",
            reason: "lengths must match QP dimensions and sparsity",
        });
    }
    Ok(())
}
