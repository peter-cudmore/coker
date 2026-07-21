mod context;
mod lower;
mod model;
#[cfg(test)]
mod tests;
mod util;

use coker_bytecode::encode_module;
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
