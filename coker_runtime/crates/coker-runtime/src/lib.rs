#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod execute;
mod ops;
#[cfg(test)]
mod tests;
mod validate;
mod workspace;

use alloc::{string::{String, ToString}, vec::Vec};
use coker_bytecode::{decode_module, BytecodeModule, InputSpec, OutputSpec, Program};
use thiserror::Error;

const UNUSED_OPERAND: u16 = u16::MAX;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("bytecode error: {0}")]
    Bytecode(#[from] coker_bytecode::BytecodeError),
    #[error("input count mismatch: expected {expected}, got {actual}")]
    InputCountMismatch { expected: usize, actual: usize },
    #[error("input {index} size mismatch: expected {expected}, got {actual}")]
    InputSizeMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("workspace buffer too small: expected at least {expected}, got {actual}")]
    WorkspaceTooSmall { expected: usize, actual: usize },
    #[error("program validation failed: {0}")]
    Validation(String),
}

#[derive(Debug, Clone)]
pub struct ProgramInfo {
    pub workspace_size: usize,
    pub required_workspace_size: usize,
    pub input_specs: Vec<InputSpec>,
    pub output_specs: Vec<OutputSpec>,
}

pub fn parse_module(module_bytes: &[u8]) -> Result<BytecodeModule, RuntimeError> {
    Ok(decode_module(module_bytes)?)
}

pub fn program_info(module_bytes: &[u8]) -> Result<ProgramInfo, RuntimeError> {
    let module = validate_module(module_bytes)?;
    Ok(program_info_from_program(entry_program(&module)?))
}

pub fn validate_module(module_bytes: &[u8]) -> Result<BytecodeModule, RuntimeError> {
    let module = parse_module(module_bytes)?;
    validate::validate_module_struct(&module)?;
    Ok(module)
}

pub fn execute(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    workspace: &mut [f32],
) -> Result<Vec<Vec<f32>>, RuntimeError> {
    let program = entry_program(module)?;
    execute_in_place(module, inputs, workspace)?;
    Ok(workspace::collect_outputs(program, workspace))
}

pub fn push_forward(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    tangents: &[&[f32]],
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), RuntimeError> {
    let program = entry_program(module)?;
    push_forward_in_place(module, inputs, tangents, workspace, tangent_workspace)?;
    Ok((
        workspace::collect_outputs(program, workspace),
        workspace::collect_outputs(program, tangent_workspace),
    ))
}

pub fn execute_in_place(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let program = entry_program(module)?;
    validate::validate_inputs(program, inputs)?;
    validate::validate_workspace(program, workspace)?;

    workspace.fill(0.0);
    workspace::pack_inputs(&program.input_specs, inputs, workspace);
    execute::execute_program_layers(module, program, workspace)
}

pub fn push_forward_in_place(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    tangents: &[&[f32]],
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let program = entry_program(module)?;
    validate::validate_inputs(program, inputs)?;
    validate::validate_inputs(program, tangents)?;
    validate::validate_workspace(program, workspace)?;
    validate::validate_workspace(program, tangent_workspace)?;

    workspace.fill(0.0);
    tangent_workspace.fill(0.0);
    workspace::pack_inputs(&program.input_specs, inputs, workspace);
    workspace::pack_inputs(&program.input_specs, tangents, tangent_workspace);
    execute::push_forward_program_layers(module, program, workspace, tangent_workspace)
}

pub fn program_info_from_program(program: &Program) -> ProgramInfo {
    ProgramInfo {
        workspace_size: program.workspace_size as usize,
        required_workspace_size: program.required_workspace_size as usize,
        input_specs: program.input_specs.clone(),
        output_specs: program.output_specs.clone(),
    }
}

pub fn entry_program(module: &BytecodeModule) -> Result<&Program, RuntimeError> {
    find_function(module, 0)
        .ok_or_else(|| RuntimeError::Validation("missing entry function_id 0".to_string()))
}

pub(crate) fn find_function(module: &BytecodeModule, function_id: u16) -> Option<&Program> {
    module
        .functions
        .iter()
        .find(|program| program.function_id == function_id)
}
