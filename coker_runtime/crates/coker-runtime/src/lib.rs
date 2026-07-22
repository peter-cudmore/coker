#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod execute;
mod ops;
#[cfg(test)]
mod tests;
mod validate;
mod workspace;

use crate::workspace::Workspace;
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
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
    #[error("output buffer size mismatch: expected {expected}, got {actual}")]
    OutputBufferSizeMismatch { expected: usize, actual: usize },
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

#[derive(Debug)]
pub struct ExecutionInputs<'a> {
    inputs: &'a [&'a [f32]],
}

#[derive(Debug)]
pub struct ExecutionOutputs<'a> {
    outputs: &'a mut [f32],
}

#[derive(Debug)]
pub struct PushForwardInputs<'a> {
    inputs: &'a [&'a [f32]],
    tangents: &'a [&'a [f32]],
}

#[derive(Debug)]
pub struct PushForwardOutputs<'a> {
    outputs: &'a mut [f32],
    tangent_outputs: &'a mut [f32],
}

#[derive(Debug)]
pub struct ModuleBuilder {
    bytecode_module: BytecodeModule,
    workspace: Option<Vec<f32>>,
    tangent_workspace: Option<Vec<f32>>,
}

impl ModuleBuilder {
    pub fn new(bytecode_module: BytecodeModule) -> Result<Self, RuntimeError> {
        validate::validate_module_struct(&bytecode_module)?;
        Ok(Self {
            bytecode_module,
            workspace: None,
            tangent_workspace: None,
        })
    }

    pub fn new_from_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        Self::new(parse_module(bytes)?)
    }

    pub fn with_workspace(mut self, workspace: Vec<f32>) -> Self {
        self.workspace = Some(workspace);
        self
    }

    pub fn with_tangent_workspace(mut self, tangent_workspace: Vec<f32>) -> Self {
        self.tangent_workspace = Some(tangent_workspace);
        self
    }

    pub fn build(self) -> Result<Module, RuntimeError> {
        let required_workspace_size =
            entry_program_unchecked(&self.bytecode_module).required_workspace_size as usize;
        let mut workspace = self
            .workspace
            .unwrap_or_else(|| vec![0.0; required_workspace_size]);
        validate::validate_workspace_size(required_workspace_size, workspace.len())?;
        workspace.fill(0.0);

        let mut tangent_workspace = self
            .tangent_workspace
            .unwrap_or_else(|| vec![0.0; required_workspace_size]);
        validate::validate_workspace_size(required_workspace_size, tangent_workspace.len())?;
        tangent_workspace.fill(0.0);

        Ok(Module {
            bytecode_module: self.bytecode_module,
            workspace,
            tangent_workspace,
        })
    }
}

#[derive(Debug)]
pub struct Module {
    bytecode_module: BytecodeModule,
    workspace: Vec<f32>,
    tangent_workspace: Vec<f32>,
}

impl Module {
    pub fn info(&self) -> ProgramInfo {
        program_info_from_program(self.entry_program())
    }

    pub fn validate_inputs<'a>(
        &self,
        inputs: &'a [&'a [f32]],
    ) -> Result<ExecutionInputs<'a>, RuntimeError> {
        validate::validate_inputs(self.entry_program(), inputs)?;
        Ok(ExecutionInputs { inputs })
    }

    pub fn validate_outputs<'a>(
        &self,
        outputs: &'a mut [f32],
    ) -> Result<ExecutionOutputs<'a>, RuntimeError> {
        validate::validate_outputs(self.entry_program(), outputs)?;
        Ok(ExecutionOutputs { outputs })
    }

    pub fn validate_push_forward_inputs<'a>(
        &self,
        inputs: &'a [&'a [f32]],
        tangents: &'a [&'a [f32]],
    ) -> Result<PushForwardInputs<'a>, RuntimeError> {
        validate::validate_inputs(self.entry_program(), inputs)?;
        validate::validate_inputs(self.entry_program(), tangents)?;
        Ok(PushForwardInputs { inputs, tangents })
    }

    pub fn validate_push_forward_outputs<'a>(
        &self,
        outputs: &'a mut [f32],
        tangent_outputs: &'a mut [f32],
    ) -> Result<PushForwardOutputs<'a>, RuntimeError> {
        validate::validate_outputs(self.entry_program(), outputs)?;
        validate::validate_outputs(self.entry_program(), tangent_outputs)?;
        Ok(PushForwardOutputs {
            outputs,
            tangent_outputs,
        })
    }

    pub fn execute(
        &mut self,
        execution_inputs: ExecutionInputs<'_>,
        execution_outputs: ExecutionOutputs<'_>,
    ) {
        let bytecode_module = &self.bytecode_module;
        let entry_program = entry_program_unchecked(bytecode_module);
        let workspace = &mut self.workspace;
        let wrote_direct_outputs = execute_in_place_unchecked(
            bytecode_module,
            entry_program,
            execution_inputs.inputs,
            workspace,
            Some(execution_outputs.outputs),
        );
        if !wrote_direct_outputs {
            workspace::write_outputs(entry_program, workspace, execution_outputs.outputs);
        }
    }

    pub fn push_forward(
        &mut self,
        push_forward_inputs: PushForwardInputs<'_>,
        push_forward_outputs: PushForwardOutputs<'_>,
    ) {
        let bytecode_module = &self.bytecode_module;
        let entry_program = entry_program_unchecked(bytecode_module);
        let workspace = &mut self.workspace;
        let tangent_workspace = &mut self.tangent_workspace;
        let wrote_direct_outputs = push_forward_in_place_unchecked(
            bytecode_module,
            entry_program,
            push_forward_inputs.inputs,
            push_forward_inputs.tangents,
            workspace,
            tangent_workspace,
            Some(push_forward_outputs.outputs),
            Some(push_forward_outputs.tangent_outputs),
        );
        if !wrote_direct_outputs {
            workspace::write_outputs(entry_program, workspace, push_forward_outputs.outputs);
            workspace::write_outputs(
                entry_program,
                tangent_workspace,
                push_forward_outputs.tangent_outputs,
            );
        }
    }

    pub fn workspace(&self) -> &[f32] {
        &self.workspace
    }

    pub fn tangent_workspace(&self) -> &[f32] {
        &self.tangent_workspace
    }

    fn entry_program(&self) -> &Program {
        entry_program_unchecked(&self.bytecode_module)
    }
}

pub fn parse_module(module_bytes: &[u8]) -> Result<BytecodeModule, RuntimeError> {
    Ok(decode_module(module_bytes)?)
}

pub fn program_info(module_bytes: &[u8]) -> Result<ProgramInfo, RuntimeError> {
    let module = validate_module(module_bytes)?;
    Ok(program_info_from_program(entry_program_unchecked(&module)))
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
    outputs: &mut [f32],
) -> Result<(), RuntimeError> {
    let entry_program = entry_program(module)?;
    validate::validate_inputs(entry_program, inputs)?;
    validate::validate_workspace(entry_program, workspace)?;
    validate::validate_outputs(entry_program, outputs)?;
    let wrote_direct_outputs =
        execute_in_place_unchecked(module, entry_program, inputs, workspace, Some(outputs));
    if !wrote_direct_outputs {
        workspace::write_outputs(entry_program, workspace, outputs);
    }
    Ok(())
}

pub fn push_forward(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    tangents: &[&[f32]],
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
    outputs: &mut [f32],
    tangent_outputs: &mut [f32],
) -> Result<(), RuntimeError> {
    let entry_program = entry_program(module)?;
    validate::validate_inputs(entry_program, inputs)?;
    validate::validate_inputs(entry_program, tangents)?;
    validate::validate_workspace(entry_program, workspace)?;
    validate::validate_workspace(entry_program, tangent_workspace)?;
    validate::validate_outputs(entry_program, outputs)?;
    validate::validate_outputs(entry_program, tangent_outputs)?;
    let wrote_direct_outputs = push_forward_in_place_unchecked(
        module,
        entry_program,
        inputs,
        tangents,
        workspace,
        tangent_workspace,
        Some(outputs),
        Some(tangent_outputs),
    );
    if !wrote_direct_outputs {
        workspace::write_outputs(entry_program, workspace, outputs);
        workspace::write_outputs(entry_program, tangent_workspace, tangent_outputs);
    }
    Ok(())
}

pub fn execute_in_place(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let entry_program = entry_program(module)?;
    validate::validate_inputs(entry_program, inputs)?;
    validate::validate_workspace(entry_program, workspace)?;
    execute_in_place_unchecked(module, entry_program, inputs, workspace, None);
    Ok(())
}

pub fn push_forward_in_place(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    tangents: &[&[f32]],
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let entry_program = entry_program(module)?;
    validate::validate_inputs(entry_program, inputs)?;
    validate::validate_inputs(entry_program, tangents)?;
    validate::validate_workspace(entry_program, workspace)?;
    validate::validate_workspace(entry_program, tangent_workspace)?;
    push_forward_in_place_unchecked(
        module,
        entry_program,
        inputs,
        tangents,
        workspace,
        tangent_workspace,
        None,
        None,
    );
    Ok(())
}

fn execute_in_place_unchecked(
    module: &BytecodeModule,
    entry_program: &Program,
    inputs: &[&[f32]],
    workspace: &mut [f32],
    outputs: Option<&mut [f32]>,
) -> bool {
    let mut workspace = Workspace::new(workspace);
    workspace.fill(0.0);
    workspace.pack_inputs(&entry_program.input_specs, inputs);
    execute::execute_program_layers(module, entry_program, &mut workspace, outputs)
}

fn push_forward_in_place_unchecked(
    module: &BytecodeModule,
    entry_program: &Program,
    inputs: &[&[f32]],
    tangents: &[&[f32]],
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
    outputs: Option<&mut [f32]>,
    tangent_outputs: Option<&mut [f32]>,
) -> bool {
    let mut workspace = Workspace::new(workspace);
    let mut tangent_workspace = Workspace::new(tangent_workspace);
    workspace.fill(0.0);
    tangent_workspace.fill(0.0);
    workspace.pack_inputs(&entry_program.input_specs, inputs);
    tangent_workspace.pack_inputs(&entry_program.input_specs, tangents);
    execute::push_forward_program_layers(
        module,
        entry_program,
        &mut workspace,
        &mut tangent_workspace,
        outputs,
        tangent_outputs,
    )
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

pub(crate) fn entry_program_unchecked(module: &BytecodeModule) -> &Program {
    find_function_unchecked(module, 0)
}

pub(crate) fn find_function(module: &BytecodeModule, function_id: u16) -> Option<&Program> {
    module
        .functions
        .iter()
        .find(|program| program.function_id == function_id)
}

pub(crate) fn find_function_unchecked(module: &BytecodeModule, function_id: u16) -> &Program {
    find_function(module, function_id).expect("validated module missing referenced function")
}
