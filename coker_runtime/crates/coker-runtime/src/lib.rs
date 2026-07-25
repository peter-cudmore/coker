#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod execute;
mod ops;
mod qp;
mod static_module;
#[cfg(all(test, feature = "std"))]
mod tests;
mod validate;
mod validation_common;
mod workspace;

#[cfg(osqp_embedded)]
pub use crate::qp::BoundMappedQpProgram;
pub use crate::qp::{
    MappedQpProgram, MappedQpPushForwardWorkspace, MappedQpWorkspace, QpSolveDiagnostics,
    QpSolveStatus, QpWorkspaceRequirements,
};
#[cfg(all(feature = "std", not(osqp_embedded)))]
pub use crate::qp::{QpRuntime, QpSolveResult, QpWorkspaceLayout, QpWorkspaceRegion};
pub use crate::static_module::{MappedExecutable, MappedModule, MappedProgram};
use crate::workspace::Workspace;
use alloc::string::{String, ToString};
use coker_bytecode::{decode_module, BytecodeModule, InputSpec, OutputSpec, Program};
use thiserror::Error;

const UNUSED_OPERAND: u16 = u16::MAX;

/// Errors produced while decoding, validating, or executing a bytecode module.
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
    #[error("QP solver error: {0}")]
    QpSolver(String),
    #[error(
        "embedded QP buffers overlap, have an invalid range, or do not match the solver layout"
    )]
    EmbeddedQpBuffersInvalid,
    #[error("embedded QP workspace overlaps or does not match the validated archive")]
    EmbeddedQpWorkspaceInvalid,
    #[error("embedded QP evaluator produced non-finite coefficients or invalid bounds")]
    EmbeddedQpNumericInvalid,
    #[error("embedded OSQP ABI {operation} failed with status {status}")]
    EmbeddedQpAbi {
        operation: &'static str,
        status: i32,
    },
    #[error("QP push-forward is unsupported: {reason}")]
    QpPushForwardUnsupported { reason: &'static str },
    #[error("QP push-forward is nondifferentiable at solver status {status:?}: {reason}")]
    QpPushForwardNondifferentiable {
        status: crate::qp::QpSolveStatus,
        reason: &'static str,
    },
}

/// Trait implemented by input and output spec views that expose workspace layout metadata.
pub trait SpecInfo {
    fn workspace_offset(&self) -> usize;
    fn length(&self) -> usize;
}

impl SpecInfo for InputSpec {
    fn workspace_offset(&self) -> usize {
        self.workspace_offset as usize
    }

    fn length(&self) -> usize {
        self.length as usize
    }
}

impl SpecInfo for OutputSpec {
    fn workspace_offset(&self) -> usize {
        self.workspace_offset as usize
    }

    fn length(&self) -> usize {
        self.length as usize
    }
}

/// Static metadata for a plain bytecode program entry point.
#[derive(Debug, Clone, Copy)]
pub struct ProgramInfo<'a, I: SpecInfo = InputSpec, O: SpecInfo = OutputSpec> {
    pub workspace_size: usize,
    pub required_workspace_size: usize,
    pub input_specs: &'a [I],
    pub output_specs: &'a [O],
}

/// Static metadata for a QP-backed bytecode entry point.
#[derive(Debug, Clone, Copy)]
pub struct QpProgramInfo<'a, I: SpecInfo = InputSpec, O: SpecInfo = OutputSpec> {
    pub required_primal_workspace_size: usize,
    pub required_tangent_workspace_size: usize,
    pub input_specs: &'a [I],
    pub output_spec: &'a O,
}

/// Metadata for either a plain program or a QP program entry point.
#[derive(Debug, Clone, Copy)]
pub enum ExecutableInfo<'a, I: SpecInfo = InputSpec, O: SpecInfo = OutputSpec> {
    Program(ProgramInfo<'a, I, O>),
    QpProgram(QpProgramInfo<'a, I, O>),
}

/// Validating builder for an owned bytecode module on `std` targets.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct ModuleBuilder {
    bytecode_module: BytecodeModule,
}

#[cfg(feature = "std")]
impl ModuleBuilder {
    /// Validates an already-decoded module before storing it.
    pub fn new(bytecode_module: BytecodeModule) -> Result<Self, RuntimeError> {
        validate::validate_module_struct(&bytecode_module)?;
        Ok(Self { bytecode_module })
    }

    /// Decodes and validates a module from serialized bytes.
    pub fn new_from_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        Self::new(decode_module(bytes)?)
    }

    /// Finishes validation and returns an executable owned module wrapper.
    pub fn build(self) -> Result<Module, RuntimeError> {
        Ok(Module {
            bytecode_module: self.bytecode_module,
        })
    }
}

/// Owned, validated bytecode module for `std` callers that prefer a single handle.
#[derive(Debug)]
pub struct Module {
    bytecode_module: BytecodeModule,
}

impl Module {
    /// Returns the entry program metadata.
    pub fn info(&self) -> ProgramInfo<'_, InputSpec, OutputSpec> {
        program_info_from_program(self.entry_program())
    }

    /// Validates borrowed input slices against the entry program signature.
    pub fn validate_inputs(&self, inputs: &[&[f32]]) -> Result<(), RuntimeError> {
        validate::validate_inputs(self.entry_program(), inputs)
    }

    /// Validates borrowed output storage against the entry program signature.
    pub fn validate_outputs(&self, outputs: &mut [f32]) -> Result<(), RuntimeError> {
        validate::validate_outputs(self.entry_program(), outputs)
    }

    /// Validates primal inputs and tangents for push-forward execution.
    pub fn validate_push_forward_inputs(
        &self,
        inputs: &[&[f32]],
        tangents: &[&[f32]],
    ) -> Result<(), RuntimeError> {
        validate::validate_inputs(self.entry_program(), inputs)?;
        validate::validate_inputs(self.entry_program(), tangents)
    }

    /// Validates primal and tangent output storage for push-forward execution.
    pub fn validate_push_forward_outputs(
        &self,
        outputs: &mut [f32],
        tangent_outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        validate::validate_outputs(self.entry_program(), outputs)?;
        validate::validate_outputs(self.entry_program(), tangent_outputs)
    }

    /// Executes the entry program and writes its outputs into `outputs`.
    pub fn execute(
        &self,
        inputs: &[&[f32]],
        workspace: &mut [f32],
        outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        execute(&self.bytecode_module, inputs, workspace, outputs)
    }

    /// Executes the entry program push-forward and writes primal and tangent outputs.
    pub fn push_forward(
        &self,
        inputs: &[&[f32]],
        tangents: &[&[f32]],
        workspace: &mut [f32],
        tangent_workspace: &mut [f32],
        outputs: &mut [f32],
        tangent_outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        push_forward(
            &self.bytecode_module,
            inputs,
            tangents,
            workspace,
            tangent_workspace,
            outputs,
            tangent_outputs,
        )
    }

    fn entry_program(&self) -> &Program {
        entry_program(&self.bytecode_module)
            .expect("validated module missing referenced entry function")
    }
}

/// Returns metadata for the validated entry program in `module`.
pub fn program_info(module: &BytecodeModule) -> ProgramInfo<'_> {
    program_info_from_program(entry_program(module).expect("validated module missing entry"))
}

/// Decodes a serialized module and validates its structure and semantics.
pub fn validate_module(module_bytes: &[u8]) -> Result<BytecodeModule, RuntimeError> {
    let module = decode_module(module_bytes)?;
    validate::validate_module_struct(&module)?;
    Ok(module)
}

/// Executes the module entry program into a caller-provided workspace and output slice.
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
        workspace::write_outputs(&entry_program.output_specs, workspace, outputs);
    }
    Ok(())
}

/// Executes the module entry program push-forward using caller-provided workspaces and outputs.
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
        workspace::write_outputs(&entry_program.output_specs, workspace, outputs);
        workspace::write_outputs(
            &entry_program.output_specs,
            tangent_workspace,
            tangent_outputs,
        );
    }
    Ok(())
}

/// Executes the module entry program in-place, leaving results in the workspace layout.
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

/// Executes the module entry program push-forward in-place, leaving results in both workspaces.
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

#[allow(clippy::too_many_arguments)]
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

/// Builds [`ProgramInfo`] directly from a decoded bytecode program record.
pub fn program_info_from_program(program: &Program) -> ProgramInfo<'_> {
    ProgramInfo {
        workspace_size: program.workspace_size as usize,
        required_workspace_size: program.required_workspace_size as usize,
        input_specs: &program.input_specs,
        output_specs: &program.output_specs,
    }
}

/// Returns the validated entry program (`function_id == 0`) from a decoded module.
pub fn entry_program(module: &BytecodeModule) -> Result<&Program, RuntimeError> {
    find_function(module, 0)
        .ok_or_else(|| RuntimeError::Validation("missing entry function_id 0".to_string()))
}
pub(crate) fn find_function(module: &BytecodeModule, function_id: u16) -> Option<&Program> {
    module.program(function_id)
}

pub(crate) fn find_function_unchecked(module: &BytecodeModule, function_id: u16) -> &Program {
    find_function(module, function_id).expect("validated module missing referenced function")
}
