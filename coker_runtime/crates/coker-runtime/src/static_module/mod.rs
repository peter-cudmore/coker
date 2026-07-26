
use coker_bytecode::{
    archived_module, ArchivedBilinearLayer, ArchivedBytecodeModule, ArchivedEvaluateInputBinding,
    ArchivedEvaluateLayer, ArchivedGenericLayer, ArchivedInputSpec, ArchivedLayer,
    ArchivedOutputSpec, ArchivedProgram, ArchivedRowOp, ArchivedScalarOp, RowOp, ScalarOp,
};

use crate::{
    ops::{
        evaluate_generic_push_forward, evaluate_generic_value, homogeneous_tangent,
        homogeneous_value,
    },
    workspace::Workspace,
    ExecutableInfo, MappedQpProgram, ProgramInfo, RuntimeError, SpecInfo, UNUSED_OPERAND,
};

mod execute;
mod support;
mod validate;

#[allow(unused_imports)]
use self::execute::*;
#[allow(unused_imports)]
use self::support::*;
#[allow(unused_imports)]
use self::validate::*;

#[derive(Clone, Copy)]
/// Borrowed runtime view over mapped archived bytecode.
///
/// The mapped bytes must outlive the returned module; the type carries that
/// lifetime instead of cloning a `BytecodeModule`.
pub struct MappedModule<'a> {
    bytecode_module: &'a ArchivedBytecodeModule,
}

#[derive(Clone, Copy)]
pub struct MappedProgram<'a> {
    function_id: u16,
    bytecode_module: &'a ArchivedBytecodeModule,
    program: &'a ArchivedProgram,
}

#[derive(Clone, Copy)]
pub enum MappedExecutable<'a> {
    Program(MappedProgram<'a>),
    QpProgram(MappedQpProgram<'a>),
}

impl SpecInfo for ArchivedInputSpec {
    fn workspace_offset(&self) -> usize {
        us32(self.workspace_offset)
    }

    fn length(&self) -> usize {
        us16(self.length)
    }
}

impl SpecInfo for ArchivedOutputSpec {
    fn workspace_offset(&self) -> usize {
        us32(self.workspace_offset)
    }

    fn length(&self) -> usize {
        us16(self.length)
    }
}

impl<'a> MappedModule<'a> {
    pub fn new(bytecode_module: &'a ArchivedBytecodeModule) -> Result<Self, RuntimeError> {
        validate_module_struct(bytecode_module)?;
        Ok(Self { bytecode_module })
    }

    pub(crate) fn from_archived_unchecked(bytecode_module: &'a ArchivedBytecodeModule) -> Self {
        Self { bytecode_module }
    }

    pub fn new_from_bytes(bytes: &'a [u8]) -> Result<Self, RuntimeError> {
        Self::new(archived_module(bytes)?)
    }

    pub fn new_with_workspace_capacities(
        bytecode_module: &'a ArchivedBytecodeModule,
        workspace_capacity: usize,
        tangent_workspace_capacity: usize,
    ) -> Result<Self, RuntimeError> {
        let module = Self::new(bytecode_module)?;
        module
            .entry_program()
            .validate_workspace_capacities(workspace_capacity, tangent_workspace_capacity)?;
        Ok(module)
    }

    pub fn new_with_workspace_capacity(
        bytecode_module: &'a ArchivedBytecodeModule,
        workspace_capacity: usize,
        tangent_workspace_capacity: usize,
    ) -> Result<Self, RuntimeError> {
        Self::new_with_workspace_capacities(
            bytecode_module,
            workspace_capacity,
            tangent_workspace_capacity,
        )
    }

    pub fn new_from_bytes_with_workspace_capacities(
        bytes: &'a [u8],
        workspace_capacity: usize,
        tangent_workspace_capacity: usize,
    ) -> Result<Self, RuntimeError> {
        Self::new_with_workspace_capacities(
            archived_module(bytes)?,
            workspace_capacity,
            tangent_workspace_capacity,
        )
    }

    pub fn new_from_bytes_with_workspace_capacity(
        bytes: &'a [u8],
        workspace_capacity: usize,
        tangent_workspace_capacity: usize,
    ) -> Result<Self, RuntimeError> {
        Self::new_from_bytes_with_workspace_capacities(
            bytes,
            workspace_capacity,
            tangent_workspace_capacity,
        )
    }

    pub fn info(&self) -> ProgramInfo<'_, ArchivedInputSpec, ArchivedOutputSpec> {
        program_info_from_program(self.entry_program().program())
    }

    pub fn program(&self, function_id: u16) -> Result<MappedProgram<'a>, RuntimeError> {
        let program =
            find_function(self.bytecode_module, function_id).ok_or(RuntimeError::MissingFunction {
                function_id,
            })?;
        Ok(MappedProgram {
            function_id,
            bytecode_module: self.bytecode_module,
            program,
        })
    }

    pub fn executable(&self, function_id: u16) -> Result<MappedExecutable<'a>, RuntimeError> {
        if let Some(program) = find_function(self.bytecode_module, function_id) {
            return Ok(MappedExecutable::Program(MappedProgram {
                function_id,
                bytecode_module: self.bytecode_module,
                program,
            }));
        }
        if self.bytecode_module.qp_program(function_id).is_some() {
            return Ok(MappedExecutable::QpProgram(MappedQpProgram::new(
                *self,
                function_id,
            )?));
        }
        Err(RuntimeError::MissingExecutable { function_id })
    }

    pub fn execute(
        &self,
        inputs: &[&[f32]],
        workspace: &mut [f32],
        outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        self.entry_program().execute(inputs, workspace, outputs)
    }

    pub fn push_forward(
        &self,
        inputs: &[&[f32]],
        tangents: &[&[f32]],
        workspace: &mut [f32],
        tangent_workspace: &mut [f32],
        outputs: &mut [f32],
        tangent_outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        self.entry_program().push_forward(
            inputs,
            tangents,
            workspace,
            tangent_workspace,
            outputs,
            tangent_outputs,
        )
    }

    fn entry_program(&self) -> MappedProgram<'a> {
        self.program(0)
            .expect("validated module missing referenced entry function")
    }

    pub(crate) fn bytecode_module(&self) -> &'a ArchivedBytecodeModule {
        self.bytecode_module
    }
}

impl<'a> MappedProgram<'a> {
    pub fn function_id(&self) -> u16 {
        self.function_id
    }

    pub fn info(&self) -> ProgramInfo<'_, ArchivedInputSpec, ArchivedOutputSpec> {
        program_info_from_program(self.program)
    }

    pub fn execute(
        &self,
        inputs: &[&[f32]],
        workspace: &mut [f32],
        outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        validate_inputs(self.program, inputs)?;
        validate_workspace(self.program, workspace)?;
        validate_outputs(self.program, outputs)?;
        let wrote_direct_outputs = execute_in_place_unchecked(
            self.bytecode_module,
            self.program,
            inputs,
            workspace,
            Some(outputs),
        );
        if !wrote_direct_outputs {
            crate::workspace::write_outputs(self.program.output_specs(), workspace, outputs);
        }
        Ok(())
    }

    pub fn push_forward(
        &self,
        inputs: &[&[f32]],
        tangents: &[&[f32]],
        workspace: &mut [f32],
        tangent_workspace: &mut [f32],
        outputs: &mut [f32],
        tangent_outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        validate_inputs(self.program, inputs)?;
        validate_inputs(self.program, tangents)?;
        validate_workspace(self.program, workspace)?;
        validate_workspace(self.program, tangent_workspace)?;
        validate_outputs(self.program, outputs)?;
        validate_outputs(self.program, tangent_outputs)?;
        let wrote_direct_outputs = push_forward_in_place_unchecked(
            self.bytecode_module,
            self.program,
            inputs,
            tangents,
            workspace,
            tangent_workspace,
            Some(outputs),
            Some(tangent_outputs),
        );
        if !wrote_direct_outputs {
            crate::workspace::write_outputs(self.program.output_specs(), workspace, outputs);
            crate::workspace::write_outputs(
                self.program.output_specs(),
                tangent_workspace,
                tangent_outputs,
            );
        }
        Ok(())
    }

    fn validate_workspace_capacities(
        &self,
        workspace_capacity: usize,
        tangent_workspace_capacity: usize,
    ) -> Result<(), RuntimeError> {
        let required_workspace_size = us32(self.program.required_workspace_size);
        if workspace_capacity < required_workspace_size {
            return Err(RuntimeError::WorkspaceTooSmall {
                expected: required_workspace_size,
                actual: workspace_capacity,
            });
        }
        if tangent_workspace_capacity < required_workspace_size {
            return Err(RuntimeError::WorkspaceTooSmall {
                expected: required_workspace_size,
                actual: tangent_workspace_capacity,
            });
        }
        Ok(())
    }

    pub(crate) fn program(&self) -> &'a ArchivedProgram {
        self.program
    }
}

impl<'a> MappedExecutable<'a> {
    pub fn function_id(&self) -> u16 {
        match self {
            Self::Program(program) => program.function_id(),
            Self::QpProgram(program) => program.function_id(),
        }
    }

    pub fn info(&self) -> ExecutableInfo<'_, ArchivedInputSpec, ArchivedOutputSpec> {
        match self {
            Self::Program(program) => ExecutableInfo::Program(program.info()),
            Self::QpProgram(program) => ExecutableInfo::QpProgram(program.info()),
        }
    }
}
