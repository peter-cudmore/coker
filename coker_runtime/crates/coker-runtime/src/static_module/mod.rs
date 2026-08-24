use crate::{
    ops::{
        evaluate_generic_push_forward, evaluate_generic_value, homogeneous_tangent,
        homogeneous_value,
    },
    workspace::Workspace,
    ExecutableInfo, MappedQpProgram, ProgramInfo, RuntimeError, SpecInfo, UNUSED_OPERAND,
};
#[cfg(osqp_embedded)]
use crate::{MappedQpWorkspace, PreparedQpProgram};
use coker_bytecode::{
    archived_module, ArchivedBilinearLayer, ArchivedBytecodeModule, ArchivedEvaluateInputBinding,
    ArchivedEvaluateLayer, ArchivedGenericLayer, ArchivedInputSpec, ArchivedLayer,
    ArchivedOutputSpec, ArchivedProgram, ArchivedRowOp, ArchivedScalarOp, RowOp, ScalarOp,
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
/// Mutable, caller-owned buffers and prepared solver state for one [`QpCall`](coker_bytecode::QpCallLayer).
///
/// `prepared` must have been created from the same mapped QP executable as
/// `qp_function_id`. Its OSQP arena remains owned by the embedding application
/// through [`MappedQpProgram::prepare_detached`].
#[cfg(osqp_embedded)]
pub struct QpCallContext<'a> {
    pub call_slot: u16,
    pub qp_function_id: u16,
    pub prepared: &'a mut PreparedQpProgram,
    pub evaluator_workspace: &'a mut [f32],
    pub coefficient_outputs: &'a mut [f32],
    pub parameters: &'a mut [f32],
    pub primal_solution: &'a mut [f32],
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
        let program = find_function(self.bytecode_module, function_id)
            .ok_or(RuntimeError::MissingFunction { function_id })?;
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
    /// Executes the entry program using one prepared, caller-owned context for every QP call layer.
    #[cfg(osqp_embedded)]
    pub fn execute_with_qp_contexts(
        &self,
        inputs: &[&[f32]],
        workspace: &mut [f32],
        outputs: &mut [f32],
        contexts: &mut [QpCallContext<'_>],
    ) -> Result<(), RuntimeError> {
        self.entry_program()
            .execute_with_qp_contexts(inputs, workspace, outputs, contexts)
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

    pub(crate) fn execute_flat_inputs(
        &self,
        inputs: &[f32],
        workspace: &mut [f32],
        outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        validate_workspace(self.program, workspace)?;
        validate_outputs(self.program, outputs)?;
        let expected = self
            .program
            .input_specs
            .iter()
            .map(|spec| us16(spec.length))
            .sum::<usize>();
        if inputs.len() != expected {
            return Err(RuntimeError::Validation(
                "flat QP parameter buffer does not match evaluator inputs",
            ));
        }
        let mut workspace_view = Workspace::new(workspace);
        workspace_view.fill(0.0);
        let mut source_start = 0;
        for spec in self.program.input_specs.iter() {
            let source_stop = source_start + us16(spec.length);
            let destination_start = us32(spec.workspace_offset);
            let destination_stop = destination_start + us16(spec.length);
            workspace_view.as_mut_slice()[destination_start..destination_stop]
                .copy_from_slice(&inputs[source_start..source_stop]);
            source_start = source_stop;
        }
        execute_program_layers(
            self.bytecode_module,
            self.program,
            &mut workspace_view,
            None,
        );
        crate::workspace::write_outputs(
            self.program.output_specs(),
            workspace_view.as_slice(),
            outputs,
        );
        Ok(())
    }

    #[cfg(osqp_embedded)]
    fn execute_with_qp_contexts(
        &self,
        inputs: &[&[f32]],
        workspace: &mut [f32],
        outputs: &mut [f32],
        contexts: &mut [QpCallContext<'_>],
    ) -> Result<(), RuntimeError> {
        validate_inputs(self.program, inputs)?;
        validate_workspace(self.program, workspace)?;
        validate_outputs(self.program, outputs)?;
        for layer in self.program.intermediate_layers.iter() {
            let ArchivedLayer::QpCall(call) = layer else {
                continue;
            };
            let mut matches = 0;
            for context in contexts.iter() {
                if context.call_slot == u16n(call.call_slot) {
                    matches += 1;
                    if context.qp_function_id != u16n(call.qp_function_id) {
                        return Err(RuntimeError::Validation(
                            "QP call context function id does not match call layer",
                        ));
                    }
                }
            }
            if matches != 1 {
                return Err(RuntimeError::Validation(
                    "QP call layer requires exactly one matching context",
                ));
            }
        }
        for (index, left) in contexts.iter().enumerate() {
            if contexts[index + 1..]
                .iter()
                .any(|right| left.call_slot == right.call_slot)
            {
                return Err(RuntimeError::Validation(
                    "duplicate QP call context binding",
                ));
            }
        }
        let mut workspace_view = Workspace::new(workspace);
        workspace_view.fill(0.0);
        workspace_view.pack_inputs(self.program.input_specs(), inputs);
        for layer in self.program.intermediate_layers.iter() {
            match layer {
                ArchivedLayer::Bilinear(layer) => {
                    execute::execute_bilinear_layer(layer, &mut workspace_view)
                }
                ArchivedLayer::Generic(layer) => {
                    execute::execute_generic_layer(layer, &mut workspace_view)
                }
                ArchivedLayer::Evaluate(layer) => execute::execute_evaluate_layer(
                    self.bytecode_module,
                    layer,
                    &mut workspace_view,
                ),
                ArchivedLayer::QpCall(call) => {
                    let context = contexts
                        .iter_mut()
                        .find(|context| context.call_slot == u16n(call.call_slot))
                        .ok_or(RuntimeError::Validation("missing QP call context"))?;
                    let mut parameter_start = 0;
                    for binding in call.input_bindings.iter() {
                        let length = match binding {
                            ArchivedEvaluateInputBinding::WorkspaceSlice { length, .. }
                            | ArchivedEvaluateInputBinding::ConstantSlice { length, .. } => {
                                us16(*length)
                            }
                        };
                        let parameter_stop = parameter_start + length;
                        match binding {
                            ArchivedEvaluateInputBinding::WorkspaceSlice { offset, .. } => {
                                let source = us32(*offset);
                                context.parameters[parameter_start..parameter_stop]
                                    .copy_from_slice(
                                        &workspace_view.as_slice()[source..source + length],
                                    );
                            }
                            ArchivedEvaluateInputBinding::ConstantSlice { values, .. } => {
                                for (destination, value) in context.parameters
                                    [parameter_start..parameter_stop]
                                    .iter_mut()
                                    .zip(values.iter())
                                {
                                    *destination = value.to_native();
                                }
                            }
                        }
                        parameter_start = parameter_stop;
                    }
                    let qp = MappedQpProgram::new(
                        MappedModule::from_archived_unchecked(self.bytecode_module),
                        u16n(call.qp_function_id),
                    )?;
                    if context.parameters.len() != parameter_start {
                        return Err(RuntimeError::Validation(
                            "flat QP parameter buffer width does not match call bindings",
                        ));
                    }
                    context.prepared.execute_flat(
                        qp,
                        context.parameters,
                        MappedQpWorkspace::new(
                            context.evaluator_workspace,
                            context.coefficient_outputs,
                        ),
                        context.primal_solution,
                    )?;
                    let output_start = us32(call.output_binding.destination_offset);
                    let output_length = us16(call.output_binding.length);
                    workspace_view.as_mut_slice()[output_start..output_start + output_length]
                        .copy_from_slice(&context.primal_solution[..output_length]);
                }
            }
        }
        crate::workspace::write_outputs(
            self.program.output_specs(),
            workspace_view.as_slice(),
            outputs,
        );
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
