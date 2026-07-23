use alloc::{format, string::ToString, vec::Vec};
use coker_bytecode::{
    archived_module, ArchivedBilinearLayer, ArchivedBytecodeModule, ArchivedEvaluateInputBinding,
    ArchivedEvaluateLayer, ArchivedGenericLayer, ArchivedLayer, ArchivedProgram, ArchivedRowOp,
    ArchivedScalarOp, InputSpec, OutputSpec, RowOp, ScalarOp,
};

use crate::{
    ops::{
        evaluate_generic_push_forward, evaluate_generic_value, homogeneous_tangent,
        homogeneous_value,
    },
    workspace::Workspace,
    ProgramInfo, RuntimeError, UNUSED_OPERAND,
};

#[derive(Clone, Copy)]
pub struct StaticModule<'a> {
    bytecode_module: &'a ArchivedBytecodeModule,
}

impl<'a> StaticModule<'a> {
    pub fn new(bytecode_module: &'a ArchivedBytecodeModule) -> Result<Self, RuntimeError> {
        validate_module_struct(bytecode_module)?;
        Ok(Self { bytecode_module })
    }

    pub fn new_from_bytes(bytes: &'a [u8]) -> Result<Self, RuntimeError> {
        Self::new(archived_module(bytes)?)
    }

    pub fn info(&self) -> ProgramInfo {
        program_info_from_program(self.entry_program())
    }

    pub fn execute(
        &self,
        inputs: &[&[f32]],
        workspace: &mut [f32],
        outputs: &mut [f32],
    ) -> Result<(), RuntimeError> {
        let entry_program = self.entry_program();
        validate_inputs(entry_program, inputs)?;
        validate_workspace(entry_program, workspace)?;
        validate_outputs(entry_program, outputs)?;
        let wrote_direct_outputs = execute_in_place_unchecked(
            self.bytecode_module,
            entry_program,
            inputs,
            workspace,
            Some(outputs),
        );
        if !wrote_direct_outputs {
            write_outputs(entry_program, workspace, outputs);
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
        let entry_program = self.entry_program();
        validate_inputs(entry_program, inputs)?;
        validate_inputs(entry_program, tangents)?;
        validate_workspace(entry_program, workspace)?;
        validate_workspace(entry_program, tangent_workspace)?;
        validate_outputs(entry_program, outputs)?;
        validate_outputs(entry_program, tangent_outputs)?;
        let wrote_direct_outputs = push_forward_in_place_unchecked(
            self.bytecode_module,
            entry_program,
            inputs,
            tangents,
            workspace,
            tangent_workspace,
            Some(outputs),
            Some(tangent_outputs),
        );
        if !wrote_direct_outputs {
            write_outputs(entry_program, workspace, outputs);
            write_outputs(entry_program, tangent_workspace, tangent_outputs);
        }
        Ok(())
    }

    fn entry_program(&self) -> &'a ArchivedProgram {
        entry_program(self.bytecode_module)
            .expect("validated module missing referenced entry function")
    }
}

fn validate_module_struct(module: &ArchivedBytecodeModule) -> Result<(), RuntimeError> {
    if module.functions.is_empty() {
        return Err(RuntimeError::Validation(
            "bytecode module must contain at least one function".to_string(),
        ));
    }
    entry_program(module)?;

    for (function_index, function_program) in module.functions.iter().enumerate() {
        let function_id = u16n(function_program.function_id);
        if module
            .functions
            .iter()
            .take(function_index)
            .any(|prior_program| u16n(prior_program.function_id) == function_id)
        {
            return Err(RuntimeError::Validation(
                "duplicate function id".to_string(),
            ));
        }
        validate_program_struct(module, function_program)?;
    }
    Ok(())
}

fn validate_program_struct(
    module: &ArchivedBytecodeModule,
    program: &ArchivedProgram,
) -> Result<(), RuntimeError> {
    let workspace_size = us32(program.workspace_size);
    let required_workspace_size = us32(program.required_workspace_size);
    if required_workspace_size < workspace_size {
        return Err(RuntimeError::Validation(
            "required workspace smaller than primary workspace".to_string(),
        ));
    }

    for input_spec in program.input_specs.iter() {
        validate_range(
            u32n(input_spec.workspace_offset),
            u16n(input_spec.length),
            workspace_size,
            "input",
        )?;
    }
    for output_spec in program.output_specs.iter() {
        validate_range(
            u32n(output_spec.workspace_offset),
            u16n(output_spec.length),
            workspace_size,
            "output",
        )?;
    }
    for layer in program.intermediate_layers.iter() {
        match layer {
            ArchivedLayer::Bilinear(bilinear_layer) => {
                validate_bilinear_layer(bilinear_layer, workspace_size, required_workspace_size)?
            }
            ArchivedLayer::Generic(generic_layer) => {
                validate_generic_layer(generic_layer, workspace_size, required_workspace_size)?
            }
            ArchivedLayer::Evaluate(evaluate_layer) => {
                validate_evaluate_layer(module, evaluate_layer, program, workspace_size)?
            }
        }
    }
    Ok(())
}

fn validate_bilinear_layer(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace_size: usize,
    required_workspace_size: usize,
) -> Result<(), RuntimeError> {
    validate_range(
        u32n(bilinear_layer.in_offset),
        u16n(bilinear_layer.in_length),
        workspace_size,
        "bilinear input",
    )?;
    validate_range(
        u32n(bilinear_layer.out_offset),
        u16n(bilinear_layer.out_length),
        workspace_size,
        "bilinear output",
    )?;
    validate_layer_scratch(
        u32n(bilinear_layer.in_offset),
        u16n(bilinear_layer.in_length),
        u32n(bilinear_layer.out_offset),
        u16n(bilinear_layer.out_length),
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
        workspace_size,
        required_workspace_size,
        "bilinear layer",
    )?;

    let in_length = u16n(bilinear_layer.in_length);
    let out_length = u16n(bilinear_layer.out_length);
    let expected_shape = (
        out_length,
        in_length
            .checked_add(1)
            .ok_or_else(|| RuntimeError::Validation("bilinear input too large".to_string()))?,
        in_length
            .checked_add(1)
            .ok_or_else(|| RuntimeError::Validation("bilinear input too large".to_string()))?,
    );
    let shape = (
        u16n(bilinear_layer.quadratic.shape.0),
        u16n(bilinear_layer.quadratic.shape.1),
        u16n(bilinear_layer.quadratic.shape.2),
    );
    if shape != expected_shape {
        return Err(RuntimeError::Validation(
            "bilinear tensor shape does not match layer dimensions".to_string(),
        ));
    }

    for entry in bilinear_layer.quadratic.entries.iter() {
        let row_index = u16n(entry.index.0);
        let left_index = u16n(entry.index.1);
        let right_index = u16n(entry.index.2);
        if row_index >= expected_shape.0 {
            return Err(RuntimeError::Validation(
                "bilinear tensor row index out of bounds".to_string(),
            ));
        }
        if left_index >= expected_shape.1 {
            return Err(RuntimeError::Validation(
                "bilinear tensor left index out of bounds".to_string(),
            ));
        }
        if right_index >= expected_shape.2 {
            return Err(RuntimeError::Validation(
                "bilinear tensor right index out of bounds".to_string(),
            ));
        }
    }

    Ok(())
}

fn validate_generic_layer(
    generic_layer: &ArchivedGenericLayer,
    workspace_size: usize,
    required_workspace_size: usize,
) -> Result<(), RuntimeError> {
    validate_range(
        u32n(generic_layer.in_offset),
        u16n(generic_layer.in_length),
        workspace_size,
        "generic input",
    )?;
    validate_range(
        u32n(generic_layer.out_offset),
        u16n(generic_layer.out_length),
        workspace_size,
        "generic output",
    )?;
    if generic_layer.ops.len() != us16(generic_layer.out_length) {
        return Err(RuntimeError::Validation(
            "generic layer op count must match output length".to_string(),
        ));
    }
    validate_layer_scratch(
        u32n(generic_layer.in_offset),
        u16n(generic_layer.in_length),
        u32n(generic_layer.out_offset),
        u16n(generic_layer.out_length),
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
        workspace_size,
        required_workspace_size,
        "generic layer",
    )?;

    let input_length = u16n(generic_layer.in_length);
    for row_operation in generic_layer.ops.iter() {
        validate_generic_operand(u16n(row_operation.first), input_length)?;
        validate_generic_operand(u16n(row_operation.second), input_length)?;
        validate_generic_operand(u16n(row_operation.third), input_length)?;
        validate_generic_row_operation(row_operation)?;
    }

    Ok(())
}

fn validate_evaluate_layer(
    module: &ArchivedBytecodeModule,
    evaluate_layer: &ArchivedEvaluateLayer,
    caller_program: &ArchivedProgram,
    caller_workspace_size: usize,
) -> Result<(), RuntimeError> {
    let callee_program = find_function(module, u16n(evaluate_layer.callee_function_id))
        .ok_or_else(|| {
            RuntimeError::Validation("evaluate callee function id missing".to_string())
        })?;

    if evaluate_layer.input_bindings.len() != callee_program.input_specs.len() {
        return Err(RuntimeError::Validation(
            "evaluate input binding count does not match callee inputs".to_string(),
        ));
    }
    if evaluate_layer.output_bindings.len() != callee_program.output_specs.len() {
        return Err(RuntimeError::Validation(
            "evaluate output binding count does not match callee outputs".to_string(),
        ));
    }
    if us32(evaluate_layer.scratch_offset) < caller_workspace_size {
        return Err(RuntimeError::Validation(
            "evaluate scratch offset overlaps caller workspace".to_string(),
        ));
    }

    let scratch_end =
        us32(evaluate_layer.scratch_offset) + us32(callee_program.required_workspace_size);
    if scratch_end > us32(caller_program.required_workspace_size) {
        return Err(RuntimeError::Validation(
            "evaluate scratch range exceeds caller required workspace".to_string(),
        ));
    }

    for (binding, input_spec) in evaluate_layer
        .input_bindings
        .iter()
        .zip(callee_program.input_specs.iter())
    {
        validate_evaluate_input_binding(binding, u16n(input_spec.length), caller_workspace_size)?;
    }
    for (binding, output_spec) in evaluate_layer
        .output_bindings
        .iter()
        .zip(callee_program.output_specs.iter())
    {
        if u16n(binding.length) != u16n(output_spec.length) {
            return Err(RuntimeError::Validation(
                "evaluate output binding length mismatch".to_string(),
            ));
        }
        validate_range(
            u32n(binding.destination_offset),
            u16n(binding.length),
            caller_workspace_size,
            "evaluate output",
        )?;
    }

    Ok(())
}

fn validate_evaluate_input_binding(
    binding: &ArchivedEvaluateInputBinding,
    expected_length: u16,
    caller_workspace_size: usize,
) -> Result<(), RuntimeError> {
    match binding {
        ArchivedEvaluateInputBinding::WorkspaceSlice { offset, length } => {
            let length = u16n(*length);
            if length != expected_length {
                return Err(RuntimeError::Validation(
                    "evaluate input binding length mismatch".to_string(),
                ));
            }
            validate_range(
                u32n(*offset),
                length,
                caller_workspace_size,
                "evaluate input",
            )
        }
        ArchivedEvaluateInputBinding::ConstantSlice { length, values } => {
            let length = u16n(*length);
            if length != expected_length || values.len() != length as usize {
                return Err(RuntimeError::Validation(
                    "evaluate constant input length mismatch".to_string(),
                ));
            }
            Ok(())
        }
    }
}

fn validate_generic_operand(operand_index: u16, input_length: u16) -> Result<(), RuntimeError> {
    if operand_index != UNUSED_OPERAND && operand_index >= input_length {
        return Err(RuntimeError::Validation(
            "generic operand index out of bounds".to_string(),
        ));
    }
    Ok(())
}

fn validate_generic_row_operation(row_operation: &ArchivedRowOp) -> Result<(), RuntimeError> {
    let operand_indices = [
        u16n(row_operation.first),
        u16n(row_operation.second),
        u16n(row_operation.third),
    ];
    for required_index in 0..required_operand_count(&row_operation.op) as usize {
        if operand_indices[required_index] == UNUSED_OPERAND {
            return Err(RuntimeError::Validation(
                "generic operation missing required operand".to_string(),
            ));
        }
    }
    Ok(())
}

fn required_operand_count(operation: &ArchivedScalarOp) -> u8 {
    match operation {
        ArchivedScalarOp::Identity
        | ArchivedScalarOp::Sin
        | ArchivedScalarOp::Cos
        | ArchivedScalarOp::Tan
        | ArchivedScalarOp::Exp
        | ArchivedScalarOp::Sqrt
        | ArchivedScalarOp::Log
        | ArchivedScalarOp::Neg
        | ArchivedScalarOp::Abs => 1,
        ArchivedScalarOp::Add
        | ArchivedScalarOp::Sub
        | ArchivedScalarOp::Mul
        | ArchivedScalarOp::Div
        | ArchivedScalarOp::Pow
        | ArchivedScalarOp::IntPow
        | ArchivedScalarOp::Atan2
        | ArchivedScalarOp::Equal
        | ArchivedScalarOp::LessThan
        | ArchivedScalarOp::LessEqual => 2,
        ArchivedScalarOp::Case => 3,
    }
}

fn validate_layer_scratch(
    input_offset: u32,
    input_length: u16,
    output_offset: u32,
    output_length: u16,
    scratch_offset: u32,
    scratch_length: u16,
    workspace_size: usize,
    required_workspace_size: usize,
    context: &str,
) -> Result<(), RuntimeError> {
    let ranges_overlap = range_end(input_offset, input_length) > output_offset as usize
        && range_end(output_offset, output_length) > input_offset as usize;
    if !ranges_overlap {
        if scratch_length != 0 {
            return Err(RuntimeError::Validation(format!(
                "{context} scratch storage must be zero when ranges are disjoint"
            )));
        }
        return Ok(());
    }

    if scratch_length != input_length {
        return Err(RuntimeError::Validation(format!(
            "{context} scratch length must match input length"
        )));
    }
    if (scratch_offset as usize) < workspace_size {
        return Err(RuntimeError::Validation(format!(
            "{context} scratch storage overlaps primary workspace"
        )));
    }
    validate_range(
        scratch_offset,
        scratch_length,
        required_workspace_size,
        "layer scratch",
    )
}

fn range_end(workspace_offset: u32, length: u16) -> usize {
    workspace_offset as usize + length as usize
}

fn validate_range(
    workspace_offset: u32,
    length: u16,
    workspace_size: usize,
    context: &str,
) -> Result<(), RuntimeError> {
    let end = workspace_offset as usize + length as usize;
    if end > workspace_size {
        return Err(RuntimeError::Validation(format!(
            "{context} range exceeds workspace"
        )));
    }
    Ok(())
}

fn validate_inputs(program: &ArchivedProgram, inputs: &[&[f32]]) -> Result<(), RuntimeError> {
    if inputs.len() != program.input_specs.len() {
        return Err(RuntimeError::InputCountMismatch {
            expected: program.input_specs.len(),
            actual: inputs.len(),
        });
    }

    for (index, (input_spec, input_value)) in
        program.input_specs.iter().zip(inputs.iter()).enumerate()
    {
        let expected_count = us16(input_spec.length);
        let actual_count = input_value.len();
        if expected_count != actual_count {
            return Err(RuntimeError::InputSizeMismatch {
                index,
                expected: expected_count,
                actual: actual_count,
            });
        }
    }
    Ok(())
}

fn validate_outputs(program: &ArchivedProgram, outputs: &[f32]) -> Result<(), RuntimeError> {
    let expected_size: usize = program
        .output_specs
        .iter()
        .map(|output_spec| us16(output_spec.length))
        .sum();
    let actual_size = outputs.len();
    if actual_size != expected_size {
        return Err(RuntimeError::OutputBufferSizeMismatch {
            expected: expected_size,
            actual: actual_size,
        });
    }
    Ok(())
}

fn validate_workspace(program: &ArchivedProgram, workspace: &[f32]) -> Result<(), RuntimeError> {
    validate_workspace_size(us32(program.required_workspace_size), workspace.len())
}

fn validate_workspace_size(expected_size: usize, actual_size: usize) -> Result<(), RuntimeError> {
    if actual_size < expected_size {
        return Err(RuntimeError::WorkspaceTooSmall {
            expected: expected_size,
            actual: actual_size,
        });
    }
    Ok(())
}

fn execute_in_place_unchecked(
    module: &ArchivedBytecodeModule,
    entry_program: &ArchivedProgram,
    inputs: &[&[f32]],
    workspace: &mut [f32],
    outputs: Option<&mut [f32]>,
) -> bool {
    let mut workspace = Workspace::new(workspace);
    workspace.fill(0.0);
    pack_inputs(&mut workspace, entry_program, inputs);
    execute_program_layers(module, entry_program, &mut workspace, outputs)
}

fn push_forward_in_place_unchecked(
    module: &ArchivedBytecodeModule,
    entry_program: &ArchivedProgram,
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
    pack_inputs(&mut workspace, entry_program, inputs);
    pack_inputs(&mut tangent_workspace, entry_program, tangents);
    push_forward_program_layers(
        module,
        entry_program,
        &mut workspace,
        &mut tangent_workspace,
        outputs,
        tangent_outputs,
    )
}

fn execute_program_layers(
    module: &ArchivedBytecodeModule,
    program: &ArchivedProgram,
    workspace: &mut Workspace<'_>,
    final_outputs: Option<&mut [f32]>,
) -> bool {
    let last_layer_index = program.intermediate_layers.len().saturating_sub(1);
    let mut final_outputs = final_outputs;
    for (layer_index, layer) in program.intermediate_layers.iter().enumerate() {
        let is_final_layer = layer_index == last_layer_index;
        let wrote_direct_outputs = if is_final_layer {
            match layer {
                ArchivedLayer::Bilinear(bilinear_layer) => final_outputs
                    .as_deref_mut()
                    .filter(|_| {
                        final_layer_matches_outputs(
                            program,
                            u32n(bilinear_layer.out_offset),
                            u16n(bilinear_layer.out_length),
                        )
                    })
                    .map(|output_buffer| {
                        execute_bilinear_layer_to_output_buffer(
                            bilinear_layer,
                            workspace,
                            output_buffer,
                        );
                    })
                    .is_some(),
                ArchivedLayer::Generic(generic_layer) => final_outputs
                    .as_deref_mut()
                    .filter(|_| {
                        final_layer_matches_outputs(
                            program,
                            u32n(generic_layer.out_offset),
                            u16n(generic_layer.out_length),
                        )
                    })
                    .map(|output_buffer| {
                        execute_generic_layer_to_output_buffer(
                            generic_layer,
                            workspace,
                            output_buffer,
                        );
                    })
                    .is_some(),
                ArchivedLayer::Evaluate(_) => false,
            }
        } else {
            false
        };

        if wrote_direct_outputs {
            return true;
        }

        match layer {
            ArchivedLayer::Bilinear(bilinear_layer) => {
                execute_bilinear_layer(bilinear_layer, workspace)
            }
            ArchivedLayer::Generic(generic_layer) => {
                execute_generic_layer(generic_layer, workspace)
            }
            ArchivedLayer::Evaluate(evaluate_layer) => {
                execute_evaluate_layer(module, evaluate_layer, workspace)
            }
        }
    }
    false
}

fn push_forward_program_layers(
    module: &ArchivedBytecodeModule,
    program: &ArchivedProgram,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
    final_outputs: Option<&mut [f32]>,
    final_tangent_outputs: Option<&mut [f32]>,
) -> bool {
    let last_layer_index = program.intermediate_layers.len().saturating_sub(1);
    let mut final_outputs = final_outputs;
    let mut final_tangent_outputs = final_tangent_outputs;
    for (layer_index, layer) in program.intermediate_layers.iter().enumerate() {
        let is_final_layer = layer_index == last_layer_index;
        let wrote_direct_outputs = if is_final_layer {
            match layer {
                ArchivedLayer::Bilinear(bilinear_layer) => match (
                    final_outputs.as_deref_mut(),
                    final_tangent_outputs.as_deref_mut(),
                ) {
                    (Some(output_buffer), Some(tangent_output_buffer))
                        if final_layer_matches_outputs(
                            program,
                            u32n(bilinear_layer.out_offset),
                            u16n(bilinear_layer.out_length),
                        ) =>
                    {
                        execute_bilinear_push_forward_to_output_buffer(
                            bilinear_layer,
                            workspace,
                            tangent_workspace,
                            output_buffer,
                            tangent_output_buffer,
                        );
                        true
                    }
                    _ => false,
                },
                ArchivedLayer::Generic(generic_layer) => match (
                    final_outputs.as_deref_mut(),
                    final_tangent_outputs.as_deref_mut(),
                ) {
                    (Some(output_buffer), Some(tangent_output_buffer))
                        if final_layer_matches_outputs(
                            program,
                            u32n(generic_layer.out_offset),
                            u16n(generic_layer.out_length),
                        ) =>
                    {
                        execute_generic_push_forward_to_output_buffer(
                            generic_layer,
                            workspace,
                            tangent_workspace,
                            output_buffer,
                            tangent_output_buffer,
                        );
                        true
                    }
                    _ => false,
                },
                ArchivedLayer::Evaluate(_) => false,
            }
        } else {
            false
        };

        if wrote_direct_outputs {
            return true;
        }

        match layer {
            ArchivedLayer::Bilinear(bilinear_layer) => {
                execute_bilinear_push_forward(bilinear_layer, workspace, tangent_workspace)
            }
            ArchivedLayer::Generic(generic_layer) => {
                execute_generic_push_forward(generic_layer, workspace, tangent_workspace)
            }
            ArchivedLayer::Evaluate(evaluate_layer) => {
                execute_evaluate_push_forward(module, evaluate_layer, workspace, tangent_workspace)
            }
        }
    }
    false
}

fn prepare_input_range(
    workspace: &mut Workspace<'_>,
    input_start: usize,
    input_stop: usize,
    scratch_offset: u32,
    scratch_length: u16,
) -> (usize, usize) {
    if scratch_length == 0 {
        return (input_start, input_stop);
    }

    let scratch_start = scratch_offset as usize;
    workspace.copy_range_to_scratch(input_start, input_stop, scratch_start);
    (scratch_start, scratch_start + scratch_length as usize)
}

fn execute_bilinear_layer(bilinear_layer: &ArchivedBilinearLayer, workspace: &mut Workspace<'_>) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let output_start = us32(bilinear_layer.out_offset);
    let output_stop = output_start + us16(bilinear_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let (input_slice, output_slice) = workspace.input_output_slices(
        prepared_input_range.0,
        prepared_input_range.1,
        output_start,
        output_stop,
    );
    execute_bilinear_into_slice(bilinear_layer, input_slice, output_slice);
}

fn execute_bilinear_layer_to_output_buffer(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    execute_bilinear_into_slice(bilinear_layer, input_slice, output_buffer);
}

fn execute_bilinear_into_slice(
    bilinear_layer: &ArchivedBilinearLayer,
    input_slice: &[f32],
    output_slice: &mut [f32],
) {
    output_slice.fill(0.0);
    for entry in bilinear_layer.quadratic.entries.iter() {
        let row_index = us16(entry.index.0);
        let left_value = homogeneous_value(input_slice, u16n(entry.index.1));
        let right_value = homogeneous_value(input_slice, u16n(entry.index.2));
        output_slice[row_index] += f32n(entry.value) * left_value * right_value;
    }
}

fn execute_bilinear_push_forward(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let output_start = us32(bilinear_layer.out_offset);
    let output_stop = output_start + us16(bilinear_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let (input_slice, output_slice) = workspace.input_output_slices(
        prepared_input_range.0,
        prepared_input_range.1,
        output_start,
        output_stop,
    );
    let (tangent_input_slice, tangent_output_slice) = tangent_workspace.input_output_slices(
        prepared_tangent_input_range.0,
        prepared_tangent_input_range.1,
        output_start,
        output_stop,
    );
    execute_bilinear_push_forward_into_slices(
        bilinear_layer,
        input_slice,
        tangent_input_slice,
        output_slice,
        tangent_output_slice,
    );
}

fn execute_bilinear_push_forward_to_output_buffer(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
    tangent_output_buffer: &mut [f32],
) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    let tangent_input_slice = &tangent_workspace.as_slice()
        [prepared_tangent_input_range.0..prepared_tangent_input_range.1];
    execute_bilinear_push_forward_into_slices(
        bilinear_layer,
        input_slice,
        tangent_input_slice,
        output_buffer,
        tangent_output_buffer,
    );
}

fn execute_bilinear_push_forward_into_slices(
    bilinear_layer: &ArchivedBilinearLayer,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
    output_slice: &mut [f32],
    tangent_output_slice: &mut [f32],
) {
    output_slice.fill(0.0);
    tangent_output_slice.fill(0.0);
    for entry in bilinear_layer.quadratic.entries.iter() {
        let row_index = us16(entry.index.0);
        let left_index = u16n(entry.index.1);
        let right_index = u16n(entry.index.2);
        let left_value = homogeneous_value(input_slice, left_index);
        let right_value = homogeneous_value(input_slice, right_index);
        let left_tangent = homogeneous_tangent(tangent_input_slice, left_index);
        let right_tangent = homogeneous_tangent(tangent_input_slice, right_index);
        let value = f32n(entry.value);
        output_slice[row_index] += value * left_value * right_value;
        tangent_output_slice[row_index] +=
            value * (left_tangent * right_value + left_value * right_tangent);
    }
}

fn execute_generic_layer(generic_layer: &ArchivedGenericLayer, workspace: &mut Workspace<'_>) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let output_start = us32(generic_layer.out_offset);
    let output_stop = output_start + us16(generic_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let (input_slice, output_slice) = workspace.input_output_slices(
        prepared_input_range.0,
        prepared_input_range.1,
        output_start,
        output_stop,
    );
    execute_generic_into_slice(generic_layer, input_slice, output_slice);
}

fn execute_generic_layer_to_output_buffer(
    generic_layer: &ArchivedGenericLayer,
    workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    execute_generic_into_slice(generic_layer, input_slice, output_buffer);
}

fn execute_generic_into_slice(
    generic_layer: &ArchivedGenericLayer,
    input_slice: &[f32],
    output_slice: &mut [f32],
) {
    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        let row_operation = row_op_from_archived(row_operation);
        output_slice[row_index] = evaluate_generic_value(&row_operation, input_slice);
    }
}

fn execute_generic_push_forward(
    generic_layer: &ArchivedGenericLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let output_start = us32(generic_layer.out_offset);
    let output_stop = output_start + us16(generic_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let (input_slice, output_slice) = workspace.input_output_slices(
        prepared_input_range.0,
        prepared_input_range.1,
        output_start,
        output_stop,
    );
    let (tangent_input_slice, tangent_output_slice) = tangent_workspace.input_output_slices(
        prepared_tangent_input_range.0,
        prepared_tangent_input_range.1,
        output_start,
        output_stop,
    );
    execute_generic_push_forward_into_slices(
        generic_layer,
        input_slice,
        tangent_input_slice,
        output_slice,
        tangent_output_slice,
    );
}

fn execute_generic_push_forward_to_output_buffer(
    generic_layer: &ArchivedGenericLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
    tangent_output_buffer: &mut [f32],
) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    let tangent_input_slice = &tangent_workspace.as_slice()
        [prepared_tangent_input_range.0..prepared_tangent_input_range.1];
    execute_generic_push_forward_into_slices(
        generic_layer,
        input_slice,
        tangent_input_slice,
        output_buffer,
        tangent_output_buffer,
    );
}

fn execute_generic_push_forward_into_slices(
    generic_layer: &ArchivedGenericLayer,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
    output_slice: &mut [f32],
    tangent_output_slice: &mut [f32],
) {
    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        let row_operation = row_op_from_archived(row_operation);
        let (value, tangent) =
            evaluate_generic_push_forward(&row_operation, input_slice, tangent_input_slice);
        output_slice[row_index] = value;
        tangent_output_slice[row_index] = tangent;
    }
}

fn execute_evaluate_layer(
    module: &ArchivedBytecodeModule,
    evaluate_layer: &ArchivedEvaluateLayer,
    workspace: &mut Workspace<'_>,
) {
    let callee_program = find_function_unchecked(module, u16n(evaluate_layer.callee_function_id));
    let scratch_start = us32(evaluate_layer.scratch_offset);
    let scratch_length = us32(callee_program.required_workspace_size);
    let caller_workspace = Workspace::new(workspace.as_mut_slice());
    let (mut caller_workspace, scratch_workspace) = caller_workspace.split_at_mut(scratch_start);
    let mut nested_workspace = scratch_workspace.truncate(scratch_length);
    nested_workspace.fill(0.0);
    pack_evaluate_inputs(
        callee_program,
        evaluate_layer,
        caller_workspace.as_slice(),
        nested_workspace.as_mut_slice(),
    );
    execute_program_layers(module, callee_program, &mut nested_workspace, None);
    copy_evaluate_outputs(
        callee_program,
        evaluate_layer,
        nested_workspace.as_slice(),
        caller_workspace.as_mut_slice(),
    );
}

fn execute_evaluate_push_forward(
    module: &ArchivedBytecodeModule,
    evaluate_layer: &ArchivedEvaluateLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let callee_program = find_function_unchecked(module, u16n(evaluate_layer.callee_function_id));
    let scratch_start = us32(evaluate_layer.scratch_offset);
    let scratch_length = us32(callee_program.required_workspace_size);
    let caller_workspace = Workspace::new(workspace.as_mut_slice());
    let (mut caller_workspace, scratch_workspace) = caller_workspace.split_at_mut(scratch_start);
    let mut nested_workspace = scratch_workspace.truncate(scratch_length);
    let caller_tangent_workspace = Workspace::new(tangent_workspace.as_mut_slice());
    let (mut caller_tangent_workspace, tangent_scratch_workspace) =
        caller_tangent_workspace.split_at_mut(scratch_start);
    let mut nested_tangent_workspace = tangent_scratch_workspace.truncate(scratch_length);
    nested_workspace.fill(0.0);
    nested_tangent_workspace.fill(0.0);
    pack_evaluate_inputs(
        callee_program,
        evaluate_layer,
        caller_workspace.as_slice(),
        nested_workspace.as_mut_slice(),
    );
    pack_evaluate_tangents(
        callee_program,
        evaluate_layer,
        caller_tangent_workspace.as_slice(),
        nested_tangent_workspace.as_mut_slice(),
    );
    push_forward_program_layers(
        module,
        callee_program,
        &mut nested_workspace,
        &mut nested_tangent_workspace,
        None,
        None,
    );
    copy_evaluate_outputs(
        callee_program,
        evaluate_layer,
        nested_workspace.as_slice(),
        caller_workspace.as_mut_slice(),
    );
    copy_evaluate_outputs(
        callee_program,
        evaluate_layer,
        nested_tangent_workspace.as_slice(),
        caller_tangent_workspace.as_mut_slice(),
    );
}

fn pack_inputs(workspace: &mut Workspace<'_>, program: &ArchivedProgram, inputs: &[&[f32]]) {
    for (input_spec, input_value) in program.input_specs.iter().zip(inputs.iter()) {
        let start = us32(input_spec.workspace_offset);
        let stop = start + us16(input_spec.length);
        workspace.as_mut_slice()[start..stop].copy_from_slice(input_value);
    }
}

fn write_outputs(program: &ArchivedProgram, workspace: &[f32], outputs: &mut [f32]) {
    let mut output_cursor = 0usize;
    for output_spec in program.output_specs.iter() {
        let start = us32(output_spec.workspace_offset);
        let stop = start + us16(output_spec.length);
        let output_stop = output_cursor + us16(output_spec.length);
        outputs[output_cursor..output_stop].copy_from_slice(&workspace[start..stop]);
        output_cursor = output_stop;
    }
}

fn final_layer_matches_outputs(
    program: &ArchivedProgram,
    layer_output_offset: u32,
    layer_output_length: u16,
) -> bool {
    let mut expected_offset = layer_output_offset;
    let mut expected_length = 0u32;
    for output_spec in program.output_specs.iter() {
        let output_offset = u32n(output_spec.workspace_offset);
        if output_offset != expected_offset {
            return false;
        }
        let output_length = u16n(output_spec.length) as u32;
        expected_offset += output_length;
        expected_length += output_length;
    }
    expected_length == layer_output_length as u32
}

fn pack_evaluate_inputs(
    program: &ArchivedProgram,
    evaluate_layer: &ArchivedEvaluateLayer,
    caller_workspace: &[f32],
    callee_workspace: &mut [f32],
) {
    for (input_spec, binding) in program
        .input_specs
        .iter()
        .zip(evaluate_layer.input_bindings.iter())
    {
        let destination_start = us32(input_spec.workspace_offset);
        let destination_stop = destination_start + us16(input_spec.length);
        match binding {
            ArchivedEvaluateInputBinding::WorkspaceSlice { offset, length } => {
                let source_start = us32(*offset);
                let source_stop = source_start + us16(*length);
                callee_workspace[destination_start..destination_stop]
                    .copy_from_slice(&caller_workspace[source_start..source_stop]);
            }
            ArchivedEvaluateInputBinding::ConstantSlice { values, .. } => {
                for (destination, value) in callee_workspace[destination_start..destination_stop]
                    .iter_mut()
                    .zip(values.iter())
                {
                    *destination = f32n(*value);
                }
            }
        }
    }
}

fn pack_evaluate_tangents(
    program: &ArchivedProgram,
    evaluate_layer: &ArchivedEvaluateLayer,
    caller_tangent_workspace: &[f32],
    callee_tangent_workspace: &mut [f32],
) {
    for (input_spec, binding) in program
        .input_specs
        .iter()
        .zip(evaluate_layer.input_bindings.iter())
    {
        let destination_start = us32(input_spec.workspace_offset);
        let destination_stop = destination_start + us16(input_spec.length);
        match binding {
            ArchivedEvaluateInputBinding::WorkspaceSlice { offset, length } => {
                let source_start = us32(*offset);
                let source_stop = source_start + us16(*length);
                callee_tangent_workspace[destination_start..destination_stop]
                    .copy_from_slice(&caller_tangent_workspace[source_start..source_stop]);
            }
            ArchivedEvaluateInputBinding::ConstantSlice { .. } => {
                callee_tangent_workspace[destination_start..destination_stop].fill(0.0);
            }
        }
    }
}

fn copy_evaluate_outputs(
    program: &ArchivedProgram,
    evaluate_layer: &ArchivedEvaluateLayer,
    callee_workspace: &[f32],
    caller_workspace: &mut [f32],
) {
    for (output_spec, binding) in program
        .output_specs
        .iter()
        .zip(evaluate_layer.output_bindings.iter())
    {
        let source_start = us32(output_spec.workspace_offset);
        let source_stop = source_start + us16(output_spec.length);
        let destination_start = us32(binding.destination_offset);
        let destination_stop = destination_start + us16(binding.length);
        caller_workspace[destination_start..destination_stop]
            .copy_from_slice(&callee_workspace[source_start..source_stop]);
    }
}

fn program_info_from_program(program: &ArchivedProgram) -> ProgramInfo {
    ProgramInfo {
        workspace_size: us32(program.workspace_size),
        required_workspace_size: us32(program.required_workspace_size),
        input_specs: program
            .input_specs
            .iter()
            .map(|input_spec| InputSpec {
                workspace_offset: u32n(input_spec.workspace_offset),
                length: u16n(input_spec.length),
            })
            .collect::<Vec<_>>(),
        output_specs: program
            .output_specs
            .iter()
            .map(|output_spec| OutputSpec {
                workspace_offset: u32n(output_spec.workspace_offset),
                length: u16n(output_spec.length),
            })
            .collect::<Vec<_>>(),
    }
}

fn entry_program(module: &ArchivedBytecodeModule) -> Result<&ArchivedProgram, RuntimeError> {
    find_function(module, 0)
        .ok_or_else(|| RuntimeError::Validation("missing entry function_id 0".to_string()))
}

fn find_function(module: &ArchivedBytecodeModule, function_id: u16) -> Option<&ArchivedProgram> {
    module
        .functions
        .iter()
        .find(|program| u16n(program.function_id) == function_id)
}

fn find_function_unchecked(module: &ArchivedBytecodeModule, function_id: u16) -> &ArchivedProgram {
    find_function(module, function_id).expect("validated module missing referenced function")
}

fn row_op_from_archived(row_op: &ArchivedRowOp) -> RowOp {
    RowOp {
        first: u16n(row_op.first),
        second: u16n(row_op.second),
        third: u16n(row_op.third),
        op: scalar_op_from_archived(&row_op.op),
    }
}

fn scalar_op_from_archived(scalar_op: &ArchivedScalarOp) -> ScalarOp {
    match scalar_op {
        ArchivedScalarOp::Identity => ScalarOp::Identity,
        ArchivedScalarOp::Sin => ScalarOp::Sin,
        ArchivedScalarOp::Cos => ScalarOp::Cos,
        ArchivedScalarOp::Tan => ScalarOp::Tan,
        ArchivedScalarOp::Exp => ScalarOp::Exp,
        ArchivedScalarOp::Sqrt => ScalarOp::Sqrt,
        ArchivedScalarOp::Log => ScalarOp::Log,
        ArchivedScalarOp::Neg => ScalarOp::Neg,
        ArchivedScalarOp::Abs => ScalarOp::Abs,
        ArchivedScalarOp::Add => ScalarOp::Add,
        ArchivedScalarOp::Sub => ScalarOp::Sub,
        ArchivedScalarOp::Mul => ScalarOp::Mul,
        ArchivedScalarOp::Div => ScalarOp::Div,
        ArchivedScalarOp::Pow => ScalarOp::Pow,
        ArchivedScalarOp::IntPow => ScalarOp::IntPow,
        ArchivedScalarOp::Atan2 => ScalarOp::Atan2,
        ArchivedScalarOp::Equal => ScalarOp::Equal,
        ArchivedScalarOp::LessThan => ScalarOp::LessThan,
        ArchivedScalarOp::LessEqual => ScalarOp::LessEqual,
        ArchivedScalarOp::Case => ScalarOp::Case,
    }
}

fn u16n(value: impl Into<u16>) -> u16 {
    value.into()
}

fn u32n(value: impl Into<u32>) -> u32 {
    value.into()
}

fn us16(value: impl Into<u16>) -> usize {
    u16n(value) as usize
}

fn us32(value: impl Into<u32>) -> usize {
    u32n(value) as usize
}

fn f32n(value: impl Into<f32>) -> f32 {
    value.into()
}
