use super::*;
use crate::validation_common::{
    validate_inputs as validate_spec_inputs, validate_layer_scratch,
    validate_outputs as validate_spec_outputs, validate_range, validate_workspace_size,
};

pub(super) fn validate_module_struct(module: &ArchivedBytecodeModule) -> Result<(), RuntimeError> {
    module.validate_semantics()?;
    let _entry_program = entry_program(module)?;
    for (_, function_program) in module.programs() {
        validate_program_struct(module, function_program)?;
    }
    Ok(())
}

pub(super) fn validate_program_struct(
    module: &ArchivedBytecodeModule,
    program: &ArchivedProgram,
) -> Result<(), RuntimeError> {
    let workspace_size = us32(program.workspace_size);
    let required_workspace_size = us32(program.required_workspace_size);
    if required_workspace_size < workspace_size {
        return Err(RuntimeError::Validation(
            "required workspace smaller than primary workspace",
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
            ArchivedLayer::QpCall(_) => {}
        }
    }
    Ok(())
}

pub(super) fn validate_bilinear_layer(
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
            .ok_or(RuntimeError::Validation("bilinear input too large"))?,
        in_length
            .checked_add(1)
            .ok_or(RuntimeError::Validation("bilinear input too large"))?,
    );
    let shape = (
        u16n(bilinear_layer.quadratic.shape.0),
        u16n(bilinear_layer.quadratic.shape.1),
        u16n(bilinear_layer.quadratic.shape.2),
    );
    if shape != expected_shape {
        return Err(RuntimeError::Validation(
            "bilinear tensor shape does not match layer dimensions",
        ));
    }

    for entry in bilinear_layer.quadratic.entries.iter() {
        let row_index = u16n(entry.index.0);
        let left_index = u16n(entry.index.1);
        let right_index = u16n(entry.index.2);
        if row_index >= expected_shape.0 {
            return Err(RuntimeError::Validation(
                "bilinear tensor row index out of bounds",
            ));
        }
        if left_index >= expected_shape.1 {
            return Err(RuntimeError::Validation(
                "bilinear tensor left index out of bounds",
            ));
        }
        if right_index >= expected_shape.2 {
            return Err(RuntimeError::Validation(
                "bilinear tensor right index out of bounds",
            ));
        }
    }

    Ok(())
}

pub(super) fn validate_generic_layer(
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
            "generic layer op count must match output length",
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

pub(super) fn validate_evaluate_layer(
    module: &ArchivedBytecodeModule,
    evaluate_layer: &ArchivedEvaluateLayer,
    caller_program: &ArchivedProgram,
    caller_workspace_size: usize,
) -> Result<(), RuntimeError> {
    let callee_program = find_function(module, u16n(evaluate_layer.callee_function_id)).ok_or(
        RuntimeError::Validation("evaluate callee function id missing"),
    )?;

    if evaluate_layer.input_bindings.len() != callee_program.input_specs.len() {
        return Err(RuntimeError::Validation(
            "evaluate input binding count does not match callee inputs",
        ));
    }
    if evaluate_layer.output_bindings.len() != callee_program.output_specs.len() {
        return Err(RuntimeError::Validation(
            "evaluate output binding count does not match callee outputs",
        ));
    }
    if us32(evaluate_layer.scratch_offset) < caller_workspace_size {
        return Err(RuntimeError::Validation(
            "evaluate scratch offset overlaps caller workspace",
        ));
    }

    let scratch_end =
        us32(evaluate_layer.scratch_offset) + us32(callee_program.required_workspace_size);
    if scratch_end > us32(caller_program.required_workspace_size) {
        return Err(RuntimeError::Validation(
            "evaluate scratch range exceeds caller required workspace",
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
                "evaluate output binding length mismatch",
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

pub(super) fn validate_evaluate_input_binding(
    binding: &ArchivedEvaluateInputBinding,
    expected_length: u16,
    caller_workspace_size: usize,
) -> Result<(), RuntimeError> {
    match binding {
        ArchivedEvaluateInputBinding::WorkspaceSlice { offset, length } => {
            let length = u16n(*length);
            if length != expected_length {
                return Err(RuntimeError::Validation(
                    "evaluate input binding length mismatch",
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
                    "evaluate constant input length mismatch",
                ));
            }
            Ok(())
        }
    }
}

pub(super) fn validate_generic_operand(
    operand_index: u16,
    input_length: u16,
) -> Result<(), RuntimeError> {
    if operand_index != UNUSED_OPERAND && operand_index >= input_length {
        return Err(RuntimeError::Validation(
            "generic operand index out of bounds",
        ));
    }
    Ok(())
}

pub(super) fn validate_generic_row_operation(
    row_operation: &ArchivedRowOp,
) -> Result<(), RuntimeError> {
    let operand_indices = [
        u16n(row_operation.first),
        u16n(row_operation.second),
        u16n(row_operation.third),
    ];
    for operand_index in operand_indices
        .iter()
        .take(required_operand_count(&row_operation.op) as usize)
    {
        if *operand_index == UNUSED_OPERAND {
            return Err(RuntimeError::Validation(
                "generic operation missing required operand",
            ));
        }
    }
    Ok(())
}

pub(super) fn required_operand_count(operation: &ArchivedScalarOp) -> u8 {
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

pub(super) fn validate_inputs(
    program: &ArchivedProgram,
    inputs: &[&[f32]],
) -> Result<(), RuntimeError> {
    validate_spec_inputs(program.input_specs(), inputs)
}

pub(super) fn validate_outputs(
    program: &ArchivedProgram,
    outputs: &[f32],
) -> Result<(), RuntimeError> {
    validate_spec_outputs(program.output_specs(), outputs)
}

pub(super) fn validate_workspace(
    program: &ArchivedProgram,
    workspace: &[f32],
) -> Result<(), RuntimeError> {
    validate_workspace_size(us32(program.required_workspace_size), workspace.len())
}
