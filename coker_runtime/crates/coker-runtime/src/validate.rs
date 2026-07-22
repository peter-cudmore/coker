use alloc::{format, string::ToString};
use coker_bytecode::{
    BilinearLayer, BytecodeModule, EvaluateInputBinding, EvaluateLayer, GenericLayer, Layer,
    Program, RowOp, ScalarOp,
};

use crate::{entry_program, find_function, RuntimeError, UNUSED_OPERAND};

pub(crate) fn validate_module_struct(module: &BytecodeModule) -> Result<(), RuntimeError> {
    if module.functions.is_empty() {
        return Err(RuntimeError::Validation(
            "bytecode module must contain at least one function".to_string(),
        ));
    }
    entry_program(module)?;

    for (function_index, function_program) in module.functions.iter().enumerate() {
        if module.functions[..function_index]
            .iter()
            .any(|prior_program| prior_program.function_id == function_program.function_id)
        {
            return Err(RuntimeError::Validation(
                "duplicate function id".to_string(),
            ));
        }
        validate_program_struct(module, function_program)?;
    }
    Ok(())
}

fn validate_program_struct(module: &BytecodeModule, program: &Program) -> Result<(), RuntimeError> {
    let workspace_size = program.workspace_size as usize;
    let required_workspace_size = program.required_workspace_size as usize;
    if required_workspace_size < workspace_size {
        return Err(RuntimeError::Validation(
            "required workspace smaller than primary workspace".to_string(),
        ));
    }

    for input_spec in &program.input_specs {
        validate_range(
            input_spec.workspace_offset,
            input_spec.length,
            workspace_size,
            "input",
        )?;
    }
    for output_spec in &program.output_specs {
        validate_range(
            output_spec.workspace_offset,
            output_spec.length,
            workspace_size,
            "output",
        )?;
    }
    for layer in &program.intermediate_layers {
        match layer {
            Layer::Bilinear(bilinear_layer) => {
                validate_bilinear_layer(bilinear_layer, workspace_size, required_workspace_size)?
            }
            Layer::Generic(generic_layer) => {
                validate_generic_layer(generic_layer, workspace_size, required_workspace_size)?
            }
            Layer::Evaluate(evaluate_layer) => {
                validate_evaluate_layer(module, evaluate_layer, program, workspace_size)?
            }
        }
    }
    Ok(())
}

fn validate_bilinear_layer(
    bilinear_layer: &BilinearLayer,
    workspace_size: usize,
    required_workspace_size: usize,
) -> Result<(), RuntimeError> {
    validate_range(
        bilinear_layer.in_offset,
        bilinear_layer.in_length,
        workspace_size,
        "bilinear input",
    )?;
    validate_range(
        bilinear_layer.out_offset,
        bilinear_layer.out_length,
        workspace_size,
        "bilinear output",
    )?;
    validate_layer_scratch(
        bilinear_layer.in_offset,
        bilinear_layer.in_length,
        bilinear_layer.out_offset,
        bilinear_layer.out_length,
        bilinear_layer.scratch_offset,
        bilinear_layer.scratch_length,
        workspace_size,
        required_workspace_size,
        "bilinear layer",
    )?;

    let expected_shape = (
        bilinear_layer.out_length,
        bilinear_layer
            .in_length
            .checked_add(1)
            .ok_or_else(|| RuntimeError::Validation("bilinear input too large".to_string()))?,
        bilinear_layer
            .in_length
            .checked_add(1)
            .ok_or_else(|| RuntimeError::Validation("bilinear input too large".to_string()))?,
    );
    if bilinear_layer.quadratic.shape != expected_shape {
        return Err(RuntimeError::Validation(
            "bilinear tensor shape does not match layer dimensions".to_string(),
        ));
    }

    for entry in &bilinear_layer.quadratic.entries {
        let (row_index, left_index, right_index) = entry.index;
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
    generic_layer: &GenericLayer,
    workspace_size: usize,
    required_workspace_size: usize,
) -> Result<(), RuntimeError> {
    validate_range(
        generic_layer.in_offset,
        generic_layer.in_length,
        workspace_size,
        "generic input",
    )?;
    validate_range(
        generic_layer.out_offset,
        generic_layer.out_length,
        workspace_size,
        "generic output",
    )?;
    validate_layer_scratch(
        generic_layer.in_offset,
        generic_layer.in_length,
        generic_layer.out_offset,
        generic_layer.out_length,
        generic_layer.scratch_offset,
        generic_layer.scratch_length,
        workspace_size,
        required_workspace_size,
        "generic layer",
    )?;

    for row_operation in &generic_layer.ops {
        validate_generic_operand(row_operation.first, generic_layer.in_length)?;
        validate_generic_operand(row_operation.second, generic_layer.in_length)?;
        validate_generic_operand(row_operation.third, generic_layer.in_length)?;
        validate_generic_row_operation(row_operation)?;
    }

    Ok(())
}

fn validate_evaluate_layer(
    module: &BytecodeModule,
    evaluate_layer: &EvaluateLayer,
    caller_program: &Program,
    caller_workspace_size: usize,
) -> Result<(), RuntimeError> {
    let callee_program =
        find_function(module, evaluate_layer.callee_function_id).ok_or_else(|| {
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
    if (evaluate_layer.scratch_offset as usize) < caller_workspace_size {
        return Err(RuntimeError::Validation(
            "evaluate scratch offset overlaps caller workspace".to_string(),
        ));
    }

    let scratch_end =
        evaluate_layer.scratch_offset as usize + callee_program.required_workspace_size as usize;
    if scratch_end > caller_program.required_workspace_size as usize {
        return Err(RuntimeError::Validation(
            "evaluate scratch range exceeds caller required workspace".to_string(),
        ));
    }

    for (binding, input_spec) in evaluate_layer
        .input_bindings
        .iter()
        .zip(callee_program.input_specs.iter())
    {
        validate_evaluate_input_binding(binding, input_spec.length, caller_workspace_size)?;
    }
    for (binding, output_spec) in evaluate_layer
        .output_bindings
        .iter()
        .zip(callee_program.output_specs.iter())
    {
        if binding.length != output_spec.length {
            return Err(RuntimeError::Validation(
                "evaluate output binding length mismatch".to_string(),
            ));
        }
        validate_range(
            binding.destination_offset,
            binding.length,
            caller_workspace_size,
            "evaluate output",
        )?;
    }

    Ok(())
}

fn validate_evaluate_input_binding(
    binding: &EvaluateInputBinding,
    expected_length: u16,
    caller_workspace_size: usize,
) -> Result<(), RuntimeError> {
    match binding {
        EvaluateInputBinding::WorkspaceSlice { offset, length } => {
            if *length != expected_length {
                return Err(RuntimeError::Validation(
                    "evaluate input binding length mismatch".to_string(),
                ));
            }
            validate_range(*offset, *length, caller_workspace_size, "evaluate input")
        }
        EvaluateInputBinding::ConstantSlice { length, values } => {
            if *length != expected_length || values.len() != *length as usize {
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

fn validate_generic_row_operation(row_operation: &RowOp) -> Result<(), RuntimeError> {
    let operand_indices = [
        row_operation.first,
        row_operation.second,
        row_operation.third,
    ];
    for required_index in 0..required_operand_count(row_operation.op) as usize {
        if operand_indices[required_index] == UNUSED_OPERAND {
            return Err(RuntimeError::Validation(format!(
                "generic operation {:?} missing required operand",
                row_operation.op
            )));
        }
    }
    Ok(())
}

fn required_operand_count(operation: ScalarOp) -> u8 {
    match operation {
        ScalarOp::Identity
        | ScalarOp::Sin
        | ScalarOp::Cos
        | ScalarOp::Tan
        | ScalarOp::Exp
        | ScalarOp::Sqrt
        | ScalarOp::Log
        | ScalarOp::Neg
        | ScalarOp::Abs => 1,
        ScalarOp::Add
        | ScalarOp::Sub
        | ScalarOp::Mul
        | ScalarOp::Div
        | ScalarOp::Pow
        | ScalarOp::IntPow
        | ScalarOp::Atan2
        | ScalarOp::Equal
        | ScalarOp::LessThan
        | ScalarOp::LessEqual => 2,
        ScalarOp::Case => 3,
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

pub(crate) fn validate_inputs(program: &Program, inputs: &[&[f32]]) -> Result<(), RuntimeError> {
    if inputs.len() != program.input_specs.len() {
        return Err(RuntimeError::InputCountMismatch {
            expected: program.input_specs.len(),
            actual: inputs.len(),
        });
    }

    for (index, (input_spec, input_value)) in
        program.input_specs.iter().zip(inputs.iter()).enumerate()
    {
        let expected_count = input_spec.length as usize;
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

pub(crate) fn validate_workspace(program: &Program, workspace: &[f32]) -> Result<(), RuntimeError> {
    validate_workspace_size(program.required_workspace_size as usize, workspace.len())
}

pub(crate) fn validate_workspace_size(
    expected_size: usize,
    actual_size: usize,
) -> Result<(), RuntimeError> {
    if actual_size < expected_size {
        return Err(RuntimeError::WorkspaceTooSmall {
            expected: expected_size,
            actual: actual_size,
        });
    }
    Ok(())
}
