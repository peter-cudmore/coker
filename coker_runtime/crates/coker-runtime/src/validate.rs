#![cfg_attr(not(feature = "std"), allow(dead_code))]

use coker_bytecode::{
    BilinearLayer, BytecodeModule, EvaluateInputBinding, EvaluateLayer, GenericLayer, Layer,
    Program, RowOp, ScalarOp,
};

use crate::{
    entry_program, find_function,
    validation_common::{
        validate_inputs as validate_spec_inputs, validate_layer_scratch,
        validate_outputs as validate_spec_outputs, validate_range, validate_workspace_size,
    },
    RuntimeError, UNUSED_OPERAND,
};

pub(crate) fn validate_module_struct(module: &BytecodeModule) -> Result<(), RuntimeError> {
    let _entry_program = entry_program(module)?;
    for (_, function_program) in module.functions() {
        validate_program_struct(module, function_program)?;
    }
    Ok(())
}

fn validate_program_struct(module: &BytecodeModule, program: &Program) -> Result<(), RuntimeError> {
    let workspace_size = program.workspace_size as usize;
    let required_workspace_size = program.required_workspace_size as usize;
    if required_workspace_size < workspace_size {
        return Err(RuntimeError::Validation(
            "required workspace smaller than primary workspace",
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
            Layer::QpCall(_) => {}
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
            .ok_or(RuntimeError::Validation("bilinear input too large"))?,
        bilinear_layer
            .in_length
            .checked_add(1)
            .ok_or(RuntimeError::Validation("bilinear input too large"))?,
    );
    if bilinear_layer.quadratic.shape != expected_shape {
        return Err(RuntimeError::Validation(
            "bilinear tensor shape does not match layer dimensions",
        ));
    }

    for entry in &bilinear_layer.quadratic.entries {
        let (row_index, left_index, right_index) = entry.index;
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
    if generic_layer.ops.len() != generic_layer.out_length as usize {
        return Err(RuntimeError::Validation(
            "generic layer op count must match output length",
        ));
    }
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
    let callee_program = find_function(module, evaluate_layer.callee_function_id).ok_or(
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
    if (evaluate_layer.scratch_offset as usize) < caller_workspace_size {
        return Err(RuntimeError::Validation(
            "evaluate scratch offset overlaps caller workspace",
        ));
    }

    let scratch_end =
        evaluate_layer.scratch_offset as usize + callee_program.required_workspace_size as usize;
    if scratch_end > caller_program.required_workspace_size as usize {
        return Err(RuntimeError::Validation(
            "evaluate scratch range exceeds caller required workspace",
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
                "evaluate output binding length mismatch",
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
                    "evaluate input binding length mismatch",
                ));
            }
            validate_range(*offset, *length, caller_workspace_size, "evaluate input")
        }
        EvaluateInputBinding::ConstantSlice { length, values } => {
            if *length != expected_length || values.len() != *length as usize {
                return Err(RuntimeError::Validation(
                    "evaluate constant input length mismatch",
                ));
            }
            Ok(())
        }
    }
}

fn validate_generic_operand(operand_index: u16, input_length: u16) -> Result<(), RuntimeError> {
    if operand_index != UNUSED_OPERAND && operand_index >= input_length {
        return Err(RuntimeError::Validation(
            "generic operand index out of bounds",
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
    for operand_index in operand_indices
        .iter()
        .take(required_operand_count(row_operation.op) as usize)
    {
        if *operand_index == UNUSED_OPERAND {
            return Err(RuntimeError::Validation(
                "generic operation missing required operand",
            ));
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

pub(crate) fn validate_inputs(program: &Program, inputs: &[&[f32]]) -> Result<(), RuntimeError> {
    validate_spec_inputs(&program.input_specs, inputs)
}

pub(crate) fn validate_outputs(program: &Program, outputs: &[f32]) -> Result<(), RuntimeError> {
    validate_spec_outputs(&program.output_specs, outputs)
}

pub(crate) fn validate_workspace(program: &Program, workspace: &[f32]) -> Result<(), RuntimeError> {
    validate_workspace_size(program.required_workspace_size as usize, workspace.len())
}
