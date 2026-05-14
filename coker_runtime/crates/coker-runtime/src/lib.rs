#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{format, string::{String, ToString}, vec, vec::Vec};
use coker_bytecode::{
    decode_module, BilinearLayer, BytecodeModule, EvaluateInputBinding,
    EvaluateLayer, GenericLayer, InputSpec, Layer, OutputSpec, Program, RowOp,
    ScalarOp,
};
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
    validate_module_struct(&module)?;
    Ok(module)
}

pub fn execute(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    workspace: &mut [f32],
) -> Result<Vec<Vec<f32>>, RuntimeError> {
    let program = entry_program(module)?;
    execute_in_place(module, inputs, workspace)?;
    Ok(collect_outputs(program, workspace))
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
        collect_outputs(program, workspace),
        collect_outputs(program, tangent_workspace),
    ))
}

pub fn execute_in_place(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let program = entry_program(module)?;
    validate_inputs(program, inputs)?;
    validate_workspace(program, workspace)?;

    workspace.fill(0.0);
    pack_inputs(&program.input_specs, inputs, workspace);
    execute_program_layers(module, program, workspace)
}

pub fn push_forward_in_place(
    module: &BytecodeModule,
    inputs: &[&[f32]],
    tangents: &[&[f32]],
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let program = entry_program(module)?;
    validate_inputs(program, inputs)?;
    validate_inputs(program, tangents)?;
    validate_workspace(program, workspace)?;
    validate_workspace(program, tangent_workspace)?;

    workspace.fill(0.0);
    tangent_workspace.fill(0.0);
    pack_inputs(&program.input_specs, inputs, workspace);
    pack_inputs(&program.input_specs, tangents, tangent_workspace);
    push_forward_program_layers(module, program, workspace, tangent_workspace)
}

pub fn program_info_from_program(program: &Program) -> ProgramInfo {
    ProgramInfo {
        workspace_size: program.workspace_size as usize,
        required_workspace_size: program.required_workspace_size as usize,
        input_specs: program.input_specs.clone(),
        output_specs: program.output_specs.clone(),
    }
}

pub fn validate_module_struct(module: &BytecodeModule) -> Result<(), RuntimeError> {
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

pub fn entry_program(module: &BytecodeModule) -> Result<&Program, RuntimeError> {
    find_function(module, 0).ok_or_else(|| {
        RuntimeError::Validation("missing entry function_id 0".to_string())
    })
}

fn find_function(module: &BytecodeModule, function_id: u16) -> Option<&Program> {
    module
        .functions
        .iter()
        .find(|program| program.function_id == function_id)
}

fn validate_program_struct(
    module: &BytecodeModule,
    program: &Program,
) -> Result<(), RuntimeError> {
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
                validate_bilinear_layer(bilinear_layer, workspace_size)?
            }
            Layer::Generic(generic_layer) => {
                validate_generic_layer(generic_layer, workspace_size)?
            }
            Layer::Evaluate(evaluate_layer) => validate_evaluate_layer(
                module,
                evaluate_layer,
                program,
                workspace_size,
            )?,
        }
    }
    Ok(())
}

fn validate_bilinear_layer(
    bilinear_layer: &BilinearLayer,
    workspace_size: usize,
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

    let expected_shape = (
        bilinear_layer.out_length,
        bilinear_layer.in_length.checked_add(1).ok_or_else(|| {
            RuntimeError::Validation("bilinear input too large".to_string())
        })?,
        bilinear_layer.in_length.checked_add(1).ok_or_else(|| {
            RuntimeError::Validation("bilinear input too large".to_string())
        })?,
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

    for row_operation in &generic_layer.ops {
        validate_generic_operand(row_operation.first, generic_layer.in_length)?;
        validate_generic_operand(row_operation.second, generic_layer.in_length)?;
        validate_generic_operand(row_operation.third, generic_layer.in_length)?;
    }

    Ok(())
}

fn validate_evaluate_layer(
    module: &BytecodeModule,
    evaluate_layer: &EvaluateLayer,
    caller_program: &Program,
    caller_workspace_size: usize,
) -> Result<(), RuntimeError> {
    let callee_program = find_function(module, evaluate_layer.callee_function_id)
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
    if (evaluate_layer.scratch_offset as usize) < caller_workspace_size {
        return Err(RuntimeError::Validation(
            "evaluate scratch offset overlaps caller workspace".to_string(),
        ));
    }

    let scratch_end = evaluate_layer.scratch_offset as usize
        + callee_program.required_workspace_size as usize;
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
            validate_range(
                *offset,
                *length,
                caller_workspace_size,
                "evaluate input",
            )
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

fn validate_generic_operand(
    operand_index: u16,
    input_length: u16,
) -> Result<(), RuntimeError> {
    if operand_index != UNUSED_OPERAND && operand_index >= input_length {
        return Err(RuntimeError::Validation(
            "generic operand index out of bounds".to_string(),
        ));
    }
    Ok(())
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

fn validate_inputs(program: &Program, inputs: &[&[f32]]) -> Result<(), RuntimeError> {
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

fn validate_workspace(program: &Program, workspace: &[f32]) -> Result<(), RuntimeError> {
    let expected_size = program.required_workspace_size as usize;
    if workspace.len() < expected_size {
        return Err(RuntimeError::WorkspaceTooSmall {
            expected: expected_size,
            actual: workspace.len(),
        });
    }
    Ok(())
}

fn pack_inputs(input_specs: &[InputSpec], inputs: &[&[f32]], workspace: &mut [f32]) {
    for (input_spec, input_value) in input_specs.iter().zip(inputs.iter()) {
        let start = input_spec.workspace_offset as usize;
        let stop = start + input_spec.length as usize;
        workspace[start..stop].copy_from_slice(input_value);
    }
}

fn pack_owned_inputs(input_specs: &[InputSpec], inputs: &[Vec<f32>], workspace: &mut [f32]) {
    for (input_spec, input_value) in input_specs.iter().zip(inputs.iter()) {
        let start = input_spec.workspace_offset as usize;
        let stop = start + input_spec.length as usize;
        workspace[start..stop].copy_from_slice(input_value);
    }
}

fn collect_outputs(program: &Program, workspace: &[f32]) -> Vec<Vec<f32>> {
    program
        .output_specs
        .iter()
        .map(|output_spec| {
            let start = output_spec.workspace_offset as usize;
            let stop = start + output_spec.length as usize;
            workspace[start..stop].to_vec()
        })
        .collect()
}

fn execute_program_layers(
    module: &BytecodeModule,
    program: &Program,
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    for layer in &program.intermediate_layers {
        match layer {
            Layer::Bilinear(bilinear_layer) => {
                execute_bilinear_layer(bilinear_layer, workspace)?
            }
            Layer::Generic(generic_layer) => execute_generic_layer(generic_layer, workspace)?,
            Layer::Evaluate(evaluate_layer) => {
                execute_evaluate_layer(module, evaluate_layer, workspace)?
            }
        }
    }
    Ok(())
}

fn push_forward_program_layers(
    module: &BytecodeModule,
    program: &Program,
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    for layer in &program.intermediate_layers {
        match layer {
            Layer::Bilinear(bilinear_layer) => {
                execute_bilinear_push_forward(bilinear_layer, workspace, tangent_workspace)?
            }
            Layer::Generic(generic_layer) => {
                execute_generic_push_forward(generic_layer, workspace, tangent_workspace)?
            }
            Layer::Evaluate(evaluate_layer) => {
                execute_evaluate_push_forward(
                    module,
                    evaluate_layer,
                    workspace,
                    tangent_workspace,
                )?
            }
        }
    }
    Ok(())
}

fn execute_bilinear_layer(
    bilinear_layer: &BilinearLayer,
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let input_start = bilinear_layer.in_offset as usize;
    let input_stop = input_start + bilinear_layer.in_length as usize;
    let output_start = bilinear_layer.out_offset as usize;
    let output_stop = output_start + bilinear_layer.out_length as usize;
    let input_slice = workspace[input_start..input_stop].to_vec();
    let output_slice = &mut workspace[output_start..output_stop];
    output_slice.fill(0.0);

    for entry in &bilinear_layer.quadratic.entries {
        let row_index = entry.index.0 as usize;
        let left_value = homogeneous_value(&input_slice, entry.index.1)?;
        let right_value = homogeneous_value(&input_slice, entry.index.2)?;
        output_slice[row_index] += entry.value * left_value * right_value;
    }

    Ok(())
}

fn execute_bilinear_push_forward(
    bilinear_layer: &BilinearLayer,
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let input_start = bilinear_layer.in_offset as usize;
    let input_stop = input_start + bilinear_layer.in_length as usize;
    let output_start = bilinear_layer.out_offset as usize;
    let output_stop = output_start + bilinear_layer.out_length as usize;

    let input_slice = workspace[input_start..input_stop].to_vec();
    let tangent_input_slice = tangent_workspace[input_start..input_stop].to_vec();
    let output_slice = &mut workspace[output_start..output_stop];
    let tangent_output_slice = &mut tangent_workspace[output_start..output_stop];
    output_slice.fill(0.0);
    tangent_output_slice.fill(0.0);

    for entry in &bilinear_layer.quadratic.entries {
        let row_index = entry.index.0 as usize;
        let left_value = homogeneous_value(&input_slice, entry.index.1)?;
        let right_value = homogeneous_value(&input_slice, entry.index.2)?;
        let left_tangent = homogeneous_tangent(&tangent_input_slice, entry.index.1)?;
        let right_tangent = homogeneous_tangent(&tangent_input_slice, entry.index.2)?;
        output_slice[row_index] += entry.value * left_value * right_value;
        tangent_output_slice[row_index] +=
            entry.value * (left_tangent * right_value + left_value * right_tangent);
    }

    Ok(())
}

fn execute_generic_layer(
    generic_layer: &GenericLayer,
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let input_start = generic_layer.in_offset as usize;
    let input_stop = input_start + generic_layer.in_length as usize;
    let output_start = generic_layer.out_offset as usize;
    let output_stop = output_start + generic_layer.out_length as usize;
    let input_slice = workspace[input_start..input_stop].to_vec();
    let output_slice = &mut workspace[output_start..output_stop];

    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        output_slice[row_index] = evaluate_generic_value(row_operation, &input_slice)?;
    }

    Ok(())
}

fn execute_generic_push_forward(
    generic_layer: &GenericLayer,
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let input_start = generic_layer.in_offset as usize;
    let input_stop = input_start + generic_layer.in_length as usize;
    let output_start = generic_layer.out_offset as usize;
    let output_stop = output_start + generic_layer.out_length as usize;

    let input_slice = workspace[input_start..input_stop].to_vec();
    let tangent_input_slice = tangent_workspace[input_start..input_stop].to_vec();
    let output_slice = &mut workspace[output_start..output_stop];
    let tangent_output_slice = &mut tangent_workspace[output_start..output_stop];

    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        let (value, tangent) =
            evaluate_generic_push_forward(row_operation, &input_slice, &tangent_input_slice)?;
        output_slice[row_index] = value;
        tangent_output_slice[row_index] = tangent;
    }

    Ok(())
}

fn execute_evaluate_layer(
    module: &BytecodeModule,
    evaluate_layer: &EvaluateLayer,
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let callee_program = find_function(module, evaluate_layer.callee_function_id)
        .ok_or_else(|| {
            RuntimeError::Validation("evaluate callee function id missing".to_string())
        })?;
    let input_values = materialize_evaluate_inputs(&evaluate_layer.input_bindings, workspace);
    let nested_outputs = {
        let scratch_start = evaluate_layer.scratch_offset as usize;
        let scratch_stop = scratch_start + callee_program.required_workspace_size as usize;
        let nested_workspace = &mut workspace[scratch_start..scratch_stop];
        nested_workspace.fill(0.0);
        pack_owned_inputs(&callee_program.input_specs, &input_values, nested_workspace);
        execute_program_layers(module, callee_program, nested_workspace)?;
        collect_outputs(callee_program, nested_workspace)
    };
    write_evaluate_outputs(&evaluate_layer.output_bindings, &nested_outputs, workspace);
    Ok(())
}

fn execute_evaluate_push_forward(
    module: &BytecodeModule,
    evaluate_layer: &EvaluateLayer,
    workspace: &mut [f32],
    tangent_workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let callee_program = find_function(module, evaluate_layer.callee_function_id)
        .ok_or_else(|| {
            RuntimeError::Validation("evaluate callee function id missing".to_string())
        })?;
    let input_values = materialize_evaluate_inputs(&evaluate_layer.input_bindings, workspace);
    let tangent_values =
        materialize_evaluate_tangents(&evaluate_layer.input_bindings, tangent_workspace);
    let (nested_outputs, nested_tangent_outputs) = {
        let scratch_start = evaluate_layer.scratch_offset as usize;
        let scratch_stop = scratch_start + callee_program.required_workspace_size as usize;
        let nested_workspace = &mut workspace[scratch_start..scratch_stop];
        let nested_tangent_workspace = &mut tangent_workspace[scratch_start..scratch_stop];
        nested_workspace.fill(0.0);
        nested_tangent_workspace.fill(0.0);
        pack_owned_inputs(&callee_program.input_specs, &input_values, nested_workspace);
        pack_owned_inputs(
            &callee_program.input_specs,
            &tangent_values,
            nested_tangent_workspace,
        );
        push_forward_program_layers(
            module,
            callee_program,
            nested_workspace,
            nested_tangent_workspace,
        )?;
        (
            collect_outputs(callee_program, nested_workspace),
            collect_outputs(callee_program, nested_tangent_workspace),
        )
    };
    write_evaluate_outputs(&evaluate_layer.output_bindings, &nested_outputs, workspace);
    write_evaluate_outputs(
        &evaluate_layer.output_bindings,
        &nested_tangent_outputs,
        tangent_workspace,
    );
    Ok(())
}

fn materialize_evaluate_inputs(
    bindings: &[EvaluateInputBinding],
    workspace: &[f32],
) -> Vec<Vec<f32>> {
    bindings
        .iter()
        .map(|binding| match binding {
            EvaluateInputBinding::WorkspaceSlice { offset, length } => {
                let start = *offset as usize;
                let stop = start + *length as usize;
                workspace[start..stop].to_vec()
            }
            EvaluateInputBinding::ConstantSlice { values, .. } => values.clone(),
        })
        .collect()
}

fn materialize_evaluate_tangents(
    bindings: &[EvaluateInputBinding],
    tangent_workspace: &[f32],
) -> Vec<Vec<f32>> {
    bindings
        .iter()
        .map(|binding| match binding {
            EvaluateInputBinding::WorkspaceSlice { offset, length } => {
                let start = *offset as usize;
                let stop = start + *length as usize;
                tangent_workspace[start..stop].to_vec()
            }
            EvaluateInputBinding::ConstantSlice { length, .. } => {
                vec![0.0; *length as usize]
            }
        })
        .collect()
}

fn write_evaluate_outputs(
    bindings: &[coker_bytecode::EvaluateOutputBinding],
    output_values: &[Vec<f32>],
    workspace: &mut [f32],
) {
    for (binding, output_value) in bindings.iter().zip(output_values.iter()) {
        let start = binding.destination_offset as usize;
        let stop = start + binding.length as usize;
        workspace[start..stop].copy_from_slice(output_value);
    }
}

fn evaluate_generic_value(
    row_operation: &RowOp,
    input_slice: &[f32],
) -> Result<f32, RuntimeError> {
    let first = operand_value(row_operation.first, input_slice)?;
    let second = operand_value(row_operation.second, input_slice)?;
    let third = operand_value(row_operation.third, input_slice)?;

    match row_operation.op {
        ScalarOp::Identity => Ok(required_operand(first, "identity")?),
        ScalarOp::Sin => Ok(required_operand(first, "sin")?.sin()),
        ScalarOp::Cos => Ok(required_operand(first, "cos")?.cos()),
        ScalarOp::Tan => Ok(required_operand(first, "tan")?.tan()),
        ScalarOp::Exp => Ok(required_operand(first, "exp")?.exp()),
        ScalarOp::Sqrt => Ok(required_operand(first, "sqrt")?.sqrt()),
        ScalarOp::Log => Ok(required_operand(first, "log")?.ln()),
        ScalarOp::Neg => Ok(-required_operand(first, "neg")?),
        ScalarOp::Abs => Ok(required_operand(first, "abs")?.abs()),
        ScalarOp::Add => {
            Ok(required_operand(first, "add")? + required_operand(second, "add")?)
        }
        ScalarOp::Sub => {
            Ok(required_operand(first, "sub")? - required_operand(second, "sub")?)
        }
        ScalarOp::Mul => {
            Ok(required_operand(first, "mul")? * required_operand(second, "mul")?)
        }
        ScalarOp::Div => Ok(divide(
            required_operand(first, "div")?,
            required_operand(second, "div")?,
        )),
        ScalarOp::Pow | ScalarOp::IntPow => {
            Ok(required_operand(first, "pow")?.powf(required_operand(second, "pow")?))
        }
        ScalarOp::Atan2 => Ok(
            required_operand(first, "atan2")?
                .atan2(required_operand(second, "atan2")?),
        ),
        ScalarOp::Equal => Ok(
            (required_operand(first, "equal")? == required_operand(second, "equal")?)
                as u8 as f32,
        ),
        ScalarOp::LessThan => Ok(
            (required_operand(first, "less_than")?
                < required_operand(second, "less_than")?) as u8 as f32,
        ),
        ScalarOp::LessEqual => Ok(
            (required_operand(first, "less_equal")?
                <= required_operand(second, "less_equal")?) as u8 as f32,
        ),
        ScalarOp::Case => {
            if required_operand(first, "case")? != 0.0 {
                Ok(required_operand(second, "case")?)
            } else {
                Ok(required_operand(third, "case")?)
            }
        }
    }
}

fn evaluate_generic_push_forward(
    row_operation: &RowOp,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
) -> Result<(f32, f32), RuntimeError> {
    let first = operand_value(row_operation.first, input_slice)?;
    let second = operand_value(row_operation.second, input_slice)?;
    let third = operand_value(row_operation.third, input_slice)?;
    let first_tangent = operand_tangent(row_operation.first, tangent_input_slice)?;
    let second_tangent = operand_tangent(row_operation.second, tangent_input_slice)?;
    let third_tangent = operand_tangent(row_operation.third, tangent_input_slice)?;

    match row_operation.op {
        ScalarOp::Identity => Ok((
            required_operand(first, "identity")?,
            required_operand(first_tangent, "identity")?,
        )),
        ScalarOp::Sin => {
            let value = required_operand(first, "sin")?;
            let tangent = required_operand(first_tangent, "sin")?;
            Ok((value.sin(), value.cos() * tangent))
        }
        ScalarOp::Cos => {
            let value = required_operand(first, "cos")?;
            let tangent = required_operand(first_tangent, "cos")?;
            Ok((value.cos(), -value.sin() * tangent))
        }
        ScalarOp::Tan => {
            let value = required_operand(first, "tan")?;
            let tangent = required_operand(first_tangent, "tan")?;
            Ok((value.tan(), tangent / (value.cos() * value.cos())))
        }
        ScalarOp::Exp => {
            let value = required_operand(first, "exp")?.exp();
            Ok((value, value * required_operand(first_tangent, "exp")?))
        }
        ScalarOp::Sqrt => {
            let value = required_operand(first, "sqrt")?.sqrt();
            Ok((value, required_operand(first_tangent, "sqrt")? / (2.0 * value)))
        }
        ScalarOp::Log => {
            let value = required_operand(first, "log")?;
            Ok((value.ln(), required_operand(first_tangent, "log")? / value))
        }
        ScalarOp::Neg => Ok((
            -required_operand(first, "neg")?,
            -required_operand(first_tangent, "neg")?,
        )),
        ScalarOp::Abs => {
            let value = required_operand(first, "abs")?;
            Ok((value.abs(), value.signum() * required_operand(first_tangent, "abs")?))
        }
        ScalarOp::Add => Ok((
            required_operand(first, "add")? + required_operand(second, "add")?,
            required_operand(first_tangent, "add")?
                + required_operand(second_tangent, "add")?,
        )),
        ScalarOp::Sub => Ok((
            required_operand(first, "sub")? - required_operand(second, "sub")?,
            required_operand(first_tangent, "sub")?
                - required_operand(second_tangent, "sub")?,
        )),
        ScalarOp::Mul => {
            let first_value = required_operand(first, "mul")?;
            let second_value = required_operand(second, "mul")?;
            Ok((
                first_value * second_value,
                second_value * required_operand(first_tangent, "mul")?
                    + first_value * required_operand(second_tangent, "mul")?,
            ))
        }
        ScalarOp::Div => {
            let numerator = required_operand(first, "div")?;
            let denominator = required_operand(second, "div")?;
            if denominator == 0.0 {
                return Ok((f32::NAN, f32::NAN));
            }
            Ok((
                divide(numerator, denominator),
                divide(
                    required_operand(first_tangent, "div")? * denominator
                        - numerator * required_operand(second_tangent, "div")?,
                    denominator * denominator,
                ),
            ))
        }
        ScalarOp::Pow | ScalarOp::IntPow => {
            let base = required_operand(first, "pow")?;
            let exponent = required_operand(second, "pow")?;
            let value = base.powf(exponent);
            if base == 0.0 {
                return Ok((value, 0.0));
            }
            Ok((
                value,
                value
                    * (required_operand(second_tangent, "pow")? * base.ln()
                        + exponent * required_operand(first_tangent, "pow")? / base),
            ))
        }
        ScalarOp::Atan2 => {
            let first_value = required_operand(first, "atan2")?;
            let second_value = required_operand(second, "atan2")?;
            let denominator = first_value * first_value + second_value * second_value;
            Ok((
                first_value.atan2(second_value),
                (second_value * required_operand(first_tangent, "atan2")?
                    - first_value * required_operand(second_tangent, "atan2")?)
                    / denominator,
            ))
        }
        ScalarOp::Equal => Ok((
            (required_operand(first, "equal")? == required_operand(second, "equal")?)
                as u8 as f32,
            0.0,
        )),
        ScalarOp::LessThan => Ok((
            (required_operand(first, "less_than")?
                < required_operand(second, "less_than")?) as u8 as f32,
            0.0,
        )),
        ScalarOp::LessEqual => Ok((
            (required_operand(first, "less_equal")?
                <= required_operand(second, "less_equal")?) as u8 as f32,
            0.0,
        )),
        ScalarOp::Case => {
            if required_operand(first, "case")? != 0.0 {
                Ok((
                    required_operand(second, "case")?,
                    required_operand(second_tangent, "case")?,
                ))
            } else {
                Ok((
                    required_operand(third, "case")?,
                    required_operand(third_tangent, "case")?,
                ))
            }
        }
    }
}

fn operand_value(
    operand_index: u16,
    input_slice: &[f32],
) -> Result<Option<f32>, RuntimeError> {
    if operand_index == UNUSED_OPERAND {
        return Ok(None);
    }
    Ok(Some(input_slice[operand_index as usize]))
}

fn operand_tangent(
    operand_index: u16,
    tangent_input_slice: &[f32],
) -> Result<Option<f32>, RuntimeError> {
    if operand_index == UNUSED_OPERAND {
        return Ok(None);
    }
    Ok(Some(tangent_input_slice[operand_index as usize]))
}

fn required_operand(
    operand_value: Option<f32>,
    context: &'static str,
) -> Result<f32, RuntimeError> {
    operand_value.ok_or_else(|| {
        RuntimeError::Validation(format!(
            "missing operand for generic operation {context}"
        ))
    })
}

fn homogeneous_value(input_slice: &[f32], index: u16) -> Result<f32, RuntimeError> {
    if index == 0 {
        return Ok(1.0);
    }
    let offset = index as usize - 1;
    Ok(input_slice[offset])
}

fn homogeneous_tangent(
    tangent_input_slice: &[f32],
    index: u16,
) -> Result<f32, RuntimeError> {
    if index == 0 {
        return Ok(0.0);
    }
    let offset = index as usize - 1;
    Ok(tangent_input_slice[offset])
}

fn divide(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        f32::NAN
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coker_bytecode::{
        encode_module, EvaluateOutputBinding, SparseEntry, SparseTensor,
    };

    fn build_nested_module() -> BytecodeModule {
        let callee_program = Program::new(
            1,
            2,
            2,
            vec![InputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![OutputSpec {
                workspace_offset: 1,
                length: 1,
            }],
            vec![Layer::Generic(GenericLayer {
                in_offset: 0,
                out_offset: 0,
                in_length: 1,
                out_length: 2,
                ops: vec![
                    RowOp {
                        first: 0,
                        second: UNUSED_OPERAND,
                        third: UNUSED_OPERAND,
                        op: ScalarOp::Identity,
                    },
                    RowOp {
                        first: 0,
                        second: UNUSED_OPERAND,
                        third: UNUSED_OPERAND,
                        op: ScalarOp::Sin,
                    },
                ],
            })],
        );
        let entry_program = Program::new(
            0,
            1,
            3,
            vec![],
            vec![OutputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![Layer::Evaluate(EvaluateLayer {
                scratch_offset: 1,
                callee_function_id: 1,
                input_count: 1,
                output_count: 1,
                input_bindings: vec![EvaluateInputBinding::ConstantSlice {
                    length: 1,
                    values: vec![2.0],
                }],
                output_bindings: vec![EvaluateOutputBinding {
                    destination_offset: 0,
                    length: 1,
                }],
            })],
        );
        BytecodeModule::new(vec![entry_program, callee_program])
    }

    #[test]
    fn execute_bilinear_homogeneous_tensor() {
        let module = BytecodeModule::new(vec![Program::new(
            0,
            2,
            2,
            vec![InputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![OutputSpec {
                workspace_offset: 1,
                length: 1,
            }],
            vec![Layer::Bilinear(BilinearLayer {
                in_offset: 0,
                out_offset: 1,
                in_length: 1,
                out_length: 1,
                quadratic: SparseTensor {
                    shape: (1, 2, 2),
                    entries: vec![
                        SparseEntry {
                            index: (0, 0, 0),
                            value: 3.0,
                        },
                        SparseEntry {
                            index: (0, 0, 1),
                            value: 2.0,
                        },
                        SparseEntry {
                            index: (0, 1, 1),
                            value: 4.0,
                        },
                    ],
                },
            })],
        )]);
        let mut workspace = vec![0.0; 2];
        let outputs = execute(&module, &[&[1.5]], &mut workspace).unwrap();
        assert_eq!(outputs[0], vec![15.0]);
    }

    #[test]
    fn push_forward_bilinear_homogeneous_tensor() {
        let module = BytecodeModule::new(vec![Program::new(
            0,
            2,
            2,
            vec![InputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![OutputSpec {
                workspace_offset: 1,
                length: 1,
            }],
            vec![Layer::Bilinear(BilinearLayer {
                in_offset: 0,
                out_offset: 1,
                in_length: 1,
                out_length: 1,
                quadratic: SparseTensor {
                    shape: (1, 2, 2),
                    entries: vec![
                        SparseEntry {
                            index: (0, 0, 1),
                            value: 3.0,
                        },
                        SparseEntry {
                            index: (0, 1, 1),
                            value: 2.0,
                        },
                    ],
                },
            })],
        )]);
        let mut workspace = vec![0.0; 2];
        let mut tangent_workspace = vec![0.0; 2];
        let (outputs, tangents) = push_forward(
            &module,
            &[&[2.0]],
            &[&[0.5]],
            &mut workspace,
            &mut tangent_workspace,
        )
        .unwrap();
        assert_eq!(outputs[0], vec![14.0]);
        assert_eq!(tangents[0], vec![5.5]);
    }

    #[test]
    fn execute_generic_layer_operations() {
        let module = BytecodeModule::new(vec![Program::new(
            0,
            2,
            2,
            vec![InputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![OutputSpec {
                workspace_offset: 1,
                length: 1,
            }],
            vec![Layer::Generic(GenericLayer {
                in_offset: 0,
                out_offset: 0,
                in_length: 1,
                out_length: 2,
                ops: vec![
                    RowOp {
                        first: 0,
                        second: UNUSED_OPERAND,
                        third: UNUSED_OPERAND,
                        op: ScalarOp::Identity,
                    },
                    RowOp {
                        first: 0,
                        second: UNUSED_OPERAND,
                        third: UNUSED_OPERAND,
                        op: ScalarOp::Sin,
                    },
                ],
            })],
        )]);
        let mut workspace = vec![0.0; 2];
        let outputs = execute(&module, &[&[1.0]], &mut workspace).unwrap();
        assert_eq!(outputs[0][0], 1.0f32.sin());
    }

    #[test]
    fn execute_evaluate_layer_calls_nested_function() {
        let module = build_nested_module();
        let mut workspace = vec![0.0; 3];
        let outputs = execute(&module, &[], &mut workspace).unwrap();
        assert_eq!(outputs[0], vec![2.0f32.sin()]);
    }

    #[test]
    fn push_forward_evaluate_layer_calls_nested_function() {
        let callee_program = Program::new(
            1,
            3,
            3,
            vec![InputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![OutputSpec {
                workspace_offset: 2,
                length: 1,
            }],
            vec![Layer::Bilinear(BilinearLayer {
                in_offset: 0,
                out_offset: 2,
                in_length: 1,
                out_length: 1,
                quadratic: SparseTensor {
                    shape: (1, 2, 2),
                    entries: vec![
                        SparseEntry {
                            index: (0, 0, 1),
                            value: 1.0,
                        },
                        SparseEntry {
                            index: (0, 1, 1),
                            value: 1.0,
                        },
                    ],
                },
            })],
        );
        let entry_program = Program::new(
            0,
            1,
            4,
            vec![InputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![OutputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![Layer::Evaluate(EvaluateLayer {
                scratch_offset: 1,
                callee_function_id: 1,
                input_count: 1,
                output_count: 1,
                input_bindings: vec![EvaluateInputBinding::WorkspaceSlice {
                    offset: 0,
                    length: 1,
                }],
                output_bindings: vec![EvaluateOutputBinding {
                    destination_offset: 0,
                    length: 1,
                }],
            })],
        );
        let module = BytecodeModule::new(vec![entry_program, callee_program]);
        let mut workspace = vec![0.0; 4];
        let mut tangent_workspace = vec![0.0; 4];
        let (outputs, tangents) = push_forward(
            &module,
            &[&[2.0]],
            &[&[0.5]],
            &mut workspace,
            &mut tangent_workspace,
        )
        .unwrap();
        assert_eq!(outputs[0], vec![6.0]);
        assert_eq!(tangents[0], vec![2.5]);
    }

    #[test]
    fn parse_and_validate_round_trip() {
        let module = BytecodeModule::new(vec![Program::new(0, 0, 0, vec![], vec![], vec![])]);
        let encoded = encode_module(&module).unwrap();
        let decoded = validate_module(&encoded).unwrap();
        assert_eq!(decoded.functions[0].workspace_size, 0);
    }
}
