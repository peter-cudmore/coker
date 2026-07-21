use coker_bytecode::{BilinearLayer, BytecodeModule, EvaluateLayer, GenericLayer, Layer, Program};

use crate::{
    find_function,
    ops::{
        evaluate_generic_push_forward, evaluate_generic_value, homogeneous_tangent,
        homogeneous_value,
    },
    workspace::{
        collect_outputs, materialize_evaluate_inputs, materialize_evaluate_tangents,
        pack_owned_inputs, write_evaluate_outputs,
    },
    RuntimeError,
};

fn with_input_and_output_slice<T, F>(
    workspace: &mut [f32],
    input_start: usize,
    input_stop: usize,
    output_start: usize,
    output_stop: usize,
    f: F,
) -> T
where
    F: FnOnce(&[f32], &mut [f32]) -> T,
{
    // Runtime target: allocate the whole workspace up front, then give each layer
    // exclusive write access to its output slice. Other layers must not write that
    // slice, and only later layers may read it. When a program does not yet meet
    // that contract, fall back to an input copy instead of aliasing overlapping
    // regions.
    if input_stop <= output_start {
        let (before_output, output_and_after) = workspace.split_at_mut(output_start);
        f(
            &before_output[input_start..input_stop],
            &mut output_and_after[..output_stop - output_start],
        )
    } else if output_stop <= input_start {
        let (output_and_before, after_output) = workspace.split_at_mut(input_start);
        f(
            &after_output[..input_stop - input_start],
            &mut output_and_before[output_start..output_stop],
        )
    } else {
        let input_slice = workspace[input_start..input_stop].to_vec();
        let output_slice = &mut workspace[output_start..output_stop];
        f(&input_slice, output_slice)
    }
}

pub(crate) fn execute_program_layers(
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

pub(crate) fn push_forward_program_layers(
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

    with_input_and_output_slice(
        workspace,
        input_start,
        input_stop,
        output_start,
        output_stop,
        |input_slice, output_slice| {
            output_slice.fill(0.0);

            for entry in &bilinear_layer.quadratic.entries {
                let row_index = entry.index.0 as usize;
                let left_value = homogeneous_value(input_slice, entry.index.1)?;
                let right_value = homogeneous_value(input_slice, entry.index.2)?;
                output_slice[row_index] += entry.value * left_value * right_value;
            }

            Ok(())
        },
    )
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

    with_input_and_output_slice(
        workspace,
        input_start,
        input_stop,
        output_start,
        output_stop,
        |input_slice, output_slice| {
            with_input_and_output_slice(
                tangent_workspace,
                input_start,
                input_stop,
                output_start,
                output_stop,
                |tangent_input_slice, tangent_output_slice| {
                    output_slice.fill(0.0);
                    tangent_output_slice.fill(0.0);

                    for entry in &bilinear_layer.quadratic.entries {
                        let row_index = entry.index.0 as usize;
                        let left_value = homogeneous_value(input_slice, entry.index.1)?;
                        let right_value = homogeneous_value(input_slice, entry.index.2)?;
                        let left_tangent =
                            homogeneous_tangent(tangent_input_slice, entry.index.1)?;
                        let right_tangent =
                            homogeneous_tangent(tangent_input_slice, entry.index.2)?;
                        output_slice[row_index] += entry.value * left_value * right_value;
                        tangent_output_slice[row_index] += entry.value
                            * (left_tangent * right_value + left_value * right_tangent);
                    }

                    Ok(())
                },
            )
        },
    )
}

fn execute_generic_layer(
    generic_layer: &GenericLayer,
    workspace: &mut [f32],
) -> Result<(), RuntimeError> {
    let input_start = generic_layer.in_offset as usize;
    let input_stop = input_start + generic_layer.in_length as usize;
    let output_start = generic_layer.out_offset as usize;
    let output_stop = output_start + generic_layer.out_length as usize;

    with_input_and_output_slice(
        workspace,
        input_start,
        input_stop,
        output_start,
        output_stop,
        |input_slice, output_slice| {
            for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
                output_slice[row_index] = evaluate_generic_value(row_operation, input_slice)?;
            }

            Ok(())
        },
    )
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

    with_input_and_output_slice(
        workspace,
        input_start,
        input_stop,
        output_start,
        output_stop,
        |input_slice, output_slice| {
            with_input_and_output_slice(
                tangent_workspace,
                input_start,
                input_stop,
                output_start,
                output_stop,
                |tangent_input_slice, tangent_output_slice| {
                    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
                        let (value, tangent) = evaluate_generic_push_forward(
                            row_operation,
                            input_slice,
                            tangent_input_slice,
                        )?;
                        output_slice[row_index] = value;
                        tangent_output_slice[row_index] = tangent;
                    }

                    Ok(())
                },
            )
        },
    )
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
