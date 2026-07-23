use coker_bytecode::{BilinearLayer, BytecodeModule, EvaluateLayer, GenericLayer, Layer, Program};

use crate::{
    find_function_unchecked,
    ops::{
        evaluate_generic_push_forward, evaluate_generic_value, homogeneous_tangent,
        homogeneous_value,
    },
    workspace::{
        copy_evaluate_outputs, final_layer_matches_outputs, pack_evaluate_inputs,
        pack_evaluate_tangents, Workspace,
    },
};

pub(crate) fn execute_program_layers(
    module: &BytecodeModule,
    program: &Program,
    workspace: &mut Workspace<'_>,
    final_outputs: Option<&mut [f32]>,
) -> bool {
    let last_layer_index = program.intermediate_layers.len().saturating_sub(1);
    let mut final_outputs = final_outputs;
    for (layer_index, layer) in program.intermediate_layers.iter().enumerate() {
        let is_final_layer = layer_index == last_layer_index;
        let wrote_direct_outputs = if is_final_layer {
            match layer {
                Layer::Bilinear(bilinear_layer) => final_outputs
                    .as_deref_mut()
                    .filter(|_| {
                        final_layer_matches_outputs(
                            &program.output_specs,
                            bilinear_layer.out_offset,
                            bilinear_layer.out_length,
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
                Layer::Generic(generic_layer) => final_outputs
                    .as_deref_mut()
                    .filter(|_| {
                        final_layer_matches_outputs(
                            &program.output_specs,
                            generic_layer.out_offset,
                            generic_layer.out_length,
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
                Layer::Evaluate(_) => false,
            }
        } else {
            false
        };

        if wrote_direct_outputs {
            return true;
        }

        match layer {
            Layer::Bilinear(bilinear_layer) => execute_bilinear_layer(bilinear_layer, workspace),
            Layer::Generic(generic_layer) => execute_generic_layer(generic_layer, workspace),
            Layer::Evaluate(evaluate_layer) => {
                execute_evaluate_layer(module, evaluate_layer, workspace)
            }
        }
    }
    false
}

pub(crate) fn push_forward_program_layers(
    module: &BytecodeModule,
    program: &Program,
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
                Layer::Bilinear(bilinear_layer) => match (
                    final_outputs.as_deref_mut(),
                    final_tangent_outputs.as_deref_mut(),
                ) {
                    (Some(output_buffer), Some(tangent_output_buffer))
                        if final_layer_matches_outputs(
                            &program.output_specs,
                            bilinear_layer.out_offset,
                            bilinear_layer.out_length,
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
                Layer::Generic(generic_layer) => match (
                    final_outputs.as_deref_mut(),
                    final_tangent_outputs.as_deref_mut(),
                ) {
                    (Some(output_buffer), Some(tangent_output_buffer))
                        if final_layer_matches_outputs(
                            &program.output_specs,
                            generic_layer.out_offset,
                            generic_layer.out_length,
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
                Layer::Evaluate(_) => false,
            }
        } else {
            false
        };

        if wrote_direct_outputs {
            return true;
        }

        match layer {
            Layer::Bilinear(bilinear_layer) => {
                execute_bilinear_push_forward(bilinear_layer, workspace, tangent_workspace)
            }
            Layer::Generic(generic_layer) => {
                execute_generic_push_forward(generic_layer, workspace, tangent_workspace)
            }
            Layer::Evaluate(evaluate_layer) => {
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

fn execute_bilinear_layer(bilinear_layer: &BilinearLayer, workspace: &mut Workspace<'_>) {
    let input_start = bilinear_layer.in_offset as usize;
    let input_stop = input_start + bilinear_layer.in_length as usize;
    let output_start = bilinear_layer.out_offset as usize;
    let output_stop = output_start + bilinear_layer.out_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        bilinear_layer.scratch_offset,
        bilinear_layer.scratch_length,
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
    bilinear_layer: &BilinearLayer,
    workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
) {
    let input_start = bilinear_layer.in_offset as usize;
    let input_stop = input_start + bilinear_layer.in_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        bilinear_layer.scratch_offset,
        bilinear_layer.scratch_length,
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    execute_bilinear_into_slice(bilinear_layer, input_slice, output_buffer);
}

fn execute_bilinear_into_slice(
    bilinear_layer: &BilinearLayer,
    input_slice: &[f32],
    output_slice: &mut [f32],
) {
    output_slice.fill(0.0);
    for entry in &bilinear_layer.quadratic.entries {
        let row_index = entry.index.0 as usize;
        let left_value = homogeneous_value(input_slice, entry.index.1);
        let right_value = homogeneous_value(input_slice, entry.index.2);
        output_slice[row_index] += entry.value * left_value * right_value;
    }
}

fn execute_bilinear_push_forward(
    bilinear_layer: &BilinearLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let input_start = bilinear_layer.in_offset as usize;
    let input_stop = input_start + bilinear_layer.in_length as usize;
    let output_start = bilinear_layer.out_offset as usize;
    let output_stop = output_start + bilinear_layer.out_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        bilinear_layer.scratch_offset,
        bilinear_layer.scratch_length,
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        bilinear_layer.scratch_offset,
        bilinear_layer.scratch_length,
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
    bilinear_layer: &BilinearLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
    tangent_output_buffer: &mut [f32],
) {
    let input_start = bilinear_layer.in_offset as usize;
    let input_stop = input_start + bilinear_layer.in_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        bilinear_layer.scratch_offset,
        bilinear_layer.scratch_length,
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        bilinear_layer.scratch_offset,
        bilinear_layer.scratch_length,
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
    bilinear_layer: &BilinearLayer,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
    output_slice: &mut [f32],
    tangent_output_slice: &mut [f32],
) {
    output_slice.fill(0.0);
    tangent_output_slice.fill(0.0);
    for entry in &bilinear_layer.quadratic.entries {
        let row_index = entry.index.0 as usize;
        let left_value = homogeneous_value(input_slice, entry.index.1);
        let right_value = homogeneous_value(input_slice, entry.index.2);
        let left_tangent = homogeneous_tangent(tangent_input_slice, entry.index.1);
        let right_tangent = homogeneous_tangent(tangent_input_slice, entry.index.2);
        output_slice[row_index] += entry.value * left_value * right_value;
        tangent_output_slice[row_index] +=
            entry.value * (left_tangent * right_value + left_value * right_tangent);
    }
}

fn execute_generic_layer(generic_layer: &GenericLayer, workspace: &mut Workspace<'_>) {
    let input_start = generic_layer.in_offset as usize;
    let input_stop = input_start + generic_layer.in_length as usize;
    let output_start = generic_layer.out_offset as usize;
    let output_stop = output_start + generic_layer.out_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        generic_layer.scratch_offset,
        generic_layer.scratch_length,
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
    generic_layer: &GenericLayer,
    workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
) {
    let input_start = generic_layer.in_offset as usize;
    let input_stop = input_start + generic_layer.in_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        generic_layer.scratch_offset,
        generic_layer.scratch_length,
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    execute_generic_into_slice(generic_layer, input_slice, output_buffer);
}

fn execute_generic_into_slice(
    generic_layer: &GenericLayer,
    input_slice: &[f32],
    output_slice: &mut [f32],
) {
    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        output_slice[row_index] = evaluate_generic_value(row_operation, input_slice);
    }
}

fn execute_generic_push_forward(
    generic_layer: &GenericLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let input_start = generic_layer.in_offset as usize;
    let input_stop = input_start + generic_layer.in_length as usize;
    let output_start = generic_layer.out_offset as usize;
    let output_stop = output_start + generic_layer.out_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        generic_layer.scratch_offset,
        generic_layer.scratch_length,
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        generic_layer.scratch_offset,
        generic_layer.scratch_length,
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
    generic_layer: &GenericLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
    tangent_output_buffer: &mut [f32],
) {
    let input_start = generic_layer.in_offset as usize;
    let input_stop = input_start + generic_layer.in_length as usize;
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        generic_layer.scratch_offset,
        generic_layer.scratch_length,
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        generic_layer.scratch_offset,
        generic_layer.scratch_length,
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
    generic_layer: &GenericLayer,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
    output_slice: &mut [f32],
    tangent_output_slice: &mut [f32],
) {
    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        let (value, tangent) =
            evaluate_generic_push_forward(row_operation, input_slice, tangent_input_slice);
        output_slice[row_index] = value;
        tangent_output_slice[row_index] = tangent;
    }
}

fn execute_evaluate_layer(
    module: &BytecodeModule,
    evaluate_layer: &EvaluateLayer,
    workspace: &mut Workspace<'_>,
) {
    let callee_program = find_function_unchecked(module, evaluate_layer.callee_function_id);
    let scratch_start = evaluate_layer.scratch_offset as usize;
    let scratch_length = callee_program.required_workspace_size as usize;
    let caller_workspace = Workspace::new(workspace.as_mut_slice());
    let (mut caller_workspace, scratch_workspace) = caller_workspace.split_at_mut(scratch_start);
    let mut nested_workspace = scratch_workspace.truncate(scratch_length);
    nested_workspace.fill(0.0);
    pack_evaluate_inputs(
        &callee_program.input_specs,
        &evaluate_layer.input_bindings,
        caller_workspace.as_slice(),
        nested_workspace.as_mut_slice(),
    );
    execute_program_layers(module, callee_program, &mut nested_workspace, None);
    copy_evaluate_outputs(
        &callee_program.output_specs,
        &evaluate_layer.output_bindings,
        nested_workspace.as_slice(),
        caller_workspace.as_mut_slice(),
    );
}

fn execute_evaluate_push_forward(
    module: &BytecodeModule,
    evaluate_layer: &EvaluateLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let callee_program = find_function_unchecked(module, evaluate_layer.callee_function_id);
    let scratch_start = evaluate_layer.scratch_offset as usize;
    let scratch_length = callee_program.required_workspace_size as usize;
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
        &callee_program.input_specs,
        &evaluate_layer.input_bindings,
        caller_workspace.as_slice(),
        nested_workspace.as_mut_slice(),
    );
    pack_evaluate_tangents(
        &callee_program.input_specs,
        &evaluate_layer.input_bindings,
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
        &callee_program.output_specs,
        &evaluate_layer.output_bindings,
        nested_workspace.as_slice(),
        caller_workspace.as_mut_slice(),
    );
    copy_evaluate_outputs(
        &callee_program.output_specs,
        &evaluate_layer.output_bindings,
        nested_tangent_workspace.as_slice(),
        caller_tangent_workspace.as_mut_slice(),
    );
}
