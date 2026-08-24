use super::*;
use crate::workspace::final_layer_matches_outputs;

mod evaluate_execute;
mod layer_execute;

#[allow(unused_imports)]
pub(super) use self::{evaluate_execute::*, layer_execute::*};

pub(super) use crate::workspace::prepare_input_range;

pub(super) fn execute_in_place_unchecked(
    module: &ArchivedBytecodeModule,
    entry_program: &ArchivedProgram,
    inputs: &[&[f32]],
    workspace: &mut [f32],
    outputs: Option<&mut [f32]>,
) -> bool {
    let mut workspace = Workspace::new(workspace);
    workspace.fill(0.0);
    workspace.pack_inputs(entry_program.input_specs(), inputs);
    execute_program_layers(module, entry_program, &mut workspace, outputs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn push_forward_in_place_unchecked(
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
    workspace.pack_inputs(entry_program.input_specs(), inputs);
    tangent_workspace.pack_inputs(entry_program.input_specs(), tangents);
    push_forward_program_layers(
        module,
        entry_program,
        &mut workspace,
        &mut tangent_workspace,
        outputs,
        tangent_outputs,
    )
}

pub(super) fn execute_program_layers(
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
                            program.output_specs(),
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
                            program.output_specs(),
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
                ArchivedLayer::Evaluate(_) | ArchivedLayer::QpCall(_) => false,
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
            ArchivedLayer::QpCall(_) => {}
        }
    }
    false
}

pub(super) fn push_forward_program_layers(
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
                            program.output_specs(),
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
                            program.output_specs(),
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
                ArchivedLayer::Evaluate(_) | ArchivedLayer::QpCall(_) => false,
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
            ArchivedLayer::QpCall(_) => {}
        }
    }
    false
}

pub(super) fn pack_evaluate_inputs(
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

pub(super) fn pack_evaluate_tangents(
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

pub(super) fn copy_evaluate_outputs(
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
