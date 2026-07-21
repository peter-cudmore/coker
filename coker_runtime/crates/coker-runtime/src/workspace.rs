use alloc::{vec, vec::Vec};
use coker_bytecode::{EvaluateInputBinding, InputSpec, Program};

pub(crate) fn pack_inputs(input_specs: &[InputSpec], inputs: &[&[f32]], workspace: &mut [f32]) {
    for (input_spec, input_value) in input_specs.iter().zip(inputs.iter()) {
        let start = input_spec.workspace_offset as usize;
        let stop = start + input_spec.length as usize;
        workspace[start..stop].copy_from_slice(input_value);
    }
}

pub(crate) fn pack_owned_inputs(
    input_specs: &[InputSpec],
    inputs: &[Vec<f32>],
    workspace: &mut [f32],
) {
    for (input_spec, input_value) in input_specs.iter().zip(inputs.iter()) {
        let start = input_spec.workspace_offset as usize;
        let stop = start + input_spec.length as usize;
        workspace[start..stop].copy_from_slice(input_value);
    }
}

pub(crate) fn collect_outputs(program: &Program, workspace: &[f32]) -> Vec<Vec<f32>> {
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

pub(crate) fn materialize_evaluate_inputs(
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

pub(crate) fn materialize_evaluate_tangents(
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

pub(crate) fn write_evaluate_outputs(
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
