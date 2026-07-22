use alloc::vec::Vec;
use coker_bytecode::{EvaluateInputBinding, EvaluateOutputBinding, InputSpec, OutputSpec, Program};

pub(crate) struct Workspace<'a> {
    values: &'a mut [f32],
}

impl<'a> Workspace<'a> {
    pub(crate) fn new(values: &'a mut [f32]) -> Self {
        Self { values }
    }

    pub(crate) fn fill(&mut self, value: f32) {
        self.values.fill(value);
    }

    pub(crate) fn pack_inputs(&mut self, input_specs: &[InputSpec], inputs: &[&[f32]]) {
        for (input_spec, input_value) in input_specs.iter().zip(inputs.iter()) {
            let start = input_spec.workspace_offset as usize;
            let stop = start + input_spec.length as usize;
            self.values[start..stop].copy_from_slice(input_value);
        }
    }

    pub(crate) fn input_output_slices(
        &mut self,
        input_start: usize,
        input_stop: usize,
        output_start: usize,
        output_stop: usize,
    ) -> (&[f32], &mut [f32]) {
        if input_stop <= output_start {
            let (before_output, output_and_after) = self.values.split_at_mut(output_start);
            (
                &before_output[input_start..input_stop],
                &mut output_and_after[..output_stop - output_start],
            )
        } else if output_stop <= input_start {
            let (output_and_before, after_output) = self.values.split_at_mut(input_start);
            (
                &after_output[..input_stop - input_start],
                &mut output_and_before[output_start..output_stop],
            )
        } else {
            unreachable!("validated layer input and output ranges must be disjoint")
        }
    }

    pub(crate) fn copy_range_to_scratch(
        &mut self,
        source_start: usize,
        source_stop: usize,
        scratch_start: usize,
    ) {
        let scratch_stop = scratch_start + (source_stop - source_start);
        let (primary_workspace, scratch_and_after) = self.values.split_at_mut(scratch_start);
        scratch_and_after[..scratch_stop - scratch_start]
            .copy_from_slice(&primary_workspace[source_start..source_stop]);
    }

    pub(crate) fn split_at_mut(self, index: usize) -> (Workspace<'a>, Workspace<'a>) {
        let (prefix, suffix) = self.values.split_at_mut(index);
        (Workspace::new(prefix), Workspace::new(suffix))
    }

    pub(crate) fn truncate(self, length: usize) -> Workspace<'a> {
        Workspace::new(&mut self.values[..length])
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        self.values
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [f32] {
        self.values
    }
}

pub(crate) fn write_outputs(program: &Program, workspace: &[f32], outputs: &mut [Vec<f32>]) {
    for (output_spec, output_buffer) in program.output_specs.iter().zip(outputs.iter_mut()) {
        let start = output_spec.workspace_offset as usize;
        let stop = start + output_spec.length as usize;
        output_buffer.copy_from_slice(&workspace[start..stop]);
    }
}

pub(crate) fn pack_evaluate_inputs(
    input_specs: &[InputSpec],
    bindings: &[EvaluateInputBinding],
    caller_workspace: &[f32],
    callee_workspace: &mut [f32],
) {
    for (input_spec, binding) in input_specs.iter().zip(bindings.iter()) {
        let destination_start = input_spec.workspace_offset as usize;
        let destination_stop = destination_start + input_spec.length as usize;
        match binding {
            EvaluateInputBinding::WorkspaceSlice { offset, length } => {
                let source_start = *offset as usize;
                let source_stop = source_start + *length as usize;
                callee_workspace[destination_start..destination_stop]
                    .copy_from_slice(&caller_workspace[source_start..source_stop]);
            }
            EvaluateInputBinding::ConstantSlice { values, .. } => {
                callee_workspace[destination_start..destination_stop].copy_from_slice(values);
            }
        }
    }
}

pub(crate) fn pack_evaluate_tangents(
    input_specs: &[InputSpec],
    bindings: &[EvaluateInputBinding],
    caller_tangent_workspace: &[f32],
    callee_tangent_workspace: &mut [f32],
) {
    for (input_spec, binding) in input_specs.iter().zip(bindings.iter()) {
        let destination_start = input_spec.workspace_offset as usize;
        let destination_stop = destination_start + input_spec.length as usize;
        match binding {
            EvaluateInputBinding::WorkspaceSlice { offset, length } => {
                let source_start = *offset as usize;
                let source_stop = source_start + *length as usize;
                callee_tangent_workspace[destination_start..destination_stop]
                    .copy_from_slice(&caller_tangent_workspace[source_start..source_stop]);
            }
            EvaluateInputBinding::ConstantSlice { .. } => {
                callee_tangent_workspace[destination_start..destination_stop].fill(0.0);
            }
        }
    }
}

pub(crate) fn copy_evaluate_outputs(
    output_specs: &[OutputSpec],
    bindings: &[EvaluateOutputBinding],
    callee_workspace: &[f32],
    caller_workspace: &mut [f32],
) {
    for (output_spec, binding) in output_specs.iter().zip(bindings.iter()) {
        let source_start = output_spec.workspace_offset as usize;
        let source_stop = source_start + output_spec.length as usize;
        let destination_start = binding.destination_offset as usize;
        let destination_stop = destination_start + binding.length as usize;
        caller_workspace[destination_start..destination_stop]
            .copy_from_slice(&callee_workspace[source_start..source_stop]);
    }
}
