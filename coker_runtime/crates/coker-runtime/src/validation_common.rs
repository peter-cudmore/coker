use alloc::format;

use crate::{RuntimeError, SpecInfo};

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_layer_scratch(
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

pub(crate) fn range_end(workspace_offset: u32, length: u16) -> usize {
    workspace_offset as usize + length as usize
}

pub(crate) fn validate_range(
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

pub(crate) fn validate_inputs<I: SpecInfo>(
    input_specs: &[I],
    inputs: &[&[f32]],
) -> Result<(), RuntimeError> {
    if inputs.len() != input_specs.len() {
        return Err(RuntimeError::InputCountMismatch {
            expected: input_specs.len(),
            actual: inputs.len(),
        });
    }

    for (index, (input_spec, input_value)) in input_specs.iter().zip(inputs.iter()).enumerate() {
        let expected_count = input_spec.length();
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

pub(crate) fn validate_outputs<O: SpecInfo>(
    output_specs: &[O],
    outputs: &[f32],
) -> Result<(), RuntimeError> {
    let expected_size: usize = output_specs.iter().map(SpecInfo::length).sum();
    let actual_size = outputs.len();
    if actual_size != expected_size {
        return Err(RuntimeError::OutputBufferSizeMismatch {
            expected: expected_size,
            actual: actual_size,
        });
    }
    Ok(())
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
