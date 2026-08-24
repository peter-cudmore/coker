use super::*;

pub(crate) fn execute_bilinear_layer(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace: &mut Workspace<'_>,
) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let output_start = us32(bilinear_layer.out_offset);
    let output_stop = output_start + us16(bilinear_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let (input_slice, output_slice) = workspace.input_output_slices(
        prepared_input_range.0,
        prepared_input_range.1,
        output_start,
        output_stop,
    );
    execute_bilinear_into_slice(bilinear_layer, input_slice, output_slice);
}

pub(super) fn execute_bilinear_layer_to_output_buffer(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    execute_bilinear_into_slice(bilinear_layer, input_slice, output_buffer);
}

pub(super) fn execute_bilinear_into_slice(
    bilinear_layer: &ArchivedBilinearLayer,
    input_slice: &[f32],
    output_slice: &mut [f32],
) {
    output_slice.fill(0.0);
    for entry in bilinear_layer.quadratic.entries.iter() {
        let row_index = us16(entry.index.0);
        let left_value = homogeneous_value(input_slice, u16n(entry.index.1));
        let right_value = homogeneous_value(input_slice, u16n(entry.index.2));
        output_slice[row_index] += f32n(entry.value) * left_value * right_value;
    }
}

pub(super) fn execute_bilinear_push_forward(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let output_start = us32(bilinear_layer.out_offset);
    let output_stop = output_start + us16(bilinear_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
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

pub(super) fn execute_bilinear_push_forward_to_output_buffer(
    bilinear_layer: &ArchivedBilinearLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
    tangent_output_buffer: &mut [f32],
) {
    let input_start = us32(bilinear_layer.in_offset);
    let input_stop = input_start + us16(bilinear_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(bilinear_layer.scratch_offset),
        u16n(bilinear_layer.scratch_length),
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

pub(super) fn execute_bilinear_push_forward_into_slices(
    bilinear_layer: &ArchivedBilinearLayer,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
    output_slice: &mut [f32],
    tangent_output_slice: &mut [f32],
) {
    output_slice.fill(0.0);
    tangent_output_slice.fill(0.0);
    for entry in bilinear_layer.quadratic.entries.iter() {
        let row_index = us16(entry.index.0);
        let left_index = u16n(entry.index.1);
        let right_index = u16n(entry.index.2);
        let left_value = homogeneous_value(input_slice, left_index);
        let right_value = homogeneous_value(input_slice, right_index);
        let left_tangent = homogeneous_tangent(tangent_input_slice, left_index);
        let right_tangent = homogeneous_tangent(tangent_input_slice, right_index);
        let value = f32n(entry.value);
        output_slice[row_index] += value * left_value * right_value;
        tangent_output_slice[row_index] +=
            value * (left_tangent * right_value + left_value * right_tangent);
    }
}

pub(crate) fn execute_generic_layer(
    generic_layer: &ArchivedGenericLayer,
    workspace: &mut Workspace<'_>,
) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let output_start = us32(generic_layer.out_offset);
    let output_stop = output_start + us16(generic_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let (input_slice, output_slice) = workspace.input_output_slices(
        prepared_input_range.0,
        prepared_input_range.1,
        output_start,
        output_stop,
    );
    execute_generic_into_slice(generic_layer, input_slice, output_slice);
}

pub(super) fn execute_generic_layer_to_output_buffer(
    generic_layer: &ArchivedGenericLayer,
    workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let input_slice = &workspace.as_slice()[prepared_input_range.0..prepared_input_range.1];
    execute_generic_into_slice(generic_layer, input_slice, output_buffer);
}

pub(super) fn execute_generic_into_slice(
    generic_layer: &ArchivedGenericLayer,
    input_slice: &[f32],
    output_slice: &mut [f32],
) {
    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        let row_operation = row_op_from_archived(row_operation);
        output_slice[row_index] = evaluate_generic_value(&row_operation, input_slice);
    }
}

pub(super) fn execute_generic_push_forward(
    generic_layer: &ArchivedGenericLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let output_start = us32(generic_layer.out_offset);
    let output_stop = output_start + us16(generic_layer.out_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
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

pub(super) fn execute_generic_push_forward_to_output_buffer(
    generic_layer: &ArchivedGenericLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
    output_buffer: &mut [f32],
    tangent_output_buffer: &mut [f32],
) {
    let input_start = us32(generic_layer.in_offset);
    let input_stop = input_start + us16(generic_layer.in_length);
    let prepared_input_range = prepare_input_range(
        workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
    );
    let prepared_tangent_input_range = prepare_input_range(
        tangent_workspace,
        input_start,
        input_stop,
        u32n(generic_layer.scratch_offset),
        u16n(generic_layer.scratch_length),
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

pub(super) fn execute_generic_push_forward_into_slices(
    generic_layer: &ArchivedGenericLayer,
    input_slice: &[f32],
    tangent_input_slice: &[f32],
    output_slice: &mut [f32],
    tangent_output_slice: &mut [f32],
) {
    for (row_index, row_operation) in generic_layer.ops.iter().enumerate() {
        let row_operation = row_op_from_archived(row_operation);
        let (value, tangent) =
            evaluate_generic_push_forward(&row_operation, input_slice, tangent_input_slice);
        output_slice[row_index] = value;
        tangent_output_slice[row_index] = tangent;
    }
}
