use super::*;

pub(super) fn execute_evaluate_layer(
    module: &ArchivedBytecodeModule,
    evaluate_layer: &ArchivedEvaluateLayer,
    workspace: &mut Workspace<'_>,
) {
    let callee_program = find_function_unchecked(module, u16n(evaluate_layer.callee_function_id));
    let scratch_start = us32(evaluate_layer.scratch_offset);
    let scratch_length = us32(callee_program.required_workspace_size);
    let caller_workspace = Workspace::new(workspace.as_mut_slice());
    let (mut caller_workspace, scratch_workspace) = caller_workspace.split_at_mut(scratch_start);
    let mut nested_workspace = scratch_workspace.truncate(scratch_length);
    nested_workspace.fill(0.0);
    pack_evaluate_inputs(
        callee_program,
        evaluate_layer,
        caller_workspace.as_slice(),
        nested_workspace.as_mut_slice(),
    );
    execute_program_layers(module, callee_program, &mut nested_workspace, None);
    copy_evaluate_outputs(
        callee_program,
        evaluate_layer,
        nested_workspace.as_slice(),
        caller_workspace.as_mut_slice(),
    );
}

pub(super) fn execute_evaluate_push_forward(
    module: &ArchivedBytecodeModule,
    evaluate_layer: &ArchivedEvaluateLayer,
    workspace: &mut Workspace<'_>,
    tangent_workspace: &mut Workspace<'_>,
) {
    let callee_program = find_function_unchecked(module, u16n(evaluate_layer.callee_function_id));
    let scratch_start = us32(evaluate_layer.scratch_offset);
    let scratch_length = us32(callee_program.required_workspace_size);
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
        callee_program,
        evaluate_layer,
        caller_workspace.as_slice(),
        nested_workspace.as_mut_slice(),
    );
    pack_evaluate_tangents(
        callee_program,
        evaluate_layer,
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
        callee_program,
        evaluate_layer,
        nested_workspace.as_slice(),
        caller_workspace.as_mut_slice(),
    );
    copy_evaluate_outputs(
        callee_program,
        evaluate_layer,
        nested_tangent_workspace.as_slice(),
        caller_tangent_workspace.as_mut_slice(),
    );
}
