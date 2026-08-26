pub(crate) fn checked_flat_input_specs(
    specs: &[InputSpec],
    field: &'static str,
) -> Result<u32, BytecodeError> {
    checked_flat_input_specs_impl(specs.len(), |index| u32::from(specs[index].length), field)
}

pub(crate) fn checked_flat_output_specs(
    specs: &[OutputSpec],
    field: &'static str,
) -> Result<u32, BytecodeError> {
    checked_flat_input_specs_impl(specs.len(), |index| u32::from(specs[index].length), field)
}

pub(crate) fn checked_archived_flat_input_specs(
    specs: &[ArchivedInputSpec],
    field: &'static str,
) -> Result<u32, BytecodeError> {
    checked_flat_input_specs_impl(
        specs.len(),
        |index| u32::from(specs[index].length.to_native()),
        field,
    )
}

pub(crate) fn checked_archived_flat_output_specs(
    specs: &[ArchivedOutputSpec],
    field: &'static str,
) -> Result<u32, BytecodeError> {
    checked_flat_input_specs_impl(
        specs.len(),
        |index| u32::from(specs[index].length.to_native()),
        field,
    )
}

pub(crate) fn checked_flat_input_specs_impl(
    len: usize,
    value_at: impl Fn(usize) -> u32,
    field: &'static str,
) -> Result<u32, BytecodeError> {
    let mut total = 0u32;
    for index in 0..len {
        total = total
            .checked_add(value_at(index))
            .ok_or_else(|| BytecodeError::Decode(format!("{field} total length overflows u32")))?;
    }
    Ok(total)
}

pub(crate) fn validate_spec_range(
    offset: u32,
    length: u16,
    capacity: u32,
    field: &'static str,
) -> Result<(), BytecodeError> {
    let end = offset
        .checked_add(u32::from(length))
        .ok_or_else(|| BytecodeError::Decode(format!("{field} range overflows u32")))?;
    if end > capacity {
        return Err(BytecodeError::Decode(format!(
            "{field} range exceeds declared workspace capacity"
        )));
    }
    Ok(())
}



pub(crate) fn validate_bytecode_module_semantics(
    module: &BytecodeModule,
) -> Result<(), BytecodeError> {
    let entry_program = module.entry_program().ok_or_else(|| {
        BytecodeError::Decode(
            "bytecode module must contain an ordinary entry program at index 0".to_string(),
        )
    })?;
    for layer in &entry_program.intermediate_layers {
        validate_owned_program_call_layer(module, entry_program, layer)?;
    }
    for executable in &module.executables {
        match executable {
            Executable::Program(program) => {
                for layer in &program.intermediate_layers {
                    validate_owned_program_call_layer(module, program, layer)?;
                }
            }
            Executable::QpProgram(qp_program) => {
                validate_owned_qp_program(module, qp_program)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_archived_bytecode_module_semantics(
    module: &ArchivedBytecodeModule,
) -> Result<(), BytecodeError> {
    let entry_program = module.entry_program().ok_or_else(|| {
        BytecodeError::Decode(
            "bytecode module must contain an ordinary entry program at index 0".to_string(),
        )
    })?;
    for layer in entry_program.intermediate_layers.iter() {
        validate_archived_program_call_layer(module, entry_program, layer)?;
    }
    for executable in module.executables.iter() {
        match executable {
            ArchivedExecutable::Program(program) => {
                for layer in program.intermediate_layers.iter() {
                    validate_archived_program_call_layer(module, program, layer)?;
                }
            }
            ArchivedExecutable::QpProgram(qp_program) => {
                validate_archived_qp_program(module, qp_program)?;
            }
        }
    }
    Ok(())
}

fn validate_owned_program_call_layer(
    module: &BytecodeModule,
    caller: &Program,
    layer: &Layer,
) -> Result<(), BytecodeError> {
    match layer {
        Layer::Evaluate(evaluate_layer)
            if module
                .qp_program(evaluate_layer.callee_function_id)
                .is_some() =>
        {
            Err(BytecodeError::Decode(
                "ordinary evaluate layers must not target QP programs".to_string(),
            ))
        }
        Layer::QpCall(qp_call) => {
            let qp = module.qp_program(qp_call.qp_function_id).ok_or_else(|| {
                BytecodeError::Decode(
                    "QP call function id must reference a QP executable".to_string(),
                )
            })?;
            if qp_call.input_bindings.len() != qp.input_specs.len() {
                return Err(BytecodeError::Decode(
                    "QP call input binding count does not match QP inputs".to_string(),
                ));
            }
            for (binding, input) in qp_call.input_bindings.iter().zip(&qp.input_specs) {
                validate_owned_qp_call_input(binding, input.length)?;
            }
            validate_owned_qp_call_output(
                &qp_call.output_binding,
                qp.output_spec.length,
                caller.workspace_size,
            )?;
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_archived_program_call_layer(
    module: &ArchivedBytecodeModule,
    caller: &ArchivedProgram,
    layer: &ArchivedLayer,
) -> Result<(), BytecodeError> {
    match layer {
        ArchivedLayer::Evaluate(evaluate_layer)
            if module
                .qp_program(evaluate_layer.callee_function_id.to_native())
                .is_some() =>
        {
            Err(BytecodeError::Decode(
                "ordinary evaluate layers must not target QP programs".to_string(),
            ))
        }
        ArchivedLayer::QpCall(qp_call) => {
            let qp = module
                .qp_program(qp_call.qp_function_id.to_native())
                .ok_or_else(|| {
                    BytecodeError::Decode(
                        "QP call function id must reference a QP executable".to_string(),
                    )
                })?;
            if qp_call.input_bindings.len() != qp.input_specs.len() {
                return Err(BytecodeError::Decode(
                    "QP call input binding count does not match QP inputs".to_string(),
                ));
            }
            for (binding, input) in qp_call.input_bindings.iter().zip(qp.input_specs.iter()) {
                validate_archived_qp_call_input(binding, input.length.to_native())?;
            }
            validate_archived_qp_call_output(
                &qp_call.output_binding,
                qp.output_spec.length.to_native(),
                caller.workspace_size.to_native(),
            )?;
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_owned_qp_call_input(
    binding: &EvaluateInputBinding,
    expected_length: u16,
) -> Result<(), BytecodeError> {
    match binding {
        EvaluateInputBinding::WorkspaceSlice { length, .. }
        | EvaluateInputBinding::ConstantSlice { length, .. }
            if *length == expected_length =>
        {
            Ok(())
        }
        _ => Err(BytecodeError::Decode(
            "QP call input binding width does not match QP input".to_string(),
        )),
    }
}

fn validate_archived_qp_call_input(
    binding: &ArchivedEvaluateInputBinding,
    expected_length: u16,
) -> Result<(), BytecodeError> {
    match binding {
        ArchivedEvaluateInputBinding::WorkspaceSlice { length, .. }
        | ArchivedEvaluateInputBinding::ConstantSlice { length, .. }
            if length.to_native() == expected_length =>
        {
            Ok(())
        }
        _ => Err(BytecodeError::Decode(
            "QP call input binding width does not match QP input".to_string(),
        )),
    }
}

fn validate_owned_qp_call_output(
    binding: &EvaluateOutputBinding,
    expected_length: u16,
    workspace_size: u32,
) -> Result<(), BytecodeError> {
    if binding.length != expected_length
        || binding
            .destination_offset
            .checked_add(u32::from(binding.length))
            > Some(workspace_size)
    {
        return Err(BytecodeError::Decode(
            "QP call output binding does not match QP output or caller workspace".to_string(),
        ));
    }
    Ok(())
}

fn validate_archived_qp_call_output(
    binding: &ArchivedEvaluateOutputBinding,
    expected_length: u16,
    workspace_size: u32,
) -> Result<(), BytecodeError> {
    if binding.length.to_native() != expected_length
        || binding
            .destination_offset
            .to_native()
            .checked_add(u32::from(binding.length.to_native()))
            > Some(workspace_size)
    {
        return Err(BytecodeError::Decode(
            "QP call output binding does not match QP output or caller workspace".to_string(),
        ));
    }
    Ok(())
}

