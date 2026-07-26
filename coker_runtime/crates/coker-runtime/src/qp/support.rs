use super::*;


pub(super) fn validate_mapped_qp_program(
    qp_program: &ArchivedQpProgram,
    evaluator: MappedProgram<'_>,
) -> Result<(usize, usize, usize, usize, QpWorkspaceRequirements), RuntimeError> {
    validate_mapped_qp_target()?;
    let coefficient_output_size = validate_embedded_evaluator(qp_program, evaluator)?;
    let n = checked_embedded_osqp_index(qp_program.p_pattern().ncols.to_native(), "QP n")?;
    if qp_program.output_spec().length() != n {
        return Err(RuntimeError::Validation("QP output spec length must equal n"));
    }
    let m = checked_embedded_osqp_index(qp_program.a_pattern().nrows.to_native(), "QP m")?;
    let p_nnz = checked_embedded_slice_len(qp_program.p_pattern().indices.len(), "QP P nnz")?;
    let a_nnz = checked_embedded_slice_len(qp_program.a_pattern().indices.len(), "QP A nnz")?;
    let arena_layout = qp_program.embedded_plan().arena_layout();
    let arena_bytes = checked_embedded_usize(arena_layout.total_bytes(), "QP arena bytes")?;
    let arena_alignment =
        checked_embedded_usize(arena_layout.arena_alignment(), "QP arena alignment")?;
    let requirements = QpWorkspaceRequirements {
        evaluator_workspace_size: evaluator.info().required_workspace_size,
        tangent_workspace_size: qp_program.required_tangent_workspace_size() as usize,
        coefficient_output_size,
        arena_bytes,
        arena_alignment,
    };
    if requirements.arena_alignment == 0 || !requirements.arena_alignment.is_power_of_two() {
        return Err(RuntimeError::Validation(
            "QP arena alignment must be a nonzero power of two",
        ));
    }
    Ok((n, m, p_nnz, a_nnz, requirements))
}

pub(super) fn validate_embedded_evaluator(
    qp_program: &ArchivedQpProgram,
    evaluator: MappedProgram<'_>,
) -> Result<usize, RuntimeError> {
    let info = evaluator.info();
    let expected_output_len = expected_output_length(qp_program.coefficient_outputs())?;
    let actual_output_len = info.output_specs.iter().try_fold(0usize, |total, spec| {
        total.checked_add(spec.length())
            .ok_or(RuntimeError::Validation("QP evaluator output lengths overflow"))
    })?;
    if actual_output_len != expected_output_len {
        return Err(RuntimeError::Validation(
            "QP evaluator output lengths do not match coefficient slices",
        ));
    }
    if info.input_specs.len() != qp_program.input_specs().len()
        || info
            .input_specs
            .iter()
            .zip(qp_program.input_specs().iter())
            .any(|(actual, expected)| {
                actual.workspace_offset() != expected.workspace_offset()
                    || actual.length() != expected.length()
            })
    {
        return Err(RuntimeError::Validation(
            "QP input specs must match the referenced coefficient evaluator inputs",
        ));
    }
    Ok(expected_output_len)
}

pub(super) fn validate_mapped_qp_target() -> Result<(), RuntimeError> {
    if !cfg!(target_endian = "little") {
        return Err(RuntimeError::Validation(
            "mapped embedded QP runtime requires a little-endian target",
        ));
    }
    Ok(())
}

pub(super) fn checked_embedded_osqp_index(
    value: u32,
    field: &'static str,
) -> Result<usize, RuntimeError> {
    if value > i32::MAX as u32 {
        return Err(RuntimeError::ValidationField {
            field,
            problem: "exceeds the embedded OSQP i32 index range",
        });
    }
    Ok(value as usize)
}

pub(super) fn checked_embedded_slice_len(
    length: usize,
    field: &'static str,
) -> Result<usize, RuntimeError> {
    let value = u32::try_from(length).map_err(|_| RuntimeError::ValidationField {
        field,
        problem: "exceeds the embedded OSQP i32 index range",
    })?;
    checked_embedded_osqp_index(value, field)
}

pub(super) fn checked_embedded_usize(
    value: u32,
    field: &'static str,
) -> Result<usize, RuntimeError> {
    usize::try_from(value).map_err(|_| RuntimeError::ValidationField {
        field,
        problem: "exceeds usize",
    })
}

pub(super) fn checked_embedded_ffi_length(length: usize) -> Result<i32, RuntimeError> {
    i32::try_from(length).map_err(|_| RuntimeError::EmbeddedQpWorkspaceInvalid)
}

pub(super) fn checked_f32_setting(value: f64, field: &'static str) -> Result<f32, RuntimeError> {
    if !value.is_finite() || value < -(f32::MAX as f64) || value > f32::MAX as f64 {
        return Err(RuntimeError::ValidationField {
            field,
            problem: "must be finite and representable as f32",
        });
    }
    Ok(value as f32)
}


pub(super) fn qp_program_info_from_program(
    qp_program: &ArchivedQpProgram,
) -> QpProgramInfo<'_, ArchivedInputSpec, ArchivedOutputSpec> {
    QpProgramInfo {
        required_primal_workspace_size: qp_program.required_primal_workspace_size() as usize,
        required_tangent_workspace_size: qp_program.required_tangent_workspace_size() as usize,
        input_specs: qp_program.input_specs(),
        output_spec: qp_program.output_spec(),
    }
}

pub(super) fn expected_output_length(
    outputs: &ArchivedQpCoefficientOutputs,
) -> Result<usize, RuntimeError> {
    (outputs.r.start.to_native() as usize)
        .checked_add(outputs.r.length.to_native() as usize)
        .ok_or(RuntimeError::Validation("QP output offsets overflow"))
}
