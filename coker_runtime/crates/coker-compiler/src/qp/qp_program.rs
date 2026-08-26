//! QP program lowering.

use super::*;

pub(crate) fn build_exported_qp_program(
    functions: &[Program],
    exported_qp: model::ExportedQpProgram,
) -> Result<QpProgram, CompileError> {
    let coefficient_function_id = checked_u16(
        exported_qp.coefficient_function_id,
        "coefficient_function_id",
    )?;
    let required_primal_workspace_size = checked_embedded_qp_u32(
        exported_qp.required_primal_workspace_size,
        "required_primal_workspace_size",
    )?;
    let required_tangent_workspace_size = checked_embedded_qp_u32(
        exported_qp.required_tangent_workspace_size,
        "required_tangent_workspace_size",
    )?;
    let input_specs = lower_input_specs(exported_qp.input_specs)?;
    let output_spec = lower_output_spec(exported_qp.output_spec)?;
    let p_pattern = lower_embedded_csc_pattern(
        exported_qp.p_pattern,
        "p_pattern.nrows",
        "p_pattern.ncols",
        "p_pattern.indptr",
        "p_pattern.indices",
    )?;
    let a_pattern = lower_embedded_csc_pattern(
        exported_qp.a_pattern,
        "a_pattern.nrows",
        "a_pattern.ncols",
        "a_pattern.indptr",
        "a_pattern.indices",
    )?;
    let coefficient_outputs = lower_qp_coefficient_outputs(&exported_qp.coefficient_outputs)?;
    let embedded_plan = lower_qp_program_plan(exported_qp.embedded_plan, &p_pattern, &a_pattern)?;

    let coefficient_program =
        functions
            .get(coefficient_function_id as usize)
            .ok_or(CompileError::InvalidField {
                field: "coefficient_function_id",
                reason: "must reference an ordinary function in the same module",
            })?;
    validate_qp_program_fields(
        &input_specs,
        &output_spec,
        &p_pattern,
        &a_pattern,
        &coefficient_outputs,
        &embedded_plan,
        required_primal_workspace_size,
        required_tangent_workspace_size,
        coefficient_program,
    )?;

    Ok(QpProgram::new(
        coefficient_function_id,
        required_primal_workspace_size,
        required_tangent_workspace_size,
        input_specs,
        output_spec,
        p_pattern,
        a_pattern,
        coefficient_outputs,
        embedded_plan,
    ))
}
