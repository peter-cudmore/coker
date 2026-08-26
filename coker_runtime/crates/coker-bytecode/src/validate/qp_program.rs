pub(crate) fn validate_qp_program_arena_region(
    region: &QpProgramArenaRegion,
    field: &'static str,
) -> Result<(), BytecodeError> {
    if region.byte_alignment == 0 || !region.byte_alignment.is_power_of_two() {
        return Err(BytecodeError::Decode(format!(
            "{field} byte alignment must be a nonzero power of two"
        )));
    }
    if !region.byte_offset.is_multiple_of(region.byte_alignment) {
        return Err(BytecodeError::Decode(format!(
            "{field} byte offset must respect byte_alignment"
        )));
    }
    Ok(())
}

pub(crate) fn validate_archived_qp_program_arena_region(
    region: &ArchivedQpProgramArenaRegion,
    field: &'static str,
) -> Result<(), BytecodeError> {
    let alignment = region.byte_alignment.to_native();
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(BytecodeError::Decode(format!(
            "{field} byte alignment must be a nonzero power of two"
        )));
    }
    let offset = region.byte_offset.to_native();
    if !offset.is_multiple_of(alignment) {
        return Err(BytecodeError::Decode(format!(
            "{field} byte offset must respect byte_alignment"
        )));
    }
    Ok(())
}

pub(crate) fn validate_qp_program_arena_layout(
    layout: &QpProgramArenaLayout,
) -> Result<(), BytecodeError> {
    validate_qp_program_arena_layout_impl(
        layout.total_bytes,
        layout.arena_alignment,
        &[
            ("arena_layout.pdata_x", &layout.pdata_x),
            ("arena_layout.pdata", &layout.pdata),
            ("arena_layout.adata_x", &layout.adata_x),
            ("arena_layout.adata", &layout.adata),
            ("arena_layout.qdata", &layout.qdata),
            ("arena_layout.ldata", &layout.ldata),
            ("arena_layout.udata", &layout.udata),
            ("arena_layout.data", &layout.data),
            ("arena_layout.settings", &layout.settings),
            ("arena_layout.xsolution", &layout.xsolution),
            ("arena_layout.ysolution", &layout.ysolution),
            ("arena_layout.solution", &layout.solution),
            ("arena_layout.info", &layout.info),
            ("arena_layout.qdldl_l_x", &layout.qdldl_l_x),
            ("arena_layout.qdldl_l_p", &layout.qdldl_l_p),
            ("arena_layout.qdldl_l_i", &layout.qdldl_l_i),
            ("arena_layout.qdldl_l", &layout.qdldl_l),
            ("arena_layout.qdldl_kkt_x", &layout.qdldl_kkt_x),
            ("arena_layout.qdldl_kkt", &layout.qdldl_kkt),
            ("arena_layout.qdldl", &layout.qdldl),
            ("arena_layout.qdldl_dinv", &layout.qdldl_dinv),
            ("arena_layout.qdldl_bp", &layout.qdldl_bp),
            ("arena_layout.qdldl_sol", &layout.qdldl_sol),
            ("arena_layout.qdldl_rho_inv_vec", &layout.qdldl_rho_inv_vec),
            ("arena_layout.qdldl_d", &layout.qdldl_d),
            ("arena_layout.qdldl_iwork", &layout.qdldl_iwork),
            ("arena_layout.qdldl_bwork", &layout.qdldl_bwork),
            ("arena_layout.qdldl_fwork", &layout.qdldl_fwork),
            ("arena_layout.work_rho_vec", &layout.work_rho_vec),
            ("arena_layout.work_rho_inv_vec", &layout.work_rho_inv_vec),
            ("arena_layout.work_constr_type", &layout.work_constr_type),
            ("arena_layout.work_x", &layout.work_x),
            ("arena_layout.work_y", &layout.work_y),
            ("arena_layout.work_z", &layout.work_z),
            ("arena_layout.work_xz_tilde", &layout.work_xz_tilde),
            ("arena_layout.work_x_prev", &layout.work_x_prev),
            ("arena_layout.work_z_prev", &layout.work_z_prev),
            ("arena_layout.work_ax", &layout.work_ax),
            ("arena_layout.work_px", &layout.work_px),
            ("arena_layout.work_aty", &layout.work_aty),
            ("arena_layout.work_delta_y", &layout.work_delta_y),
            ("arena_layout.work_atdelta_y", &layout.work_atdelta_y),
            ("arena_layout.work_delta_x", &layout.work_delta_x),
            ("arena_layout.work_pdelta_x", &layout.work_pdelta_x),
            ("arena_layout.work_adelta_x", &layout.work_adelta_x),
            ("arena_layout.workspace", &layout.workspace),
        ],
    )
}

pub(crate) fn validate_archived_qp_program_arena_layout(
    layout: &ArchivedQpProgramArenaLayout,
) -> Result<(), BytecodeError> {
    validate_archived_qp_program_arena_layout_impl(
        layout.total_bytes.to_native(),
        layout.arena_alignment.to_native(),
        &[
            ("arena_layout.pdata_x", &layout.pdata_x),
            ("arena_layout.pdata", &layout.pdata),
            ("arena_layout.adata_x", &layout.adata_x),
            ("arena_layout.adata", &layout.adata),
            ("arena_layout.qdata", &layout.qdata),
            ("arena_layout.ldata", &layout.ldata),
            ("arena_layout.udata", &layout.udata),
            ("arena_layout.data", &layout.data),
            ("arena_layout.settings", &layout.settings),
            ("arena_layout.xsolution", &layout.xsolution),
            ("arena_layout.ysolution", &layout.ysolution),
            ("arena_layout.solution", &layout.solution),
            ("arena_layout.info", &layout.info),
            ("arena_layout.qdldl_l_x", &layout.qdldl_l_x),
            ("arena_layout.qdldl_l_p", &layout.qdldl_l_p),
            ("arena_layout.qdldl_l_i", &layout.qdldl_l_i),
            ("arena_layout.qdldl_l", &layout.qdldl_l),
            ("arena_layout.qdldl_kkt_x", &layout.qdldl_kkt_x),
            ("arena_layout.qdldl_kkt", &layout.qdldl_kkt),
            ("arena_layout.qdldl", &layout.qdldl),
            ("arena_layout.qdldl_dinv", &layout.qdldl_dinv),
            ("arena_layout.qdldl_bp", &layout.qdldl_bp),
            ("arena_layout.qdldl_sol", &layout.qdldl_sol),
            ("arena_layout.qdldl_rho_inv_vec", &layout.qdldl_rho_inv_vec),
            ("arena_layout.qdldl_d", &layout.qdldl_d),
            ("arena_layout.qdldl_iwork", &layout.qdldl_iwork),
            ("arena_layout.qdldl_bwork", &layout.qdldl_bwork),
            ("arena_layout.qdldl_fwork", &layout.qdldl_fwork),
            ("arena_layout.work_rho_vec", &layout.work_rho_vec),
            ("arena_layout.work_rho_inv_vec", &layout.work_rho_inv_vec),
            ("arena_layout.work_constr_type", &layout.work_constr_type),
            ("arena_layout.work_x", &layout.work_x),
            ("arena_layout.work_y", &layout.work_y),
            ("arena_layout.work_z", &layout.work_z),
            ("arena_layout.work_xz_tilde", &layout.work_xz_tilde),
            ("arena_layout.work_x_prev", &layout.work_x_prev),
            ("arena_layout.work_z_prev", &layout.work_z_prev),
            ("arena_layout.work_ax", &layout.work_ax),
            ("arena_layout.work_px", &layout.work_px),
            ("arena_layout.work_aty", &layout.work_aty),
            ("arena_layout.work_delta_y", &layout.work_delta_y),
            ("arena_layout.work_atdelta_y", &layout.work_atdelta_y),
            ("arena_layout.work_delta_x", &layout.work_delta_x),
            ("arena_layout.work_pdelta_x", &layout.work_pdelta_x),
            ("arena_layout.work_adelta_x", &layout.work_adelta_x),
            ("arena_layout.workspace", &layout.workspace),
        ],
    )
}

pub(crate) fn validate_qp_program_arena_layout_impl(
    total_bytes: u32,
    arena_alignment: u32,
    regions: &[(&'static str, &QpProgramArenaRegion)],
) -> Result<(), BytecodeError> {
    if arena_alignment == 0 || !arena_alignment.is_power_of_two() {
        return Err(BytecodeError::Decode(
            "embedded QP arena alignment must be a nonzero power of two".to_string(),
        ));
    }
    for (field, region) in regions {
        region.validate(field)?;
        if region.byte_alignment > arena_alignment {
            return Err(BytecodeError::Decode(format!(
                "{field} alignment exceeds arena_alignment"
            )));
        }
        let end = region
            .byte_offset
            .checked_add(region.byte_len)
            .ok_or_else(|| BytecodeError::Decode(format!("{field} byte range overflows u32")))?;
        if end > total_bytes {
            return Err(BytecodeError::Decode(format!(
                "{field} byte range exceeds arena_layout.total_bytes"
            )));
        }
    }
    for (index, (field, region)) in regions.iter().enumerate() {
        if region.byte_len == 0 {
            continue;
        }
        let start = region.byte_offset;
        let end = start + region.byte_len;
        for (other_field, other_region) in regions.iter().skip(index + 1) {
            if other_region.byte_len == 0 {
                continue;
            }
            let other_start = other_region.byte_offset;
            let other_end = other_start + other_region.byte_len;
            if start < other_end && other_start < end {
                return Err(BytecodeError::Decode(format!(
                    "arena layout regions {field} and {other_field} must not overlap"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_archived_qp_program_arena_layout_impl(
    total_bytes: u32,
    arena_alignment: u32,
    regions: &[(&'static str, &ArchivedQpProgramArenaRegion)],
) -> Result<(), BytecodeError> {
    if arena_alignment == 0 || !arena_alignment.is_power_of_two() {
        return Err(BytecodeError::Decode(
            "embedded QP arena alignment must be a nonzero power of two".to_string(),
        ));
    }
    for (field, region) in regions {
        region.validate(field)?;
        if region.byte_alignment.to_native() > arena_alignment {
            return Err(BytecodeError::Decode(format!(
                "{field} alignment exceeds arena_alignment"
            )));
        }
        let end = region
            .byte_offset
            .to_native()
            .checked_add(region.byte_len.to_native())
            .ok_or_else(|| BytecodeError::Decode(format!("{field} byte range overflows u32")))?;
        if end > total_bytes {
            return Err(BytecodeError::Decode(format!(
                "{field} byte range exceeds arena_layout.total_bytes"
            )));
        }
    }
    for (index, (field, region)) in regions.iter().enumerate() {
        let len = region.byte_len.to_native();
        if len == 0 {
            continue;
        }
        let start = region.byte_offset.to_native();
        let end = start + len;
        for (other_field, other_region) in regions.iter().skip(index + 1) {
            let other_len = other_region.byte_len.to_native();
            if other_len == 0 {
                continue;
            }
            let other_start = other_region.byte_offset.to_native();
            let other_end = other_start + other_len;
            if start < other_end && other_start < end {
                return Err(BytecodeError::Decode(format!(
                    "arena layout regions {field} and {other_field} must not overlap"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_archived_qp_program_plan(
    plan: &ArchivedQpProgramPlan,
) -> Result<(), BytecodeError> {
    if plan.abi_version.to_native() != QpProgramPlan::ABI_VERSION {
        return Err(BytecodeError::Decode(format!(
            "unsupported embedded QP plan abi version: expected {}, found {}",
            QpProgramPlan::ABI_VERSION,
            plan.abi_version.to_native()
        )));
    }
    if !matches!(
        plan.profile,
        ArchivedEmbeddedQpProfile::Osqp063Embedded2Qdldl
    ) {
        return Err(BytecodeError::Decode(
            "unsupported embedded QP plan profile".to_string(),
        ));
    }
    if plan.version.to_native() != QpProgramPlan::VERSION {
        return Err(BytecodeError::Decode(format!(
            "unsupported embedded QP plan version: expected {}, found {}",
            QpProgramPlan::VERSION,
            plan.version.to_native()
        )));
    }
    validate_archived_embedded_osqp_settings(&plan.settings)?;
    plan.arena_layout.validate()?;
    plan.qdldl_plan.validate()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_qp_output_slices(
    px: (u32, u32),
    q: (u32, u32),
    ax: (u32, u32),
    l: (u32, u32),
    u: (u32, u32),
    r: (u32, u32),
    n: u32,
    m: u32,
    p_nnz: u32,
    a_nnz: u32,
    field: &'static str,
) -> Result<u32, BytecodeError> {
    let slices = [
        ("px", px, p_nnz),
        ("q", q, n),
        ("ax", ax, a_nnz),
        ("l", l, m),
        ("u", u, m),
        ("r", r, 1),
    ];
    let mut expected_start = 0u32;
    for (name, (start, length), expected_length) in slices {
        if start != expected_start {
            return Err(BytecodeError::Decode(format!(
                "{field}.{name} must start at the previous slice end"
            )));
        }
        if length != expected_length {
            return Err(BytecodeError::Decode(format!(
                "{field}.{name} length must match the QP dimensions and sparsity"
            )));
        }
        expected_start = start
            .checked_add(length)
            .ok_or_else(|| BytecodeError::Decode(format!("{field}.{name} range overflows u32")))?;
    }
    Ok(expected_start)
}

pub(crate) fn validate_owned_qp_program(
    module: &BytecodeModule,
    qp_program: &QpProgram,
) -> Result<(), BytecodeError> {
    let coefficient_program = module
        .program(qp_program.coefficient_function_id)
        .ok_or_else(|| {
            BytecodeError::Decode(
                "QP coefficient_function_id must reference an ordinary function".to_string(),
            )
        })?;
    if coefficient_program.input_specs != qp_program.input_specs {
        return Err(BytecodeError::Decode(
            "QP input specs must match the referenced coefficient evaluator inputs".to_string(),
        ));
    }
    validate_spec_range(
        qp_program.output_spec.workspace_offset,
        qp_program.output_spec.length,
        qp_program.required_primal_workspace_size,
        "QP output spec in primal workspace",
    )?;
    validate_spec_range(
        qp_program.output_spec.workspace_offset,
        qp_program.output_spec.length,
        qp_program.required_tangent_workspace_size,
        "QP output spec in tangent workspace",
    )?;
    qp_program.p_pattern.validate("QP p_pattern")?;
    qp_program.a_pattern.validate("QP a_pattern")?;
    let n = qp_program.p_pattern.ncols;
    let m = qp_program.a_pattern.nrows;
    if qp_program.p_pattern.nrows != n {
        return Err(BytecodeError::Decode(
            "QP p_pattern must be square".to_string(),
        ));
    }
    if qp_program.a_pattern.ncols != n {
        return Err(BytecodeError::Decode(
            "QP a_pattern column count must match p_pattern".to_string(),
        ));
    }
    if u32::from(qp_program.output_spec.length) != n {
        return Err(BytecodeError::Decode(
            "QP output spec length must match the decision-vector dimension".to_string(),
        ));
    }
    let p_nnz = u32::try_from(qp_program.p_pattern.indices.len())
        .map_err(|_| BytecodeError::Decode("QP p_pattern nnz exceeds u32".to_string()))?;
    let a_nnz = u32::try_from(qp_program.a_pattern.indices.len())
        .map_err(|_| BytecodeError::Decode("QP a_pattern nnz exceeds u32".to_string()))?;
    validate_csc_structure(
        qp_program.p_pattern.indptr.len(),
        qp_program.p_pattern.indices.len(),
        |index| {
            usize::try_from(qp_program.p_pattern.indptr[index])
                .map_err(|_| BytecodeError::Decode("QP p_pattern indptr exceeds usize".to_string()))
        },
        |index| {
            usize::try_from(qp_program.p_pattern.indices[index])
                .map_err(|_| BytecodeError::Decode("QP p_pattern indices exceed usize".to_string()))
        },
        usize::try_from(qp_program.p_pattern.nrows).map_err(|_| {
            BytecodeError::Decode("QP p_pattern row count exceeds usize".to_string())
        })?,
        usize::try_from(qp_program.p_pattern.ncols).map_err(|_| {
            BytecodeError::Decode("QP p_pattern column count exceeds usize".to_string())
        })?,
        true,
        "QP p_pattern",
    )?;
    validate_csc_structure(
        qp_program.a_pattern.indptr.len(),
        qp_program.a_pattern.indices.len(),
        |index| {
            usize::try_from(qp_program.a_pattern.indptr[index])
                .map_err(|_| BytecodeError::Decode("QP a_pattern indptr exceeds usize".to_string()))
        },
        |index| {
            usize::try_from(qp_program.a_pattern.indices[index])
                .map_err(|_| BytecodeError::Decode("QP a_pattern indices exceed usize".to_string()))
        },
        usize::try_from(qp_program.a_pattern.nrows).map_err(|_| {
            BytecodeError::Decode("QP a_pattern row count exceeds usize".to_string())
        })?,
        usize::try_from(qp_program.a_pattern.ncols).map_err(|_| {
            BytecodeError::Decode("QP a_pattern column count exceeds usize".to_string())
        })?,
        false,
        "QP a_pattern",
    )?;
    let expected_output_len = validate_qp_output_slices(
        (
            qp_program.coefficient_outputs.px.start,
            qp_program.coefficient_outputs.px.length,
        ),
        (
            qp_program.coefficient_outputs.q.start,
            qp_program.coefficient_outputs.q.length,
        ),
        (
            qp_program.coefficient_outputs.ax.start,
            qp_program.coefficient_outputs.ax.length,
        ),
        (
            qp_program.coefficient_outputs.l.start,
            qp_program.coefficient_outputs.l.length,
        ),
        (
            qp_program.coefficient_outputs.u.start,
            qp_program.coefficient_outputs.u.length,
        ),
        (
            qp_program.coefficient_outputs.r.start,
            qp_program.coefficient_outputs.r.length,
        ),
        n,
        m,
        p_nnz,
        a_nnz,
        "QP coefficient_outputs",
    )?;
    if coefficient_program.checked_flat_output_size()? != expected_output_len {
        return Err(BytecodeError::Decode(
            "QP coefficient evaluator output lengths do not match coefficient slices".to_string(),
        ));
    }
    qp_program.embedded_plan.validate()?;
    if qp_program.embedded_plan.qdldl_plan.p_pattern != qp_program.p_pattern {
        return Err(BytecodeError::Decode(
            "QP embedded plan P pattern must match the QP program P pattern".to_string(),
        ));
    }
    if qp_program.embedded_plan.qdldl_plan.a_pattern != qp_program.a_pattern {
        return Err(BytecodeError::Decode(
            "QP embedded plan A pattern must match the QP program A pattern".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_archived_qp_program(
    module: &ArchivedBytecodeModule,
    qp_program: &ArchivedQpProgram,
) -> Result<(), BytecodeError> {
    let coefficient_program = module
        .program(qp_program.coefficient_function_id())
        .ok_or_else(|| {
            BytecodeError::Decode(
                "QP coefficient_function_id must reference an ordinary function".to_string(),
            )
        })?;
    if qp_program.input_specs.len() != coefficient_program.input_specs.len()
        || qp_program
            .input_specs
            .iter()
            .zip(coefficient_program.input_specs.iter())
            .any(|(lhs, rhs)| {
                lhs.workspace_offset.to_native() != rhs.workspace_offset.to_native()
                    || lhs.length.to_native() != rhs.length.to_native()
            })
    {
        return Err(BytecodeError::Decode(
            "QP input specs must match the referenced coefficient evaluator inputs".to_string(),
        ));
    }
    validate_spec_range(
        qp_program.output_spec.workspace_offset.to_native(),
        qp_program.output_spec.length.to_native(),
        qp_program.required_primal_workspace_size.to_native(),
        "QP output spec in primal workspace",
    )?;
    validate_spec_range(
        qp_program.output_spec.workspace_offset.to_native(),
        qp_program.output_spec.length.to_native(),
        qp_program.required_tangent_workspace_size.to_native(),
        "QP output spec in tangent workspace",
    )?;
    validate_archived_embedded_csc_pattern(&qp_program.p_pattern, "QP p_pattern")?;
    validate_archived_embedded_csc_pattern(&qp_program.a_pattern, "QP a_pattern")?;
    let n = qp_program.p_pattern.ncols.to_native();
    let m = qp_program.a_pattern.nrows.to_native();
    if qp_program.p_pattern.nrows.to_native() != n {
        return Err(BytecodeError::Decode(
            "QP p_pattern must be square".to_string(),
        ));
    }
    if qp_program.a_pattern.ncols.to_native() != n {
        return Err(BytecodeError::Decode(
            "QP a_pattern column count must match p_pattern".to_string(),
        ));
    }
    if u32::from(qp_program.output_spec.length.to_native()) != n {
        return Err(BytecodeError::Decode(
            "QP output spec length must match the decision-vector dimension".to_string(),
        ));
    }
    let p_nnz = u32::try_from(qp_program.p_pattern.indices.len())
        .map_err(|_| BytecodeError::Decode("QP p_pattern nnz exceeds u32".to_string()))?;
    let a_nnz = u32::try_from(qp_program.a_pattern.indices.len())
        .map_err(|_| BytecodeError::Decode("QP a_pattern nnz exceeds u32".to_string()))?;
    validate_csc_structure(
        qp_program.p_pattern.indptr.len(),
        qp_program.p_pattern.indices.len(),
        |index| {
            usize::try_from(qp_program.p_pattern.indptr[index].to_native())
                .map_err(|_| BytecodeError::Decode("QP p_pattern indptr exceeds usize".to_string()))
        },
        |index| {
            usize::try_from(qp_program.p_pattern.indices[index].to_native())
                .map_err(|_| BytecodeError::Decode("QP p_pattern indices exceed usize".to_string()))
        },
        usize::try_from(qp_program.p_pattern.nrows.to_native()).map_err(|_| {
            BytecodeError::Decode("QP p_pattern row count exceeds usize".to_string())
        })?,
        usize::try_from(qp_program.p_pattern.ncols.to_native()).map_err(|_| {
            BytecodeError::Decode("QP p_pattern column count exceeds usize".to_string())
        })?,
        true,
        "QP p_pattern",
    )?;
    validate_csc_structure(
        qp_program.a_pattern.indptr.len(),
        qp_program.a_pattern.indices.len(),
        |index| {
            usize::try_from(qp_program.a_pattern.indptr[index].to_native())
                .map_err(|_| BytecodeError::Decode("QP a_pattern indptr exceeds usize".to_string()))
        },
        |index| {
            usize::try_from(qp_program.a_pattern.indices[index].to_native())
                .map_err(|_| BytecodeError::Decode("QP a_pattern indices exceed usize".to_string()))
        },
        usize::try_from(qp_program.a_pattern.nrows.to_native()).map_err(|_| {
            BytecodeError::Decode("QP a_pattern row count exceeds usize".to_string())
        })?,
        usize::try_from(qp_program.a_pattern.ncols.to_native()).map_err(|_| {
            BytecodeError::Decode("QP a_pattern column count exceeds usize".to_string())
        })?,
        false,
        "QP a_pattern",
    )?;
    let expected_output_len = validate_qp_output_slices(
        (
            qp_program.coefficient_outputs.px.start.to_native(),
            qp_program.coefficient_outputs.px.length.to_native(),
        ),
        (
            qp_program.coefficient_outputs.q.start.to_native(),
            qp_program.coefficient_outputs.q.length.to_native(),
        ),
        (
            qp_program.coefficient_outputs.ax.start.to_native(),
            qp_program.coefficient_outputs.ax.length.to_native(),
        ),
        (
            qp_program.coefficient_outputs.l.start.to_native(),
            qp_program.coefficient_outputs.l.length.to_native(),
        ),
        (
            qp_program.coefficient_outputs.u.start.to_native(),
            qp_program.coefficient_outputs.u.length.to_native(),
        ),
        (
            qp_program.coefficient_outputs.r.start.to_native(),
            qp_program.coefficient_outputs.r.length.to_native(),
        ),
        n,
        m,
        p_nnz,
        a_nnz,
        "QP coefficient_outputs",
    )?;
    if coefficient_program.checked_flat_output_size()? != expected_output_len {
        return Err(BytecodeError::Decode(
            "QP coefficient evaluator output lengths do not match coefficient slices".to_string(),
        ));
    }
    qp_program.embedded_plan.validate()?;
    if !archived_csc_patterns_match(
        &qp_program.embedded_plan.qdldl_plan.p_pattern,
        &qp_program.p_pattern,
    ) {
        return Err(BytecodeError::Decode(
            "QP embedded plan P pattern must match the QP program P pattern".to_string(),
        ));
    }
    if !archived_csc_patterns_match(
        &qp_program.embedded_plan.qdldl_plan.a_pattern,
        &qp_program.a_pattern,
    ) {
        return Err(BytecodeError::Decode(
            "QP embedded plan A pattern must match the QP program A pattern".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn archived_csc_patterns_match(
    lhs: &ArchivedEmbeddedCscPattern,
    rhs: &ArchivedEmbeddedCscPattern,
) -> bool {
    lhs.nrows.to_native() == rhs.nrows.to_native()
        && lhs.ncols.to_native() == rhs.ncols.to_native()
        && lhs.nnz.to_native() == rhs.nnz.to_native()
        && lhs.indptr.len() == rhs.indptr.len()
        && lhs.indices.len() == rhs.indices.len()
        && lhs
            .indptr
            .iter()
            .zip(rhs.indptr.iter())
            .all(|(lhs, rhs)| lhs.to_native() == rhs.to_native())
        && lhs
            .indices
            .iter()
            .zip(rhs.indices.iter())
            .all(|(lhs, rhs)| lhs.to_native() == rhs.to_native())
}

pub(crate) fn validate_archived_embedded_csc_pattern(
    pattern: &ArchivedEmbeddedCscPattern,
    field: &'static str,
) -> Result<(), BytecodeError> {
    let nrows = usize::try_from(pattern.nrows.to_native())
        .map_err(|_| BytecodeError::Decode(format!("{field} row count exceeds usize")))?;
    let ncols = usize::try_from(pattern.ncols.to_native())
        .map_err(|_| BytecodeError::Decode(format!("{field} column count exceeds usize")))?;
    if pattern.indptr.len() != ncols + 1 {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr length must be column_count + 1"
        )));
    }
    let first = pattern
        .indptr
        .iter()
        .next()
        .map(|value| value.to_native())
        .unwrap_or(0);
    if first != 0 {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr must start at zero"
        )));
    }
    let mut indptr_iter = pattern.indptr.iter();
    let mut next_iter = pattern.indptr.iter().skip(1);
    while let (Some(start), Some(end)) = (indptr_iter.next(), next_iter.next()) {
        if start.to_native() > end.to_native() {
            return Err(BytecodeError::Decode(format!(
                "{field} indptr must be nondecreasing"
            )));
        }
    }
    let terminal = pattern.indptr[ncols].to_native();
    if terminal < 0 || terminal as u32 != pattern.nnz.to_native() {
        return Err(BytecodeError::Decode(format!(
            "{field} terminal indptr must match nnz"
        )));
    }
    if usize::try_from(pattern.nnz.to_native())
        .map_err(|_| BytecodeError::Decode(format!("{field} nnz exceeds usize")))?
        != pattern.indices.len()
    {
        return Err(BytecodeError::Decode(format!(
            "{field} nnz must match the number of indices"
        )));
    }
    for col in 0..ncols {
        let start = usize::try_from(pattern.indptr[col].to_native())
            .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
        let end = usize::try_from(pattern.indptr[col + 1].to_native())
            .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
        let mut previous_row = None;
        for index in start..end {
            let row = usize::try_from(pattern.indices[index].to_native())
                .map_err(|_| BytecodeError::Decode(format!("{field} row index exceeds usize")))?;
            if row >= nrows {
                return Err(BytecodeError::Decode(format!(
                    "{field} row index out of bounds"
                )));
            }
            if let Some(previous_row) = previous_row {
                if row <= previous_row {
                    return Err(BytecodeError::Decode(format!(
                        "{field} row indices must be strictly increasing within each column"
                    )));
                }
            }
            previous_row = Some(row);
        }
    }
    Ok(())
}
