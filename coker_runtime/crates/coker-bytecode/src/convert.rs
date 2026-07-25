use super::*;

pub(crate) fn embedded_qp_plan_from_archived(archived: &ArchivedEmbeddedQpPlan) -> EmbeddedQpPlan {
    EmbeddedQpPlan {
        profile: embedded_qp_profile_from_archived(&archived.profile),
        version: archived.version.into(),
        settings: embedded_osqp_settings_from_archived(&archived.settings),
        qdldl_plan: embedded_qdldl_plan_from_archived(&archived.qdldl_plan),
    }
}

pub(crate) fn embedded_qdldl_plan_from_archived(
    archived: &ArchivedEmbeddedQdldlPlan,
) -> EmbeddedQdldlPlan {
    EmbeddedQdldlPlan {
        p_pattern: embedded_csc_pattern_from_archived(&archived.p_pattern),
        a_pattern: embedded_csc_pattern_from_archived(&archived.a_pattern),
        kkt_pattern: embedded_csc_pattern_from_archived(&archived.kkt_pattern),
        p_diag_indices: archived
            .p_diag_indices
            .iter()
            .map(|value| value.to_native())
            .collect(),
        kkt_permutation: archived
            .kkt_permutation
            .iter()
            .map(|value| value.to_native())
            .collect(),
        p_to_kkt: archived
            .p_to_kkt
            .iter()
            .map(|value| value.to_native())
            .collect(),
        a_to_kkt: archived
            .a_to_kkt
            .iter()
            .map(|value| value.to_native())
            .collect(),
        rho_to_kkt: archived
            .rho_to_kkt
            .iter()
            .map(|value| value.to_native())
            .collect(),
    }
}

pub(crate) fn embedded_qp_profile_from_archived(
    profile: &ArchivedEmbeddedQpProfile,
) -> EmbeddedQpProfile {
    match profile {
        ArchivedEmbeddedQpProfile::Osqp063Embedded2Qdldl => {
            EmbeddedQpProfile::Osqp063Embedded2Qdldl
        }
    }
}

pub(crate) fn embedded_linsys_solver_from_archived(
    solver: &ArchivedEmbeddedLinsysSolver,
) -> EmbeddedLinsysSolver {
    match solver {
        ArchivedEmbeddedLinsysSolver::Qdldl => EmbeddedLinsysSolver::Qdldl,
    }
}

pub(crate) fn embedded_csc_pattern_from_archived(
    pattern: &ArchivedEmbeddedCscPattern,
) -> EmbeddedCscPattern {
    EmbeddedCscPattern {
        nrows: pattern.nrows.into(),
        ncols: pattern.ncols.into(),
        indptr: pattern.indptr.iter().map(|value| (*value).into()).collect(),
        indices: pattern
            .indices
            .iter()
            .map(|value| (*value).into())
            .collect(),
    }
}

pub(crate) fn embedded_osqp_settings_from_archived(
    settings: &ArchivedEmbeddedOsqpSettings,
) -> EmbeddedOsqpSettings {
    EmbeddedOsqpSettings {
        rho: settings.rho.into(),
        sigma: settings.sigma.into(),
        alpha: settings.alpha.into(),
        adaptive_rho: settings.adaptive_rho,
        adaptive_rho_interval: settings.adaptive_rho_interval.into(),
        adaptive_rho_tolerance: settings.adaptive_rho_tolerance.into(),
        max_iter: settings.max_iter.into(),
        eps_abs: settings.eps_abs.into(),
        eps_rel: settings.eps_rel.into(),
        eps_prim_inf: settings.eps_prim_inf.into(),
        eps_dual_inf: settings.eps_dual_inf.into(),
        scaling: settings.scaling.into(),
        scaled_termination: settings.scaled_termination,
        check_termination: settings.check_termination.into(),
        warm_start: settings.warm_start,
        linsys_solver: embedded_linsys_solver_from_archived(&settings.linsys_solver),
    }
}

pub(crate) fn validate_embedded_osqp_settings(
    settings: &EmbeddedOsqpSettings,
) -> Result<(), BytecodeError> {
    if settings.linsys_solver != EmbeddedLinsysSolver::Qdldl {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use the QDLDL solver".to_string(),
        ));
    }
    if settings.scaling != 0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must disable scaling".to_string(),
        ));
    }
    if !settings.adaptive_rho {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must enable adaptive rho".to_string(),
        ));
    }
    if !(settings.rho.is_finite() && settings.rho > 0.0) {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use a positive rho".to_string(),
        ));
    }
    if !(settings.sigma.is_finite() && settings.sigma > 0.0) {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use a positive sigma".to_string(),
        ));
    }
    if !(settings.alpha.is_finite() && settings.alpha > 0.0 && settings.alpha < 2.0) {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use an alpha in (0, 2)".to_string(),
        ));
    }
    if !(settings.adaptive_rho_tolerance.is_finite() && settings.adaptive_rho_tolerance >= 1.0) {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use an adaptive_rho_tolerance of at least 1".to_string(),
        ));
    }
    if settings.max_iter == 0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must allow at least one iteration".to_string(),
        ));
    }
    if settings.eps_abs < 0.0 || settings.eps_rel < 0.0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings absolute and relative tolerances must be non-negative"
                .to_string(),
        ));
    }
    if settings.eps_abs == 0.0 && settings.eps_rel == 0.0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must not disable both eps_abs and eps_rel".to_string(),
        ));
    }
    if settings.eps_prim_inf <= 0.0 || settings.eps_dual_inf <= 0.0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings infeasibility tolerances must be positive".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_embedded_csc_pattern(
    nrows: u32,
    ncols: u32,
    indptr: &[u32],
    indices: &[u32],
    field: &'static str,
) -> Result<(), BytecodeError> {
    let nrows = usize::try_from(nrows)
        .map_err(|_| BytecodeError::Decode(format!("{field} row count exceeds usize")))?;
    let ncols = usize::try_from(ncols)
        .map_err(|_| BytecodeError::Decode(format!("{field} column count exceeds usize")))?;
    if indptr.len() != ncols + 1 {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr length must be column_count + 1"
        )));
    }
    if indptr.first().copied().unwrap_or(0) != 0 {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr must start at zero"
        )));
    }
    if indptr.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr must be nondecreasing"
        )));
    }
    if usize::try_from(*indptr.last().unwrap_or(&0))
        .map_err(|_| BytecodeError::Decode(format!("{field} terminal indptr exceeds usize")))?
        != indices.len()
    {
        return Err(BytecodeError::Decode(format!(
            "{field} terminal indptr must match the number of indices"
        )));
    }
    for col in 0..ncols {
        let start = usize::try_from(indptr[col])
            .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
        let end = usize::try_from(indptr[col + 1])
            .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
        let mut previous_row = None;
        for &row in &indices[start..end] {
            let row = usize::try_from(row)
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

pub(crate) fn validate_index_entries_in_range(
    values: &[u32],
    limit: usize,
    field: &'static str,
) -> Result<(), BytecodeError> {
    for (idx, &value) in values.iter().enumerate() {
        let value = usize::try_from(value).map_err(|_| {
            BytecodeError::Decode(format!("{field} entry at index {idx} exceeds usize"))
        })?;
        if value >= limit {
            return Err(BytecodeError::Decode(format!(
                "{field} entry at index {idx} is out of bounds"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_permutation(
    values: &[u32],
    len: usize,
    field: &'static str,
) -> Result<(), BytecodeError> {
    validate_index_entries_in_range(values, len, field)?;
    if values.len() != len {
        return Err(BytecodeError::Decode(format!(
            "{field} length must match the KKT dimension"
        )));
    }
    let mut seen = vec![false; len];
    for (idx, &value) in values.iter().enumerate() {
        let value = usize::try_from(value).map_err(|_| {
            BytecodeError::Decode(format!("{field} entry at index {idx} exceeds usize"))
        })?;
        if seen[value] {
            return Err(BytecodeError::Decode(format!(
                "{field} entries must be unique"
            )));
        }
        seen[value] = true;
    }
    Ok(())
}

pub(crate) fn validate_upper_triangular_pattern(
    indptr: &[u32],
    indices: &[u32],
    field: &'static str,
) -> Result<(), BytecodeError> {
    for col in 0..indptr.len().saturating_sub(1) {
        let start = usize::try_from(indptr[col])
            .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
        let end = usize::try_from(indptr[col + 1])
            .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
        for &row in &indices[start..end] {
            let row = usize::try_from(row)
                .map_err(|_| BytecodeError::Decode(format!("{field} row index exceeds usize")))?;
            if row > col {
                return Err(BytecodeError::Decode(format!(
                    "{field} entries must be upper triangular"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_p_diagonal_indices_exact(
    ncols: usize,
    indptr_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    p_diag_indices_len: usize,
    p_diag_index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
) -> Result<usize, BytecodeError> {
    let mut diagonal_count = 0usize;
    for col in 0..ncols {
        let start = indptr_at(col)?;
        let end = indptr_at(col + 1)?;
        for entry in start..end {
            if index_at(entry)? == col {
                if diagonal_count >= p_diag_indices_len {
                    return Err(BytecodeError::Decode(
                        "qdldl_plan.p_diag_indices length must match the actual P diagonal count"
                            .to_string(),
                    ));
                }
                if p_diag_index_at(diagonal_count)? != entry {
                    return Err(BytecodeError::Decode(
                        "qdldl_plan.p_diag_indices must reference the actual P diagonal entries in column order"
                            .to_string(),
                    ));
                }
                diagonal_count += 1;
                break;
            }
        }
    }
    if p_diag_indices_len != diagonal_count {
        return Err(BytecodeError::Decode(
            "qdldl_plan.p_diag_indices length must match the actual P diagonal count".to_string(),
        ));
    }
    Ok(diagonal_count)
}

pub(crate) fn validate_p_to_kkt_exact(
    p_cols: usize,
    p_indptr_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    p_index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    p_to_kkt_len: usize,
    p_to_kkt_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    kkt_indptr_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    kkt_index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
) -> Result<(), BytecodeError> {
    if p_to_kkt_len != p_indptr_at(p_cols)? {
        return Err(BytecodeError::Decode(
            "qdldl_plan.p_to_kkt length must match the P sparsity pattern".to_string(),
        ));
    }
    for col in 0..p_cols {
        let p_start = p_indptr_at(col)?;
        let p_end = p_indptr_at(col + 1)?;
        let kkt_start = kkt_indptr_at(col)?;
        let kkt_end = kkt_indptr_at(col + 1)?;
        for entry in p_start..p_end {
            let mapped = p_to_kkt_at(entry)?;
            if mapped < kkt_start || mapped >= kkt_end {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.p_to_kkt must map each P entry into the matching KKT column"
                        .to_string(),
                ));
            }
            if kkt_index_at(mapped)? != p_index_at(entry)? {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.p_to_kkt must reference the matching KKT row for each P entry"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_a_to_kkt_exact(
    p_cols: usize,
    a_cols: usize,
    a_indptr_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    a_index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    a_to_kkt_len: usize,
    a_to_kkt_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    kkt_indptr_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    kkt_index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
) -> Result<(), BytecodeError> {
    if a_to_kkt_len != a_indptr_at(a_cols)? {
        return Err(BytecodeError::Decode(
            "qdldl_plan.a_to_kkt length must match the A sparsity pattern".to_string(),
        ));
    }
    for col in 0..a_cols {
        let a_start = a_indptr_at(col)?;
        let a_end = a_indptr_at(col + 1)?;
        for entry in a_start..a_end {
            let row = a_index_at(entry)?;
            let kkt_col = p_cols.checked_add(row).ok_or_else(|| {
                BytecodeError::Decode("qdldl_plan.a_to_kkt KKT column exceeds usize".to_string())
            })?;
            let kkt_start = kkt_indptr_at(kkt_col)?;
            let kkt_end = kkt_indptr_at(kkt_col + 1)?;
            let mapped = a_to_kkt_at(entry)?;
            if mapped < kkt_start || mapped >= kkt_end {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.a_to_kkt must map each A entry into the matching KKT column"
                        .to_string(),
                ));
            }
            if kkt_index_at(mapped)? != col {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.a_to_kkt must reference the matching KKT row for each A entry"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_rho_to_kkt_exact(
    p_cols: usize,
    a_rows: usize,
    kkt_indptr_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    kkt_index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    rho_to_kkt_len: usize,
    rho_to_kkt_at: impl Fn(usize) -> Result<usize, BytecodeError>,
) -> Result<(), BytecodeError> {
    if rho_to_kkt_len != a_rows {
        return Err(BytecodeError::Decode(
            "qdldl_plan.rho_to_kkt length must match the number of constraints".to_string(),
        ));
    }
    for row in 0..a_rows {
        let column = p_cols + row;
        let start = kkt_indptr_at(column)?;
        let end = kkt_indptr_at(column + 1)?;
        if start >= end || kkt_index_at(end - 1)? != column {
            return Err(BytecodeError::Decode(
                "qdldl_plan.kkt_pattern must contain each lower KKT diagonal entry at the end of its constraint column"
                    .to_string(),
            ));
        }
        if rho_to_kkt_at(row)? != end - 1 {
            return Err(BytecodeError::Decode(
                "qdldl_plan.rho_to_kkt must reference the lower KKT diagonal entries".to_string(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn module_from_archived(module: &ArchivedBytecodeModule) -> BytecodeModule {
    BytecodeModule::from_executables(
        module
            .executables
            .iter()
            .map(executable_from_archived)
            .collect(),
    )
}

pub(crate) fn module_from_split_archived(module: &ArchivedSplitBytecodeModule) -> BytecodeModule {
    let executable_count = module
        .functions
        .iter()
        .map(|program| program.function_id.to_native() as usize)
        .chain(
            module
                .qp_programs
                .iter()
                .map(|program| program.function_id.to_native() as usize),
        )
        .max()
        .map_or(0, |index| index + 1);
    let mut executables = vec![None; executable_count];
    for program in module.functions.iter() {
        executables[program.function_id.to_native() as usize] =
            Some(Executable::Program(program_from_split_archived(program)));
    }
    for program in module.qp_programs.iter() {
        executables[program.function_id.to_native() as usize] = Some(Executable::QpProgram(
            qp_program_from_split_archived(program),
        ));
    }
    BytecodeModule::from_executables(executables.into_iter().flatten().collect())
}

pub(crate) fn module_from_legacy_archived(module: &ArchivedLegacyBytecodeModule) -> BytecodeModule {
    let executable_count = module
        .functions
        .iter()
        .map(|program| program.function_id.to_native() as usize)
        .max()
        .map_or(0, |index| index + 1);
    let mut executables = vec![None; executable_count];
    for program in module.functions.iter() {
        executables[program.function_id.to_native() as usize] =
            Some(Executable::Program(program_from_split_archived(program)));
    }
    BytecodeModule::from_executables(executables.into_iter().flatten().collect())
}

pub(crate) fn executable_from_archived(executable: &ArchivedExecutable) -> Executable {
    match executable {
        ArchivedExecutable::Program(program) => Executable::Program(program_from_archived(program)),
        ArchivedExecutable::QpProgram(program) => {
            Executable::QpProgram(qp_program_from_archived(program))
        }
    }
}

pub(crate) fn program_from_archived(program: &ArchivedProgram) -> Program {
    Program::new(
        program.workspace_size.into(),
        program.required_workspace_size.into(),
        program
            .input_specs
            .iter()
            .map(input_spec_from_archived)
            .collect(),
        program
            .output_specs
            .iter()
            .map(output_spec_from_archived)
            .collect(),
        program
            .intermediate_layers
            .iter()
            .map(layer_from_archived)
            .collect(),
    )
}

pub(crate) fn qp_program_from_archived(program: &ArchivedQpProgram) -> QpProgram {
    QpProgram::new(
        program.coefficient_function_id.into(),
        program.required_primal_workspace_size.into(),
        program.required_tangent_workspace_size.into(),
        program
            .input_specs
            .iter()
            .map(input_spec_from_archived)
            .collect(),
        output_spec_from_archived(&program.output_spec),
        embedded_csc_pattern_from_archived(&program.p_pattern),
        embedded_csc_pattern_from_archived(&program.a_pattern),
        qp_coefficient_outputs_from_archived(&program.coefficient_outputs),
        qp_program_plan_from_archived(&program.embedded_plan),
    )
}

pub(crate) fn program_from_split_archived(program: &ArchivedSplitProgram) -> Program {
    Program::new(
        program.workspace_size.into(),
        program.required_workspace_size.into(),
        program
            .input_specs
            .iter()
            .map(input_spec_from_archived)
            .collect(),
        program
            .output_specs
            .iter()
            .map(output_spec_from_archived)
            .collect(),
        program
            .intermediate_layers
            .iter()
            .map(layer_from_archived)
            .collect(),
    )
}

pub(crate) fn qp_program_from_split_archived(program: &ArchivedSplitQpProgram) -> QpProgram {
    QpProgram::new(
        program.coefficient_function_id.into(),
        program.required_primal_workspace_size.into(),
        program.required_tangent_workspace_size.into(),
        program
            .input_specs
            .iter()
            .map(input_spec_from_archived)
            .collect(),
        output_spec_from_archived(&program.output_spec),
        embedded_csc_pattern_from_archived(&program.p_pattern),
        embedded_csc_pattern_from_archived(&program.a_pattern),
        qp_coefficient_outputs_from_archived(&program.coefficient_outputs),
        qp_program_plan_from_archived(&program.embedded_plan),
    )
}

pub(crate) fn qp_coefficient_outputs_from_archived(
    outputs: &ArchivedQpCoefficientOutputs,
) -> QpCoefficientOutputs {
    QpCoefficientOutputs {
        px: qp_output_slice_from_archived(&outputs.px),
        q: qp_output_slice_from_archived(&outputs.q),
        ax: qp_output_slice_from_archived(&outputs.ax),
        l: qp_output_slice_from_archived(&outputs.l),
        u: qp_output_slice_from_archived(&outputs.u),
        r: qp_output_slice_from_archived(&outputs.r),
    }
}

pub(crate) fn qp_output_slice_from_archived(slice: &ArchivedQpOutputSlice) -> QpOutputSlice {
    QpOutputSlice {
        start: slice.start.into(),
        length: slice.length.into(),
    }
}

pub(crate) fn qp_program_plan_from_archived(plan: &ArchivedQpProgramPlan) -> QpProgramPlan {
    QpProgramPlan {
        abi_version: plan.abi_version.into(),
        profile: embedded_qp_profile_from_archived(&plan.profile),
        version: plan.version.into(),
        settings: embedded_osqp_settings_from_archived(&plan.settings),
        arena_layout: qp_program_arena_layout_from_archived(&plan.arena_layout),
        qdldl_plan: qp_program_qdldl_plan_from_archived(&plan.qdldl_plan),
    }
}

pub(crate) fn qp_program_arena_layout_from_archived(
    layout: &ArchivedQpProgramArenaLayout,
) -> QpProgramArenaLayout {
    QpProgramArenaLayout {
        total_bytes: layout.total_bytes.into(),
        arena_alignment: layout.arena_alignment.into(),
        pdata_x: qp_program_arena_region_from_archived(&layout.pdata_x),
        pdata: qp_program_arena_region_from_archived(&layout.pdata),
        adata_x: qp_program_arena_region_from_archived(&layout.adata_x),
        adata: qp_program_arena_region_from_archived(&layout.adata),
        qdata: qp_program_arena_region_from_archived(&layout.qdata),
        ldata: qp_program_arena_region_from_archived(&layout.ldata),
        udata: qp_program_arena_region_from_archived(&layout.udata),
        data: qp_program_arena_region_from_archived(&layout.data),
        settings: qp_program_arena_region_from_archived(&layout.settings),
        xsolution: qp_program_arena_region_from_archived(&layout.xsolution),
        ysolution: qp_program_arena_region_from_archived(&layout.ysolution),
        solution: qp_program_arena_region_from_archived(&layout.solution),
        info: qp_program_arena_region_from_archived(&layout.info),
        qdldl_l_x: qp_program_arena_region_from_archived(&layout.qdldl_l_x),
        qdldl_l: qp_program_arena_region_from_archived(&layout.qdldl_l),
        qdldl_kkt_x: qp_program_arena_region_from_archived(&layout.qdldl_kkt_x),
        qdldl_kkt: qp_program_arena_region_from_archived(&layout.qdldl_kkt),
        qdldl: qp_program_arena_region_from_archived(&layout.qdldl),
        qdldl_dinv: qp_program_arena_region_from_archived(&layout.qdldl_dinv),
        qdldl_bp: qp_program_arena_region_from_archived(&layout.qdldl_bp),
        qdldl_sol: qp_program_arena_region_from_archived(&layout.qdldl_sol),
        qdldl_rho_inv_vec: qp_program_arena_region_from_archived(&layout.qdldl_rho_inv_vec),
        qdldl_d: qp_program_arena_region_from_archived(&layout.qdldl_d),
        qdldl_iwork: qp_program_arena_region_from_archived(&layout.qdldl_iwork),
        qdldl_bwork: qp_program_arena_region_from_archived(&layout.qdldl_bwork),
        qdldl_fwork: qp_program_arena_region_from_archived(&layout.qdldl_fwork),
        work_rho_vec: qp_program_arena_region_from_archived(&layout.work_rho_vec),
        work_rho_inv_vec: qp_program_arena_region_from_archived(&layout.work_rho_inv_vec),
        work_constr_type: qp_program_arena_region_from_archived(&layout.work_constr_type),
        work_x: qp_program_arena_region_from_archived(&layout.work_x),
        work_y: qp_program_arena_region_from_archived(&layout.work_y),
        work_z: qp_program_arena_region_from_archived(&layout.work_z),
        work_xz_tilde: qp_program_arena_region_from_archived(&layout.work_xz_tilde),
        work_x_prev: qp_program_arena_region_from_archived(&layout.work_x_prev),
        work_z_prev: qp_program_arena_region_from_archived(&layout.work_z_prev),
        work_ax: qp_program_arena_region_from_archived(&layout.work_ax),
        work_px: qp_program_arena_region_from_archived(&layout.work_px),
        work_aty: qp_program_arena_region_from_archived(&layout.work_aty),
        work_delta_y: qp_program_arena_region_from_archived(&layout.work_delta_y),
        work_atdelta_y: qp_program_arena_region_from_archived(&layout.work_atdelta_y),
        work_delta_x: qp_program_arena_region_from_archived(&layout.work_delta_x),
        work_pdelta_x: qp_program_arena_region_from_archived(&layout.work_pdelta_x),
        work_adelta_x: qp_program_arena_region_from_archived(&layout.work_adelta_x),
        workspace: qp_program_arena_region_from_archived(&layout.workspace),
    }
}

pub(crate) fn qp_program_arena_region_from_archived(
    region: &ArchivedQpProgramArenaRegion,
) -> QpProgramArenaRegion {
    QpProgramArenaRegion {
        byte_offset: region.byte_offset.into(),
        byte_len: region.byte_len.into(),
        byte_alignment: region.byte_alignment.into(),
    }
}

pub(crate) fn qp_program_qdldl_plan_from_archived(
    plan: &ArchivedQpProgramQdldlPlan,
) -> QpProgramQdldlPlan {
    QpProgramQdldlPlan {
        p_pattern: embedded_csc_pattern_from_archived(&plan.p_pattern),
        a_pattern: embedded_csc_pattern_from_archived(&plan.a_pattern),
        kkt_pattern: embedded_csc_pattern_from_archived(&plan.kkt_pattern),
        p_diag_indices: plan
            .p_diag_indices
            .iter()
            .map(|value| value.to_native())
            .collect(),
        kkt_permutation: plan
            .kkt_permutation
            .iter()
            .map(|value| value.to_native())
            .collect(),
        p_to_kkt: plan
            .p_to_kkt
            .iter()
            .map(|value| value.to_native())
            .collect(),
        a_to_kkt: plan
            .a_to_kkt
            .iter()
            .map(|value| value.to_native())
            .collect(),
        rho_to_kkt: plan
            .rho_to_kkt
            .iter()
            .map(|value| value.to_native())
            .collect(),
        symbolic_l: qdldl_symbolic_l_from_archived(&plan.symbolic_l),
    }
}

pub(crate) fn qdldl_symbolic_l_from_archived(
    symbolic_l: &ArchivedQdldlSymbolicL,
) -> QdldlSymbolicL {
    QdldlSymbolicL {
        l_pattern: embedded_csc_pattern_from_archived(&symbolic_l.l_pattern),
        etree: symbolic_l
            .etree
            .iter()
            .map(|value| value.to_native())
            .collect(),
        lnz: symbolic_l
            .lnz
            .iter()
            .map(|value| value.to_native())
            .collect(),
    }
}

pub(crate) fn input_spec_from_archived(input_spec: &ArchivedInputSpec) -> InputSpec {
    InputSpec {
        workspace_offset: input_spec.workspace_offset.into(),
        length: input_spec.length.into(),
    }
}

pub(crate) fn output_spec_from_archived(output_spec: &ArchivedOutputSpec) -> OutputSpec {
    OutputSpec {
        workspace_offset: output_spec.workspace_offset.into(),
        length: output_spec.length.into(),
    }
}

pub(crate) fn layer_from_archived(layer: &ArchivedLayer) -> Layer {
    match layer {
        ArchivedLayer::Bilinear(bilinear_layer) => {
            Layer::Bilinear(bilinear_layer_from_archived(bilinear_layer))
        }
        ArchivedLayer::Generic(generic_layer) => {
            Layer::Generic(generic_layer_from_archived(generic_layer))
        }
        ArchivedLayer::Evaluate(evaluate_layer) => {
            Layer::Evaluate(evaluate_layer_from_archived(evaluate_layer))
        }
    }
}

pub(crate) fn bilinear_layer_from_archived(
    bilinear_layer: &ArchivedBilinearLayer,
) -> BilinearLayer {
    BilinearLayer {
        in_offset: bilinear_layer.in_offset.into(),
        out_offset: bilinear_layer.out_offset.into(),
        in_length: bilinear_layer.in_length.into(),
        out_length: bilinear_layer.out_length.into(),
        scratch_offset: bilinear_layer.scratch_offset.into(),
        scratch_length: bilinear_layer.scratch_length.into(),
        quadratic: sparse_tensor_from_archived(&bilinear_layer.quadratic),
    }
}

pub(crate) fn generic_layer_from_archived(generic_layer: &ArchivedGenericLayer) -> GenericLayer {
    GenericLayer {
        in_offset: generic_layer.in_offset.into(),
        out_offset: generic_layer.out_offset.into(),
        in_length: generic_layer.in_length.into(),
        out_length: generic_layer.out_length.into(),
        scratch_offset: generic_layer.scratch_offset.into(),
        scratch_length: generic_layer.scratch_length.into(),
        ops: generic_layer.ops.iter().map(row_op_from_archived).collect(),
    }
}

pub(crate) fn evaluate_layer_from_archived(
    evaluate_layer: &ArchivedEvaluateLayer,
) -> EvaluateLayer {
    EvaluateLayer {
        scratch_offset: evaluate_layer.scratch_offset.into(),
        callee_function_id: evaluate_layer.callee_function_id.into(),
        input_bindings: evaluate_layer
            .input_bindings
            .iter()
            .map(evaluate_input_binding_from_archived)
            .collect(),
        output_bindings: evaluate_layer
            .output_bindings
            .iter()
            .map(evaluate_output_binding_from_archived)
            .collect(),
    }
}

pub(crate) fn evaluate_input_binding_from_archived(
    binding: &ArchivedEvaluateInputBinding,
) -> EvaluateInputBinding {
    match binding {
        ArchivedEvaluateInputBinding::WorkspaceSlice { offset, length } => {
            EvaluateInputBinding::WorkspaceSlice {
                offset: (*offset).into(),
                length: (*length).into(),
            }
        }
        ArchivedEvaluateInputBinding::ConstantSlice { length, values } => {
            EvaluateInputBinding::ConstantSlice {
                length: (*length).into(),
                values: values.iter().map(|value| (*value).into()).collect(),
            }
        }
    }
}

pub(crate) fn evaluate_output_binding_from_archived(
    binding: &ArchivedEvaluateOutputBinding,
) -> EvaluateOutputBinding {
    EvaluateOutputBinding {
        destination_offset: binding.destination_offset.into(),
        length: binding.length.into(),
    }
}

pub(crate) fn row_op_from_archived(row_op: &ArchivedRowOp) -> RowOp {
    RowOp {
        first: row_op.first.into(),
        second: row_op.second.into(),
        third: row_op.third.into(),
        op: scalar_op_from_archived(&row_op.op),
    }
}

pub(crate) fn scalar_op_from_archived(scalar_op: &ArchivedScalarOp) -> ScalarOp {
    match scalar_op {
        ArchivedScalarOp::Identity => ScalarOp::Identity,
        ArchivedScalarOp::Sin => ScalarOp::Sin,
        ArchivedScalarOp::Cos => ScalarOp::Cos,
        ArchivedScalarOp::Tan => ScalarOp::Tan,
        ArchivedScalarOp::Exp => ScalarOp::Exp,
        ArchivedScalarOp::Sqrt => ScalarOp::Sqrt,
        ArchivedScalarOp::Log => ScalarOp::Log,
        ArchivedScalarOp::Neg => ScalarOp::Neg,
        ArchivedScalarOp::Abs => ScalarOp::Abs,
        ArchivedScalarOp::Add => ScalarOp::Add,
        ArchivedScalarOp::Sub => ScalarOp::Sub,
        ArchivedScalarOp::Mul => ScalarOp::Mul,
        ArchivedScalarOp::Div => ScalarOp::Div,
        ArchivedScalarOp::Pow => ScalarOp::Pow,
        ArchivedScalarOp::IntPow => ScalarOp::IntPow,
        ArchivedScalarOp::Atan2 => ScalarOp::Atan2,
        ArchivedScalarOp::Equal => ScalarOp::Equal,
        ArchivedScalarOp::LessThan => ScalarOp::LessThan,
        ArchivedScalarOp::LessEqual => ScalarOp::LessEqual,
        ArchivedScalarOp::Case => ScalarOp::Case,
    }
}

pub(crate) fn sparse_tensor_from_archived(tensor: &ArchivedSparseTensor) -> SparseTensor {
    SparseTensor {
        shape: (
            tensor.shape.0.into(),
            tensor.shape.1.into(),
            tensor.shape.2.into(),
        ),
        entries: tensor
            .entries
            .iter()
            .map(sparse_entry_from_archived)
            .collect(),
    }
}

pub(crate) fn sparse_entry_from_archived(entry: &ArchivedSparseEntry) -> SparseEntry {
    SparseEntry {
        index: (
            entry.index.0.into(),
            entry.index.1.into(),
            entry.index.2.into(),
        ),
        value: entry.value.into(),
    }
}
