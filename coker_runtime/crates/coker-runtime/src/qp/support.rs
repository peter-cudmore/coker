use super::*;
#[cfg(all(feature = "std", not(osqp_embedded)))]
use alloc::vec::Vec;

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
    #[cfg(any(osqp_embedded, not(feature = "std")))]
    let requirements = {
        let plan = qp_program.embedded_plan();
        QpWorkspaceRequirements {
            evaluator_workspace_size: evaluator.info().required_workspace_size,
            tangent_workspace_size: qp_program.required_tangent_workspace_size() as usize,
            coefficient_output_size,
            arena_bytes: checked_embedded_usize(
                plan.arena_layout().total_bytes(),
                "QP arena bytes",
            )?,
            arena_alignment: checked_embedded_usize(
                plan.arena_layout().arena_alignment(),
                "QP arena alignment",
            )?,
        }
    };
    #[cfg(all(feature = "std", not(osqp_embedded)))]
    let requirements = {
        let host_layout = QpWorkspaceLayout::from_validated_parts(
            evaluator.info().required_workspace_size,
            coefficient_output_size,
            p_nnz,
            a_nnz,
            n,
            m,
        )?;
        QpWorkspaceRequirements {
            evaluator_workspace_size: evaluator.info().required_workspace_size,
            tangent_workspace_size: qp_program.required_tangent_workspace_size() as usize,
            coefficient_output_size,
            arena_bytes: host_layout.required_f64_capacity() * size_of::<f64>(),
            arena_alignment: align_of::<f64>(),
        }
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

#[cfg(any(osqp_embedded, not(feature = "std")))]
pub(super) fn checked_embedded_usize(
    value: u32,
    field: &'static str,
) -> Result<usize, RuntimeError> {
    usize::try_from(value).map_err(|_| RuntimeError::ValidationField {
        field,
        problem: "exceeds usize",
    })
}

#[cfg(osqp_embedded)]
pub(super) fn checked_embedded_ffi_length(length: usize) -> Result<i32, RuntimeError> {
    i32::try_from(length).map_err(|_| RuntimeError::EmbeddedQpWorkspaceInvalid)
}

#[cfg(osqp_embedded)]
pub(super) fn checked_f32_setting(value: f64, field: &'static str) -> Result<f32, RuntimeError> {
    if !value.is_finite() || value < -(f32::MAX as f64) || value > f32::MAX as f64 {
        return Err(RuntimeError::ValidationField {
            field,
            problem: "must be finite and representable as f32",
        });
    }
    Ok(value as f32)
}
#[cfg(all(feature = "std", not(osqp_embedded)))]
pub(super) fn checked_host_ffi_length(
    length: usize,
    field: &'static str,
) -> Result<ffi::c_int, RuntimeError> {
    ffi::c_int::try_from(length).map_err(|_| RuntimeError::ValidationField {
        field,
        problem: "exceeds OSQP index range",
    })
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
#[allow(clippy::unnecessary_fallible_conversions)]
pub(super) fn collect_host_osqp_indices(
    values: &[u32_le],
    field: &'static str,
) -> Result<Vec<ffi::c_int>, RuntimeError> {
    values
        .iter()
        .map(|value| {
            let value = value.to_native();
            if size_of::<ffi::c_int>() > size_of::<u32>() {
                Ok(value.into())
            } else {
                ffi::c_int::try_from(value).map_err(|_| RuntimeError::ValidationField {
                    field,
                    problem: "exceeds OSQP index range",
                })
            }
        })
        .collect()
}

#[cfg(osqp_embedded)]
pub(super) fn ffi_plan_from_program(
    qp_program: &ArchivedQpProgram,
) -> Result<ffi::CokerOsqpPlan, RuntimeError> {
    let plan = qp_program.embedded_plan();
    let qdldl = plan.qdldl_plan();
    let symbolic_l = qdldl.symbolic_l();
    let n = checked_embedded_osqp_index(qp_program.p_pattern().ncols.to_native(), "QP n")?;
    let m = checked_embedded_osqp_index(qp_program.a_pattern().nrows.to_native(), "QP m")?;
    let n_plus_m = n
        .checked_add(m)
        .ok_or(RuntimeError::Validation("QP dimensions overflow"))?;
    Ok(ffi::CokerOsqpPlan {
        abi_version: u32::from(plan.abi_version()),
        n: checked_embedded_ffi_length(n)?,
        m: checked_embedded_ffi_length(m)?,
        n_plus_m: checked_embedded_ffi_length(n_plus_m)?,
        p: ffi_csc_view_from_pattern(qp_program.p_pattern())?,
        a: ffi_csc_view_from_pattern(qp_program.a_pattern())?,
        kkt: ffi_csc_view_from_pattern(qdldl.kkt_pattern())?,
        qdldl_l: ffi_csc_view_from_pattern(symbolic_l.l_pattern())?,
        p_to_kkt: ffi_index_view_from_u32(&qdldl.p_to_kkt)?,
        a_to_kkt: ffi_index_view_from_u32(&qdldl.a_to_kkt)?,
        rho_to_kkt: ffi_index_view_from_u32(&qdldl.rho_to_kkt)?,
        p_diagonal_idx: ffi_index_view_from_u32(&qdldl.p_diag_indices)?,
        permutation: ffi_index_view_from_u32(&qdldl.kkt_permutation)?,
        qdldl_etree: ffi_index_view_from_u32(symbolic_l.etree())?,
        qdldl_lnz: ffi_index_view_from_u32(symbolic_l.lnz())?,
        settings: ffi_settings_from_archived(plan)?,
        arena_layout: ffi_arena_layout_from_archived(plan.arena_layout())?,
    })
}

#[cfg(osqp_embedded)]
pub(super) fn ffi_csc_view_from_pattern(
    pattern: &coker_bytecode::ArchivedEmbeddedCscPattern,
) -> Result<ffi::CokerOsqpCscView, RuntimeError> {
    Ok(ffi::CokerOsqpCscView {
        col_ptr: pattern.indptr.as_ptr().cast(),
        row_idx: pattern.indices.as_ptr().cast(),
        nnz: checked_embedded_ffi_length(pattern.indices.len())?,
    })
}

#[cfg(osqp_embedded)]
pub(super) fn ffi_index_view_from_u32(
    values: &[u32_le],
) -> Result<ffi::CokerOsqpIndexView, RuntimeError> {
    Ok(ffi::CokerOsqpIndexView {
        ptr: values.as_ptr().cast(),
        len: checked_embedded_ffi_length(values.len())?,
    })
}

#[cfg(osqp_embedded)]
pub(super) fn ffi_settings_from_archived(
    plan: &coker_bytecode::ArchivedQpProgramPlan,
) -> Result<ffi::CokerOsqpSettings, RuntimeError> {
    let settings = plan.settings();
    let linsys_solver = match settings.linsys_solver {
        coker_bytecode::ArchivedEmbeddedLinsysSolver::Qdldl => 0,
    };
    Ok(ffi::CokerOsqpSettings {
        rho: checked_f32_setting(settings.rho.to_native(), "embedded QP rho")?,
        sigma: checked_f32_setting(settings.sigma.to_native(), "embedded QP sigma")?,
        scaling: i32::try_from(settings.scaling.to_native())
            .map_err(|_| RuntimeError::Validation("embedded QP scaling exceeds i32"))?,
        adaptive_rho: if settings.adaptive_rho { 1 } else { 0 },
        adaptive_rho_interval: i32::try_from(settings.adaptive_rho_interval.to_native()).map_err(
            |_| RuntimeError::Validation("embedded QP adaptive_rho_interval exceeds i32"),
        )?,
        adaptive_rho_tolerance: checked_f32_setting(
            settings.adaptive_rho_tolerance.to_native(),
            "embedded QP adaptive_rho_tolerance",
        )?,
        max_iter: i32::try_from(settings.max_iter.to_native())
            .map_err(|_| RuntimeError::Validation("embedded QP max_iter exceeds i32"))?,
        eps_abs: checked_f32_setting(settings.eps_abs.to_native(), "embedded QP eps_abs")?,
        eps_rel: checked_f32_setting(settings.eps_rel.to_native(), "embedded QP eps_rel")?,
        eps_prim_inf: checked_f32_setting(
            settings.eps_prim_inf.to_native(),
            "embedded QP eps_prim_inf",
        )?,
        eps_dual_inf: checked_f32_setting(
            settings.eps_dual_inf.to_native(),
            "embedded QP eps_dual_inf",
        )?,
        alpha: checked_f32_setting(settings.alpha.to_native(), "embedded QP alpha")?,
        linsys_solver,
        scaled_termination: if settings.scaled_termination { 1 } else { 0 },
        check_termination: i32::try_from(settings.check_termination.to_native())
            .map_err(|_| RuntimeError::Validation("embedded QP check_termination exceeds i32"))?,
        warm_start: if settings.warm_start { 1 } else { 0 },
    })
}

#[cfg(osqp_embedded)]
pub(super) fn ffi_arena_layout_from_archived(
    layout: &coker_bytecode::ArchivedQpProgramArenaLayout,
) -> Result<ffi::CokerOsqpArenaLayout, RuntimeError> {
    Ok(ffi::CokerOsqpArenaLayout {
        bytes: checked_embedded_usize(layout.total_bytes(), "QP arena bytes")?,
        alignment: checked_embedded_usize(layout.arena_alignment(), "QP arena alignment")?,
        pdata_x: ffi_arena_region_from_archived(layout.pdata_x())?,
        pdata: ffi_arena_region_from_archived(layout.pdata())?,
        adata_x: ffi_arena_region_from_archived(layout.adata_x())?,
        adata: ffi_arena_region_from_archived(layout.adata())?,
        qdata: ffi_arena_region_from_archived(layout.qdata())?,
        ldata: ffi_arena_region_from_archived(layout.ldata())?,
        udata: ffi_arena_region_from_archived(layout.udata())?,
        data: ffi_arena_region_from_archived(layout.data())?,
        settings: ffi_arena_region_from_archived(layout.settings())?,
        xsolution: ffi_arena_region_from_archived(layout.xsolution())?,
        ysolution: ffi_arena_region_from_archived(layout.ysolution())?,
        solution: ffi_arena_region_from_archived(layout.solution())?,
        info: ffi_arena_region_from_archived(layout.info())?,
        qdldl_L_x: ffi_arena_region_from_archived(layout.qdldl_l_x())?,
        qdldl_L: ffi_arena_region_from_archived(layout.qdldl_l())?,
        qdldl_KKT_x: ffi_arena_region_from_archived(layout.qdldl_kkt_x())?,
        qdldl_KKT: ffi_arena_region_from_archived(layout.qdldl_kkt())?,
        qdldl: ffi_arena_region_from_archived(layout.qdldl())?,
        qdldl_Dinv: ffi_arena_region_from_archived(layout.qdldl_dinv())?,
        qdldl_bp: ffi_arena_region_from_archived(layout.qdldl_bp())?,
        qdldl_sol: ffi_arena_region_from_archived(layout.qdldl_sol())?,
        qdldl_rho_inv_vec: ffi_arena_region_from_archived(layout.qdldl_rho_inv_vec())?,
        qdldl_D: ffi_arena_region_from_archived(layout.qdldl_d())?,
        qdldl_iwork: ffi_arena_region_from_archived(layout.qdldl_iwork())?,
        qdldl_bwork: ffi_arena_region_from_archived(layout.qdldl_bwork())?,
        qdldl_fwork: ffi_arena_region_from_archived(layout.qdldl_fwork())?,
        work_rho_vec: ffi_arena_region_from_archived(layout.work_rho_vec())?,
        work_rho_inv_vec: ffi_arena_region_from_archived(layout.work_rho_inv_vec())?,
        work_constr_type: ffi_arena_region_from_archived(layout.work_constr_type())?,
        work_x: ffi_arena_region_from_archived(layout.work_x())?,
        work_y: ffi_arena_region_from_archived(layout.work_y())?,
        work_z: ffi_arena_region_from_archived(layout.work_z())?,
        work_xz_tilde: ffi_arena_region_from_archived(layout.work_xz_tilde())?,
        work_x_prev: ffi_arena_region_from_archived(layout.work_x_prev())?,
        work_z_prev: ffi_arena_region_from_archived(layout.work_z_prev())?,
        work_Ax: ffi_arena_region_from_archived(layout.work_ax())?,
        work_Px: ffi_arena_region_from_archived(layout.work_px())?,
        work_Aty: ffi_arena_region_from_archived(layout.work_aty())?,
        work_delta_y: ffi_arena_region_from_archived(layout.work_delta_y())?,
        work_Atdelta_y: ffi_arena_region_from_archived(layout.work_atdelta_y())?,
        work_delta_x: ffi_arena_region_from_archived(layout.work_delta_x())?,
        work_Pdelta_x: ffi_arena_region_from_archived(layout.work_pdelta_x())?,
        work_Adelta_x: ffi_arena_region_from_archived(layout.work_adelta_x())?,
        workspace: ffi_arena_region_from_archived(layout.workspace())?,
    })
}

#[cfg(osqp_embedded)]
pub(super) fn ffi_arena_region_from_archived(
    region: &coker_bytecode::ArchivedQpProgramArenaRegion,
) -> Result<ffi::CokerOsqpArenaRegion, RuntimeError> {
    Ok(ffi::CokerOsqpArenaRegion {
        offset: checked_embedded_usize(region.byte_offset(), "QP arena region offset")?,
        bytes: checked_embedded_usize(region.byte_len(), "QP arena region length")?,
        alignment: checked_embedded_usize(region.byte_alignment(), "QP arena region alignment")?,
    })
}

#[cfg(osqp_embedded)]
pub(super) unsafe fn arena_region_slice_mut<'a, T>(
    base: *mut u8,
    bytes: usize,
    region: &coker_bytecode::ArchivedQpProgramArenaRegion,
) -> Result<&'a mut [T], RuntimeError> {
    let offset = checked_embedded_usize(region.byte_offset(), "QP arena region offset")?;
    let byte_len = checked_embedded_usize(region.byte_len(), "QP arena region length")?;
    let alignment = checked_embedded_usize(region.byte_alignment(), "QP arena region alignment")?;
    if alignment == 0
        || !alignment.is_power_of_two()
        || alignment < align_of::<T>()
        || byte_len % size_of::<T>() != 0
    {
        return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
    }
    let address = (base as usize)
        .checked_add(offset)
        .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?;
    let end = offset
        .checked_add(byte_len)
        .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?;
    if address % alignment != 0 || end > bytes {
        return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
    }
    Ok(slice::from_raw_parts_mut(
        base.add(offset).cast::<T>(),
        byte_len / size_of::<T>(),
    ))
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
