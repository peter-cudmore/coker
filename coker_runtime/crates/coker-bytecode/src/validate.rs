use super::*;

pub(crate) fn validate_archived_embedded_qp_plan(
    archived: &ArchivedEmbeddedQpPlan,
) -> Result<(), BytecodeError> {
    if !matches!(
        archived.profile,
        ArchivedEmbeddedQpProfile::Osqp063Embedded2Qdldl
    ) {
        return Err(BytecodeError::Decode(
            "unsupported embedded QP plan profile".to_string(),
        ));
    }
    if archived.version.to_native() != EmbeddedQpPlan::VERSION {
        return Err(BytecodeError::Decode(format!(
            "unsupported embedded QP plan version: expected {}, found {}",
            EmbeddedQpPlan::VERSION,
            archived.version.to_native()
        )));
    }
    validate_archived_embedded_osqp_settings(&archived.settings)?;
    validate_archived_qdldl_plan_dimensions(
        &archived.qdldl_plan.p_pattern,
        &archived.qdldl_plan.a_pattern,
        &archived.qdldl_plan.kkt_pattern,
        &archived.qdldl_plan.p_diag_indices,
        &archived.qdldl_plan.kkt_permutation,
        &archived.qdldl_plan.p_to_kkt,
        &archived.qdldl_plan.a_to_kkt,
        &archived.qdldl_plan.rho_to_kkt,
    )?;
    Ok(())
}

pub(crate) fn validate_archived_embedded_osqp_settings(
    settings: &ArchivedEmbeddedOsqpSettings,
) -> Result<(), BytecodeError> {
    if !matches!(settings.linsys_solver, ArchivedEmbeddedLinsysSolver::Qdldl) {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use the QDLDL solver".to_string(),
        ));
    }
    if settings.scaling.to_native() != 0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must disable scaling".to_string(),
        ));
    }
    if !settings.adaptive_rho {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must enable adaptive rho".to_string(),
        ));
    }
    if !(settings.rho.to_native().is_finite() && settings.rho.to_native() > 0.0) {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use a positive rho".to_string(),
        ));
    }
    if !(settings.sigma.to_native().is_finite() && settings.sigma.to_native() > 0.0) {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use a positive sigma".to_string(),
        ));
    }
    if !(settings.alpha.to_native().is_finite()
        && settings.alpha.to_native() > 0.0
        && settings.alpha.to_native() < 2.0)
    {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use an alpha in (0, 2)".to_string(),
        ));
    }
    if !(settings.adaptive_rho_tolerance.to_native().is_finite()
        && settings.adaptive_rho_tolerance.to_native() >= 1.0)
    {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must use an adaptive_rho_tolerance of at least 1".to_string(),
        ));
    }
    if settings.max_iter.to_native() == 0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must allow at least one iteration".to_string(),
        ));
    }
    if settings.eps_abs.to_native() < 0.0 || settings.eps_rel.to_native() < 0.0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings absolute and relative tolerances must be non-negative"
                .to_string(),
        ));
    }
    if settings.eps_abs.to_native() == 0.0 && settings.eps_rel.to_native() == 0.0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings must not disable both eps_abs and eps_rel".to_string(),
        ));
    }
    if settings.eps_prim_inf.to_native() <= 0.0 || settings.eps_dual_inf.to_native() <= 0.0 {
        return Err(BytecodeError::Decode(
            "embedded OSQP settings infeasibility tolerances must be positive".to_string(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_archived_qdldl_plan_dimensions(
    p_pattern: &ArchivedEmbeddedCscPattern,
    a_pattern: &ArchivedEmbeddedCscPattern,
    kkt_pattern: &ArchivedEmbeddedCscPattern,
    p_diag_indices: &ArchivedU32Vec,
    kkt_permutation: &ArchivedU32Vec,
    p_to_kkt: &ArchivedU32Vec,
    a_to_kkt: &ArchivedU32Vec,
    rho_to_kkt: &ArchivedU32Vec,
) -> Result<(), BytecodeError> {
    let validate_csc_pattern = |pattern: &ArchivedEmbeddedCscPattern,
                                upper_triangular: bool,
                                field: &'static str|
     -> Result<(), BytecodeError> {
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
        if usize::try_from(terminal)
            .map_err(|_| BytecodeError::Decode(format!("{field} terminal indptr exceeds usize")))?
            != pattern.indices.len()
        {
            return Err(BytecodeError::Decode(format!(
                "{field} terminal indptr must match the number of indices"
            )));
        }
        for col in 0..ncols {
            let start = usize::try_from(pattern.indptr[col].to_native())
                .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
            let end = usize::try_from(pattern.indptr[col + 1].to_native())
                .map_err(|_| BytecodeError::Decode(format!("{field} indptr exceeds usize")))?;
            let mut previous_row = None;
            for index in start..end {
                let row = usize::try_from(pattern.indices[index].to_native()).map_err(|_| {
                    BytecodeError::Decode(format!("{field} row index exceeds usize"))
                })?;
                if row >= nrows {
                    return Err(BytecodeError::Decode(format!(
                        "{field} row index out of bounds"
                    )));
                }
                if upper_triangular && row > col {
                    return Err(BytecodeError::Decode(format!(
                        "{field} entries must be upper triangular"
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
    };
    let p_indptr_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(p_pattern.indptr[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.p_pattern indptr exceeds usize".to_string())
        })
    };
    let p_index_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(p_pattern.indices[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.p_pattern indices exceed usize".to_string())
        })
    };
    let kkt_indptr_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(kkt_pattern.indptr[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.kkt_pattern indptr exceeds usize".to_string())
        })
    };
    let kkt_index_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(kkt_pattern.indices[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.kkt_pattern indices exceed usize".to_string())
        })
    };
    let p_diag_index_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(p_diag_indices[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.p_diag_indices entry exceeds usize".to_string())
        })
    };
    let a_indptr_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(a_pattern.indptr[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.a_pattern indptr exceeds usize".to_string())
        })
    };
    let a_index_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(a_pattern.indices[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.a_pattern indices exceed usize".to_string())
        })
    };
    let p_to_kkt_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(p_to_kkt[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.p_to_kkt entry exceeds usize".to_string())
        })
    };
    let a_to_kkt_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(a_to_kkt[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.a_to_kkt entry exceeds usize".to_string())
        })
    };
    let rho_to_kkt_at = |index: usize| -> Result<usize, BytecodeError> {
        usize::try_from(rho_to_kkt[index].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.rho_to_kkt entry exceeds usize".to_string())
        })
    };

    validate_csc_pattern(p_pattern, true, "qdldl_plan.p_pattern")?;
    validate_csc_pattern(a_pattern, false, "qdldl_plan.a_pattern")?;
    validate_csc_pattern(kkt_pattern, true, "qdldl_plan.kkt_pattern")?;

    let p_rows = usize::try_from(p_pattern.nrows.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.p_pattern row count exceeds usize".to_string())
    })?;
    let p_cols = usize::try_from(p_pattern.ncols.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.p_pattern column count exceeds usize".to_string())
    })?;
    let a_rows = usize::try_from(a_pattern.nrows.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.a_pattern row count exceeds usize".to_string())
    })?;
    let a_cols = usize::try_from(a_pattern.ncols.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.a_pattern column count exceeds usize".to_string())
    })?;
    let kkt_rows = usize::try_from(kkt_pattern.nrows.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern row count exceeds usize".to_string())
    })?;
    let kkt_cols = usize::try_from(kkt_pattern.ncols.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern column count exceeds usize".to_string())
    })?;

    if p_rows != p_cols {
        return Err(BytecodeError::Decode(
            "p_pattern must be square for embedded QDLDL".to_string(),
        ));
    }
    if a_cols != p_cols {
        return Err(BytecodeError::Decode(
            "a_pattern must have the same column count as p_pattern".to_string(),
        ));
    }
    if kkt_rows != p_rows + a_rows || kkt_cols != p_rows + a_rows {
        return Err(BytecodeError::Decode(
            "kkt_pattern dimensions must match the combined QP dimensions".to_string(),
        ));
    }

    let p_nnz = p_pattern.indices.len();
    let a_nnz = a_pattern.indices.len();
    let kkt_nnz = kkt_pattern.indices.len();
    let p_diagonal_count = validate_p_diagonal_indices_exact(
        p_cols,
        p_indptr_at,
        p_index_at,
        p_diag_indices.len(),
        p_diag_index_at,
    )?;
    let expected_kkt_nnz = p_nnz
        .checked_add(p_cols - p_diagonal_count)
        .and_then(|value| value.checked_add(a_nnz))
        .and_then(|value| value.checked_add(a_rows))
        .ok_or_else(|| {
            BytecodeError::Decode("qdldl_plan.kkt_pattern nnz exceeds usize".to_string())
        })?;
    if kkt_nnz != expected_kkt_nnz {
        return Err(BytecodeError::Decode(
            "qdldl_plan.kkt_pattern nnz must equal P nnz + missing P diagonal entries + A nnz + constraint count"
                .to_string(),
        ));
    }
    for (idx, value) in kkt_permutation.iter().enumerate() {
        let value = usize::try_from(value.to_native()).map_err(|_| {
            BytecodeError::Decode(format!(
                "qdldl_plan.kkt_permutation entry at index {idx} exceeds usize"
            ))
        })?;
        if value >= kkt_rows {
            return Err(BytecodeError::Decode(format!(
                "qdldl_plan.kkt_permutation entry at index {idx} is out of bounds"
            )));
        }
        if kkt_permutation
            .iter()
            .take(idx)
            .any(|previous| usize::try_from(previous.to_native()).ok() == Some(value))
        {
            return Err(BytecodeError::Decode(
                "qdldl_plan.kkt_permutation entries must be unique".to_string(),
            ));
        }
    }
    if kkt_permutation.len() != kkt_rows {
        return Err(BytecodeError::Decode(
            "qdldl_plan.kkt_permutation length must match the KKT dimension".to_string(),
        ));
    }
    for (idx, value) in p_to_kkt.iter().enumerate() {
        let value = usize::try_from(value.to_native()).map_err(|_| {
            BytecodeError::Decode(format!(
                "qdldl_plan.p_to_kkt entry at index {idx} exceeds usize"
            ))
        })?;
        if value >= kkt_nnz {
            return Err(BytecodeError::Decode(format!(
                "qdldl_plan.p_to_kkt entry at index {idx} is out of bounds"
            )));
        }
    }
    for (idx, value) in a_to_kkt.iter().enumerate() {
        let value = usize::try_from(value.to_native()).map_err(|_| {
            BytecodeError::Decode(format!(
                "qdldl_plan.a_to_kkt entry at index {idx} exceeds usize"
            ))
        })?;
        if value >= kkt_nnz {
            return Err(BytecodeError::Decode(format!(
                "qdldl_plan.a_to_kkt entry at index {idx} is out of bounds"
            )));
        }
    }
    for (idx, value) in rho_to_kkt.iter().enumerate() {
        let value = usize::try_from(value.to_native()).map_err(|_| {
            BytecodeError::Decode(format!(
                "qdldl_plan.rho_to_kkt entry at index {idx} exceeds usize"
            ))
        })?;
        if value >= kkt_nnz {
            return Err(BytecodeError::Decode(format!(
                "qdldl_plan.rho_to_kkt entry at index {idx} is out of bounds"
            )));
        }
    }

    validate_p_to_kkt_exact(
        p_cols,
        p_indptr_at,
        p_index_at,
        p_to_kkt.len(),
        p_to_kkt_at,
        kkt_indptr_at,
        kkt_index_at,
    )?;
    validate_a_to_kkt_exact(
        p_cols,
        a_cols,
        a_indptr_at,
        a_index_at,
        a_to_kkt.len(),
        a_to_kkt_at,
        kkt_indptr_at,
        kkt_index_at,
    )?;
    validate_rho_to_kkt_exact(
        p_cols,
        a_rows,
        kkt_indptr_at,
        kkt_index_at,
        rho_to_kkt.len(),
        rho_to_kkt_at,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_embedded_qp_plan_dimensions_impl(
    p_nrows: u32,
    p_ncols: u32,
    p_indptr: &[i32],
    p_indices: &[i32],
    a_nrows: u32,
    a_ncols: u32,
    a_indptr: &[i32],
    a_indices: &[i32],
    kkt_nrows: u32,
    kkt_ncols: u32,
    kkt_indptr: &[i32],
    kkt_indices: &[i32],
    p_diag_indices: &[u32],
    kkt_permutation: &[u32],
    p_to_kkt: &[u32],
    a_to_kkt: &[u32],
    rho_to_kkt: &[u32],
) -> Result<(), BytecodeError> {
    validate_embedded_csc_pattern(
        p_nrows,
        p_ncols,
        p_indptr,
        p_indices,
        "qdldl_plan.p_pattern",
    )?;
    validate_embedded_csc_pattern(
        a_nrows,
        a_ncols,
        a_indptr,
        a_indices,
        "qdldl_plan.a_pattern",
    )?;
    validate_embedded_csc_pattern(
        kkt_nrows,
        kkt_ncols,
        kkt_indptr,
        kkt_indices,
        "qdldl_plan.kkt_pattern",
    )?;
    validate_upper_triangular_pattern(p_indptr, p_indices, "qdldl_plan.p_pattern")?;
    validate_upper_triangular_pattern(kkt_indptr, kkt_indices, "qdldl_plan.kkt_pattern")?;

    let p_rows = usize::try_from(p_nrows).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.p_pattern row count exceeds usize".to_string())
    })?;
    let p_cols = usize::try_from(p_ncols).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.p_pattern column count exceeds usize".to_string())
    })?;
    let a_rows = usize::try_from(a_nrows).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.a_pattern row count exceeds usize".to_string())
    })?;
    let a_cols = usize::try_from(a_ncols).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.a_pattern column count exceeds usize".to_string())
    })?;
    let kkt_rows = usize::try_from(kkt_nrows).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern row count exceeds usize".to_string())
    })?;
    let kkt_cols = usize::try_from(kkt_ncols).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern column count exceeds usize".to_string())
    })?;

    if p_rows != p_cols {
        return Err(BytecodeError::Decode(
            "p_pattern must be square for embedded QDLDL".to_string(),
        ));
    }
    if a_cols != p_cols {
        return Err(BytecodeError::Decode(
            "a_pattern must have the same column count as p_pattern".to_string(),
        ));
    }
    if kkt_rows != p_rows + a_rows || kkt_cols != p_rows + a_rows {
        return Err(BytecodeError::Decode(
            "kkt_pattern dimensions must match the combined QP dimensions".to_string(),
        ));
    }

    let p_nnz = p_indices.len();
    let a_nnz = a_indices.len();
    let kkt_nnz = kkt_indices.len();
    let p_diagonal_count = validate_p_diagonal_indices_exact(
        p_cols,
        |index| {
            usize::try_from(p_indptr[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.p_pattern indptr exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(p_indices[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.p_pattern indices exceed usize".to_string())
            })
        },
        p_diag_indices.len(),
        |index| {
            usize::try_from(p_diag_indices[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.p_diag_indices entry exceeds usize".to_string())
            })
        },
    )?;
    let expected_kkt_nnz = p_nnz
        .checked_add(p_cols - p_diagonal_count)
        .and_then(|value| value.checked_add(a_nnz))
        .and_then(|value| value.checked_add(a_rows))
        .ok_or_else(|| {
            BytecodeError::Decode("qdldl_plan.kkt_pattern nnz exceeds usize".to_string())
        })?;
    if kkt_nnz != expected_kkt_nnz {
        return Err(BytecodeError::Decode(
            "qdldl_plan.kkt_pattern nnz must equal P nnz + missing P diagonal entries + A nnz + constraint count"
                .to_string(),
        ));
    }
    validate_permutation(kkt_permutation, kkt_rows, "qdldl_plan.kkt_permutation")?;
    validate_index_entries_in_range(rho_to_kkt, kkt_nnz, "qdldl_plan.rho_to_kkt")?;

    validate_p_to_kkt_exact(
        p_cols,
        |index| {
            usize::try_from(p_indptr[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.p_pattern indptr exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(p_indices[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.p_pattern indices exceed usize".to_string())
            })
        },
        p_to_kkt.len(),
        |index| {
            usize::try_from(p_to_kkt[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.p_to_kkt entry exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(kkt_indptr[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.kkt_pattern indptr exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(kkt_indices[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.kkt_pattern indices exceed usize".to_string())
            })
        },
    )?;
    validate_a_to_kkt_exact(
        p_cols,
        a_cols,
        |index| {
            usize::try_from(a_indptr[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.a_pattern indptr exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(a_indices[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.a_pattern indices exceed usize".to_string())
            })
        },
        a_to_kkt.len(),
        |index| {
            usize::try_from(a_to_kkt[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.a_to_kkt entry exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(kkt_indptr[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.kkt_pattern indptr exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(kkt_indices[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.kkt_pattern indices exceed usize".to_string())
            })
        },
    )?;
    validate_rho_to_kkt_exact(
        p_cols,
        a_rows,
        |index| {
            usize::try_from(kkt_indptr[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.kkt_pattern indptr exceeds usize".to_string())
            })
        },
        |index| {
            usize::try_from(kkt_indices[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.kkt_pattern indices exceed usize".to_string())
            })
        },
        rho_to_kkt.len(),
        |index| {
            usize::try_from(rho_to_kkt[index]).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.rho_to_kkt entry exceeds usize".to_string())
            })
        },
    )?;

    Ok(())
}

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_csc_structure(
    indptr_len: usize,
    indices_len: usize,
    indptr_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    index_at: impl Fn(usize) -> Result<usize, BytecodeError>,
    nrows: usize,
    ncols: usize,
    upper_triangular: bool,
    field: &'static str,
) -> Result<(), BytecodeError> {
    if indptr_len != ncols + 1 {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr length must equal columns + 1"
        )));
    }
    if indptr_at(0)? != 0 {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr must start at zero"
        )));
    }
    if indptr_at(ncols)? != indices_len {
        return Err(BytecodeError::Decode(format!(
            "{field} indptr terminal offset must match index count"
        )));
    }
    for col in 0..ncols {
        let start = indptr_at(col)?;
        let end = indptr_at(col + 1)?;
        if start > end || end > indices_len {
            return Err(BytecodeError::Decode(format!(
                "{field} indptr entries must stay in bounds"
            )));
        }
        let mut previous_row = None;
        for index in start..end {
            let row = index_at(index)?;
            if row >= nrows {
                return Err(BytecodeError::Decode(format!(
                    "{field} row index out of bounds"
                )));
            }
            if upper_triangular && row > col {
                return Err(BytecodeError::Decode(format!(
                    "{field} entries must be upper triangular"
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

pub(crate) fn validate_qdldl_symbolic_l(
    symbolic_l: &QdldlSymbolicL,
    kkt_pattern: &EmbeddedCscPattern,
) -> Result<(), BytecodeError> {
    symbolic_l
        .l_pattern
        .validate("qdldl_plan.symbolic_l.l_pattern")?;
    let kkt_nrows = usize::try_from(kkt_pattern.nrows).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern row count exceeds usize".to_string())
    })?;
    let kkt_ncols = usize::try_from(kkt_pattern.ncols).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern column count exceeds usize".to_string())
    })?;
    if symbolic_l.l_pattern.nrows != kkt_pattern.nrows
        || symbolic_l.l_pattern.ncols != kkt_pattern.ncols
    {
        return Err(BytecodeError::Decode(
            "qdldl_plan.symbolic_l.l_pattern dimensions must match the KKT pattern".to_string(),
        ));
    }
    if symbolic_l.etree.len() != kkt_ncols {
        return Err(BytecodeError::Decode(
            "qdldl_plan.symbolic_l.etree length must match the KKT column count".to_string(),
        ));
    }
    if symbolic_l.lnz.len() != kkt_ncols {
        return Err(BytecodeError::Decode(
            "qdldl_plan.symbolic_l.lnz length must match the KKT column count".to_string(),
        ));
    }
    validate_csc_structure(
        symbolic_l.l_pattern.indptr.len(),
        symbolic_l.l_pattern.indices.len(),
        |index| {
            usize::try_from(symbolic_l.l_pattern.indptr[index]).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indptr exceeds usize".to_string(),
                )
            })
        },
        |index| {
            usize::try_from(symbolic_l.l_pattern.indices[index]).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indices exceed usize".to_string(),
                )
            })
        },
        kkt_nrows,
        kkt_ncols,
        false,
        "qdldl_plan.symbolic_l.l_pattern",
    )?;
    for col in 0..kkt_ncols {
        let start = usize::try_from(symbolic_l.l_pattern.indptr[col]).map_err(|_| {
            BytecodeError::Decode(
                "qdldl_plan.symbolic_l.l_pattern indptr exceeds usize".to_string(),
            )
        })?;
        let end = usize::try_from(symbolic_l.l_pattern.indptr[col + 1]).map_err(|_| {
            BytecodeError::Decode(
                "qdldl_plan.symbolic_l.l_pattern indptr exceeds usize".to_string(),
            )
        })?;
        let nnz = usize::try_from(symbolic_l.lnz[col]).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.symbolic_l.lnz entry exceeds usize".to_string())
        })?;
        if end - start != nnz {
            return Err(BytecodeError::Decode(
                "qdldl_plan.symbolic_l.lnz must match the L column counts".to_string(),
            ));
        }
        for row in &symbolic_l.l_pattern.indices[start..end] {
            let row = usize::try_from(*row).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indices exceed usize".to_string(),
                )
            })?;
            if row <= col {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern must store strictly lower-triangular rows"
                        .to_string(),
                ));
            }
        }
        let parent = symbolic_l.etree[col];
        if parent != u32::MAX {
            let parent = usize::try_from(parent).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.symbolic_l.etree entry exceeds usize".to_string())
            })?;
            if parent >= kkt_ncols || parent <= col {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.etree parents must point to a later KKT column or u32::MAX"
                        .to_string(),
                ));
            }
        }
    }
    for col in 0..kkt_ncols {
        let mut cursor = symbolic_l.etree[col];
        let mut steps = 0usize;
        while cursor != u32::MAX {
            let parent = usize::try_from(cursor).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.symbolic_l.etree entry exceeds usize".to_string())
            })?;
            steps += 1;
            if steps > kkt_ncols {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.etree must be acyclic".to_string(),
                ));
            }
            cursor = symbolic_l.etree[parent];
        }
    }
    Ok(())
}

pub(crate) fn validate_archived_qdldl_symbolic_l(
    symbolic_l: &ArchivedQdldlSymbolicL,
    kkt_pattern: &ArchivedEmbeddedCscPattern,
) -> Result<(), BytecodeError> {
    validate_archived_embedded_csc_pattern(
        &symbolic_l.l_pattern,
        "qdldl_plan.symbolic_l.l_pattern",
    )?;
    let kkt_nrows = usize::try_from(kkt_pattern.nrows.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern row count exceeds usize".to_string())
    })?;
    let kkt_ncols = usize::try_from(kkt_pattern.ncols.to_native()).map_err(|_| {
        BytecodeError::Decode("qdldl_plan.kkt_pattern column count exceeds usize".to_string())
    })?;
    if symbolic_l.l_pattern.nrows.to_native() != kkt_pattern.nrows.to_native()
        || symbolic_l.l_pattern.ncols.to_native() != kkt_pattern.ncols.to_native()
    {
        return Err(BytecodeError::Decode(
            "qdldl_plan.symbolic_l.l_pattern dimensions must match the KKT pattern".to_string(),
        ));
    }
    if symbolic_l.etree.len() != kkt_ncols {
        return Err(BytecodeError::Decode(
            "qdldl_plan.symbolic_l.etree length must match the KKT column count".to_string(),
        ));
    }
    if symbolic_l.lnz.len() != kkt_ncols {
        return Err(BytecodeError::Decode(
            "qdldl_plan.symbolic_l.lnz length must match the KKT column count".to_string(),
        ));
    }
    validate_csc_structure(
        symbolic_l.l_pattern.indptr.len(),
        symbolic_l.l_pattern.indices.len(),
        |index| {
            usize::try_from(symbolic_l.l_pattern.indptr[index].to_native()).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indptr exceeds usize".to_string(),
                )
            })
        },
        |index| {
            usize::try_from(symbolic_l.l_pattern.indices[index].to_native()).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indices exceed usize".to_string(),
                )
            })
        },
        kkt_nrows,
        kkt_ncols,
        false,
        "qdldl_plan.symbolic_l.l_pattern",
    )?;
    for col in 0..kkt_ncols {
        let start =
            usize::try_from(symbolic_l.l_pattern.indptr[col].to_native()).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indptr exceeds usize".to_string(),
                )
            })?;
        let end =
            usize::try_from(symbolic_l.l_pattern.indptr[col + 1].to_native()).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indptr exceeds usize".to_string(),
                )
            })?;
        let nnz = usize::try_from(symbolic_l.lnz[col].to_native()).map_err(|_| {
            BytecodeError::Decode("qdldl_plan.symbolic_l.lnz entry exceeds usize".to_string())
        })?;
        if end - start != nnz {
            return Err(BytecodeError::Decode(
                "qdldl_plan.symbolic_l.lnz must match the L column counts".to_string(),
            ));
        }
        for row in symbolic_l
            .l_pattern
            .indices
            .iter()
            .skip(start)
            .take(end - start)
        {
            let row = usize::try_from(row.to_native()).map_err(|_| {
                BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern indices exceed usize".to_string(),
                )
            })?;
            if row <= col {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.l_pattern must store strictly lower-triangular rows"
                        .to_string(),
                ));
            }
        }
        let parent = symbolic_l.etree[col].to_native();
        if parent != u32::MAX {
            let parent = usize::try_from(parent).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.symbolic_l.etree entry exceeds usize".to_string())
            })?;
            if parent >= kkt_ncols || parent <= col {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.etree parents must point to a later KKT column or u32::MAX"
                        .to_string(),
                ));
            }
        }
    }
    for col in 0..kkt_ncols {
        let mut cursor = symbolic_l.etree[col].to_native();
        let mut steps = 0usize;
        while cursor != u32::MAX {
            let parent = usize::try_from(cursor).map_err(|_| {
                BytecodeError::Decode("qdldl_plan.symbolic_l.etree entry exceeds usize".to_string())
            })?;
            steps += 1;
            if steps > kkt_ncols {
                return Err(BytecodeError::Decode(
                    "qdldl_plan.symbolic_l.etree must be acyclic".to_string(),
                ));
            }
            cursor = symbolic_l.etree[parent].to_native();
        }
    }
    Ok(())
}

pub(crate) fn validate_qp_program_qdldl_plan(
    plan: &QpProgramQdldlPlan,
) -> Result<(), BytecodeError> {
    plan.p_pattern.validate("qdldl_plan.p_pattern")?;
    plan.a_pattern.validate("qdldl_plan.a_pattern")?;
    plan.kkt_pattern.validate("qdldl_plan.kkt_pattern")?;
    validate_embedded_qp_plan_dimensions_impl(
        plan.p_pattern.nrows,
        plan.p_pattern.ncols,
        &plan.p_pattern.indptr,
        &plan.p_pattern.indices,
        plan.a_pattern.nrows,
        plan.a_pattern.ncols,
        &plan.a_pattern.indptr,
        &plan.a_pattern.indices,
        plan.kkt_pattern.nrows,
        plan.kkt_pattern.ncols,
        &plan.kkt_pattern.indptr,
        &plan.kkt_pattern.indices,
        &plan.p_diag_indices,
        &plan.kkt_permutation,
        &plan.p_to_kkt,
        &plan.a_to_kkt,
        &plan.rho_to_kkt,
    )?;
    plan.symbolic_l.validate(&plan.kkt_pattern)
}

pub(crate) fn validate_archived_qp_program_qdldl_plan(
    plan: &ArchivedQpProgramQdldlPlan,
) -> Result<(), BytecodeError> {
    validate_archived_qdldl_plan_dimensions(
        &plan.p_pattern,
        &plan.a_pattern,
        &plan.kkt_pattern,
        &plan.p_diag_indices,
        &plan.kkt_permutation,
        &plan.p_to_kkt,
        &plan.a_to_kkt,
        &plan.rho_to_kkt,
    )?;
    plan.symbolic_l.validate(&plan.kkt_pattern)
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
    if usize::try_from(terminal)
        .map_err(|_| BytecodeError::Decode(format!("{field} terminal indptr exceeds usize")))?
        != pattern.indices.len()
    {
        return Err(BytecodeError::Decode(format!(
            "{field} terminal indptr must match the number of indices"
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
