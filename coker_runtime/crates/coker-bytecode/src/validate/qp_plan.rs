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
