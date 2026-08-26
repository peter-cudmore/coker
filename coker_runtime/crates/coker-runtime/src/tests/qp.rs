#[cfg(not(osqp_embedded))]
#[test]
fn qp_workspace_layout_is_exact() {
    let module_bytes = encode_host_scalar_qp_module(false);
    let program =
        MappedQpProgram::new_from_bytes(module_bytes.as_slice(), HOST_QP_FUNCTION_ID).unwrap();
    let layout = program.workspace_layout().unwrap();

    assert_eq!(
        layout.evaluator_workspace,
        QpWorkspaceRegion { start: 0, len: 24 }
    );
    assert_eq!(
        layout.coefficient_outputs,
        QpWorkspaceRegion { start: 24, len: 24 }
    );
    assert_eq!(layout.p_x, QpWorkspaceRegion { start: 48, len: 8 });
    assert_eq!(layout.a_x, QpWorkspaceRegion { start: 56, len: 8 });
    assert_eq!(layout.q, QpWorkspaceRegion { start: 64, len: 8 });
    assert_eq!(layout.l, QpWorkspaceRegion { start: 72, len: 8 });
    assert_eq!(layout.u, QpWorkspaceRegion { start: 80, len: 8 });
    assert_eq!(
        layout.primal_warm_start,
        QpWorkspaceRegion { start: 88, len: 8 }
    );
    assert_eq!(
        layout.dual_warm_start,
        QpWorkspaceRegion { start: 96, len: 8 }
    );
    assert_eq!(layout.total_bytes(), 104);
    assert_eq!(layout.required_f64_capacity(), 13);
}

#[cfg(not(osqp_embedded))]
#[test]
fn qp_runtime_rejects_short_workspace() {
    let module_bytes = encode_host_scalar_qp_module(false);
    let program =
        MappedQpProgram::new_from_bytes(module_bytes.as_slice(), HOST_QP_FUNCTION_ID).unwrap();
    let layout = program.workspace_layout().unwrap();
    let mut workspace = vec![0.0; layout.required_f64_capacity() - 1];

    let error = QpRuntime::new(program, workspace.as_mut_slice()).unwrap_err();
    match error {
        RuntimeError::WorkspaceTooSmall { expected, actual } => {
            assert_eq!(expected, layout.required_f64_capacity());
            assert_eq!(actual, layout.required_f64_capacity() - 1);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[cfg(not(osqp_embedded))]
#[test]
fn qp_load_and_solve_reuse_caller_workspace_without_allocations() {
    let module_bytes = encode_host_scalar_qp_module(false);
    let program =
        MappedQpProgram::new_from_bytes(module_bytes.as_slice(), HOST_QP_FUNCTION_ID).unwrap();
    let layout = program.workspace_layout().unwrap();
    let mut workspace = vec![0.0; layout.required_f64_capacity()];
    let mut runtime = QpRuntime::new(program, workspace.as_mut_slice()).unwrap();

    let _tracker = AllocationTracker::start();
    {
        let result = runtime.solve(&[], None).unwrap();
        assert_eq!(result.primal.unwrap().len(), 1);
        assert_eq!(result.dual.unwrap().len(), 1);
    }
    {
        let result = runtime.solve(&[], None).unwrap();
        assert_eq!(result.primal.unwrap().len(), 1);
        assert_eq!(result.dual.unwrap().len(), 1);
    }
    assert_eq!(AllocationTracker::count(), 0);
}

#[cfg(not(osqp_embedded))]
#[test]
fn qp_parameter_updates_match_scalar_reference_and_reuse_solver_state() {
    const DEFAULT_QP_SOLVER_TOLERANCE: f64 = 1e-3;
    let module_bytes = encode_host_scalar_qp_module(true);
    let program =
        MappedQpProgram::new_from_bytes(module_bytes.as_slice(), HOST_QP_FUNCTION_ID).unwrap();
    let layout = program.workspace_layout().unwrap();
    let mut workspace = vec![0.0; layout.required_f64_capacity()];
    let mut runtime = QpRuntime::new(program, workspace.as_mut_slice()).unwrap();

    fn reference_solution(p: f64, q: f64, a: f64, l: f64, u: f64) -> f64 {
        let unconstrained = -q / p;
        let lower = l / a;
        let upper = u / a;
        unconstrained.max(lower).min(upper)
    }

    let first = [2.0f32, -4.0, 1.0, 0.0, 10.0, 0.0];
    let second = [4.0f32, -4.0, 2.0, 0.0, 1.5, 0.0];

    let first_result = runtime.solve(&[&first], None).unwrap();
    let first_primal = first_result.primal.expect("expected primal solution")[0];
    assert_eq!(first_result.status, QpSolveStatus::Solved);
    assert!((first_primal - reference_solution(2.0, -4.0, 1.0, 0.0, 10.0)).abs() < 1e-6);

    let (second_primal, second_status, iterations, primal_residual, dual_residual) = {
        let second_result = runtime.solve(&[&second], None).unwrap();
        (
            second_result.primal.expect("expected primal solution")[0],
            second_result.status,
            second_result.iterations,
            second_result.primal_residual,
            second_result.dual_residual,
        )
    };
    assert_eq!(second_status, QpSolveStatus::Solved);
    let expected_second = reference_solution(4.0, -4.0, 2.0, 0.0, 1.5);
    assert!(
        (second_primal - expected_second).abs() <= DEFAULT_QP_SOLVER_TOLERANCE,
        "updated solve produced {second_primal}, expected {expected_second}, iterations {iterations:?}, primal residual {primal_residual:?}, dual residual {dual_residual:?}",
    );
    assert!(
        primal_residual.expect("solved OSQP result must report a primal residual")
            <= DEFAULT_QP_SOLVER_TOLERANCE,
    );
    assert_ne!(first_primal, second_primal);
}

#[cfg(not(osqp_embedded))]
#[test]
fn qp_load_rejects_invalid_csc_structure() {
    let mut module = build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        6,
        1,
        false,
    );
    module
        .qp_program_mut(HOST_QP_FUNCTION_ID)
        .unwrap()
        .p_pattern
        .indptr = vec![0, 2];

    assert_qp_load_error(
        &module,
        "QP p_pattern terminal indptr must match the number of indices",
    );
}

#[cfg(not(osqp_embedded))]
#[test]
fn qp_load_rejects_bad_evaluator_output_layout() {
    let mut module = build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        6,
        1,
        false,
    );
    module
        .qp_program_mut(HOST_QP_FUNCTION_ID)
        .unwrap()
        .coefficient_outputs
        .r
        .start = 6;

    assert_qp_load_error(
        &module,
        "QP coefficient_outputs.r must start at the previous slice end",
    );
}

#[cfg(not(osqp_embedded))]
fn assert_qp_load_error(module: &BytecodeModule, expected_message: &str) {
    let bytes = encode_into_aligned_bytes(module);
    let message = MappedQpProgram::new_from_bytes(bytes.as_slice(), HOST_QP_FUNCTION_ID)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains(expected_message),
        "expected validation error containing {expected_message:?}, got {message:?}"
    );
}

#[cfg(not(osqp_embedded))]
#[test]
fn qp_load_rejects_p_lower_triangle_entries() {
    let mut module = build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        7,
        2,
        false,
    );
    let qp_program = module.qp_program_mut(HOST_QP_FUNCTION_ID).unwrap();
    qp_program.p_pattern = coker_bytecode::EmbeddedCscPattern {
        nrows: 2,
        ncols: 2,
        nnz: 1,
        indptr: vec![0, 1, 1],
        indices: vec![1],
    };
    qp_program.a_pattern = coker_bytecode::EmbeddedCscPattern {
        nrows: 1,
        ncols: 2,
        nnz: 1,
        indptr: vec![0, 1, 1],
        indices: vec![0],
    };
    qp_program.coefficient_outputs = QpCoefficientOutputs {
        px: QpOutputSlice {
            start: 0,
            length: 1,
        },
        q: QpOutputSlice {
            start: 1,
            length: 2,
        },
        ax: QpOutputSlice {
            start: 3,
            length: 1,
        },
        l: QpOutputSlice {
            start: 4,
            length: 1,
        },
        u: QpOutputSlice {
            start: 5,
            length: 1,
        },
        r: QpOutputSlice {
            start: 6,
            length: 1,
        },
    };

    assert_qp_load_error(&module, "QP p_pattern entries must be upper triangular");
}

#[cfg(not(osqp_embedded))]
#[test]
fn qp_load_rejects_bad_coefficient_output_layout_length() {
    let mut module = build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        7,
        1,
        false,
    );
    module
        .qp_program_mut(HOST_QP_FUNCTION_ID)
        .unwrap()
        .coefficient_outputs = QpCoefficientOutputs {
        px: QpOutputSlice {
            start: 0,
            length: 1,
        },
        q: QpOutputSlice {
            start: 1,
            length: 1,
        },
        ax: QpOutputSlice {
            start: 2,
            length: 1,
        },
        l: QpOutputSlice {
            start: 3,
            length: 1,
        },
        u: QpOutputSlice {
            start: 4,
            length: 1,
        },
        r: QpOutputSlice {
            start: 5,
            length: 2,
        },
    };

    assert_qp_load_error(
        &module,
        "QP coefficient_outputs.r length must match the QP dimensions and sparsity",
    );
}
#[cfg(osqp_embedded)]
#[test]
fn bound_mapped_qp_program_rejects_short_coefficient_outputs_and_recovers_after_failed_update() {
    let module = build_mapped_scalar_qp_module(2, 1, 6, 1, true);
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let original_bytes = aligned_bytes.as_slice().to_vec();
    let mapped = MappedModule::new_from_bytes(aligned_bytes.as_slice()).unwrap();
    let qp_program = mapped.qp_program(2).unwrap();
    let requirements = qp_program.workspace_requirements();
    let mut arena = vec![core::mem::MaybeUninit::<u8>::uninit(); requirements.arena_bytes];
    let mut bound = qp_program.bind(&mut arena).unwrap();
    let mut evaluator_workspace = [0.0; 6];
    let mut short_coefficient_outputs = [0.0; 5];
    let mut exact_coefficient_outputs = [0.0; 6];
    let mut oversized_coefficient_outputs = [123.0; 8];
    let mut outputs = [0.0; 1];

    let valid = [2.0f32, -4.0, 1.0, 0.0, 10.0, 0.0];
    let short_error = match bound.execute(
        &[&valid],
        None,
        MappedQpWorkspace::new(&mut evaluator_workspace, &mut short_coefficient_outputs),
        &mut outputs,
    ) {
        Ok(_) => panic!("expected short coefficient workspace rejection"),
        Err(error) => error,
    };
    assert!(matches!(
        short_error,
        RuntimeError::WorkspaceTooSmall {
            expected: 6,
            actual: 5,
        }
    ));

    let invalid = [-1.0f32, -4.0, 1.0, 0.0, 10.0, 0.0];
    let error = match bound.execute(
        &[&invalid],
        None,
        MappedQpWorkspace::new(&mut evaluator_workspace, &mut exact_coefficient_outputs),
        &mut outputs,
    ) {
        Ok(_) => panic!("expected mapped QP update failure"),
        Err(error) => error,
    };
    assert!(matches!(error, RuntimeError::EmbeddedQpCscUpdate { .. }));

    let diagnostics = bound
        .execute(
            &[&valid],
            None,
            MappedQpWorkspace::new(&mut evaluator_workspace, &mut oversized_coefficient_outputs),
            &mut outputs,
        )
        .unwrap();
    assert_eq!(diagnostics.status, QpSolveStatus::Solved);

    let updated = [2.0f32, -2.0, 1.0, 0.0, 10.0, 0.0];
    let diagnostics = bound
        .execute(
            &[&updated],
            None,
            MappedQpWorkspace::new(&mut evaluator_workspace, &mut oversized_coefficient_outputs),
            &mut outputs,
        )
        .unwrap();
    assert_eq!(diagnostics.status, QpSolveStatus::Solved);
    assert_eq!(&oversized_coefficient_outputs[6..], &[123.0, 123.0]);
    let flat = [2.0f32, -2.0, 1.0, 0.0, 10.0, 0.0];
    let mut flat_outputs = [0.0f32; 1];
    let flat_diagnostics = bound
        .execute_flat(
            &flat,
            MappedQpWorkspace::new(&mut evaluator_workspace, &mut oversized_coefficient_outputs),
            &mut flat_outputs,
        )
        .unwrap();
    assert_eq!(flat_diagnostics.status, QpSolveStatus::Solved);
    assert!(flat_outputs[0].is_nan() && outputs[0].is_nan());
    assert_eq!(aligned_bytes.as_slice(), original_bytes.as_slice());
}

#[cfg(osqp_embedded)]
#[test]
fn prepared_flat_qp_stream_is_deterministic_f32_reference() {
    let module = build_mapped_scalar_qp_module(2, 1, 6, 1, true);
    let encoded = encode_into_aligned_bytes(&module);
    let mapped = MappedModule::new_from_bytes(encoded.as_slice()).unwrap();
    let qp = mapped.qp_program(2).unwrap();
    let requirements = qp.workspace_requirements();
    let mut arena = vec![core::mem::MaybeUninit::<u8>::uninit(); requirements.arena_bytes];
    let mut prepared = unsafe { qp.prepare_detached(&mut arena) }.unwrap();
    let mut evaluator_workspace = [0.0f32; 6];
    let mut coefficient_outputs = [0.0f32; 6];
    let mut outputs = [0.0f32; 1];
    let stream = [
        [2.0f32, -4.0, 1.0, 0.0, 10.0, 0.0],
        [4.0f32, -4.0, 2.0, 0.0, 1.5, 0.0],
    ];
    let mut observed = [0.0f32; 2];
    let mut statuses = [QpSolveStatus::Unsolved; 2];
    for (index, parameters) in stream.iter().enumerate() {
        let diagnostics = prepared
            .execute_flat(
                qp,
                parameters,
                MappedQpWorkspace::new(&mut evaluator_workspace, &mut coefficient_outputs),
                &mut outputs,
            )
            .unwrap();
        statuses[index] = diagnostics.status;
        observed[index] = outputs[0];
    }
    assert_eq!(statuses, [QpSolveStatus::Solved, QpSolveStatus::Solved]);
    let mut repeated = [0.0f32; 2];
    for (index, parameters) in stream.iter().enumerate() {
        prepared
            .execute_flat(
                qp,
                parameters,
                MappedQpWorkspace::new(&mut evaluator_workspace, &mut coefficient_outputs),
                &mut outputs,
            )
            .unwrap();
        repeated[index] = outputs[0];
    }
    assert_eq!(observed[0].to_bits(), repeated[0].to_bits());
    assert_eq!(observed[1].to_bits(), repeated[1].to_bits());
}

#[cfg(not(osqp_embedded))]
#[test]
fn validated_embedded_qp_keeps_nonzero_csc_shape_borrowed_and_layout_cached() {
    let bytes = encode_embedded_qp_with_nonzero_csc();
    let bytes_start = bytes.as_ptr() as usize;
    let bytes_end = bytes_start + bytes.len();
    let _tracker = AllocationTracker::start();

    let validated = ValidatedEmbeddedQp::parse(bytes.as_slice()).unwrap();
    let layout = QpWorkspaceLayout::for_validated(&validated);
    let shape = validated.problem_shape();
    let ffi_shape = shape.as_ffi();

    assert_eq!(validated.dimensions(), (2, 2));
    assert_eq!(validated.nnz(), (2, 2));
    assert_eq!(shape.n(), 2);
    assert_eq!(shape.m(), 2);
    assert_eq!(shape.p_nnz(), 2);
    assert_eq!(shape.a_nnz(), 2);
    assert_eq!(
        layout.evaluator_workspace,
        QpWorkspaceRegion { start: 0, len: 44 }
    );
    assert_eq!(
        layout.coefficient_outputs,
        QpWorkspaceRegion { start: 44, len: 44 }
    );
    assert_eq!(layout.p_x, QpWorkspaceRegion { start: 88, len: 16 });
    assert_eq!(
        layout.a_x,
        QpWorkspaceRegion {
            start: 104,
            len: 16
        }
    );
    assert_eq!(
        layout.q,
        QpWorkspaceRegion {
            start: 120,
            len: 16
        }
    );
    assert_eq!(
        layout.l,
        QpWorkspaceRegion {
            start: 136,
            len: 16
        }
    );
    assert_eq!(
        layout.u,
        QpWorkspaceRegion {
            start: 152,
            len: 16
        }
    );
    assert_eq!(
        layout.primal_warm_start,
        QpWorkspaceRegion {
            start: 168,
            len: 16
        }
    );
    assert_eq!(
        layout.dual_warm_start,
        QpWorkspaceRegion {
            start: 184,
            len: 16
        }
    );
    assert_eq!(layout.total_bytes(), 200);
    assert_eq!(layout.required_f64_capacity(), 25);
    assert_eq!(ffi_shape.p.col_ptr, shape.p_indptr());
    assert_eq!(ffi_shape.p.row_idx, shape.p_indices());
    assert_eq!(ffi_shape.a.col_ptr, shape.a_indptr());
    assert_eq!(ffi_shape.a.row_idx, shape.a_indices());

    for pointer in [
        shape.p_indptr(),
        shape.p_indices(),
        shape.a_indptr(),
        shape.a_indices(),
    ] {
        let address = pointer as usize;
        assert_eq!(address % core::mem::align_of::<i32>(), 0);
        assert!(address >= bytes_start && address < bytes_end);
    }
    unsafe {
        assert_eq!(core::slice::from_raw_parts(shape.p_indptr(), 3), &[0, 1, 2]);
        assert_eq!(core::slice::from_raw_parts(shape.p_indices(), 2), &[0, 1]);
        assert_eq!(core::slice::from_raw_parts(shape.a_indptr(), 3), &[0, 1, 2]);
        assert_eq!(core::slice::from_raw_parts(shape.a_indices(), 2), &[0, 1]);
    }
    assert_eq!(AllocationTracker::count(), 0);
}

#[cfg(not(osqp_embedded))]
#[test]
fn validated_embedded_qp_rejects_indices_outside_osqp_i32_range() {
    let bytes = encode_embedded_qp_with_oversized_n();

    match ValidatedEmbeddedQp::parse(bytes.as_slice()) {
        Err(RuntimeError::ValidationField {
            field: "QP n",
            problem: "exceeds the embedded OSQP i32 index range",
        }) => {}
        Err(other) => panic!("unexpected error: {other:?}"),
        Ok(_) => panic!("oversized OSQP index must be rejected"),
    }
}

#[cfg(not(osqp_embedded))]
#[test]
fn validated_embedded_qp_rejects_misaligned_mapped_archive() {
    let encoded = encode_embedded_qp_with_nonzero_csc();
    let alignment = coker_bytecode::ARCHIVED_MODULE_ALIGNMENT;
    let mut storage = vec![0u8; encoded.len() + alignment];
    let offset = (0..alignment)
        .find(|offset| (storage.as_ptr() as usize + offset + 16) % alignment != 0)
        .expect("an offset must misalign the archive payload");
    storage[offset..offset + encoded.len()].copy_from_slice(encoded.as_slice());
    let mapped = &storage[offset..offset + encoded.len()];

    let _tracker = AllocationTracker::start();
    let error = match ValidatedEmbeddedQp::parse(mapped) {
        Err(error) => error,
        Ok(_) => panic!("misaligned archive must be rejected"),
    };
    assert!(matches!(error, RuntimeError::Bytecode(_)));
    assert_eq!(
        AllocationTracker::count(),
        0,
        "rejection must not repair alignment by allocating a copy"
    );
    drop(_tracker);
    assert!(error.to_string().contains("payload must be"));
}
