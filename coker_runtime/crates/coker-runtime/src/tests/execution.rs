#[test]
fn execute_bilinear_homogeneous_tensor() {
    let module = BytecodeModule::new(vec![Program::new(
        2,
        2,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 1,
            length: 1,
        }],
        vec![Layer::Bilinear(BilinearLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            quadratic: SparseTensor {
                shape: (1, 2, 2),
                entries: vec![
                    SparseEntry {
                        index: (0, 0, 0),
                        value: 3.0,
                    },
                    SparseEntry {
                        index: (0, 0, 1),
                        value: 2.0,
                    },
                    SparseEntry {
                        index: (0, 1, 1),
                        value: 4.0,
                    },
                ],
            },
        })],
    )]);
    let mut workspace = vec![0.0; 2];
    let mut outputs = vec![0.0; 1];
    execute(&module, &[&[1.5]], &mut workspace, &mut outputs).unwrap();
    assert_eq!(outputs, vec![15.0]);
}

#[test]
fn push_forward_bilinear_homogeneous_tensor() {
    let module = BytecodeModule::new(vec![Program::new(
        2,
        2,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 1,
            length: 1,
        }],
        vec![Layer::Bilinear(BilinearLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            quadratic: SparseTensor {
                shape: (1, 2, 2),
                entries: vec![
                    SparseEntry {
                        index: (0, 0, 1),
                        value: 3.0,
                    },
                    SparseEntry {
                        index: (0, 1, 1),
                        value: 2.0,
                    },
                ],
            },
        })],
    )]);
    let mut workspace = vec![0.0; 2];
    let mut tangent_workspace = vec![0.0; 2];
    let mut outputs = vec![0.0; 1];
    let mut tangent_outputs = vec![0.0; 1];
    push_forward(
        &module,
        &[&[2.0]],
        &[&[0.5]],
        &mut workspace,
        &mut tangent_workspace,
        &mut outputs,
        &mut tangent_outputs,
    )
    .unwrap();
    assert_eq!(outputs, vec![14.0]);
    assert_eq!(tangent_outputs, vec![5.5]);
}

#[test]
fn execute_generic_layer_operations_without_allocations_after_setup() {
    let module = BytecodeModule::new(vec![Program::new(
        2,
        2,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 1,
            length: 1,
        }],
        vec![Layer::Generic(GenericLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            ops: vec![RowOp {
                first: 0,
                second: UNUSED_OPERAND,
                third: UNUSED_OPERAND,
                op: ScalarOp::Sin,
            }],
        })],
    )]);
    let mut workspace = [0.0; 2];
    let mut outputs = [0.0; 1];
    let _tracker = AllocationTracker::start();
    execute(&module, &[&[1.0]], &mut workspace, &mut outputs).unwrap();
    assert_eq!(AllocationTracker::count(), 0);
    assert_eq!(outputs[0], 1.0f32.sin());
}

#[test]
fn push_forward_generic_layer_operations_without_allocations_after_setup() {
    let module = BytecodeModule::new(vec![Program::new(
        2,
        2,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 1,
            length: 1,
        }],
        vec![Layer::Generic(GenericLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            ops: vec![RowOp {
                first: 0,
                second: UNUSED_OPERAND,
                third: UNUSED_OPERAND,
                op: ScalarOp::Sin,
            }],
        })],
    )]);
    let mut workspace = [0.0; 2];
    let mut tangent_workspace = [0.0; 2];
    let mut outputs = [0.0; 1];
    let mut tangent_outputs = [0.0; 1];
    let _tracker = AllocationTracker::start();
    push_forward(
        &module,
        &[&[2.0]],
        &[&[0.5]],
        &mut workspace,
        &mut tangent_workspace,
        &mut outputs,
        &mut tangent_outputs,
    )
    .unwrap();
    assert_eq!(AllocationTracker::count(), 0);
    assert_eq!(outputs[0], 2.0f32.sin());
    assert_eq!(tangent_outputs[0], 2.0f32.cos() * 0.5);
}

#[test]
fn execute_evaluate_layer_calls_nested_function() {
    let module = build_nested_module();
    let mut workspace = vec![0.0; 4];
    let mut outputs = vec![0.0; 1];
    execute(&module, &[], &mut workspace, &mut outputs).unwrap();
    assert_eq!(outputs, vec![2.0f32.sin()]);
}

#[test]
fn push_forward_evaluate_layer_calls_nested_function() {
    let callee_program = Program::new(
        3,
        3,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 2,
            length: 1,
        }],
        vec![Layer::Bilinear(BilinearLayer {
            in_offset: 0,
            out_offset: 2,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            quadratic: SparseTensor {
                shape: (1, 2, 2),
                entries: vec![
                    SparseEntry {
                        index: (0, 0, 1),
                        value: 1.0,
                    },
                    SparseEntry {
                        index: (0, 1, 1),
                        value: 1.0,
                    },
                ],
            },
        })],
    );
    let entry_program = Program::new(
        1,
        4,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![Layer::Evaluate(EvaluateLayer {
            scratch_offset: 1,
            callee_function_id: 1,
            input_bindings: vec![EvaluateInputBinding::WorkspaceSlice {
                offset: 0,
                length: 1,
            }],
            output_bindings: vec![EvaluateOutputBinding {
                destination_offset: 0,
                length: 1,
            }],
        })],
    );
    let module = BytecodeModule::new(vec![entry_program, callee_program]);
    let mut workspace = vec![0.0; 4];
    let mut tangent_workspace = vec![0.0; 4];
    let mut outputs = vec![0.0; 1];
    let mut tangent_outputs = vec![0.0; 1];
    push_forward(
        &module,
        &[&[2.0]],
        &[&[0.5]],
        &mut workspace,
        &mut tangent_workspace,
        &mut outputs,
        &mut tangent_outputs,
    )
    .unwrap();
    assert_eq!(outputs, vec![6.0]);
    assert_eq!(tangent_outputs, vec![2.5]);
}

#[test]
fn execute_overlapping_bilinear_layer_uses_scratch_workspace() {
    let module = BytecodeModule::new(vec![Program::new(
        2,
        3,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 1,
            length: 1,
        }],
        vec![Layer::Bilinear(BilinearLayer {
            in_offset: 0,
            out_offset: 0,
            in_length: 1,
            out_length: 2,
            scratch_offset: 2,
            scratch_length: 1,
            quadratic: SparseTensor {
                shape: (2, 2, 2),
                entries: vec![
                    SparseEntry {
                        index: (0, 0, 1),
                        value: 1.0,
                    },
                    SparseEntry {
                        index: (1, 1, 1),
                        value: 2.0,
                    },
                ],
            },
        })],
    )]);
    let mut workspace = vec![0.0; 3];
    let mut outputs = vec![0.0; 1];
    execute(&module, &[&[3.0]], &mut workspace, &mut outputs).unwrap();
    assert_eq!(outputs, vec![18.0]);
}

#[test]
fn module_builder_executes_with_caller_workspaces_without_allocations() {
    let callee_program = Program::new(
        3,
        3,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 2,
            length: 1,
        }],
        vec![Layer::Bilinear(BilinearLayer {
            in_offset: 0,
            out_offset: 2,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            quadratic: SparseTensor {
                shape: (1, 2, 2),
                entries: vec![
                    SparseEntry {
                        index: (0, 0, 1),
                        value: 1.0,
                    },
                    SparseEntry {
                        index: (0, 1, 1),
                        value: 1.0,
                    },
                ],
            },
        })],
    );
    let entry_program = Program::new(
        1,
        4,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![Layer::Evaluate(EvaluateLayer {
            scratch_offset: 1,
            callee_function_id: 1,
            input_bindings: vec![EvaluateInputBinding::WorkspaceSlice {
                offset: 0,
                length: 1,
            }],
            output_bindings: vec![EvaluateOutputBinding {
                destination_offset: 0,
                length: 1,
            }],
        })],
    );
    let module = BytecodeModule::new(vec![entry_program, callee_program]);
    let _tracker = AllocationTracker::start();
    let module = ModuleBuilder::new(module).unwrap().build().unwrap();
    let mut workspace = [0.0; 4];
    let mut tangent_workspace = [0.0; 4];
    let mut outputs = [0.0; 1];
    let mut tangent_outputs = [0.0; 1];
    module
        .execute(&[&[2.0]], &mut workspace, &mut outputs)
        .unwrap();
    assert_eq!(outputs, [6.0]);
    module
        .push_forward(
            &[&[2.0]],
            &[&[0.5]],
            &mut workspace,
            &mut tangent_workspace,
            &mut outputs,
            &mut tangent_outputs,
        )
        .unwrap();
    assert_eq!(AllocationTracker::count(), 0);
    assert_eq!(outputs, [6.0]);
    assert_eq!(tangent_outputs, [2.5]);
}

#[test]
fn mapped_module_borrows_input_slice_lifetime() {
    fn load<'a>(bytes: &'a [u8]) -> MappedModule<'a> {
        MappedModule::new_from_bytes(bytes).unwrap()
    }

    let module = build_nested_module();
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let mapped = load(aligned_bytes.as_slice());
    assert_eq!(mapped.info().required_workspace_size, 4);
}
#[test]
fn mapped_module_looks_up_dense_program_indices() {
    let callee_program = Program::new(
        1,
        1,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![],
    );
    let entry_program = Program::new(
        1,
        2,
        vec![],
        vec![OutputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![Layer::Evaluate(EvaluateLayer {
            scratch_offset: 1,
            callee_function_id: 1,
            input_bindings: vec![EvaluateInputBinding::ConstantSlice {
                length: 1,
                values: vec![3.0],
            }],
            output_bindings: vec![EvaluateOutputBinding {
                destination_offset: 0,
                length: 1,
            }],
        })],
    );
    let module = BytecodeModule::new(vec![entry_program, callee_program]);
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let mapped = MappedModule::new_from_bytes(aligned_bytes.as_slice()).unwrap();

    assert_eq!(mapped.program(0).unwrap().function_id(), 0);
    assert_eq!(mapped.program(1).unwrap().function_id(), 1);
    assert!(matches!(
        mapped.program(2),
        Err(RuntimeError::MissingFunction { function_id: 2 })
    ));

    let mut workspace = [0.0; 2];
    let mut outputs = [0.0; 1];
    mapped.execute(&[], &mut workspace, &mut outputs).unwrap();
    assert_eq!(outputs, [3.0]);
}
#[cfg(not(osqp_embedded))]
#[test]
fn mapped_qp_program_looks_up_qp_index_from_bytes_and_module() {
    let module = build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        6,
        1,
        false,
    );
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let mapped = MappedModule::new_from_bytes(aligned_bytes.as_slice()).unwrap();
    let from_bytes =
        MappedQpProgram::new_from_bytes(aligned_bytes.as_slice(), HOST_QP_FUNCTION_ID).unwrap();
    let from_module = mapped.qp_program(HOST_QP_FUNCTION_ID).unwrap();

    assert_eq!(from_bytes.function_id(), HOST_QP_FUNCTION_ID);
    assert_eq!(from_module.function_id(), HOST_QP_FUNCTION_ID);
    assert_eq!(
        from_bytes.workspace_requirements().coefficient_output_size,
        6
    );
    assert_eq!(
        from_module.workspace_requirements().coefficient_output_size,
        6
    );
    assert_eq!(
        from_bytes.workspace_requirements().tangent_workspace_size,
        1
    );
    assert_eq!(
        from_module.workspace_requirements().tangent_workspace_size,
        1
    );
    assert!(matches!(
        MappedQpProgram::new_from_bytes(aligned_bytes.as_slice(), 0),
        Err(RuntimeError::MissingQpFunction { function_id: 0 })
    ));
    assert!(matches!(
        mapped.qp_program(0),
        Err(RuntimeError::MissingQpFunction { function_id: 0 })
    ));
}

#[cfg(not(osqp_embedded))]
#[test]
fn mapped_qp_program_rejects_output_spec_length_mismatch() {
    let module = build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        6,
        2,
        false,
    );
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let expected_message = "QP output spec length must match the decision-vector dimension";

    for error in [
        match MappedModule::new_from_bytes(aligned_bytes.as_slice()) {
            Ok(_) => panic!("expected mapped module byte validation to fail"),
            Err(error) => error,
        },
        match MappedQpProgram::new_from_bytes(aligned_bytes.as_slice(), HOST_QP_FUNCTION_ID) {
            Ok(_) => panic!("expected mapped QP byte validation to fail"),
            Err(error) => error,
        },
    ] {
        match error {
            RuntimeError::Bytecode(coker_bytecode::BytecodeError::Decode(message)) => {
                assert!(message.contains(expected_message));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[cfg(not(osqp_embedded))]
#[test]
fn mapped_qp_program_rejects_evaluator_output_length_mismatch() {
    let module = build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        7,
        1,
        false,
    );
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let expected_message =
        "QP coefficient evaluator output lengths do not match coefficient slices";

    for error in [
        match MappedModule::new_from_bytes(aligned_bytes.as_slice()) {
            Ok(_) => panic!("expected mapped module byte validation to fail"),
            Err(error) => error,
        },
        match MappedQpProgram::new_from_bytes(aligned_bytes.as_slice(), HOST_QP_FUNCTION_ID) {
            Ok(_) => panic!("expected mapped QP byte validation to fail"),
            Err(error) => error,
        },
    ] {
        match error {
            RuntimeError::Bytecode(coker_bytecode::BytecodeError::Decode(message)) => {
                assert!(message.contains(expected_message));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

#[test]
fn mapped_module_executes_from_aligned_bytes_without_allocations() {
    let module = build_nested_module();
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let _tracker = AllocationTracker::start();
    let module = MappedModule::new_from_bytes(aligned_bytes.as_slice()).unwrap();
    let mut workspace = [0.0; 4];
    let mut outputs = [0.0; 1];
    module.execute(&[], &mut workspace, &mut outputs).unwrap();
    assert_eq!(AllocationTracker::count(), 0);
    assert_eq!(outputs, [2.0f32.sin()]);
}
#[test]
fn mapped_module_from_bytes_with_workspace_capacities_is_allocation_free() {
    let module = build_nested_module();
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let _tracker = AllocationTracker::start();
    let module =
        MappedModule::new_from_bytes_with_workspace_capacities(aligned_bytes.as_slice(), 4, 4)
            .unwrap();
    let mut workspace = [0.0; 4];
    let mut outputs = [0.0; 1];
    module.execute(&[], &mut workspace, &mut outputs).unwrap();
    assert_eq!(AllocationTracker::count(), 0);
    assert_eq!(outputs, [2.0f32.sin()]);
}

#[test]
fn mapped_module_from_bytes_with_workspace_capacities_rejects_short_workspace() {
    let module = build_nested_module();
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let _tracker = AllocationTracker::start();
    let error = match MappedModule::new_from_bytes_with_workspace_capacities(
        aligned_bytes.as_slice(),
        3,
        4,
    ) {
        Ok(_) => panic!("expected workspace-capacity validation to fail"),
        Err(error) => error,
    };
    assert_eq!(AllocationTracker::count(), 0);
    assert!(matches!(
        error,
        RuntimeError::WorkspaceTooSmall {
            expected: 4,
            actual: 3,
        }
    ));
}

#[test]
fn mapped_module_push_forward_from_aligned_bytes() {
    let callee_program = Program::new(
        3,
        3,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 2,
            length: 1,
        }],
        vec![Layer::Bilinear(BilinearLayer {
            in_offset: 0,
            out_offset: 2,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            quadratic: SparseTensor {
                shape: (1, 2, 2),
                entries: vec![
                    SparseEntry {
                        index: (0, 0, 1),
                        value: 1.0,
                    },
                    SparseEntry {
                        index: (0, 1, 1),
                        value: 1.0,
                    },
                ],
            },
        })],
    );
    let entry_program = Program::new(
        1,
        4,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![Layer::Evaluate(EvaluateLayer {
            scratch_offset: 1,
            callee_function_id: 1,
            input_bindings: vec![EvaluateInputBinding::WorkspaceSlice {
                offset: 0,
                length: 1,
            }],
            output_bindings: vec![EvaluateOutputBinding {
                destination_offset: 0,
                length: 1,
            }],
        })],
    );
    let module = BytecodeModule::new(vec![entry_program, callee_program]);
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let _tracker = AllocationTracker::start();
    let module = MappedModule::new_from_bytes(aligned_bytes.as_slice()).unwrap();
    let mut workspace = [0.0; 4];
    let mut tangent_workspace = [0.0; 4];
    let mut outputs = [0.0; 1];
    let mut tangent_outputs = [0.0; 1];
    module
        .push_forward(
            &[&[2.0]],
            &[&[0.5]],
            &mut workspace,
            &mut tangent_workspace,
            &mut outputs,
            &mut tangent_outputs,
        )
        .unwrap();
    assert_eq!(AllocationTracker::count(), 0);
    assert_eq!(outputs, [6.0]);
    assert_eq!(tangent_outputs, [2.5]);
}

#[test]
fn module_execute_rejects_short_workspace() {
    let module = ModuleBuilder::new(build_nested_module())
        .unwrap()
        .build()
        .unwrap();
    let mut workspace = vec![0.0; 3];
    let mut outputs = vec![0.0; 1];
    let error = module
        .execute(&[], &mut workspace, &mut outputs)
        .unwrap_err();
    assert!(matches!(
        error,
        RuntimeError::WorkspaceTooSmall {
            expected: 4,
            actual: 3,
        }
    ));
}

#[test]
fn module_validate_inputs_rejects_wrong_shape_before_execution() {
    let module = ModuleBuilder::new(build_nested_module())
        .unwrap()
        .build()
        .unwrap();
    let error = module.validate_inputs(&[&[1.0]]).unwrap_err();
    assert!(matches!(
        error,
        RuntimeError::InputCountMismatch {
            expected: 0,
            actual: 1,
        }
    ));
}

#[test]
fn module_validate_outputs_rejects_wrong_shape_before_execution() {
    let module = ModuleBuilder::new(build_nested_module())
        .unwrap()
        .build()
        .unwrap();
    let mut outputs = vec![0.0; 2];
    let error = module.validate_outputs(&mut outputs).unwrap_err();
    assert!(matches!(
        error,
        RuntimeError::OutputBufferSizeMismatch {
            expected: 1,
            actual: 2,
        }
    ));
}

#[test]
fn validate_rejects_overlapping_generic_ranges_without_scratch() {
    let module = BytecodeModule::new(vec![Program::new(
        3,
        3,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 2,
            length: 1,
        }],
        vec![Layer::Generic(GenericLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 2,
            out_length: 2,
            scratch_offset: 0,
            scratch_length: 0,
            ops: vec![
                RowOp {
                    first: 0,
                    second: UNUSED_OPERAND,
                    third: UNUSED_OPERAND,
                    op: ScalarOp::Identity,
                },
                RowOp {
                    first: 1,
                    second: UNUSED_OPERAND,
                    third: UNUSED_OPERAND,
                    op: ScalarOp::Identity,
                },
            ],
        })],
    )]);
    let encoded = encode_module(&module).unwrap();
    let error = validate_module(&encoded).unwrap_err();
    assert!(matches!(
        error,
        RuntimeError::ValidationContext {
            context: "generic layer",
            problem: "scratch length must match input length",
        }
    ));
}

#[test]
fn validate_rejects_generic_row_count_mismatch() {
    let module = BytecodeModule::new(vec![Program::new(
        3,
        3,
        vec![InputSpec {
            workspace_offset: 0,
            length: 1,
        }],
        vec![OutputSpec {
            workspace_offset: 2,
            length: 1,
        }],
        vec![Layer::Generic(GenericLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 1,
            out_length: 1,
            scratch_offset: 0,
            scratch_length: 0,
            ops: vec![
                RowOp {
                    first: 0,
                    second: UNUSED_OPERAND,
                    third: UNUSED_OPERAND,
                    op: ScalarOp::Identity,
                },
                RowOp {
                    first: 0,
                    second: UNUSED_OPERAND,
                    third: UNUSED_OPERAND,
                    op: ScalarOp::Sin,
                },
            ],
        })],
    )]);
    let encoded = encode_module(&module).unwrap();
    let error = validate_module(&encoded).unwrap_err();
    assert!(matches!(error, RuntimeError::Validation(_)));
    assert!(error
        .to_string()
        .contains("generic layer op count must match output length"));
}

#[test]
fn parse_and_validate_round_trip() {
    let module = BytecodeModule::new(vec![Program::new(0, 0, vec![], vec![], vec![])]);
    let encoded = encode_module(&module).unwrap();
    let decoded = validate_module(&encoded).unwrap();
    assert_eq!(decoded.program(0).unwrap().workspace_size, 0);
}

