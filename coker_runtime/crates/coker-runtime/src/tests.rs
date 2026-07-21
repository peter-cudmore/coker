use super::*;
use coker_bytecode::{
    encode_module, BilinearLayer, EvaluateInputBinding, EvaluateLayer, EvaluateOutputBinding,
    GenericLayer, Layer, RowOp, ScalarOp, SparseEntry, SparseTensor,
};

    fn build_nested_module() -> BytecodeModule {
        let callee_program = Program::new(
            1,
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
                out_offset: 0,
                in_length: 1,
                out_length: 2,
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
        );
        let entry_program = Program::new(
            0,
            1,
            3,
            vec![],
            vec![OutputSpec {
                workspace_offset: 0,
                length: 1,
            }],
            vec![Layer::Evaluate(EvaluateLayer {
                scratch_offset: 1,
                callee_function_id: 1,
                input_count: 1,
                output_count: 1,
                input_bindings: vec![EvaluateInputBinding::ConstantSlice {
                    length: 1,
                    values: vec![2.0],
                }],
                output_bindings: vec![EvaluateOutputBinding {
                    destination_offset: 0,
                    length: 1,
                }],
            })],
        );
        BytecodeModule::new(vec![entry_program, callee_program])
    }

    #[test]
    fn execute_bilinear_homogeneous_tensor() {
        let module = BytecodeModule::new(vec![Program::new(
            0,
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
        let outputs = execute(&module, &[&[1.5]], &mut workspace).unwrap();
        assert_eq!(outputs[0], vec![15.0]);
    }

    #[test]
    fn push_forward_bilinear_homogeneous_tensor() {
        let module = BytecodeModule::new(vec![Program::new(
            0,
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
        let (outputs, tangents) = push_forward(
            &module,
            &[&[2.0]],
            &[&[0.5]],
            &mut workspace,
            &mut tangent_workspace,
        )
        .unwrap();
        assert_eq!(outputs[0], vec![14.0]);
        assert_eq!(tangents[0], vec![5.5]);
    }

    #[test]
    fn execute_generic_layer_operations() {
        let module = BytecodeModule::new(vec![Program::new(
            0,
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
                out_offset: 0,
                in_length: 1,
                out_length: 2,
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
        let mut workspace = vec![0.0; 2];
        let outputs = execute(&module, &[&[1.0]], &mut workspace).unwrap();
        assert_eq!(outputs[0][0], 1.0f32.sin());
    }

    #[test]
    fn execute_evaluate_layer_calls_nested_function() {
        let module = build_nested_module();
        let mut workspace = vec![0.0; 3];
        let outputs = execute(&module, &[], &mut workspace).unwrap();
        assert_eq!(outputs[0], vec![2.0f32.sin()]);
    }

    #[test]
    fn push_forward_evaluate_layer_calls_nested_function() {
        let callee_program = Program::new(
            1,
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
            0,
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
                input_count: 1,
                output_count: 1,
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
        let (outputs, tangents) = push_forward(
            &module,
            &[&[2.0]],
            &[&[0.5]],
            &mut workspace,
            &mut tangent_workspace,
        )
        .unwrap();
        assert_eq!(outputs[0], vec![6.0]);
        assert_eq!(tangents[0], vec![2.5]);
    }

    #[test]
    fn parse_and_validate_round_trip() {
        let module = BytecodeModule::new(vec![Program::new(0, 0, 0, vec![], vec![], vec![])]);
        let encoded = encode_module(&module).unwrap();
        let decoded = validate_module(&encoded).unwrap();
        assert_eq!(decoded.functions[0].workspace_size, 0);
    }
