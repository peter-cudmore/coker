use super::*;
use coker_bytecode::{decode_module, Layer, ScalarOp};

#[test]
fn compile_exported_json_builds_module_bytecode() {
    let exported_module_json = r#"
        {
            "functions": [
                {
                    "function_id": 0,
                    "program": {
                        "workspace": {"location": 0, "count": 2},
                        "input_layer": {
                            "inputs": [
                                {"memory": {"location": 0, "count": 1}}
                            ]
                        },
                        "output_layer": {
                            "outputs": [
                                {"memory": {"location": 1, "count": 1}}
                            ]
                        },
                        "intermediate_layers": [
                            {
                                "kind": "generic",
                                "memory_in": {"location": 0, "count": 1},
                                "memory_out": {"location": 0, "count": 2},
                                "ops": [
                                    {
                                        "op": {"kind": "internal", "value": "identity"},
                                        "first": 0,
                                        "second": -1,
                                        "third": -1
                                    },
                                    {
                                        "op": {"kind": "enum", "value": "SIN"},
                                        "first": 0,
                                        "second": -1,
                                        "third": -1
                                    }
                                ]
                            }
                        ]
                    }
                }
            ]
        }
        "#;

    let module_bytes = compile_exported_json(exported_module_json.as_bytes()).unwrap();
    let module = decode_module(&module_bytes).unwrap();

    assert_eq!(module.functions.len(), 1);
    let program = &module.functions[0];
    assert_eq!(program.function_id, 0);
    assert_eq!(program.workspace_size, 2);
    assert_eq!(program.required_workspace_size, 3);
    assert_eq!(program.input_specs[0].length, 1);
    assert_eq!(program.output_specs[0].length, 1);
    match &program.intermediate_layers[0] {
        Layer::Generic(generic_layer) => {
            assert_eq!(generic_layer.ops.len(), 2);
            assert_eq!(generic_layer.ops[0].second, u16::MAX);
            assert_eq!(generic_layer.ops[1].op, ScalarOp::Sin);
            assert_eq!(generic_layer.scratch_offset, 2);
            assert_eq!(generic_layer.scratch_length, 1);
        }
        _ => panic!("expected generic layer"),
    }
}

#[test]
fn compile_exported_json_builds_evaluate_layer() {
    let exported_module_json = r#"
        {
            "functions": [
                {
                    "function_id": 0,
                    "program": {
                        "workspace": {"location": 0, "count": 1},
                        "input_layer": {"inputs": []},
                        "output_layer": {
                            "outputs": [
                                {"memory": {"location": 0, "count": 1}}
                            ]
                        },
                        "intermediate_layers": [
                            {
                                "kind": "evaluate",
                                "memory_in": {"location": 0, "count": 0},
                                "memory_out": {"location": 0, "count": 1},
                                "callee_function_id": 1,
                                "inputs": [
                                    {"kind": "constant", "length": 1, "values": [2.0]}
                                ],
                                "outputs": [
                                    {"destination_offset": 0, "length": 1}
                                ]
                            }
                        ]
                    }
                },
                {
                    "function_id": 1,
                    "program": {
                        "workspace": {"location": 0, "count": 2},
                        "input_layer": {
                            "inputs": [
                                {"memory": {"location": 0, "count": 1}}
                            ]
                        },
                        "output_layer": {
                            "outputs": [
                                {"memory": {"location": 1, "count": 1}}
                            ]
                        },
                        "intermediate_layers": [
                            {
                                "kind": "generic",
                                "memory_in": {"location": 0, "count": 1},
                                "memory_out": {"location": 0, "count": 2},
                                "ops": [
                                    {
                                        "op": {"kind": "internal", "value": "identity"},
                                        "first": 0,
                                        "second": -1,
                                        "third": -1
                                    },
                                    {
                                        "op": {"kind": "enum", "value": "SIN"},
                                        "first": 0,
                                        "second": -1,
                                        "third": -1
                                    }
                                ]
                            }
                        ]
                    }
                }
            ]
        }
        "#;

    let module_bytes = compile_exported_json(exported_module_json.as_bytes()).unwrap();
    let module = decode_module(&module_bytes).unwrap();
    let program = &module.functions[0];
    assert_eq!(program.required_workspace_size, 4);
    match &program.intermediate_layers[0] {
        Layer::Evaluate(evaluate_layer) => {
            assert_eq!(evaluate_layer.callee_function_id, 1);
            assert_eq!(evaluate_layer.scratch_offset, 1);
        }
        _ => panic!("expected evaluate layer"),
    }
}

#[test]
fn compile_exported_json_rejects_opaque_programs() {
    let exported_module_json = r#"
        {
            "functions": [
                {
                    "function_id": 0,
                    "program": {
                        "workspace": {"location": 0, "count": 1},
                        "input_layer": {"inputs": []},
                        "output_layer": {"outputs": []},
                        "intermediate_layers": [
                            {
                                "kind": "generic",
                                "memory_in": {"location": 0, "count": 0},
                                "memory_out": {"location": 0, "count": 1},
                                "ops": [
                                    {
                                        "op": {"kind": "internal", "value": "identity"},
                                        "first": -1,
                                        "second": -1,
                                        "third": -1
                                    }
                                ],
                                "opaque_programs": [{}]
                            }
                        ]
                    }
                }
            ]
        }
        "#;

    let error = compile_exported_json(exported_module_json.as_bytes()).unwrap_err();
    assert!(matches!(error, CompileError::NotImplemented(_)));
}
