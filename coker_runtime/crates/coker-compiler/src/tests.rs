use super::*;
use coker_bytecode::{
    decode_module, EmbeddedCscPattern, EmbeddedLinsysSolver, EmbeddedOsqpSettings,
    EmbeddedQpProfile, Layer, QdldlSymbolicL, QpProgramArenaRegion, QpProgramQdldlPlan, ScalarOp,
};
use serde_json::{json, Value};

const QP_ARENA_REGION_NAMES: &[&str] = &[
    "pdata_x",
    "pdata",
    "adata_x",
    "adata",
    "qdata",
    "ldata",
    "udata",
    "data",
    "settings",
    "xsolution",
    "ysolution",
    "solution",
    "info",
    "qdldl_l_x",
    "qdldl_l",
    "qdldl_kkt_x",
    "qdldl_kkt",
    "qdldl",
    "qdldl_dinv",
    "qdldl_bp",
    "qdldl_sol",
    "qdldl_rho_inv_vec",
    "qdldl_d",
    "qdldl_iwork",
    "qdldl_bwork",
    "qdldl_fwork",
    "work_rho_vec",
    "work_rho_inv_vec",
    "work_constr_type",
    "work_x",
    "work_y",
    "work_z",
    "work_xz_tilde",
    "work_x_prev",
    "work_z_prev",
    "work_ax",
    "work_px",
    "work_aty",
    "work_delta_y",
    "work_atdelta_y",
    "work_delta_x",
    "work_pdelta_x",
    "work_adelta_x",
    "workspace",
];

fn coefficient_function_json(output_count: u32) -> Value {
    json!({
        "function_id": 0,
        "program": {
            "workspace": {"location": 0, "count": output_count},
            "input_layer": {
                "inputs": [
                    {"memory": {"location": 0, "count": 2}}
                ]
            },
            "output_layer": {
                "outputs": [
                    {"memory": {"location": 0, "count": output_count}}
                ]
            },
            "intermediate_layers": []
        }
    })
}

fn qp_program_arena_layout_json() -> Value {
    let mut object = serde_json::Map::new();
    let mut offset = 0u64;
    for field in QP_ARENA_REGION_NAMES {
        object.insert(
            (*field).to_string(),
            json!({
                "byte_offset": offset,
                "byte_len": 1,
                "byte_alignment": 1
            }),
        );
        offset += 1;
    }
    object.insert("total_bytes".to_string(), json!(offset));
    object.insert("arena_alignment".to_string(), json!(1));
    Value::Object(object)
}

fn qp_program_plan_json() -> Value {
    json!({
        "abi_version": 1,
        "profile": "Osqp063Embedded2Qdldl",
        "version": 1,
        "settings": {
            "rho": 0.1,
            "sigma": 1e-6,
            "alpha": 1.6,
            "adaptive_rho": true,
            "adaptive_rho_interval": 50,
            "adaptive_rho_tolerance": 5.0,
            "max_iter": 4000,
            "eps_abs": 1e-3,
            "eps_rel": 1e-3,
            "eps_prim_inf": 1e-4,
            "eps_dual_inf": 1e-4,
            "scaling": 0,
            "scaled_termination": false,
            "check_termination": 25,
            "warm_start": true,
            "linsys_solver": "Qdldl"
        },
        "arena_layout": qp_program_arena_layout_json(),
        "qdldl_plan": {
            "p_pattern": {
                "nrows": 1,
                "ncols": 1,
                "indptr": [0, 1],
                "indices": [0]
            },
            "a_pattern": {
                "nrows": 1,
                "ncols": 1,
                "indptr": [0, 1],
                "indices": [0]
            },
            "kkt_pattern": {
                "nrows": 2,
                "ncols": 2,
                "indptr": [0, 1, 3],
                "indices": [0, 0, 1]
            },
            "p_diag_indices": [0],
            "kkt_permutation": [1, 0],
            "p_to_kkt": [0],
            "a_to_kkt": [1],
            "rho_to_kkt": [2],
            "symbolic_l": {
                "l_pattern": {
                    "nrows": 2,
                    "ncols": 2,
                    "indptr": [0, 1, 1],
                    "indices": [1]
                },
                "etree": [1, 4294967295u64],
                "lnz": [1, 0]
            }
        }
    })
}

fn exported_qp_module_json() -> Value {
    json!({
        "functions": [coefficient_function_json(6)],
        "qp_programs": [
            {
                "function_id": 1,
                "coefficient_function_id": 0,
                "required_primal_workspace_size": 1,
                "required_tangent_workspace_size": 1,
                "input_specs": [
                    {"memory": {"location": 0, "count": 2}}
                ],
                "output_spec": {"memory": {"location": 0, "count": 1}},
                "p_pattern": {
                    "nrows": 1,
                    "ncols": 1,
                    "indptr": [0, 1],
                    "indices": [0]
                },
                "a_pattern": {
                    "nrows": 1,
                    "ncols": 1,
                    "indptr": [0, 1],
                    "indices": [0]
                },
                "coefficient_outputs": {
                    "px": {"start": 0, "length": 1},
                    "q": {"start": 1, "length": 1},
                    "ax": {"start": 2, "length": 1},
                    "l": {"start": 3, "length": 1},
                    "u": {"start": 4, "length": 1},
                    "r": {"start": 5, "length": 1}
                },
                "embedded_plan": qp_program_plan_json()
            }
        ]
    })
}

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

    assert_eq!(module.functions().count(), 1);
    let program = module.program(0).unwrap();
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
    let program = module.program(0).unwrap();
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
fn compile_exported_qp_json_builds_single_module_with_qp_program() {
    let exported_qp_json = exported_qp_module_json().to_string();

    let module_bytes = compile_exported_qp_json(exported_qp_json.as_bytes()).unwrap();
    let module = decode_module(&module_bytes).unwrap();

    assert_eq!(module.functions().count(), 1);
    assert_eq!(module.qp_programs().count(), 1);

    let coefficient_function = module.program(0).unwrap();
    let qp_program = module.qp_program(1).unwrap();

    assert_eq!(qp_program.coefficient_function_id, 0);
    assert_eq!(qp_program.required_primal_workspace_size, 1);
    assert_eq!(qp_program.required_tangent_workspace_size, 1);
    assert_eq!(qp_program.input_specs, coefficient_function.input_specs);
    assert_eq!(qp_program.output_spec.workspace_offset, 0);
    assert_eq!(qp_program.output_spec.length, 1);
    assert_eq!(
        qp_program.p_pattern,
        EmbeddedCscPattern {
            nrows: 1,
            ncols: 1,
            indptr: vec![0, 1],
            indices: vec![0],
        }
    );
    assert_eq!(qp_program.coefficient_outputs.r.length, 1);
    assert_eq!(qp_program.embedded_plan.abi_version, 1);
    assert_eq!(
        qp_program.embedded_plan.profile,
        EmbeddedQpProfile::Osqp063Embedded2Qdldl
    );
    assert_eq!(
        qp_program.embedded_plan.settings,
        EmbeddedOsqpSettings {
            rho: 0.1,
            sigma: 1e-6,
            alpha: 1.6,
            adaptive_rho: true,
            adaptive_rho_interval: 50,
            adaptive_rho_tolerance: 5.0,
            max_iter: 4_000,
            eps_abs: 1e-3,
            eps_rel: 1e-3,
            eps_prim_inf: 1e-4,
            eps_dual_inf: 1e-4,
            scaling: 0,
            scaled_termination: false,
            check_termination: 25,
            warm_start: true,
            linsys_solver: EmbeddedLinsysSolver::Qdldl,
        }
    );
    assert_eq!(qp_program.embedded_plan.version, 1);
    assert_eq!(
        qp_program.embedded_plan.qdldl_plan,
        QpProgramQdldlPlan {
            p_pattern: EmbeddedCscPattern {
                nrows: 1,
                ncols: 1,
                indptr: vec![0, 1],
                indices: vec![0],
            },
            a_pattern: EmbeddedCscPattern {
                nrows: 1,
                ncols: 1,
                indptr: vec![0, 1],
                indices: vec![0],
            },
            kkt_pattern: EmbeddedCscPattern {
                nrows: 2,
                ncols: 2,
                indptr: vec![0, 1, 3],
                indices: vec![0, 0, 1],
            },
            p_diag_indices: vec![0],
            kkt_permutation: vec![1, 0],
            p_to_kkt: vec![0],
            a_to_kkt: vec![1],
            rho_to_kkt: vec![2],
            symbolic_l: QdldlSymbolicL {
                l_pattern: EmbeddedCscPattern {
                    nrows: 2,
                    ncols: 2,
                    indptr: vec![0, 1, 1],
                    indices: vec![1],
                },
                etree: vec![1, u32::MAX],
                lnz: vec![1, 0],
            },
        }
    );
    assert_eq!(
        qp_program.embedded_plan.arena_layout.qdldl_l,
        QpProgramArenaRegion {
            byte_offset: 14,
            byte_len: 1,
            byte_alignment: 1,
        }
    );
}

#[test]
fn compile_exported_qp_json_rejects_mismatched_embedded_plan_pattern() {
    let mut exported_qp_json = exported_qp_module_json();
    exported_qp_json["functions"][0]["program"]["workspace"]["count"] = json!(5);
    exported_qp_json["functions"][0]["program"]["output_layer"]["outputs"][0]["memory"]["count"] =
        json!(5);
    exported_qp_json["qp_programs"][0]["p_pattern"]["indptr"] = json!([0, 0]);
    exported_qp_json["qp_programs"][0]["p_pattern"]["indices"] = json!([]);
    exported_qp_json["qp_programs"][0]["coefficient_outputs"] = json!({
        "px": {"start": 0, "length": 0},
        "q": {"start": 0, "length": 1},
        "ax": {"start": 1, "length": 1},
        "l": {"start": 2, "length": 1},
        "u": {"start": 3, "length": 1},
        "r": {"start": 4, "length": 1}
    });

    let error = compile_exported_qp_json(exported_qp_json.to_string().as_bytes()).unwrap_err();

    assert!(matches!(
        error,
        CompileError::InvalidField {
            field: "embedded_plan.qdldl_plan.p_pattern",
            ..
        }
    ));
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

    assert!(matches!(
        error,
        CompileError::NotImplemented(message)
            if message == "function evaluation and opaque programs"
    ));
}
