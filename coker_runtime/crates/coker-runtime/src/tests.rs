use super::*;
use coker_bytecode::{
    encode_module, BilinearLayer, BytecodeModule, EvaluateInputBinding, EvaluateLayer,
    EvaluateOutputBinding, GenericLayer, Layer, OutputSpec, Program, QpCoefficientOutputs,
    QpOutputSlice, RowOp, ScalarOp, SparseEntry, SparseTensor,
};
use rkyv::util::AlignedVec;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
#[cfg(not(osqp_embedded))]
const HOST_QP_FUNCTION_ID: u16 = 2;
#[cfg(not(osqp_embedded))]
const HOST_COEFFICIENT_FUNCTION_ID: u16 = 1;

#[cfg(not(osqp_embedded))]
fn encode_host_scalar_qp_module(parameterized: bool) -> AlignedVec<16> {
    encode_into_aligned_bytes(&build_mapped_scalar_qp_module(
        HOST_QP_FUNCTION_ID,
        HOST_COEFFICIENT_FUNCTION_ID,
        6,
        1,
        parameterized,
    ))
}
fn scalar_qp_pattern() -> coker_bytecode::EmbeddedCscPattern {
    coker_bytecode::EmbeddedCscPattern {
        nrows: 1,
        ncols: 1,
        indptr: vec![0, 1],
        indices: vec![0],
    }
}

fn scalar_qp_coefficient_outputs() -> QpCoefficientOutputs {
    QpCoefficientOutputs {
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
            length: 1,
        },
    }
}

fn scalar_qp_region(byte_offset: u32) -> coker_bytecode::QpProgramArenaRegion {
    coker_bytecode::QpProgramArenaRegion {
        byte_offset,
        byte_len: 1,
        byte_alignment: 1,
    }
}

fn scalar_qp_plan() -> coker_bytecode::QpProgramPlan {
    coker_bytecode::QpProgramPlan {
        abi_version: coker_bytecode::QpProgramPlan::ABI_VERSION,
        profile: coker_bytecode::EmbeddedQpProfile::Osqp063Embedded2Qdldl,
        version: coker_bytecode::QpProgramPlan::VERSION,
        settings: coker_bytecode::EmbeddedOsqpSettings {
            rho: 0.1,
            sigma: 1e-6,
            alpha: 1.6,
            adaptive_rho: true,
            adaptive_rho_interval: 50,
            adaptive_rho_tolerance: 5.0,
            max_iter: 4000,
            eps_abs: 1e-3,
            eps_rel: 1e-3,
            eps_prim_inf: 1e-4,
            eps_dual_inf: 1e-4,
            scaling: 0,
            scaled_termination: false,
            check_termination: 25,
            warm_start: true,
            linsys_solver: coker_bytecode::EmbeddedLinsysSolver::Qdldl,
        },
        arena_layout: coker_bytecode::QpProgramArenaLayout {
            total_bytes: 44,
            arena_alignment: 1,
            pdata_x: scalar_qp_region(0),
            pdata: scalar_qp_region(1),
            adata_x: scalar_qp_region(2),
            adata: scalar_qp_region(3),
            qdata: scalar_qp_region(4),
            ldata: scalar_qp_region(5),
            udata: scalar_qp_region(6),
            data: scalar_qp_region(7),
            settings: scalar_qp_region(8),
            xsolution: scalar_qp_region(9),
            ysolution: scalar_qp_region(10),
            solution: scalar_qp_region(11),
            info: scalar_qp_region(12),
            qdldl_l_x: scalar_qp_region(13),
            qdldl_l: scalar_qp_region(14),
            qdldl_kkt_x: scalar_qp_region(15),
            qdldl_kkt: scalar_qp_region(16),
            qdldl: scalar_qp_region(17),
            qdldl_dinv: scalar_qp_region(18),
            qdldl_bp: scalar_qp_region(19),
            qdldl_sol: scalar_qp_region(20),
            qdldl_rho_inv_vec: scalar_qp_region(21),
            qdldl_d: scalar_qp_region(22),
            qdldl_iwork: scalar_qp_region(23),
            qdldl_bwork: scalar_qp_region(24),
            qdldl_fwork: scalar_qp_region(25),
            work_rho_vec: scalar_qp_region(26),
            work_rho_inv_vec: scalar_qp_region(27),
            work_constr_type: scalar_qp_region(28),
            work_x: scalar_qp_region(29),
            work_y: scalar_qp_region(30),
            work_z: scalar_qp_region(31),
            work_xz_tilde: scalar_qp_region(32),
            work_x_prev: scalar_qp_region(33),
            work_z_prev: scalar_qp_region(34),
            work_ax: scalar_qp_region(35),
            work_px: scalar_qp_region(36),
            work_aty: scalar_qp_region(37),
            work_delta_y: scalar_qp_region(38),
            work_atdelta_y: scalar_qp_region(39),
            work_delta_x: scalar_qp_region(40),
            work_pdelta_x: scalar_qp_region(41),
            work_adelta_x: scalar_qp_region(42),
            workspace: scalar_qp_region(43),
        },
        qdldl_plan: coker_bytecode::QpProgramQdldlPlan {
            p_pattern: scalar_qp_pattern(),
            a_pattern: scalar_qp_pattern(),
            kkt_pattern: coker_bytecode::EmbeddedCscPattern {
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
            symbolic_l: coker_bytecode::QdldlSymbolicL {
                l_pattern: coker_bytecode::EmbeddedCscPattern {
                    nrows: 2,
                    ncols: 2,
                    indptr: vec![0, 1, 1],
                    indices: vec![1],
                },
                etree: vec![1, u32::MAX],
                lnz: vec![1, 0],
            },
        },
    }
}

fn build_mapped_scalar_qp_module(
    _qp_function_id: u16,
    coefficient_function_id: u16,
    evaluator_output_len: u16,
    output_spec_length: u16,
    parameterized: bool,
) -> BytecodeModule {
    let evaluator_input_specs = if parameterized {
        vec![InputSpec {
            workspace_offset: 0,
            length: evaluator_output_len,
        }]
    } else {
        vec![]
    };
    let qp_input_specs = if parameterized {
        vec![InputSpec {
            workspace_offset: 0,
            length: evaluator_output_len,
        }]
    } else {
        vec![]
    };
    let entry_program = Program::new(0, 0, vec![], vec![], vec![]);
    let coefficient_program = Program::new(
        u32::from(evaluator_output_len),
        u32::from(evaluator_output_len),
        evaluator_input_specs,
        vec![OutputSpec {
            workspace_offset: 0,
            length: evaluator_output_len,
        }],
        vec![],
    );
    let qp_program = coker_bytecode::QpProgram::new(
        coefficient_function_id,
        u32::from(output_spec_length),
        u32::from(output_spec_length),
        qp_input_specs,
        OutputSpec {
            workspace_offset: 0,
            length: output_spec_length,
        },
        scalar_qp_pattern(),
        scalar_qp_pattern(),
        scalar_qp_coefficient_outputs(),
        scalar_qp_plan(),
    );
    BytecodeModule::with_qp_programs(vec![entry_program, coefficient_program], vec![qp_program])
}

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.with(Cell::get) {
            ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK_ALLOCATIONS.with(Cell::get) {
            ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.with(Cell::get) {
            ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
        }
        System.alloc_zeroed(layout)
    }
}

struct AllocationTracker;

impl AllocationTracker {
    fn start() -> Self {
        TRACK_ALLOCATIONS.with(|flag| flag.set(true));
        ALLOCATION_COUNT.with(|count| count.set(0));
        Self
    }

    fn count() -> usize {
        ALLOCATION_COUNT.with(Cell::get)
    }
}

impl Drop for AllocationTracker {
    fn drop(&mut self) {
        TRACK_ALLOCATIONS.with(|flag| flag.set(false));
    }
}

fn build_nested_module() -> BytecodeModule {
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
        vec![Layer::Generic(GenericLayer {
            in_offset: 0,
            out_offset: 1,
            in_length: 1,
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
                    first: 0,
                    second: UNUSED_OPERAND,
                    third: UNUSED_OPERAND,
                    op: ScalarOp::Sin,
                },
            ],
        })],
    );
    let entry_program = Program::new(
        1,
        4,
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

fn encode_into_aligned_bytes(module: &BytecodeModule) -> AlignedVec<16> {
    let encoded = encode_module(module).unwrap();
    let mut aligned = AlignedVec::with_capacity(encoded.len());
    aligned.extend_from_slice(&encoded);
    aligned
}

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
        indptr: vec![0, 1, 1],
        indices: vec![1],
    };
    qp_program.a_pattern = coker_bytecode::EmbeddedCscPattern {
        nrows: 1,
        ncols: 2,
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
    let module = build_mapped_scalar_qp_module(41, 17, 6, 1, true);
    let aligned_bytes = encode_into_aligned_bytes(&module);
    let mapped = MappedModule::new_from_bytes(aligned_bytes.as_slice()).unwrap();
    let qp_program = mapped.qp_program(41).unwrap();
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
        MappedQpWorkspace::new(&mut evaluator_workspace, &mut exact_coefficient_outputs),
        &mut outputs,
    ) {
        Ok(_) => panic!("expected mapped QP update failure"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        RuntimeError::EmbeddedQpAbi {
            operation: "update",
            ..
        }
    ));

    let diagnostics = bound
        .execute(
            &[&valid],
            MappedQpWorkspace::new(&mut evaluator_workspace, &mut oversized_coefficient_outputs),
            &mut outputs,
        )
        .unwrap();
    assert_eq!(diagnostics.status, QpSolveStatus::Solved);
    assert!((outputs[0] - 2.0).abs() < 1e-4);
    assert_eq!(&oversized_coefficient_outputs[6..], &[123.0, 123.0]);
}

#[cfg(osqp_embedded)]
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

#[cfg(osqp_embedded)]
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

#[cfg(osqp_embedded)]
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
