use super::*;
use coker_bytecode::{
    encode_module, BilinearLayer, BytecodeModule, EvaluateInputBinding, EvaluateLayer,
    EvaluateOutputBinding, GenericLayer, InputSpec, Layer, OutputSpec, Program,
    QpCoefficientOutputs, QpOutputSlice, RowOp, ScalarOp, SparseEntry, SparseTensor,
};
#[cfg(osqp_embedded)]
use coker_osqp_ffi::raw_embedded as ffi_raw;
#[cfg(osqp_embedded)]
use core::mem::{align_of, size_of};
use rkyv::util::AlignedVec;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    string::ToString,
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
        nnz: 1,
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
#[cfg(not(osqp_embedded))]
fn scalar_qp_region(byte_offset: u32) -> coker_bytecode::QpProgramArenaRegion {
    coker_bytecode::QpProgramArenaRegion {
        byte_offset,
        byte_len: 1,
        byte_alignment: 1,
    }
}

#[cfg(not(osqp_embedded))]
fn scalar_qp_arena_layout() -> coker_bytecode::QpProgramArenaLayout {
    coker_bytecode::QpProgramArenaLayout {
        total_bytes: 46,
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
        qdldl_l_p: scalar_qp_region(14),
        qdldl_l_i: scalar_qp_region(15),
        qdldl_l: scalar_qp_region(16),
        qdldl_kkt_x: scalar_qp_region(17),
        qdldl_kkt: scalar_qp_region(18),
        qdldl: scalar_qp_region(19),
        qdldl_dinv: scalar_qp_region(20),
        qdldl_bp: scalar_qp_region(21),
        qdldl_sol: scalar_qp_region(22),
        qdldl_rho_inv_vec: scalar_qp_region(23),
        qdldl_d: scalar_qp_region(24),
        qdldl_iwork: scalar_qp_region(25),
        qdldl_bwork: scalar_qp_region(26),
        qdldl_fwork: scalar_qp_region(27),
        work_rho_vec: scalar_qp_region(28),
        work_rho_inv_vec: scalar_qp_region(29),
        work_constr_type: scalar_qp_region(30),
        work_x: scalar_qp_region(31),
        work_y: scalar_qp_region(32),
        work_z: scalar_qp_region(33),
        work_xz_tilde: scalar_qp_region(34),
        work_x_prev: scalar_qp_region(35),
        work_z_prev: scalar_qp_region(36),
        work_ax: scalar_qp_region(37),
        work_px: scalar_qp_region(38),
        work_aty: scalar_qp_region(39),
        work_delta_y: scalar_qp_region(40),
        work_atdelta_y: scalar_qp_region(41),
        work_delta_x: scalar_qp_region(42),
        work_pdelta_x: scalar_qp_region(43),
        work_adelta_x: scalar_qp_region(44),
        workspace: scalar_qp_region(45),
    }
}

#[cfg(osqp_embedded)]
fn scalar_qp_push_region<T>(
    offset: &mut usize,
    arena_alignment: &mut usize,
    count: usize,
) -> coker_bytecode::QpProgramArenaRegion {
    let alignment = align_of::<T>();
    let byte_offset = (*offset + alignment - 1) & !(alignment - 1);
    let byte_len = count * size_of::<T>();
    *offset = byte_offset + byte_len;
    *arena_alignment = (*arena_alignment).max(alignment);
    coker_bytecode::QpProgramArenaRegion {
        byte_offset: byte_offset as u32,
        byte_len: byte_len as u32,
        byte_alignment: alignment as u32,
    }
}

#[cfg(osqp_embedded)]
fn scalar_qp_arena_layout() -> coker_bytecode::QpProgramArenaLayout {
    let mut offset = 0usize;
    let mut arena_alignment = 1usize;

    let pdata_x = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let pdata =
        scalar_qp_push_region::<ffi_raw::OSQPCscMatrix>(&mut offset, &mut arena_alignment, 1);
    let adata_x = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let adata =
        scalar_qp_push_region::<ffi_raw::OSQPCscMatrix>(&mut offset, &mut arena_alignment, 1);
    let qdata = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let ldata = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let udata = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let data = scalar_qp_push_region::<ffi_raw::OSQPData>(&mut offset, &mut arena_alignment, 1);
    let settings =
        scalar_qp_push_region::<ffi_raw::OSQPSettings>(&mut offset, &mut arena_alignment, 1);
    let xsolution =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let ysolution =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let solution =
        scalar_qp_push_region::<ffi_raw::OSQPSolution>(&mut offset, &mut arena_alignment, 1);
    let info = scalar_qp_push_region::<ffi_raw::OSQPInfo>(&mut offset, &mut arena_alignment, 1);
    let qdldl_l_x =
        scalar_qp_push_region::<ffi_raw::QDLDL_float>(&mut offset, &mut arena_alignment, 1);
    let qdldl_l_p =
        scalar_qp_push_region::<ffi_raw::QDLDL_int>(&mut offset, &mut arena_alignment, 3);
    let qdldl_l_i =
        scalar_qp_push_region::<ffi_raw::QDLDL_int>(&mut offset, &mut arena_alignment, 1);
    let qdldl_l =
        scalar_qp_push_region::<ffi_raw::OSQPCscMatrix>(&mut offset, &mut arena_alignment, 1);
    let qdldl_kkt_x =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 3);
    let qdldl_kkt =
        scalar_qp_push_region::<ffi_raw::OSQPCscMatrix>(&mut offset, &mut arena_alignment, 1);
    let qdldl =
        scalar_qp_push_region::<ffi_raw::qdldl_solver>(&mut offset, &mut arena_alignment, 1);
    let qdldl_dinv =
        scalar_qp_push_region::<ffi_raw::QDLDL_float>(&mut offset, &mut arena_alignment, 2);
    let qdldl_bp =
        scalar_qp_push_region::<ffi_raw::QDLDL_float>(&mut offset, &mut arena_alignment, 2);
    let qdldl_sol =
        scalar_qp_push_region::<ffi_raw::QDLDL_float>(&mut offset, &mut arena_alignment, 2);
    let qdldl_rho_inv_vec =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let qdldl_d =
        scalar_qp_push_region::<ffi_raw::QDLDL_float>(&mut offset, &mut arena_alignment, 2);
    let qdldl_iwork =
        scalar_qp_push_region::<ffi_raw::QDLDL_int>(&mut offset, &mut arena_alignment, 6);
    let qdldl_bwork =
        scalar_qp_push_region::<ffi_raw::QDLDL_bool>(&mut offset, &mut arena_alignment, 2);
    let qdldl_fwork =
        scalar_qp_push_region::<ffi_raw::QDLDL_float>(&mut offset, &mut arena_alignment, 2);
    let work_rho_vec =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_rho_inv_vec =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_constr_type =
        scalar_qp_push_region::<ffi_raw::OSQPInt>(&mut offset, &mut arena_alignment, 1);
    let work_x = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_y = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_z = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_xz_tilde =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 2);
    let work_x_prev =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_z_prev =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_ax = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_px = scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_aty =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_delta_y =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_atdelta_y =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_delta_x =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_pdelta_x =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let work_adelta_x =
        scalar_qp_push_region::<ffi_raw::OSQPFloat>(&mut offset, &mut arena_alignment, 1);
    let workspace =
        scalar_qp_push_region::<ffi_raw::OSQPWorkspace>(&mut offset, &mut arena_alignment, 1);
    let total_bytes = ((offset + arena_alignment - 1) & !(arena_alignment - 1)) as u32;

    coker_bytecode::QpProgramArenaLayout {
        total_bytes,
        arena_alignment: arena_alignment as u32,
        pdata_x,
        pdata,
        adata_x,
        adata,
        qdata,
        ldata,
        udata,
        data,
        settings,
        xsolution,
        ysolution,
        solution,
        info,
        qdldl_l_x,
        qdldl_l_p,
        qdldl_l_i,
        qdldl_l,
        qdldl_kkt_x,
        qdldl_kkt,
        qdldl,
        qdldl_dinv,
        qdldl_bp,
        qdldl_sol,
        qdldl_rho_inv_vec,
        qdldl_d,
        qdldl_iwork,
        qdldl_bwork,
        qdldl_fwork,
        work_rho_vec,
        work_rho_inv_vec,
        work_constr_type,
        work_x,
        work_y,
        work_z,
        work_xz_tilde,
        work_x_prev,
        work_z_prev,
        work_ax,
        work_px,
        work_aty,
        work_delta_y,
        work_atdelta_y,
        work_delta_x,
        work_pdelta_x,
        work_adelta_x,
        workspace,
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
        arena_layout: scalar_qp_arena_layout(),
        qdldl_plan: coker_bytecode::QpProgramQdldlPlan {
            p_pattern: scalar_qp_pattern(),
            a_pattern: scalar_qp_pattern(),
            kkt_pattern: coker_bytecode::EmbeddedCscPattern {
                nrows: 2,
                ncols: 2,
                nnz: 3,
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
                    nnz: 1,
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


include!("tests/execution.rs");
include!("tests/qp.rs");
