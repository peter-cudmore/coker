use crate::*;

#[test]
fn embedded_shape_ffi_uses_i32_structure_and_f32_updates() {
    let col_ptr = [0_i32, 0_i32];
    let shape = CokerOsqpProblemShape {
        n: 1,
        m: 0,
        p: CokerOsqpCscView {
            col_ptr: col_ptr.as_ptr(),
            row_idx: core::ptr::null(),
            nnz: 0,
        },
        a: CokerOsqpCscView {
            col_ptr: col_ptr.as_ptr(),
            row_idx: core::ptr::null(),
            nnz: 0,
        },
    };
    let update = CokerOsqpNumericUpdate {
        p_x: core::ptr::null(),
        p_nnz: 0,
        a_x: core::ptr::null(),
        a_nnz: 0,
        q: core::ptr::null(),
        q_len: 0,
        l: core::ptr::null(),
        l_len: 0,
        u: core::ptr::null(),
        u_len: 0,
    };

    assert_eq!(shape.n, 1);
    assert_eq!(shape.p.nnz, 0);
    assert_eq!(update.q_len, 0);
    assert_eq!(COKER_OSQP_UNSUPPORTED, -6);
    assert_eq!(COKER_OSQP_SOLVE_UNSOLVED, 0);
}

#[test]
fn generated_raw_bindings_expose_osqp_1_solver_split() {
    use crate::raw_embedded as raw;
    use core::mem::{offset_of, size_of};

    assert!(size_of::<raw::OSQPSolver>() > 0);
    assert!(size_of::<raw::OSQPWorkspace>() > 0);
    assert!(size_of::<raw::OSQPData>() > 0);
    assert!(size_of::<raw::OSQPSettings>() > 0);
    assert_eq!(offset_of!(raw::OSQPSolver, settings), 0);
    assert_eq!(
        offset_of!(raw::OSQPSolver, solution),
        size_of::<*mut raw::OSQPSettings>()
    );
    assert_eq!(
        offset_of!(raw::OSQPSolver, info),
        size_of::<*mut raw::OSQPSettings>() + size_of::<*mut raw::OSQPSolution>()
    );
    assert_eq!(
        offset_of!(raw::OSQPSolver, work),
        size_of::<*mut raw::OSQPSettings>()
            + size_of::<*mut raw::OSQPSolution>()
            + size_of::<*mut raw::OSQPInfo>()
    );
    assert_eq!(raw::OSQP_EMBEDDED_MODE, 2);
}
