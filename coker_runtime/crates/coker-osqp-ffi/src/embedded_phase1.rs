#[cfg(test)]
mod tests {
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
}
