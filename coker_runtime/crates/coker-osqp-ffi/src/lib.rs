#![no_std]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/// Integer type selected by the configured OSQP ABI.
#[cfg(osqp_dlong)]
pub type osqp_int = core::ffi::c_longlong;
#[cfg(not(osqp_dlong))]
pub type osqp_int = core::ffi::c_int;
/// Floating-point type selected by the configured OSQP ABI.
#[cfg(osqp_embedded)]
pub type osqp_float = f32;
#[cfg(not(osqp_embedded))]
pub type osqp_float = f64;

/// C integer alias used by generated OSQP bindings.
pub type c_int = osqp_int;
/// C floating-point alias used by generated OSQP bindings.
pub type c_float = osqp_float;

/// Opaque timer handle used by the upstream OSQP ABI.
pub enum OSQPTimer {}

#[cfg(test)]
extern crate std;

#[cfg(osqp_embedded)]
mod bindings_embedded;
#[cfg(osqp_embedded)]
pub mod raw_embedded;
#[cfg(osqp_embedded)]
pub mod embedded_bind;
#[cfg(not(osqp_embedded))]
mod bindings;

#[cfg(osqp_embedded)]
pub use bindings_embedded::*;
#[cfg(not(osqp_embedded))]
pub use bindings::*;
mod generated_solver_contract;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;
    use std::ptr;
    #[test]
    fn abi_types_match_c_config() {
        #[cfg(osqp_embedded)]
        assert_eq!(mem::size_of::<c_float>(), mem::size_of::<f32>());

        #[cfg(not(osqp_embedded))]
        assert_eq!(mem::size_of::<c_float>(), mem::size_of::<f64>());

        #[cfg(osqp_dlong)]
        assert_eq!(mem::size_of::<c_int>(), mem::size_of::<core::ffi::c_longlong>());

        #[cfg(not(osqp_dlong))]
        assert_eq!(mem::size_of::<c_int>(), mem::size_of::<core::ffi::c_int>());
    }

    #[test]
    fn generated_solver_contract_matches_codegen_layout() {
        assert_eq!(
            generated_solver_contract::GENERATED_SOLVER_ENV,
            "COKER_OSQP_GENERATED_SOLVER_DIR"
        );
        assert_eq!(generated_solver_contract::GENERATED_SOLVER_INCLUDE_DIR, "include");
        assert_eq!(generated_solver_contract::GENERATED_SOLVER_CONFIGURE_DIR, "configure");
        assert_eq!(generated_solver_contract::GENERATED_SOLVER_SOURCE_DIR, "src/osqp");
        assert_eq!(generated_solver_contract::GENERATED_SOLVER_WORKSPACE_HEADER, "workspace.h");
        assert_eq!(
            generated_solver_contract::GENERATED_SOLVER_WORKSPACE_SOURCE,
            "src/osqp/workspace.c"
        );
        assert_eq!(generated_solver_contract::GENERATED_SOLVER_TYPES_HEADER, "types.h");
        assert_eq!(
            generated_solver_contract::GENERATED_SOLVER_QDLDL_INTERFACE_HEADER,
            "qdldl_interface.h"
        );
        assert_eq!(
            generated_solver_contract::GENERATED_SOLVER_CONFIGURE_HEADER,
            "osqp_configure.h"
        );
    }

    #[cfg(not(osqp_embedded))]
    extern "C" {
        fn free(ptr: *mut u8);
        fn csc_matrix(
            m: c_int,
            n: c_int,
            nzmax: c_int,
            x: *mut c_float,
            i: *mut c_int,
            p: *mut c_int,
        ) -> *mut csc;
    }

    // examples/osqp_demo.c converted into rust
    #[cfg(not(osqp_embedded))]
    #[test]
    fn osqp_demo_rust() {
        unsafe {
            osqp_demo_rust_unsafe();
        }
    }

    #[cfg(not(osqp_embedded))]
    unsafe fn osqp_demo_rust_unsafe() {
        // `csc_matrix` borrows these buffers; keep each backing array alive until after cleanup.
        let mut p_x = [4.0 as c_float, 1.0, 2.0];
        let mut p_i = [0 as c_int, 0, 1];
        let mut p_p = [0 as c_int, 1, 3];
        let mut a_x = [1.0 as c_float, 1.0, 1.0, 1.0];
        let mut a_i = [0 as c_int, 1, 0, 2];
        let mut a_p = [0 as c_int, 2, 4];
        let mut q = [1.0 as c_float, 1.0];
        let mut l = [1.0 as c_float, 0.0, 0.0];
        let mut u = [1.0 as c_float, 0.7, 0.7];
        let data = OSQPData {
            n: 2,
            m: 3,
            P: csc_matrix(
                2,
                2,
                p_x.len() as c_int,
                p_x.as_mut_ptr(),
                p_i.as_mut_ptr(),
                p_p.as_mut_ptr(),
            ),
            A: csc_matrix(
                3,
                2,
                a_x.len() as c_int,
                a_x.as_mut_ptr(),
                a_i.as_mut_ptr(),
                a_p.as_mut_ptr(),
            ),
            q: q.as_mut_ptr(),
            l: l.as_mut_ptr(),
            u: u.as_mut_ptr(),
        };

        let mut settings = mem::zeroed::<OSQPSettings>();
        osqp_set_default_settings(&mut settings);
        settings.verbose = 0;
        settings.warm_start = 1;

        let mut work: *mut OSQPWorkspace = ptr::null_mut();
        let exitflag = osqp_setup(&mut work, &data, &settings);
        assert_eq!(exitflag, 0);
        assert!(!work.is_null(), "successful setup must initialize a workspace");
        assert_eq!(osqp_solve(work), 0);
        assert_eq!((*work).info.as_ref().unwrap().status_val, OSQP_SOLVED as c_int);
        let solution = (*work)
            .solution
            .as_ref()
            .expect("solved workspace must contain a solution");
        assert!(!solution.x.is_null(), "solved workspace must contain primal values");
        let x = core::slice::from_raw_parts(solution.x, data.n as usize);
        let solution_tolerance = 2.0e-3 as c_float;
        assert!(
            (x[0] - 0.3).abs() <= solution_tolerance,
            "unexpected x[0]: {}",
            x[0]
        );
        assert!(
            (x[1] - 0.7).abs() <= solution_tolerance,
            "unexpected x[1]: {}",
            x[1]
        );
        assert_eq!(osqp_cleanup(work), 0);
        free(data.A.cast());
        free(data.P.cast());
    }

    #[cfg(osqp_embedded)]
    mod embedded_phase1;
}
