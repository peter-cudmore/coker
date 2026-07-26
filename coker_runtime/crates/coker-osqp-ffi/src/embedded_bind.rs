use core::{
    convert::TryFrom,
    marker::PhantomData,
    mem::{align_of, size_of, MaybeUninit},
    ptr::NonNull,
};

use crate::raw_embedded as raw;

#[derive(Debug)]
pub struct EmbeddedArena<'a> {
    base: NonNull<u8>,
    bytes: usize,
    _borrowed: PhantomData<&'a mut [MaybeUninit<u8>]>,
}

impl<'a> EmbeddedArena<'a> {
    pub fn new(storage: &'a mut [MaybeUninit<u8>]) -> Self {
        let base = NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap_or_else(NonNull::dangling);
        Self {
            base,
            bytes: storage.len(),
            _borrowed: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn zero_region(&mut self, offset: usize, len: usize) -> bool {
        match self.region_ptr::<u8>(offset, len, 1) {
            Some(ptr) => {
                unsafe { ptr.as_ptr().write_bytes(0, len) };
                true
            }
            None => false,
        }
    }

    pub fn region_ptr<T>(
        &self,
        offset: usize,
        bytes: usize,
        alignment: usize,
    ) -> Option<NonNull<T>> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return None;
        }
        let end = offset.checked_add(bytes)?;
        if end > self.bytes {
            return None;
        }
        let ptr = unsafe { self.base.as_ptr().add(offset) };
        if (ptr as usize) % align_of::<T>() != 0 {
            return None;
        }
        NonNull::new(ptr.cast::<T>())
    }

    pub fn region_value_mut<T>(
        &mut self,
        offset: usize,
        alignment: usize,
    ) -> Option<&mut T> {
        let ptr = self.region_ptr::<T>(offset, size_of::<T>(), alignment)?;
        Some(unsafe { ptr.as_ptr().as_mut()? })
    }
}

pub fn bind_csc_matrix(
    matrix: &mut raw::OSQPCscMatrix,
    rows: raw::OSQPInt,
    cols: raw::OSQPInt,
    nnz: raw::OSQPInt,
    col_ptr: *mut raw::OSQPInt,
    row_idx: *mut raw::OSQPInt,
    values: *mut raw::OSQPFloat,
) {
    matrix.m = rows;
    matrix.n = cols;
    matrix.p = col_ptr;
    matrix.i = row_idx;
    matrix.x = values;
    matrix.nzmax = nnz;
    matrix.nz = -1;
    matrix.owned = 0;
}

pub fn bind_vectorf(
    vector: &mut raw::OSQPVectorf,
    values: *mut raw::OSQPFloat,
    length: raw::OSQPInt,
) {
    vector.values = values;
    vector.length = length;
}

pub fn bind_vectori(vector: &mut raw::OSQPVectori, values: *mut raw::OSQPInt, length: raw::OSQPInt) {
    vector.values = values;
    vector.length = length;
}

pub fn bind_matrix(
    matrix: &mut raw::OSQPMatrix,
    csc: *mut raw::OSQPCscMatrix,
    symmetry: raw::OSQPMatrix_symmetry_type,
) {
    matrix.csc = csc;
    matrix.symmetry = symmetry;
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddedSolutionView<'a> {
    pub primal: &'a [raw::OSQPFloat],
    pub dual: &'a [raw::OSQPFloat],
    pub status: raw::OSQPInt,
    pub iterations: raw::OSQPInt,
    pub primal_residual: raw::OSQPFloat,
    pub dual_residual: raw::OSQPFloat,
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddedSolver {
    solver: NonNull<raw::OSQPSolver>,
}

impl EmbeddedSolver {
    pub unsafe fn from_ptr(solver: *mut raw::OSQPSolver) -> Option<Self> {
        Some(Self {
            solver: NonNull::new(solver)?,
        })
    }

    pub fn as_ptr(&self) -> *mut raw::OSQPSolver {
        self.solver.as_ptr()
    }

    pub unsafe fn dimensions(&self) -> Option<(usize, usize)> {
        let work = self.solver.as_ref().work.as_ref()?;
        let data = work.data.as_ref()?;
        let m = usize::try_from(data.m).ok()?;
        let n = usize::try_from(data.n).ok()?;
        Some((n, m))
    }

    pub unsafe fn solve(&mut self) -> raw::OSQPInt {
        raw::osqp_solve(self.as_ptr())
    }

    pub unsafe fn warm_start(
        &mut self,
        primal: Option<&[raw::OSQPFloat]>,
        dual: Option<&[raw::OSQPFloat]>,
    ) -> Option<raw::OSQPInt> {
        let (n, m) = self.dimensions()?;
        if primal.is_some_and(|values| values.len() != n)
            || dual.is_some_and(|values| values.len() != m)
        {
            return None;
        }
        Some(raw::osqp_warm_start(
            self.as_ptr(),
            primal.map_or(core::ptr::null(), |values| values.as_ptr()),
            dual.map_or(core::ptr::null(), |values| values.as_ptr()),
        ))
    }

    pub unsafe fn update_data_vec(
        &mut self,
        q: &[raw::OSQPFloat],
        l: &[raw::OSQPFloat],
        u: &[raw::OSQPFloat],
    ) -> Option<raw::OSQPInt> {
        let (n, m) = self.dimensions()?;
        if q.len() != n || l.len() != m || u.len() != m {
            return None;
        }
        Some(raw::osqp_update_data_vec(
            self.as_ptr(),
            q.as_ptr(),
            l.as_ptr(),
            u.as_ptr(),
        ))
    }

    pub unsafe fn update_data_mat(
        &mut self,
        px: &[raw::OSQPFloat],
        ax: &[raw::OSQPFloat],
    ) -> Option<raw::OSQPInt> {
        let work = self.solver.as_ref().work.as_ref()?;
        let data = work.data.as_ref()?;
        let p = data.P.as_ref()?.csc.as_ref()?;
        let a = data.A.as_ref()?.csc.as_ref()?;
        let p_nnz = usize::try_from(p.nzmax).ok()?;
        let a_nnz = usize::try_from(a.nzmax).ok()?;
        if px.len() != p_nnz || ax.len() != a_nnz {
            return None;
        }
        Some(raw::osqp_update_data_mat(
            self.as_ptr(),
            px.as_ptr(),
            core::ptr::null(),
            p.nzmax,
            ax.as_ptr(),
            core::ptr::null(),
            a.nzmax,
        ))
    }

    pub unsafe fn solution<'a>(&'a self) -> Option<EmbeddedSolutionView<'a>> {
        let solver = self.solver.as_ref();
        let info = solver.info.as_ref()?;
        let solution = solver.solution.as_ref()?;
        let (n, m) = self.dimensions()?;
        if (n != 0 && solution.x.is_null()) || (m != 0 && solution.y.is_null()) {
            return None;
        }
        let primal = core::slice::from_raw_parts(solution.x, n);
        let dual = core::slice::from_raw_parts(solution.y, m);
        Some(EmbeddedSolutionView {
            primal,
            dual,
            status: info.status_val,
            iterations: info.iter,
            primal_residual: info.prim_res,
            dual_residual: info.dual_res,
        })
    }
}
