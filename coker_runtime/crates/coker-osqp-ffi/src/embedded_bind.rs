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
        let base =
            NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap_or_else(NonNull::dangling);
        Self {
            base,
            bytes: storage.len(),
            _borrowed: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
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

    pub fn region_value_mut<T>(&mut self, offset: usize, alignment: usize) -> Option<&mut T> {
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

pub fn bind_vectori(
    vector: &mut raw::OSQPVectori,
    values: *mut raw::OSQPInt,
    length: raw::OSQPInt,
) {
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

/// C-facing CSC metadata used to verify full OSQP matrix updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddedCscCounts {
    pub nzmax: raw::OSQPInt,
    pub terminal_indptr: raw::OSQPInt,
}

/// C-facing P and A counts read from the live embedded OSQP solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddedCscUpdateCounts {
    pub p: EmbeddedCscCounts,
    pub a: EmbeddedCscCounts,
}

impl EmbeddedCscUpdateCounts {
    pub fn is_consistent(self, px_len: usize, ax_len: usize) -> bool {
        self.p.nzmax >= 0
            && self.a.nzmax >= 0
            && self.p.terminal_indptr >= 0
            && self.a.terminal_indptr >= 0
            && self.p.nzmax == self.p.terminal_indptr
            && self.a.nzmax == self.a.terminal_indptr
            && usize::try_from(self.p.nzmax).ok() == Some(px_len)
            && usize::try_from(self.a.nzmax).ok() == Some(ax_len)
    }
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

fn csc_counts(matrix: &raw::OSQPCscMatrix) -> Option<EmbeddedCscCounts> {
    let n = usize::try_from(matrix.n).ok()?;
    if matrix.nzmax < 0 || matrix.p.is_null() || matrix.i.is_null() || matrix.x.is_null() {
        return None;
    }
    let terminal_indptr = unsafe { *matrix.p.add(n) };
    Some(EmbeddedCscCounts {
        nzmax: matrix.nzmax,
        terminal_indptr,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct EmbeddedSolver {
    solver: NonNull<raw::OSQPSolver>,
}

impl EmbeddedSolver {
    /// # Safety
    ///
    /// `solver` must be a valid, non-null pointer to a live `OSQPSolver`
    /// whose pointees remain valid for the lifetime of the returned wrapper.
    pub unsafe fn from_ptr(solver: *mut raw::OSQPSolver) -> Option<Self> {
        Some(Self {
            solver: NonNull::new(solver)?,
        })
    }

    pub fn as_ptr(&self) -> *mut raw::OSQPSolver {
        self.solver.as_ptr()
    }

    /// # Safety
    ///
    /// `self` must wrap a valid live `OSQPSolver` with initialized `work`
    /// and `data` pointers.
    pub unsafe fn dimensions(&self) -> Option<(usize, usize)> {
        let work = self.solver.as_ref().work.as_ref()?;
        let data = work.data.as_ref()?;
        let m = usize::try_from(data.m).ok()?;
        let n = usize::try_from(data.n).ok()?;
        Some((n, m))
    }

    /// # Safety
    ///
    /// `self` must wrap a valid live `OSQPSolver` prepared for `osqp_solve`.
    pub unsafe fn solve(&mut self) -> raw::OSQPInt {
        raw::osqp_solve(self.as_ptr())
    }

    /// # Safety
    ///
    /// `self` must wrap a valid live `OSQPSolver`. Any provided warm-start
    /// slices must match the solver dimensions.
    pub unsafe fn warm_start(
        &mut self,
        primal: Option<&[raw::OSQPFloat]>,
        dual: Option<&[raw::OSQPFloat]>,
    ) -> Option<raw::OSQPInt> {
        let (n, m) = self.dimensions()?;
        if primal.map_or(false, |values| values.len() != n)
            || dual.map_or(false, |values| values.len() != m)
        {
            return None;
        }
        Some(raw::osqp_warm_start(
            self.as_ptr(),
            primal.map_or(core::ptr::null(), |values| values.as_ptr()),
            dual.map_or(core::ptr::null(), |values| values.as_ptr()),
        ))
    }

    /// # Safety
    ///
    /// `self` must wrap a valid live `OSQPSolver`. The provided slices must
    /// match the solver dimensions.
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

    /// Returns the live C-facing CSC metadata used by OSQP's full updates.
    ///
    /// # Safety
    ///
    /// `self` must wrap a valid live `OSQPSolver` with initialized data and
    /// matrix descriptors.
    pub unsafe fn matrix_update_counts(&self) -> Option<EmbeddedCscUpdateCounts> {
        let work = self.solver.as_ref().work.as_ref()?;
        let data = work.data.as_ref()?;
        Some(EmbeddedCscUpdateCounts {
            p: csc_counts(data.P.as_ref()?.csc.as_ref()?)?,
            a: csc_counts(data.A.as_ref()?.csc.as_ref()?)?,
        })
    }

    /// # Safety
    ///
    /// `self` must wrap a valid live `OSQPSolver`. The supplied full matrices
    /// must match the live CSC descriptors exactly.
    pub unsafe fn update_data_mat(
        &mut self,
        px: &[raw::OSQPFloat],
        ax: &[raw::OSQPFloat],
    ) -> Option<raw::OSQPInt> {
        let counts = self.matrix_update_counts()?;
        if !counts.is_consistent(px.len(), ax.len()) {
            return None;
        }
        Some(raw::osqp_update_data_mat(
            self.as_ptr(),
            px.as_ptr(),
            core::ptr::null(),
            counts.p.terminal_indptr,
            ax.as_ptr(),
            core::ptr::null(),
            counts.a.terminal_indptr,
        ))
    }

    /// # Safety
    ///
    /// `self` must wrap a valid live `OSQPSolver` whose solution and info
    /// pointers are initialized.
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
