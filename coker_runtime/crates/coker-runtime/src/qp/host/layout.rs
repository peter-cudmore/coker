use super::*;
#[cfg(all(feature = "std", not(osqp_embedded)))]
use alloc::vec::Vec;

#[cfg(all(feature = "std", not(osqp_embedded)))]
/// Byte range for one region inside a packed host QP workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QpWorkspaceRegion {
    pub start: usize,
    pub len: usize,
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
/// Layout of the packed host-side QP workspace expected by [`QpRuntime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QpWorkspaceLayout {
    pub evaluator_workspace: QpWorkspaceRegion,
    pub coefficient_outputs: QpWorkspaceRegion,
    pub p_x: QpWorkspaceRegion,
    pub a_x: QpWorkspaceRegion,
    pub q: QpWorkspaceRegion,
    pub l: QpWorkspaceRegion,
    pub u: QpWorkspaceRegion,
    pub primal_warm_start: QpWorkspaceRegion,
    pub dual_warm_start: QpWorkspaceRegion,
    total_bytes: usize,
    required_f64_capacity: usize,
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
/// Caller-owned storage for QP evaluator, coefficient, and numeric solver inputs.
///
/// This does not contain OSQP's internal setup allocation. The current OSQP FFI
/// owns that allocation during `QpRuntime` construction.
#[derive(Debug)]
pub(super) struct QpWorkspace<'a> {
    ptr: NonNull<f64>,
    len: usize,
    layout: QpWorkspaceLayout,
    _marker: PhantomData<&'a mut [f64]>,
}

/// Reusable host-side QP runtime that owns OSQP setup state and borrows caller workspace.
#[cfg(all(feature = "std", not(osqp_embedded)))]
pub struct QpRuntime<'module, 'workspace> {
    pub(super) program: MappedQpProgram<'module>,
    pub(super) p_indptr: Vec<ffi::c_int>,
    pub(super) p_indices: Vec<ffi::c_int>,
    pub(super) a_indptr: Vec<ffi::c_int>,
    pub(super) a_indices: Vec<ffi::c_int>,
    pub(super) workspace: QpWorkspace<'workspace>,
    pub(super) problem: NonNull<ffi::OSQPWorkspace>,
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
impl<'module, 'workspace> core::fmt::Debug for QpRuntime<'module, 'workspace> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QpRuntime")
            .field("function_id", &self.program.function_id())
            .field("n", &self.program.n)
            .field("m", &self.program.m)
            .field("p_indptr", &self.p_indptr)
            .field("p_indices", &self.p_indices)
            .field("a_indptr", &self.a_indptr)
            .field("a_indices", &self.a_indices)
            .field("workspace", &self.workspace)
            .finish()
    }
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
pub(super) fn csc_matrix<'a>(
    nrows: ffi::c_int,
    ncols: ffi::c_int,
    indptr: &'a [ffi::c_int],
    indices: &'a [ffi::c_int],
    data: &'a [f64],
) -> Result<ffi::csc, RuntimeError> {
    Ok(ffi::csc {
        nzmax: checked_host_ffi_length(data.len(), "QP CSC nnz")?,
        m: nrows,
        n: ncols,
        p: indptr.as_ptr() as *mut ffi::c_int,
        i: indices.as_ptr() as *mut ffi::c_int,
        x: data.as_ptr() as *mut ffi::c_float,
        nz: -1,
    })
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
pub(super) fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
pub(super) fn checked_add_region(
    start: usize,
    len: usize,
) -> Result<QpWorkspaceRegion, RuntimeError> {
    let end = start
        .checked_add(len)
        .ok_or(RuntimeError::Validation("QP workspace layout overflow"))?;
    Ok(QpWorkspaceRegion {
        start,
        len: end - start,
    })
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
impl QpWorkspaceLayout {
    pub(crate) fn from_validated_parts(
        evaluator_workspace: usize,
        coefficient_output_len: usize,
        p_nnz: usize,
        a_nnz: usize,
        n: usize,
        m: usize,
    ) -> Result<Self, RuntimeError> {
        let mut offset = 0usize;
        let evaluator_workspace = checked_add_region(
            offset,
            evaluator_workspace
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| {
                    RuntimeError::Validation("QP workspace layout overflow")
                })?,
        )?;
        offset = evaluator_workspace.start + evaluator_workspace.len;

        let coefficient_outputs = checked_add_region(
            offset,
            coefficient_output_len
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| {
                    RuntimeError::Validation("QP workspace layout overflow")
                })?,
        )?;
        offset = coefficient_outputs.start + coefficient_outputs.len;

        offset = align_up(offset, align_of::<f64>());
        let p_x = checked_add_region(
            offset,
            p_nnz.checked_mul(size_of::<f64>()).ok_or_else(|| {
                RuntimeError::Validation("QP workspace layout overflow")
            })?,
        )?;
        offset = p_x.start + p_x.len;

        let a_x = checked_add_region(
            offset,
            a_nnz.checked_mul(size_of::<f64>()).ok_or_else(|| {
                RuntimeError::Validation("QP workspace layout overflow")
            })?,
        )?;
        offset = a_x.start + a_x.len;

        let q = checked_add_region(
            offset,
            n.checked_mul(size_of::<f64>()).ok_or_else(|| {
                RuntimeError::Validation("QP workspace layout overflow")
            })?,
        )?;
        offset = q.start + q.len;

        let l = checked_add_region(
            offset,
            m.checked_mul(size_of::<f64>()).ok_or_else(|| {
                RuntimeError::Validation("QP workspace layout overflow")
            })?,
        )?;
        offset = l.start + l.len;

        let u = checked_add_region(
            offset,
            m.checked_mul(size_of::<f64>()).ok_or_else(|| {
                RuntimeError::Validation("QP workspace layout overflow")
            })?,
        )?;
        offset = u.start + u.len;

        let primal_warm_start = checked_add_region(
            offset,
            n.checked_mul(size_of::<f64>()).ok_or_else(|| {
                RuntimeError::Validation("QP workspace layout overflow")
            })?,
        )?;
        offset = primal_warm_start.start + primal_warm_start.len;

        let dual_warm_start = checked_add_region(
            offset,
            m.checked_mul(size_of::<f64>()).ok_or_else(|| {
                RuntimeError::Validation("QP workspace layout overflow")
            })?,
        )?;
        offset = dual_warm_start.start + dual_warm_start.len;

        let total_bytes = offset;
        let required_f64_capacity = total_bytes
            .checked_add(size_of::<f64>() - 1)
            .ok_or(RuntimeError::Validation("QP workspace layout overflow"))?
            / size_of::<f64>();

        Ok(Self {
            evaluator_workspace,
            coefficient_outputs,
            p_x,
            a_x,
            q,
            l,
            u,
            primal_warm_start,
            dual_warm_start,
            total_bytes,
            required_f64_capacity,
        })
    }

    /// Returns the minimum `f64` slice length required to hold this layout.
    pub fn required_f64_capacity(&self) -> usize {
        self.required_f64_capacity
    }

    /// Returns the total workspace footprint in bytes, including alignment padding.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

/// Mutable typed slices projected from a packed host QP workspace buffer.
#[cfg(all(feature = "std", not(osqp_embedded)))]
pub struct QpWorkspaceView<'a> {
    pub evaluator_workspace: &'a mut [f32],
    pub coefficient_outputs: &'a mut [f32],
    pub p_x: &'a mut [f64],
    pub a_x: &'a mut [f64],
    pub q: &'a mut [f64],
    pub l: &'a mut [f64],
    pub u: &'a mut [f64],
    pub primal_warm_start: &'a mut [f64],
    pub dual_warm_start: &'a mut [f64],
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
impl<'a> QpWorkspace<'a> {
    pub(super) fn borrowed(
        workspace: &'a mut [f64],
        layout: QpWorkspaceLayout,
    ) -> Result<Self, RuntimeError> {
        if workspace.len() < layout.required_f64_capacity() {
            return Err(RuntimeError::WorkspaceTooSmall {
                expected: layout.required_f64_capacity(),
                actual: workspace.len(),
            });
        }
        Ok(Self {
            ptr: NonNull::new(workspace.as_mut_ptr()).unwrap_or_else(NonNull::dangling),
            len: workspace.len(),
            layout,
            _marker: PhantomData,
        })
    }

    pub(super) fn layout(&self) -> QpWorkspaceLayout {
        self.layout
    }

    pub(super) fn with_view<T>(&mut self, f: impl FnOnce(QpWorkspaceView<'_>) -> T) -> T {
        let bytes_len = self.len * size_of::<f64>();
        let bytes = self.ptr.as_ptr() as *mut u8;
        let layout = self.layout;

        unsafe {
            let evaluator_workspace = f32_slice_mut(bytes, bytes_len, layout.evaluator_workspace);
            let coefficient_outputs = f32_slice_mut(bytes, bytes_len, layout.coefficient_outputs);
            let p_x = f64_slice_mut(bytes, bytes_len, layout.p_x);
            let a_x = f64_slice_mut(bytes, bytes_len, layout.a_x);
            let q = f64_slice_mut(bytes, bytes_len, layout.q);
            let l = f64_slice_mut(bytes, bytes_len, layout.l);
            let u = f64_slice_mut(bytes, bytes_len, layout.u);
            let primal_warm_start = f64_slice_mut(bytes, bytes_len, layout.primal_warm_start);
            let dual_warm_start = f64_slice_mut(bytes, bytes_len, layout.dual_warm_start);
            f(QpWorkspaceView {
                evaluator_workspace,
                coefficient_outputs,
                p_x,
                a_x,
                q,
                l,
                u,
                primal_warm_start,
                dual_warm_start,
            })
        }
    }
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
pub(super) unsafe fn f32_slice_mut<'a>(
    bytes: *mut u8,
    bytes_len: usize,
    region: QpWorkspaceRegion,
) -> &'a mut [f32] {
    debug_assert_eq!(region.start % align_of::<f32>(), 0);
    debug_assert_eq!(region.len % size_of::<f32>(), 0);
    debug_assert!(region.start + region.len <= bytes_len);
    slice::from_raw_parts_mut(
        bytes.add(region.start) as *mut f32,
        region.len / size_of::<f32>(),
    )
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
pub(super) unsafe fn f64_slice_mut<'a>(
    bytes: *mut u8,
    bytes_len: usize,
    region: QpWorkspaceRegion,
) -> &'a mut [f64] {
    debug_assert_eq!(region.start % align_of::<f64>(), 0);
    debug_assert_eq!(region.len % size_of::<f64>(), 0);
    debug_assert!(region.start + region.len <= bytes_len);
    slice::from_raw_parts_mut(
        bytes.add(region.start) as *mut f64,
        region.len / size_of::<f64>(),
    )
}
#[cfg(all(feature = "std", not(osqp_embedded)))]
pub(super) fn host_workspace_from_arena(
    arena: &mut [MaybeUninit<u8>],
    requirements: QpWorkspaceRequirements,
) -> Result<&mut [f64], RuntimeError> {
    let base = arena.as_mut_ptr().cast::<u8>() as usize;
    let alignment = align_of::<f64>();
    if arena.len() < requirements.arena_bytes
        || requirements.arena_alignment < alignment
        || !requirements.arena_alignment.is_power_of_two()
        || !base.is_multiple_of(requirements.arena_alignment)
        || !base.is_multiple_of(alignment)
    {
        return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
    }
    Ok(unsafe {
        slice::from_raw_parts_mut(
            arena.as_mut_ptr().cast::<f64>(),
            arena.len() / size_of::<f64>(),
        )
    })
}
