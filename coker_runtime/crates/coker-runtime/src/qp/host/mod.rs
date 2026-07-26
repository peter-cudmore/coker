#![cfg(all(feature = "std", not(osqp_embedded)))]

mod layout;
mod runtime;

pub use self::layout::QpRuntime;
pub use self::layout::{QpWorkspaceLayout, QpWorkspaceRegion};

#[allow(unused_imports)]
use self::layout::*;
#[allow(unused_imports)]
use self::runtime::*;

use super::*;

#[cfg(all(feature = "std", not(osqp_embedded)))]
impl<'a> MappedQpProgram<'a> {
    /// Binds this host-only QP convenience runtime to caller-provided arena storage.
    ///
    /// This path copies archived sparsity indices into host vectors and lets the
    /// OSQP C workspace keep its own setup allocation. It is not the embedded
    /// caller-owned runtime contract.
    pub fn bind_host<'arena>(
        &self,
        arena: &'arena mut [MaybeUninit<u8>],
    ) -> Result<BoundMappedQpProgram<'a, 'arena>, RuntimeError> {
        let workspace = host_workspace_from_arena(arena, self.workspace_requirements)?;
        let runtime = QpRuntime::new(*self, workspace)?;
        Ok(BoundMappedQpProgram {
            program: *self,
            runtime,
        })
    }
}
