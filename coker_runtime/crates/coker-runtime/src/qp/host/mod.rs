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
    /// Binds this host QP program to caller-provided arena storage.
    pub fn bind<'arena>(
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
