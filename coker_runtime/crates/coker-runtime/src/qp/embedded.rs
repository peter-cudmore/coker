#[cfg(osqp_embedded)]
use super::*;

#[cfg(osqp_embedded)]
impl<'a> MappedQpProgram<'a> {
    fn bind_instance(
        &self,
        arena: &BoundQpArena<'_>,
    ) -> Result<ffi::CokerOsqpInstance, RuntimeError> {
        let plan = ffi_plan_from_program(self.qp_program)?;
        let mut instance = MaybeUninit::<ffi::CokerOsqpInstance>::zeroed();
        let bind_status = unsafe {
            ffi::coker_osqp_bind_plan(
                &plan,
                arena.as_ffi(self.workspace_requirements),
                instance.as_mut_ptr(),
            )
        };
        if bind_status != ffi::COKER_OSQP_OK {
            return Err(RuntimeError::EmbeddedQpAbi {
                operation: "bind",
                status: bind_status,
            });
        }
        Ok(unsafe { instance.assume_init() })
    }

    /// Binds this embedded QP program to a caller-provided aligned OSQP arena.
    pub fn bind<'arena>(
        &self,
        arena: &'arena mut [MaybeUninit<u8>],
    ) -> Result<BoundMappedQpProgram<'a, 'arena>, RuntimeError> {
        let arena = BoundQpArena::new(arena, self.workspace_requirements)?;
        let instance = self.bind_instance(&arena)?;
        Ok(BoundMappedQpProgram {
            program: *self,
            arena,
            instance: Some(instance),
        })
    }
}

#[cfg(osqp_embedded)]
impl<'module, 'arena> BoundMappedQpProgram<'module, 'arena> {
    fn invalid_instance_error() -> RuntimeError {
        RuntimeError::Validation(
            "embedded mapped QP instance is invalid after a prior update failure".to_string(),
        )
    }

    pub fn execute(
        &mut self,
        parameters: &[&[f32]],
        workspace: MappedQpWorkspace<'_>,
        outputs: &mut [f32],
    ) -> Result<QpSolveDiagnostics, RuntimeError> {
        self.program.validate_parameters(parameters)?;
        workspace.validate_for(self.program.workspace_requirements)?;
        self.program.validate_output_buffer(outputs)?;

        let coefficient_output_size = self.program.workspace_requirements.coefficient_output_size;
        let MappedQpWorkspace {
            evaluator_workspace,
            coefficient_outputs,
        } = workspace;
        let coefficient_outputs = &mut coefficient_outputs[..coefficient_output_size];
        self.program
            .evaluator
            .execute(parameters, evaluator_workspace, coefficient_outputs)?;

        let arena_layout = self.program.qp_program.embedded_plan().arena_layout();
        let base = self.arena.base.as_ptr();
        let bytes = self.arena.bytes;
        let p_x = unsafe { arena_region_slice_mut::<f32>(base, bytes, arena_layout.pdata_x())? };
        let a_x = unsafe { arena_region_slice_mut::<f32>(base, bytes, arena_layout.adata_x())? };
        let q = unsafe { arena_region_slice_mut::<f32>(base, bytes, arena_layout.qdata())? };
        let l = unsafe { arena_region_slice_mut::<f32>(base, bytes, arena_layout.ldata())? };
        let u = unsafe { arena_region_slice_mut::<f32>(base, bytes, arena_layout.udata())? };

        scatter_embedded_qp_coefficients(
            self.program.qp_program.coefficient_outputs(),
            coefficient_outputs,
            p_x,
            q,
            a_x,
            l,
            u,
        )?;
        validate_embedded_numeric_update(p_x, a_x, q, l, u)?;

        let update = ffi::CokerOsqpNumericUpdate {
            p_x: p_x.as_ptr(),
            p_nnz: checked_embedded_ffi_length(self.program.p_nnz)?,
            a_x: a_x.as_ptr(),
            a_nnz: checked_embedded_ffi_length(self.program.a_nnz)?,
            q: q.as_ptr(),
            q_len: checked_embedded_ffi_length(self.program.n)?,
            l: l.as_ptr(),
            l_len: checked_embedded_ffi_length(self.program.m)?,
            u: u.as_ptr(),
            u_len: checked_embedded_ffi_length(self.program.m)?,
        };
        let update_status = {
            let instance = self
                .instance
                .as_mut()
                .ok_or_else(Self::invalid_instance_error)?;
            unsafe { ffi::coker_osqp_update(instance, &update) }
        };
        if update_status != ffi::COKER_OSQP_OK {
            self.instance = self.program.bind_instance(&self.arena).ok();
            return Err(RuntimeError::EmbeddedQpAbi {
                operation: "update",
                status: update_status,
            });
        }

        let mut solve_status = ffi::COKER_OSQP_SOLVE_UNSOLVED;
        let solve_abi_status = {
            let instance = self
                .instance
                .as_mut()
                .ok_or_else(Self::invalid_instance_error)?;
            unsafe { ffi::coker_osqp_solve(instance, &mut solve_status) }
        };
        if solve_abi_status != ffi::COKER_OSQP_OK {
            return Err(RuntimeError::EmbeddedQpAbi {
                operation: "solve",
                status: solve_abi_status,
            });
        }

        let mut solution = MaybeUninit::<ffi::CokerOsqpSolution>::zeroed();
        let solution_status = {
            let instance = self
                .instance
                .as_ref()
                .ok_or_else(Self::invalid_instance_error)?;
            unsafe { ffi::coker_osqp_solution(instance, solution.as_mut_ptr()) }
        };
        if solution_status != ffi::COKER_OSQP_OK {
            return Err(RuntimeError::EmbeddedQpAbi {
                operation: "solution",
                status: solution_status,
            });
        }
        let solution = unsafe { solution.assume_init() };
        if solution.primal_len != checked_embedded_ffi_length(self.program.n)?
            || solution.dual_len != checked_embedded_ffi_length(self.program.m)?
            || (self.program.n != 0 && solution.primal.is_null())
            || (self.program.m != 0 && solution.dual.is_null())
        {
            return Err(RuntimeError::EmbeddedQpAbi {
                operation: "solution",
                status: ffi::COKER_OSQP_INVALID_ARGUMENT,
            });
        }

        let primal = unsafe { slice::from_raw_parts(solution.primal, self.program.n) };
        outputs.copy_from_slice(primal);
        Ok(QpSolveDiagnostics {
            status: QpSolveStatus::from_embedded_raw(solve_status),
            iterations: solution.iterations,
            primal_residual: solution.primal_residual,
            dual_residual: solution.dual_residual,
        })
    }
    pub fn push_forward(
        &mut self,
        parameters: &[&[f32]],
        parameter_tangents: &[&[f32]],
        workspace: MappedQpPushForwardWorkspace<'_>,
        outputs: &mut [f32],
        tangent_outputs: &mut [f32],
    ) -> Result<QpSolveDiagnostics, RuntimeError> {
        self.program.validate_parameters(parameters)?;
        self.program.validate_tangents(parameter_tangents)?;
        workspace.validate_for(self.program.workspace_requirements)?;
        self.program
            .validate_push_forward_outputs(outputs, tangent_outputs)?;
        Err(MappedQpProgram::push_forward_unsupported_error())
    }
}

#[cfg(osqp_embedded)]
impl<'a> BoundQpArena<'a> {
    fn new(
        arena: &'a mut [MaybeUninit<u8>],
        requirements: QpWorkspaceRequirements,
    ) -> Result<Self, RuntimeError> {
        let base = NonNull::new(arena.as_mut_ptr().cast()).unwrap_or_else(NonNull::dangling);
        if arena.len() < requirements.arena_bytes
            || requirements.arena_alignment == 0
            || !requirements.arena_alignment.is_power_of_two()
            || (base.as_ptr() as usize) % requirements.arena_alignment != 0
        {
            return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
        }
        Ok(Self {
            base,
            bytes: arena.len(),
            _borrowed: PhantomData,
        })
    }

    fn as_ffi(&self, requirements: QpWorkspaceRequirements) -> ffi::CokerOsqpArena {
        ffi::CokerOsqpArena {
            base: self.base.as_ptr().cast(),
            bytes: self.bytes,
            alignment: requirements.arena_alignment,
        }
    }
}
