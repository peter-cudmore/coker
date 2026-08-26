use super::*;

impl<'a> MappedQpProgram<'a> {
    fn bind_instance_in_arena(
        &self,
        arena: &BoundQpArena<'_>,
    ) -> Result<EmbeddedOsqpInstance, RuntimeError> {
        EmbeddedOsqpInstance::bind(self.qp_program, arena)
    }

    /// Binds this QP program to a caller-provided aligned OSQP arena.
    pub fn bind<'arena>(
        &self,
        arena: &'arena mut [MaybeUninit<u8>],
    ) -> Result<BoundMappedQpProgram<'a, 'arena>, RuntimeError> {
        let arena = BoundQpArena::new(arena, self.workspace_requirements)?;
        let instance = self.bind_instance_in_arena(&arena)?;
        Ok(BoundMappedQpProgram {
            program: *self,
            arena,
            instance: Some(instance),
        })
    }

    /// Prepares detached solver state over an externally-owned arena.
    ///
    /// # Safety
    ///
    /// The caller must keep `arena` alive, writable, and at a stable address for the
    /// full lifetime of the returned solver state.
    pub unsafe fn prepare_detached(
        &self,
        arena: &mut [MaybeUninit<u8>],
    ) -> Result<PreparedQpProgram, RuntimeError> {
        let arena = BoundQpArena::new(arena, self.workspace_requirements)?;
        let instance = self.bind_instance_in_arena(&arena)?;
        Ok(PreparedQpProgram {
            function_id: self.function_id,
            instance: Some(instance),
            arena_base: arena.base,
            arena_bytes: arena.bytes,
        })
    }
}

impl<'module, 'arena> BoundMappedQpProgram<'module, 'arena> {
    fn invalid_instance_error() -> RuntimeError {
        RuntimeError::Validation(
            "embedded mapped QP instance is invalid after a prior update failure",
        )
    }

    pub fn execute(
        &mut self,
        parameters: &[&[f32]],
        warm_start: Option<&[f32]>,
        workspace: MappedQpWorkspace<'_>,
        outputs: &mut [f32],
    ) -> Result<QpSolveDiagnostics, RuntimeError> {
        execute_qp_program(
            self.program,
            &self.arena,
            &mut self.instance,
            QpParameters::Slices(parameters),
            warm_start,
            workspace,
            outputs,
        )
    }
    /// Executes this bound QP with one flat f32 parameter buffer.
    ///
    /// This is the source-runnable embedded reference entry point used by
    /// trace harnesses. It uses the same evaluator, f32 coefficient
    /// conversion, embedded OSQP update, solve, and status mapping as
    /// [`Self::execute`], without allocating or rebuilding solver state.
    pub fn execute_flat(
        &mut self,
        parameters: &[f32],
        workspace: MappedQpWorkspace<'_>,
        outputs: &mut [f32],
    ) -> Result<QpSolveDiagnostics, RuntimeError> {
        execute_qp_program(
            self.program,
            &self.arena,
            &mut self.instance,
            QpParameters::Flat(parameters),
            None,
            workspace,
            outputs,
        )
    }
}

impl PreparedQpProgram {
    pub fn execute(
        &mut self,
        program: MappedQpProgram<'_>,
        parameters: &[&[f32]],
        warm_start: Option<&[f32]>,
        workspace: MappedQpWorkspace<'_>,
        outputs: &mut [f32],
    ) -> Result<QpSolveDiagnostics, RuntimeError> {
        if program.function_id() != self.function_id {
            return Err(RuntimeError::Validation(
                "prepared QP instance function_id does not match the mapped program",
            ));
        }
        let arena = unsafe { BoundQpArena::from_raw(self.arena_base, self.arena_bytes) };
        execute_qp_program(
            program,
            &arena,
            &mut self.instance,
            QpParameters::Slices(parameters),
            warm_start,
            workspace,
            outputs,
        )
    }
    /// Executes a detached prepared QP with one flat f32 parameter buffer.
    ///
    /// The arithmetic and status behavior is identical to mapped execution;
    /// the caller owns all arena and scratch storage.
    pub fn execute_flat(
        &mut self,
        program: MappedQpProgram<'_>,
        parameters: &[f32],
        workspace: MappedQpWorkspace<'_>,
        outputs: &mut [f32],
    ) -> Result<QpSolveDiagnostics, RuntimeError> {
        if program.function_id() != self.function_id {
            return Err(RuntimeError::Validation(
                "prepared QP instance function_id does not match the mapped program",
            ));
        }
        let arena = unsafe { BoundQpArena::from_raw(self.arena_base, self.arena_bytes) };
        execute_qp_program(
            program,
            &arena,
            &mut self.instance,
            QpParameters::Flat(parameters),
            None,
            workspace,
            outputs,
        )
    }
}

enum QpParameters<'a> {
    Slices(&'a [&'a [f32]]),
    Flat(&'a [f32]),
}

fn execute_qp_program(
    program: MappedQpProgram<'_>,
    arena: &BoundQpArena<'_>,
    instance: &mut Option<EmbeddedOsqpInstance>,
    parameters: QpParameters<'_>,
    warm_start: Option<&[f32]>,
    workspace: MappedQpWorkspace<'_>,
    outputs: &mut [f32],
) -> Result<QpSolveDiagnostics, RuntimeError> {
    workspace.validate_for(program.workspace_requirements)?;
    program.validate_output_buffer(outputs)?;

    let coefficient_output_size = program.workspace_requirements.coefficient_output_size;
    let MappedQpWorkspace {
        evaluator_workspace,
        coefficient_outputs,
    } = workspace;
    let coefficient_outputs = &mut coefficient_outputs[..coefficient_output_size];
    match parameters {
        QpParameters::Slices(parameters) => {
            program.validate_parameters(parameters)?;
            program
                .evaluator
                .execute(parameters, evaluator_workspace, coefficient_outputs)?;
        }
        QpParameters::Flat(parameters) => program.evaluator.execute_flat_inputs(
            parameters,
            evaluator_workspace,
            coefficient_outputs,
        )?,
    }

    {
        let instance_ref = instance
            .as_mut()
            .ok_or_else(BoundMappedQpProgram::invalid_instance_error)?;
        instance_ref.refresh_self_pointers();
        let (p_x, a_x, q, l, u) =
            instance_ref.numeric_slices_mut(program.p_nnz, program.a_nnz, program.n, program.m)?;

        scatter_embedded_qp_coefficients(
            program.qp_program.coefficient_outputs(),
            coefficient_outputs,
            p_x,
            q,
            a_x,
            l,
            u,
        )?;
        validate_embedded_numeric_update(p_x, a_x, q, l, u)?;
    }

    if program.qp_program.embedded_plan().settings().warm_start {
        if let Some(initial) = warm_start {
            if initial.len() != program.n {
                return Err(RuntimeError::Validation(
                    "QP warm start length does not match decision dimension",
                ));
            }
            let instance_ref = instance
                .as_mut()
                .ok_or_else(BoundMappedQpProgram::invalid_instance_error)?;
            let mut solver = instance_ref.solver();
            let warm_start_status = unsafe {
                solver
                    .warm_start(Some(initial), None)
                    .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?
            };
            if warm_start_status != 0 {
                return Err(RuntimeError::EmbeddedQpAbi {
                    operation: "warm_start",
                    status: warm_start_status,
                });
            }
        }
    }

    let (q_ptr, l_ptr, u_ptr) = {
        let instance_ref = instance
            .as_mut()
            .ok_or_else(BoundMappedQpProgram::invalid_instance_error)?;
        let (_, _, q, l, u) =
            instance_ref.numeric_slices_mut(program.p_nnz, program.a_nnz, program.n, program.m)?;
        (q.as_ptr(), l.as_ptr(), u.as_ptr())
    };
    let update_vec_status = {
        let instance_ref = instance
            .as_mut()
            .ok_or_else(BoundMappedQpProgram::invalid_instance_error)?;
        let mut solver = instance_ref.solver();
        unsafe {
            solver
                .update_data_vec(
                    slice::from_raw_parts(q_ptr, program.n),
                    slice::from_raw_parts(l_ptr, program.m),
                    slice::from_raw_parts(u_ptr, program.m),
                )
                .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?
        }
    };
    if update_vec_status != 0 {
        *instance = program.bind_instance_in_arena(arena).ok();
        return Err(RuntimeError::EmbeddedQpAbi {
            operation: "update_data_vec",
            status: update_vec_status,
        });
    }

    let (p_x_ptr, a_x_ptr) = {
        let instance_ref = instance
            .as_mut()
            .ok_or_else(BoundMappedQpProgram::invalid_instance_error)?;
        let (p_x, a_x, _, _, _) =
            instance_ref.numeric_slices_mut(program.p_nnz, program.a_nnz, program.n, program.m)?;
        (p_x.as_ptr(), a_x.as_ptr())
    };
    let (update_mat_status, counts) = {
        let instance_ref = instance
            .as_mut()
            .ok_or_else(BoundMappedQpProgram::invalid_instance_error)?;
        let mut solver = instance_ref.solver();
        let counts = unsafe {
            solver
                .matrix_update_counts()
                .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?
        };
        if !counts.is_consistent(program.p_nnz, program.a_nnz) {
            return Err(RuntimeError::EmbeddedQpCscDescriptor {
                p_nzmax: counts.p.nzmax,
                p_terminal: counts.p.terminal_indptr,
                p_submitted: program.p_nnz,
                a_nzmax: counts.a.nzmax,
                a_terminal: counts.a.terminal_indptr,
                a_submitted: program.a_nnz,
            });
        }
        let status = unsafe {
            solver
                .update_data_mat(
                    slice::from_raw_parts(p_x_ptr, program.p_nnz),
                    slice::from_raw_parts(a_x_ptr, program.a_nnz),
                )
                .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?
        };
        (status, counts)
    };
    if update_mat_status != 0 {
        *instance = program.bind_instance_in_arena(arena).ok();
        return Err(RuntimeError::EmbeddedQpCscUpdate {
            status: update_mat_status,
            p_nzmax: counts.p.nzmax,
            p_terminal: counts.p.terminal_indptr,
            p_submitted: program.p_nnz,
            a_nzmax: counts.a.nzmax,
            a_terminal: counts.a.terminal_indptr,
            a_submitted: program.a_nnz,
        });
    }

    let (status, iterations, primal_residual, dual_residual) = {
        let instance_ref = instance
            .as_mut()
            .ok_or_else(BoundMappedQpProgram::invalid_instance_error)?;
        let mut solver = instance_ref.solver();
        let solve_status = unsafe { solver.solve() };
        if solve_status != 0 {
            return Err(RuntimeError::EmbeddedQpAbi {
                operation: "solve",
                status: solve_status,
            });
        }
        let solution = unsafe {
            solver
                .solution()
                .ok_or(RuntimeError::EmbeddedQpWorkspaceInvalid)?
        };
        if solution.primal.len() != program.n || solution.dual.len() != program.m {
            return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
        }
        outputs.copy_from_slice(solution.primal);
        (
            QpSolveStatus::from_raw(solution.status),
            solution.iterations,
            solution.primal_residual,
            solution.dual_residual,
        )
    };
    Ok(QpSolveDiagnostics {
        status,
        iterations,
        primal_residual,
        dual_residual,
    })
}

fn is_aligned(value: usize, alignment: usize) -> bool {
    value.checked_rem(alignment) == Some(0)
}

impl<'a> BoundQpArena<'a> {
    fn new(
        arena: &'a mut [MaybeUninit<u8>],
        requirements: QpWorkspaceRequirements,
    ) -> Result<Self, RuntimeError> {
        let base = NonNull::new(arena.as_mut_ptr().cast()).unwrap_or_else(NonNull::dangling);
        if arena.len() < requirements.arena_bytes
            || requirements.arena_alignment == 0
            || !requirements.arena_alignment.is_power_of_two()
            || !is_aligned(base.as_ptr() as usize, requirements.arena_alignment)
        {
            return Err(RuntimeError::EmbeddedQpWorkspaceInvalid);
        }
        Ok(Self {
            base,
            bytes: arena.len(),
            _borrowed: PhantomData,
        })
    }

    unsafe fn from_raw(base: NonNull<u8>, bytes: usize) -> Self {
        Self {
            base,
            bytes,
            _borrowed: PhantomData,
        }
    }
}

fn scatter_embedded_qp_coefficients(
    outputs: &ArchivedQpCoefficientOutputs,
    coefficient_outputs: &[f32],
    p_x: &mut [f32],
    q: &mut [f32],
    a_x: &mut [f32],
    l: &mut [f32],
    u: &mut [f32],
) -> Result<(), RuntimeError> {
    scatter_embedded_output_slice(&outputs.px, coefficient_outputs, p_x)?;
    scatter_embedded_output_slice(&outputs.q, coefficient_outputs, q)?;
    scatter_embedded_output_slice(&outputs.ax, coefficient_outputs, a_x)?;
    scatter_embedded_output_slice(&outputs.l, coefficient_outputs, l)?;
    scatter_embedded_output_slice(&outputs.u, coefficient_outputs, u)
}

fn scatter_embedded_output_slice(
    output: &coker_bytecode::ArchivedQpOutputSlice,
    source: &[f32],
    destination: &mut [f32],
) -> Result<(), RuntimeError> {
    let start = output.start.to_native() as usize;
    let length = output.length.to_native() as usize;
    let end = start
        .checked_add(length)
        .ok_or(RuntimeError::EmbeddedQpNumericInvalid)?;
    if end > source.len() || length != destination.len() {
        return Err(RuntimeError::EmbeddedQpNumericInvalid);
    }
    destination.copy_from_slice(&source[start..end]);
    Ok(())
}

fn validate_embedded_numeric_update(
    p_x: &[f32],
    a_x: &[f32],
    q: &[f32],
    l: &[f32],
    u: &[f32],
) -> Result<(), RuntimeError> {
    if p_x
        .iter()
        .chain(a_x)
        .chain(q)
        .chain(l)
        .chain(u)
        .any(|value| !value.is_finite())
        || l.iter().zip(u).any(|(lower, upper)| lower > upper)
    {
        return Err(RuntimeError::EmbeddedQpNumericInvalid);
    }
    Ok(())
}
