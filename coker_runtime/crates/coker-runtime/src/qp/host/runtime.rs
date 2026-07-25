use super::*;

#[cfg(all(feature = "std", not(osqp_embedded)))]
impl<'module, 'arena> BoundMappedQpProgram<'module, 'arena> {
    /// Evaluates coefficients, updates OSQP, solves the QP, and writes the primal solution.
    pub fn execute(
        &mut self,
        parameters: &[&[f32]],
        warm_start: Option<&[f64]>,
        workspace: MappedQpWorkspace<'_>,
        outputs: &mut [f32],
    ) -> Result<QpSolveDiagnostics, RuntimeError> {
        self.program.validate_parameters(parameters)?;
        workspace.validate_for(self.program.workspace_requirements)?;
        self.program.validate_output_buffer(outputs)?;

        let coefficient_outputs = &mut workspace.coefficient_outputs
            [..self.program.workspace_requirements.coefficient_output_size];
        self.program.evaluator.execute(
            parameters,
            workspace.evaluator_workspace,
            coefficient_outputs,
        )?;

        let p_nnz = checked_host_ffi_length(self.runtime.p_indices.len(), "QP P nnz")?;
        let a_nnz = checked_host_ffi_length(self.runtime.a_indices.len(), "QP A nnz")?;
        let problem = self.runtime.problem.as_ptr();
        let warm_start_enabled = self
            .program
            .qp_program
            .embedded_plan()
            .settings()
            .warm_start;
        let n = self.program.n;
        let m = self.program.m;

        let diagnostics = self.runtime.workspace.with_view(|view| {
            scatter_qp_coefficients(
                self.program.qp_program.coefficient_outputs(),
                coefficient_outputs,
                view.p_x,
                view.q,
                view.a_x,
                view.l,
                view.u,
            )?;

            let update_p_a = unsafe {
                ffi::osqp_update_P_A(
                    problem,
                    view.p_x.as_ptr() as *const ffi::c_float,
                    ptr::null(),
                    p_nnz,
                    view.a_x.as_ptr() as *const ffi::c_float,
                    ptr::null(),
                    a_nnz,
                )
            };
            if update_p_a != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP matrix update failed with status {update_p_a}"
                )));
            }

            let update_q = unsafe {
                ffi::osqp_update_lin_cost(problem, view.q.as_ptr() as *const ffi::c_float)
            };
            if update_q != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP linear cost update failed with status {update_q}"
                )));
            }

            let update_bounds = unsafe {
                ffi::osqp_update_bounds(
                    problem,
                    view.l.as_ptr() as *const ffi::c_float,
                    view.u.as_ptr() as *const ffi::c_float,
                )
            };
            if update_bounds != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP bounds update failed with status {update_bounds}"
                )));
            }

            if warm_start_enabled {
                if let Some(initial) = warm_start {
                    if initial.len() != n {
                        return Err(RuntimeError::Validation(
                            "QP warm start length does not match decision dimension".to_string(),
                        ));
                    }
                    view.primal_warm_start.copy_from_slice(initial);
                }
                let warm_start_x = unsafe {
                    ffi::osqp_warm_start_x(
                        problem,
                        view.primal_warm_start.as_ptr() as *const ffi::c_float,
                    )
                };
                if warm_start_x != 0 {
                    return Err(RuntimeError::QpSolver(format!(
                        "OSQP primal warm start failed with status {warm_start_x}"
                    )));
                }
                let warm_start_y = unsafe {
                    ffi::osqp_warm_start_y(
                        problem,
                        view.dual_warm_start.as_ptr() as *const ffi::c_float,
                    )
                };
                if warm_start_y != 0 {
                    return Err(RuntimeError::QpSolver(format!(
                        "OSQP dual warm start failed with status {warm_start_y}"
                    )));
                }
            }

            let solve_status = unsafe { ffi::osqp_solve(problem) };
            if solve_status != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP solve failed with status {solve_status}"
                )));
            }

            let info = unsafe { (*problem).info.as_ref() };
            let solution = unsafe { (*problem).solution.as_ref() };
            let status = info
                .map(|info| QpSolveStatus::from_raw(info.status_val))
                .unwrap_or(QpSolveStatus::Unsolved);
            let primal = solution
                .map(|solution| unsafe { slice::from_raw_parts(solution.x as *const f64, n) });
            let dual = solution
                .map(|solution| unsafe { slice::from_raw_parts(solution.y as *const f64, m) });
            if let Some(values) = primal {
                if view.primal_warm_start.len() == values.len() {
                    view.primal_warm_start.copy_from_slice(values);
                }
            }
            if let Some(values) = dual {
                if view.dual_warm_start.len() == values.len() {
                    view.dual_warm_start.copy_from_slice(values);
                }
            }
            let primal = primal.ok_or_else(|| {
                RuntimeError::QpSolver("OSQP solve returned no primal solution".to_string())
            })?;
            for (destination, value) in outputs.iter_mut().zip(primal.iter()) {
                *destination = *value as f32;
            }
            Ok::<_, RuntimeError>(QpSolveDiagnostics {
                status,
                iterations: info.map(|info| info.iter as i32).unwrap_or_default(),
                primal_residual: info.map(|info| info.pri_res as f32).unwrap_or_default(),
                dual_residual: info.map(|info| info.dua_res as f32).unwrap_or_default(),
            })
        })?;
        Ok(diagnostics)
    }

    /// Placeholder push-forward entry point for host QP solves.
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

#[cfg(all(feature = "std", not(osqp_embedded)))]
impl<'module, 'workspace> QpRuntime<'module, 'workspace> {
    /// Creates a reusable host QP runtime over caller-provided workspace storage.
    pub fn new(
        program: MappedQpProgram<'module>,
        workspace: &'workspace mut [f64],
    ) -> Result<Self, RuntimeError> {
        let layout = program.workspace_layout()?;
        let workspace = QpWorkspace::borrowed(workspace, layout)?;
        Self::from_program_and_workspace(program, workspace)
    }

    fn from_program_and_workspace(
        program: MappedQpProgram<'module>,
        mut workspace: QpWorkspace<'workspace>,
    ) -> Result<Self, RuntimeError> {
        let p_pattern = program.qp_program.p_pattern();
        let a_pattern = program.qp_program.a_pattern();
        let p_indptr = collect_host_osqp_indices(&p_pattern.indptr, "QP P indptr")?;
        let p_indices = collect_host_osqp_indices(&p_pattern.indices, "QP P indices")?;
        let a_indptr = collect_host_osqp_indices(&a_pattern.indptr, "QP A indptr")?;
        let a_indices = collect_host_osqp_indices(&a_pattern.indices, "QP A indices")?;
        let n = checked_host_ffi_length(program.n, "QP n")?;
        let m = checked_host_ffi_length(program.m, "QP m")?;
        let settings_source = program.qp_program.embedded_plan().settings();

        let problem = workspace.with_view(|view| {
            view.evaluator_workspace.fill(0.0);
            view.coefficient_outputs.fill(0.0);
            view.p_x.fill(0.0);
            view.a_x.fill(0.0);
            view.q.fill(0.0);
            view.l.fill(0.0);
            view.u.fill(0.0);
            view.primal_warm_start.fill(0.0);
            view.dual_warm_start.fill(0.0);

            let p = csc_matrix(n, n, &p_indptr, &p_indices, view.p_x)?;
            let a = csc_matrix(m, n, &a_indptr, &a_indices, view.a_x)?;
            let data = ffi::OSQPData {
                n,
                m,
                P: &p as *const ffi::csc as *mut ffi::csc,
                A: &a as *const ffi::csc as *mut ffi::csc,
                q: view.q.as_mut_ptr() as *mut ffi::c_float,
                l: view.l.as_mut_ptr() as *mut ffi::c_float,
                u: view.u.as_mut_ptr() as *mut ffi::c_float,
            };
            let mut settings = unsafe { core::mem::zeroed::<ffi::OSQPSettings>() };
            unsafe {
                ffi::osqp_set_default_settings(&mut settings);
            }
            let _ = settings_source;
            settings.verbose = 0;
            settings.warm_start = if settings_source.warm_start { 1 } else { 0 };
            settings.polish = 1;

            let mut work = ptr::null_mut();
            let exitflag = unsafe { ffi::osqp_setup(&mut work, &data, &settings) };
            if exitflag != 0 {
                if !work.is_null() {
                    unsafe {
                        let _ = ffi::osqp_cleanup(work);
                    }
                }
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP setup failed with status {exitflag}"
                )));
            }
            NonNull::new(work).ok_or_else(|| {
                RuntimeError::QpSolver("OSQP setup returned a null workspace".to_string())
            })
        })?;

        Ok(Self {
            program,
            p_indptr,
            p_indices,
            a_indptr,
            a_indices,
            workspace,
            problem,
        })
    }

    /// Returns the packed workspace layout used by this runtime.
    pub fn workspace_layout(&self) -> QpWorkspaceLayout {
        self.workspace.layout()
    }

    /// Solves the QP in-place and returns borrowed views into the live OSQP solution.
    pub fn solve(
        &mut self,
        parameters: &[&[f32]],
        warm_start: Option<&[f64]>,
    ) -> Result<QpSolveResult<'_>, RuntimeError> {
        self.program.validate_parameters(parameters)?;

        let n = self.program.n;
        let m = self.program.m;
        let warm_start_enabled = self
            .program
            .qp_program
            .embedded_plan()
            .settings()
            .warm_start;
        let p_nnz = checked_host_ffi_length(self.p_indices.len(), "QP P nnz")?;
        let a_nnz = checked_host_ffi_length(self.a_indices.len(), "QP A nnz")?;
        let problem = self.problem.as_ptr();
        let program = self.program;

        let result = self.workspace.with_view(|view| {
            if view.evaluator_workspace.len() != program.evaluator.info().required_workspace_size {
                return Err(RuntimeError::Validation(
                    "QP evaluator workspace size does not match program metadata".to_string(),
                ));
            }
            if view.coefficient_outputs.len()
                != program.workspace_requirements.coefficient_output_size
            {
                return Err(RuntimeError::Validation(
                    "QP evaluator output length does not match program metadata".to_string(),
                ));
            }

            program.evaluator.execute(
                parameters,
                view.evaluator_workspace,
                view.coefficient_outputs,
            )?;
            scatter_qp_coefficients(
                program.qp_program.coefficient_outputs(),
                view.coefficient_outputs,
                view.p_x,
                view.q,
                view.a_x,
                view.l,
                view.u,
            )?;

            let update_p_a = unsafe {
                ffi::osqp_update_P_A(
                    problem,
                    view.p_x.as_ptr() as *const ffi::c_float,
                    ptr::null(),
                    p_nnz,
                    view.a_x.as_ptr() as *const ffi::c_float,
                    ptr::null(),
                    a_nnz,
                )
            };
            if update_p_a != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP matrix update failed with status {update_p_a}"
                )));
            }

            let update_q = unsafe {
                ffi::osqp_update_lin_cost(problem, view.q.as_ptr() as *const ffi::c_float)
            };
            if update_q != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP linear cost update failed with status {update_q}"
                )));
            }

            let update_bounds = unsafe {
                ffi::osqp_update_bounds(
                    problem,
                    view.l.as_ptr() as *const ffi::c_float,
                    view.u.as_ptr() as *const ffi::c_float,
                )
            };
            if update_bounds != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP bounds update failed with status {update_bounds}"
                )));
            }

            if warm_start_enabled {
                if let Some(initial) = warm_start {
                    if initial.len() != n {
                        return Err(RuntimeError::Validation(
                            "QP warm start length does not match decision dimension".to_string(),
                        ));
                    }
                    view.primal_warm_start.copy_from_slice(initial);
                }
                let warm_start_x = unsafe {
                    ffi::osqp_warm_start_x(
                        problem,
                        view.primal_warm_start.as_ptr() as *const ffi::c_float,
                    )
                };
                if warm_start_x != 0 {
                    return Err(RuntimeError::QpSolver(format!(
                        "OSQP primal warm start failed with status {warm_start_x}"
                    )));
                }
                let warm_start_y = unsafe {
                    ffi::osqp_warm_start_y(
                        problem,
                        view.dual_warm_start.as_ptr() as *const ffi::c_float,
                    )
                };
                if warm_start_y != 0 {
                    return Err(RuntimeError::QpSolver(format!(
                        "OSQP dual warm start failed with status {warm_start_y}"
                    )));
                }
            }

            let solve_status = unsafe { ffi::osqp_solve(problem) };
            if solve_status != 0 {
                return Err(RuntimeError::QpSolver(format!(
                    "OSQP solve failed with status {solve_status}"
                )));
            }

            let info = unsafe { (*problem).info.as_ref() };
            let solution = unsafe { (*problem).solution.as_ref() };
            let status = info
                .map(|info| QpSolveStatus::from_raw(info.status_val))
                .unwrap_or(QpSolveStatus::Unsolved);
            let primal = solution
                .map(|solution| unsafe { slice::from_raw_parts(solution.x as *const f64, n) });
            let dual = solution
                .map(|solution| unsafe { slice::from_raw_parts(solution.y as *const f64, m) });
            if let Some(values) = primal {
                if view.primal_warm_start.len() == values.len() {
                    view.primal_warm_start.copy_from_slice(values);
                }
            }
            if let Some(values) = dual {
                if view.dual_warm_start.len() == values.len() {
                    view.dual_warm_start.copy_from_slice(values);
                }
            }

            Ok::<_, RuntimeError>(QpSolveResult {
                status,
                primal,
                dual,
                iterations: info.map(|info| info.iter as usize),
                primal_residual: info.map(|info| info.pri_res),
                dual_residual: info.map(|info| info.dua_res),
            })
        })?;

        Ok(result)
    }
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
impl<'module, 'workspace> Drop for QpRuntime<'module, 'workspace> {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::osqp_cleanup(self.problem.as_ptr());
        }
    }
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
fn scatter_qp_coefficients(
    outputs: &ArchivedQpCoefficientOutputs,
    coefficient_outputs: &[f32],
    p_x: &mut [f64],
    q: &mut [f64],
    a_x: &mut [f64],
    l: &mut [f64],
    u: &mut [f64],
) -> Result<(), RuntimeError> {
    scatter_output_slice(&outputs.px, coefficient_outputs, p_x)?;
    scatter_output_slice(&outputs.q, coefficient_outputs, q)?;
    scatter_output_slice(&outputs.ax, coefficient_outputs, a_x)?;
    scatter_output_slice(&outputs.l, coefficient_outputs, l)?;
    scatter_output_slice(&outputs.u, coefficient_outputs, u)?;
    Ok(())
}

#[cfg(all(feature = "std", not(osqp_embedded)))]
fn scatter_output_slice(
    slice: &ArchivedQpOutputSlice,
    source: &[f32],
    destination: &mut [f64],
) -> Result<(), RuntimeError> {
    let start = slice.start.to_native() as usize;
    let length = slice.length.to_native() as usize;
    let end = start
        .checked_add(length)
        .ok_or_else(|| RuntimeError::Validation("QP coefficient output overflow".to_string()))?;
    if end > source.len() || length != destination.len() {
        return Err(RuntimeError::Validation(
            "QP coefficient output length mismatch".to_string(),
        ));
    }
    for (dst, src) in destination.iter_mut().zip(&source[start..end]) {
        *dst = *src as f64;
    }
    Ok(())
}

#[cfg(osqp_embedded)]
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

#[cfg(osqp_embedded)]
fn scatter_embedded_output_slice(
    output: &ArchivedQpOutputSlice,
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

#[cfg(osqp_embedded)]
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
