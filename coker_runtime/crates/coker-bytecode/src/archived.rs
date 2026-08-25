use super::*;
use core::mem::align_of;
use rkyv::{access, rancor::Error as RkyvError};

impl ArchivedQdldlSymbolicL {
    /// Returns the archived sparsity pattern for the symbolic `L` factor.
    pub fn l_pattern(&self) -> &ArchivedEmbeddedCscPattern {
        &self.l_pattern
    }

    /// Returns the archived elimination tree for the symbolic factorization.
    pub fn etree(&self) -> &[rkyv::rend::u32_le] {
        &self.etree
    }

    /// Returns the archived per-column nonzero counts for the symbolic `L` factor.
    pub fn lnz(&self) -> &[rkyv::rend::u32_le] {
        &self.lnz
    }

    /// Validates the symbolic factorization against the archived KKT pattern.
    pub fn validate(&self, kkt_pattern: &ArchivedEmbeddedCscPattern) -> Result<(), BytecodeError> {
        validate_archived_qdldl_symbolic_l(self, kkt_pattern)
    }
}

impl ArchivedQpProgramArenaRegion {
    /// Returns the byte offset of this archived arena region.
    pub fn byte_offset(&self) -> u32 {
        self.byte_offset.to_native()
    }

    /// Returns the byte length of this archived arena region.
    pub fn byte_len(&self) -> u32 {
        self.byte_len.to_native()
    }

    /// Returns the required byte alignment of this archived arena region.
    pub fn byte_alignment(&self) -> u32 {
        self.byte_alignment.to_native()
    }

    /// Validates this archived arena region descriptor.
    pub fn validate(&self, field: &'static str) -> Result<(), BytecodeError> {
        validate_archived_qp_program_arena_region(self, field)
    }
}

impl ArchivedQpProgramArenaLayout {
    /// Returns the total arena size in bytes.
    pub fn total_bytes(&self) -> u32 {
        self.total_bytes.to_native()
    }

    /// Returns the required alignment for the full arena allocation.
    pub fn arena_alignment(&self) -> u32 {
        self.arena_alignment.to_native()
    }

    /// Returns the arena region for OSQP's `pdata_x` numeric buffer.
    pub fn pdata_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.pdata_x
    }

    /// Returns the arena region for OSQP's `pdata` matrix descriptor.
    pub fn pdata(&self) -> &ArchivedQpProgramArenaRegion {
        &self.pdata
    }

    /// Returns the arena region for OSQP's `adata_x` numeric buffer.
    pub fn adata_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.adata_x
    }

    /// Returns the arena region for OSQP's `adata` matrix descriptor.
    pub fn adata(&self) -> &ArchivedQpProgramArenaRegion {
        &self.adata
    }

    /// Returns the arena region for OSQP's linear term buffer.
    pub fn qdata(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdata
    }

    /// Returns the arena region for OSQP's lower-bound buffer.
    pub fn ldata(&self) -> &ArchivedQpProgramArenaRegion {
        &self.ldata
    }

    /// Returns the arena region for OSQP's upper-bound buffer.
    pub fn udata(&self) -> &ArchivedQpProgramArenaRegion {
        &self.udata
    }

    /// Returns the arena region for OSQP's shared workspace data buffer.
    pub fn data(&self) -> &ArchivedQpProgramArenaRegion {
        &self.data
    }

    /// Returns the arena region for OSQP's settings record.
    pub fn settings(&self) -> &ArchivedQpProgramArenaRegion {
        &self.settings
    }

    /// Returns the arena region for the primal solution vector.
    pub fn xsolution(&self) -> &ArchivedQpProgramArenaRegion {
        &self.xsolution
    }

    /// Returns the arena region for the dual solution vector.
    pub fn ysolution(&self) -> &ArchivedQpProgramArenaRegion {
        &self.ysolution
    }

    /// Returns the arena region for OSQP's aggregate solution record.
    pub fn solution(&self) -> &ArchivedQpProgramArenaRegion {
        &self.solution
    }

    /// Returns the arena region for OSQP's info record.
    pub fn info(&self) -> &ArchivedQpProgramArenaRegion {
        &self.info
    }

    /// Returns the arena region for QDLDL's `Lx` numeric buffer.
    pub fn qdldl_l_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_l_x
    }

    /// Returns the arena region for QDLDL's mutable `Lp` buffer.
    pub fn qdldl_l_p(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_l_p
    }

    /// Returns the arena region for QDLDL's mutable `Li` buffer.
    pub fn qdldl_l_i(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_l_i
    }

    /// Returns the arena region for QDLDL's `L` structure.
    pub fn qdldl_l(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_l
    }

    /// Returns the arena region for QDLDL's `KKTx` numeric buffer.
    pub fn qdldl_kkt_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_kkt_x
    }

    /// Returns the arena region for QDLDL's KKT structure.
    pub fn qdldl_kkt(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_kkt
    }

    /// Returns the arena region for the top-level QDLDL workspace record.
    pub fn qdldl(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl
    }

    /// Returns the arena region for QDLDL's inverse diagonal buffer.
    pub fn qdldl_dinv(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_dinv
    }

    /// Returns the arena region for QDLDL's breakpoint buffer.
    pub fn qdldl_bp(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_bp
    }

    /// Returns the arena region for QDLDL's solve scratch vector.
    pub fn qdldl_sol(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_sol
    }

    /// Returns the arena region for QDLDL's `rho_inv_vec` buffer.
    pub fn qdldl_rho_inv_vec(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_rho_inv_vec
    }

    /// Returns the arena region for QDLDL's diagonal buffer.
    pub fn qdldl_d(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_d
    }

    /// Returns the arena region for QDLDL's integer work buffer.
    pub fn qdldl_iwork(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_iwork
    }

    /// Returns the arena region for QDLDL's boolean work buffer.
    pub fn qdldl_bwork(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_bwork
    }

    /// Returns the arena region for QDLDL's floating-point work buffer.
    pub fn qdldl_fwork(&self) -> &ArchivedQpProgramArenaRegion {
        &self.qdldl_fwork
    }

    /// Returns the arena region for OSQP's `rho_vec` work buffer.
    pub fn work_rho_vec(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_rho_vec
    }

    /// Returns the arena region for OSQP's `rho_inv_vec` work buffer.
    pub fn work_rho_inv_vec(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_rho_inv_vec
    }

    /// Returns the arena region for OSQP's constraint-type work buffer.
    pub fn work_constr_type(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_constr_type
    }

    /// Returns the arena region for OSQP's primal iterate work vector.
    pub fn work_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_x
    }

    /// Returns the arena region for OSQP's dual iterate work vector.
    pub fn work_y(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_y
    }

    /// Returns the arena region for OSQP's slack iterate work vector.
    pub fn work_z(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_z
    }

    /// Returns the arena region for OSQP's `xz_tilde` work vector.
    pub fn work_xz_tilde(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_xz_tilde
    }

    /// Returns the arena region for OSQP's previous primal iterate.
    pub fn work_x_prev(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_x_prev
    }

    /// Returns the arena region for OSQP's previous slack iterate.
    pub fn work_z_prev(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_z_prev
    }

    /// Returns the arena region for OSQP's `Ax` work vector.
    pub fn work_ax(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_ax
    }

    /// Returns the arena region for OSQP's `Px` work vector.
    pub fn work_px(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_px
    }

    /// Returns the arena region for OSQP's `A^T y` work vector.
    pub fn work_aty(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_aty
    }

    /// Returns the arena region for OSQP's `delta_y` work vector.
    pub fn work_delta_y(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_delta_y
    }

    /// Returns the arena region for OSQP's `A^T delta_y` work vector.
    pub fn work_atdelta_y(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_atdelta_y
    }

    /// Returns the arena region for OSQP's `delta_x` work vector.
    pub fn work_delta_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_delta_x
    }

    /// Returns the arena region for OSQP's `P delta_x` work vector.
    pub fn work_pdelta_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_pdelta_x
    }

    /// Returns the arena region for OSQP's `A delta_x` work vector.
    pub fn work_adelta_x(&self) -> &ArchivedQpProgramArenaRegion {
        &self.work_adelta_x
    }

    /// Returns the arena region for the full embedded solver workspace.
    pub fn workspace(&self) -> &ArchivedQpProgramArenaRegion {
        &self.workspace
    }

    /// Validates the archived arena layout and every region boundary.
    pub fn validate(&self) -> Result<(), BytecodeError> {
        validate_archived_qp_program_arena_layout(self)
    }
}

impl ArchivedQpProgramQdldlPlan {
    /// Returns the archived `P` sparsity pattern used by the solver plan.
    pub fn p_pattern(&self) -> &ArchivedEmbeddedCscPattern {
        &self.p_pattern
    }

    /// Returns the archived `A` sparsity pattern used by the solver plan.
    pub fn a_pattern(&self) -> &ArchivedEmbeddedCscPattern {
        &self.a_pattern
    }

    /// Returns the archived KKT sparsity pattern used by the solver plan.
    pub fn kkt_pattern(&self) -> &ArchivedEmbeddedCscPattern {
        &self.kkt_pattern
    }

    /// Returns the archived symbolic QDLDL factorization metadata.
    pub fn symbolic_l(&self) -> &ArchivedQdldlSymbolicL {
        &self.symbolic_l
    }

    /// Validates the archived QDLDL plan against its embedded sparsity tables.
    pub fn validate(&self) -> Result<(), BytecodeError> {
        validate_archived_qp_program_qdldl_plan(self)
    }
}

impl ArchivedQpProgramPlan {
    /// Returns the embedded-plan ABI version expected by the runtime.
    pub fn abi_version(&self) -> u16 {
        self.abi_version.to_native()
    }

    /// Returns the embedded solver profile encoded in this archived plan.
    pub fn profile(&self) -> EmbeddedQpProfile {
        embedded_qp_profile_from_archived(&self.profile)
    }

    /// Returns the plan payload version.
    pub fn version(&self) -> u16 {
        self.version.to_native()
    }

    /// Returns the archived embedded OSQP settings.
    pub fn settings(&self) -> &ArchivedEmbeddedOsqpSettings {
        &self.settings
    }

    /// Returns the archived arena layout required by this plan.
    pub fn arena_layout(&self) -> &ArchivedQpProgramArenaLayout {
        &self.arena_layout
    }

    /// Returns the archived QDLDL metadata required by this plan.
    pub fn qdldl_plan(&self) -> &ArchivedQpProgramQdldlPlan {
        &self.qdldl_plan
    }

    /// Validates the archived embedded plan header, settings, layout, and QDLDL metadata.
    pub fn validate(&self) -> Result<(), BytecodeError> {
        validate_archived_qp_program_plan(self)
    }
}

impl ArchivedBytecodeModule {
    /// Iterates all archived executables with their index-derived ids.
    pub fn executables(&self) -> impl Iterator<Item = ArchivedExecutableRef<'_>> {
        self.executables
            .iter()
            .enumerate()
            .map(|(function_id, executable)| match executable {
                ArchivedExecutable::Program(program) => {
                    ArchivedExecutableRef::Program(function_id as u16, program)
                }
                ArchivedExecutable::QpProgram(program) => {
                    ArchivedExecutableRef::QpProgram(function_id as u16, program)
                }
            })
    }

    /// Iterates archived ordinary programs with their index-derived ids.
    pub fn programs(&self) -> impl Iterator<Item = (u16, &ArchivedProgram)> {
        self.executables
            .iter()
            .enumerate()
            .filter_map(|(function_id, executable)| match executable {
                ArchivedExecutable::Program(program) => Some((function_id as u16, program)),
                ArchivedExecutable::QpProgram(_) => None,
            })
    }

    /// Iterates archived QP programs with their index-derived ids.
    pub fn qp_programs(&self) -> impl Iterator<Item = (u16, &ArchivedQpProgram)> {
        self.executables
            .iter()
            .enumerate()
            .filter_map(|(function_id, executable)| match executable {
                ArchivedExecutable::Program(_) => None,
                ArchivedExecutable::QpProgram(program) => Some((function_id as u16, program)),
            })
    }

    /// Returns the archived ordinary program for `function_id`, if present.
    pub fn program(&self, function_id: u16) -> Option<&ArchivedProgram> {
        match self.executables.get(function_id as usize)? {
            ArchivedExecutable::Program(program) => Some(program),
            ArchivedExecutable::QpProgram(_) => None,
        }
    }

    /// Returns the archived QP program for `function_id`, if present.
    pub fn qp_program(&self, function_id: u16) -> Option<&ArchivedQpProgram> {
        match self.executables.get(function_id as usize)? {
            ArchivedExecutable::Program(_) => None,
            ArchivedExecutable::QpProgram(program) => Some(program),
        }
    }

    /// Returns either archived executable kind for `function_id`.
    pub fn executable(&self, function_id: u16) -> Option<ArchivedExecutableRef<'_>> {
        match self.executables.get(function_id as usize)? {
            ArchivedExecutable::Program(program) => {
                Some(ArchivedExecutableRef::Program(function_id, program))
            }
            ArchivedExecutable::QpProgram(program) => {
                Some(ArchivedExecutableRef::QpProgram(function_id, program))
            }
        }
    }

    /// Returns the archived conventional entry program (`function_id == 0`), if present.
    pub fn entry_program(&self) -> Option<&ArchivedProgram> {
        self.program(0)
    }

    /// Validates archived module-level structural and cross-reference invariants.
    pub fn validate_semantics(&self) -> Result<(), BytecodeError> {
        validate_archived_bytecode_module_semantics(self)
    }
}

impl ArchivedExecutableRef<'_> {
    /// Returns the index-derived id of the referenced archived executable.
    pub fn function_id(&self) -> u16 {
        match self {
            Self::Program(function_id, _) | Self::QpProgram(function_id, _) => *function_id,
        }
    }
}

/// Archived accessor helpers for mapped plain programs.
impl ArchivedProgram {
    /// Returns the caller-provided workspace size encoded for this program.
    pub fn workspace_size(&self) -> u32 {
        self.workspace_size.to_native()
    }

    /// Returns the minimum workspace size required to execute this program.
    pub fn required_workspace_size(&self) -> u32 {
        self.required_workspace_size.to_native()
    }

    /// Returns the archived flat input specification list.
    pub fn input_specs(&self) -> &[ArchivedInputSpec] {
        &self.input_specs
    }

    /// Returns the archived flat output specification list.
    pub fn output_specs(&self) -> &[ArchivedOutputSpec] {
        &self.output_specs
    }

    /// Returns the archived intermediate layer sequence.
    pub fn intermediate_layers(&self) -> &[ArchivedLayer] {
        &self.intermediate_layers
    }

    /// Computes the flat input width implied by the archived input specs.
    pub fn checked_flat_input_size(&self) -> Result<u32, BytecodeError> {
        checked_archived_flat_input_specs(self.input_specs(), "program input specs")
    }

    /// Computes the flat output width implied by the archived output specs.
    pub fn checked_flat_output_size(&self) -> Result<u32, BytecodeError> {
        checked_archived_flat_output_specs(self.output_specs(), "program output specs")
    }
}

/// Archived accessor helpers for direct-mapped CSC patterns.
impl ArchivedEmbeddedCscPattern {
    /// Returns the explicit number of CSC entries required by the OSQP ABI.
    pub fn nnz(&self) -> u32 {
        self.nnz.to_native()
    }
}

/// Archived accessor helpers for mapped QP program records.
impl ArchivedQpProgram {
    /// Returns the coefficient-evaluator function id referenced by this QP program.
    pub fn coefficient_function_id(&self) -> u16 {
        self.coefficient_function_id.to_native()
    }

    /// Returns the required primal workspace size for the embedded solver buffers.
    pub fn required_primal_workspace_size(&self) -> u32 {
        self.required_primal_workspace_size.to_native()
    }

    /// Returns the required tangent workspace size for coefficient evaluation.
    pub fn required_tangent_workspace_size(&self) -> u32 {
        self.required_tangent_workspace_size.to_native()
    }

    /// Returns the archived parameter input specification list.
    pub fn input_specs(&self) -> &[ArchivedInputSpec] {
        &self.input_specs
    }

    /// Returns the archived primal output specification.
    pub fn output_spec(&self) -> &ArchivedOutputSpec {
        &self.output_spec
    }

    /// Returns the archived quadratic-term sparsity pattern.
    pub fn p_pattern(&self) -> &ArchivedEmbeddedCscPattern {
        &self.p_pattern
    }

    /// Returns the archived constraint-matrix sparsity pattern.
    pub fn a_pattern(&self) -> &ArchivedEmbeddedCscPattern {
        &self.a_pattern
    }

    /// Returns the archived partition of coefficient-evaluator outputs.
    pub fn coefficient_outputs(&self) -> &ArchivedQpCoefficientOutputs {
        &self.coefficient_outputs
    }

    /// Returns the archived embedded solver plan required by this QP program.
    pub fn embedded_plan(&self) -> &ArchivedQpProgramPlan {
        &self.embedded_plan
    }

    /// Computes the flat parameter input width implied by the archived input specs.
    pub fn checked_flat_input_size(&self) -> Result<u32, BytecodeError> {
        checked_archived_flat_input_specs(self.input_specs(), "QP input specs")
    }

    /// Returns the flat primal output width encoded by the archived output spec.
    pub fn checked_flat_output_size(&self) -> u32 {
        u32::from(self.output_spec.length.to_native())
    }
}

/// Returns a validated archived module view over a mapped byte slice.
pub fn archived_module(bytes: &[u8]) -> Result<&ArchivedBytecodeModule, BytecodeError> {
    let payload = archive_payload(bytes, &MAGIC, VERSION, align_of::<ArchivedBytecodeModule>())?;
    let archived = access::<ArchivedBytecodeModule, RkyvError>(payload)
        .map_err(|error| BytecodeError::Decode(error.to_string()))?;
    archived.validate_semantics()?;
    Ok(archived)
}

/// Returns a validated archived host QP payload view over a mapped byte slice.
pub fn archived_qp_program(bytes: &[u8]) -> Result<&ArchivedQpProgramArchive, BytecodeError> {
    let payload = archive_payload(
        bytes,
        &QP_MAGIC,
        VERSION,
        align_of::<ArchivedQpProgramArchive>(),
    )?;
    access::<ArchivedQpProgramArchive, RkyvError>(payload)
        .map_err(|error| BytecodeError::Decode(error.to_string()))
}

/// Returns a validated archived standalone embedded QP plan over a mapped byte slice.
pub fn archived_embedded_qp_plan(bytes: &[u8]) -> Result<&ArchivedEmbeddedQpPlan, BytecodeError> {
    let payload = archive_payload(
        bytes,
        &EMBEDDED_QP_PLAN_MAGIC,
        EmbeddedQpPlan::VERSION,
        align_of::<ArchivedEmbeddedQpPlan>(),
    )?;
    let archived = access::<ArchivedEmbeddedQpPlan, RkyvError>(payload)
        .map_err(|error| BytecodeError::Decode(error.to_string()))?;
    validate_archived_embedded_qp_plan(archived)?;
    Ok(archived)
}
