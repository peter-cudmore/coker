#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::{convert::TryFrom, mem::align_of};
use rkyv::{Archive, Deserialize, Serialize};
use thiserror::Error;

const MAGIC: [u8; 8] = *b"COKERB03";
const VERSION: u16 = 3;
const HEADER_SIZE: usize = 16;
const QP_MAGIC: [u8; 8] = *b"COKERQ03";
const EMBEDDED_QP_PLAN_MAGIC: [u8; 8] = *b"COKERP03";
const EMBEDDED_QP_PLAN_VERSION: u16 = 1;
type ArchivedU32Vec = rkyv::vec::ArchivedVec<rkyv::rend::u32_le>;

/// Half-open slice into the flattened QP coefficient output buffer.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct QpOutputSlice {
    pub start: u32,
    pub length: u32,
}

/// Partition of coefficient evaluator outputs into OSQP numeric update slices.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct QpCoefficientOutputs {
    pub px: QpOutputSlice,
    pub q: QpOutputSlice,
    pub ax: QpOutputSlice,
    pub l: QpOutputSlice,
    pub u: QpOutputSlice,
    pub r: QpOutputSlice,
}


/// Serialized host QP problem payload used by tooling and host-side execution.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct QpProgramArchive {
    pub n: u32,
    pub m: u32,
    pub parameter_lengths: Vec<u32>,
    pub decision_input_indices: Vec<u32>,
    pub constraint_row_offsets: Vec<u32>,
    pub p_indptr: Vec<u32>,
    pub p_indices: Vec<u32>,
    pub a_indptr: Vec<u32>,
    pub a_indices: Vec<u32>,
    pub coefficient_program: BytecodeModule,
    pub coefficient_outputs: QpCoefficientOutputs,
    pub warm_start: bool,
}

/// Embedded solver profile encoded in a pointer-free QP plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum EmbeddedQpProfile {
    Osqp063Embedded2Qdldl,
}

/// Embedded linear-system backend encoded in a pointer-free QP plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum EmbeddedLinsysSolver {
    Qdldl,
}

/// Pointer-free CSC sparsity pattern used by embedded solver plans.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct EmbeddedCscPattern {
    pub nrows: u32,
    pub ncols: u32,
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
}

/// Embedded OSQP settings captured in the archived QP plan.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct EmbeddedOsqpSettings {
    pub rho: f64,
    pub sigma: f64,
    pub alpha: f64,
    pub adaptive_rho: bool,
    pub adaptive_rho_interval: u32,
    pub adaptive_rho_tolerance: f64,
    pub max_iter: u32,
    pub eps_abs: f64,
    pub eps_rel: f64,
    pub eps_prim_inf: f64,
    pub eps_dual_inf: f64,
    pub scaling: u32,
    pub scaled_termination: bool,
    pub check_termination: u32,
    pub warm_start: bool,
    pub linsys_solver: EmbeddedLinsysSolver,
}

/// Symbolic QDLDL factorization metadata for the embedded KKT solve.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct QdldlSymbolicL {
    pub l_pattern: EmbeddedCscPattern,
    pub etree: Vec<u32>,
    pub lnz: Vec<u32>,
}

/// One aligned byte region inside the embedded OSQP arena layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct QpProgramArenaRegion {
    pub byte_offset: u32,
    pub byte_len: u32,
    pub byte_alignment: u32,
}

/// Full embedded OSQP arena layout required by a validated QP plan.
#[derive(Debug, Clone, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct QpProgramArenaLayout {
    pub total_bytes: u32,
    pub arena_alignment: u32,
    pub pdata_x: QpProgramArenaRegion,
    pub pdata: QpProgramArenaRegion,
    pub adata_x: QpProgramArenaRegion,
    pub adata: QpProgramArenaRegion,
    pub qdata: QpProgramArenaRegion,
    pub ldata: QpProgramArenaRegion,
    pub udata: QpProgramArenaRegion,
    pub data: QpProgramArenaRegion,
    pub settings: QpProgramArenaRegion,
    pub xsolution: QpProgramArenaRegion,
    pub ysolution: QpProgramArenaRegion,
    pub solution: QpProgramArenaRegion,
    pub info: QpProgramArenaRegion,
    pub qdldl_l_x: QpProgramArenaRegion,
    pub qdldl_l: QpProgramArenaRegion,
    pub qdldl_kkt_x: QpProgramArenaRegion,
    pub qdldl_kkt: QpProgramArenaRegion,
    pub qdldl: QpProgramArenaRegion,
    pub qdldl_dinv: QpProgramArenaRegion,
    pub qdldl_bp: QpProgramArenaRegion,
    pub qdldl_sol: QpProgramArenaRegion,
    pub qdldl_rho_inv_vec: QpProgramArenaRegion,
    pub qdldl_d: QpProgramArenaRegion,
    pub qdldl_iwork: QpProgramArenaRegion,
    pub qdldl_bwork: QpProgramArenaRegion,
    pub qdldl_fwork: QpProgramArenaRegion,
    pub work_rho_vec: QpProgramArenaRegion,
    pub work_rho_inv_vec: QpProgramArenaRegion,
    pub work_constr_type: QpProgramArenaRegion,
    pub work_x: QpProgramArenaRegion,
    pub work_y: QpProgramArenaRegion,
    pub work_z: QpProgramArenaRegion,
    pub work_xz_tilde: QpProgramArenaRegion,
    pub work_x_prev: QpProgramArenaRegion,
    pub work_z_prev: QpProgramArenaRegion,
    pub work_ax: QpProgramArenaRegion,
    pub work_px: QpProgramArenaRegion,
    pub work_aty: QpProgramArenaRegion,
    pub work_delta_y: QpProgramArenaRegion,
    pub work_atdelta_y: QpProgramArenaRegion,
    pub work_delta_x: QpProgramArenaRegion,
    pub work_pdelta_x: QpProgramArenaRegion,
    pub work_adelta_x: QpProgramArenaRegion,
    pub workspace: QpProgramArenaRegion,
}

/// Embedded QDLDL metadata for a validated QP program plan.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct QpProgramQdldlPlan {
    pub p_pattern: EmbeddedCscPattern,
    pub a_pattern: EmbeddedCscPattern,
    pub kkt_pattern: EmbeddedCscPattern,
    pub p_diag_indices: Vec<u32>,
    pub kkt_permutation: Vec<u32>,
    pub p_to_kkt: Vec<u32>,
    pub a_to_kkt: Vec<u32>,
    pub rho_to_kkt: Vec<u32>,
    pub symbolic_l: QdldlSymbolicL,
}

/// Pointer-free embedded execution plan stored inside a mapped QP program.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct QpProgramPlan {
    pub abi_version: u16,
    pub profile: EmbeddedQpProfile,
    pub version: u16,
    pub settings: EmbeddedOsqpSettings,
    pub arena_layout: QpProgramArenaLayout,
    pub qdldl_plan: QpProgramQdldlPlan,
}

/// Compact embedded QDLDL metadata stored in the standalone solver-plan payload.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct EmbeddedQdldlPlan {
    pub p_pattern: EmbeddedCscPattern,
    pub a_pattern: EmbeddedCscPattern,
    pub kkt_pattern: EmbeddedCscPattern,
    pub p_diag_indices: Vec<u32>,
    pub kkt_permutation: Vec<u32>,
    pub p_to_kkt: Vec<u32>,
    pub a_to_kkt: Vec<u32>,
    pub rho_to_kkt: Vec<u32>,
}

/// Pointer-free solver-plan payload for the fixed OSQP 0.6.3 `EMBEDDED=2`
/// + QDLDL profile.
///
/// Embedded runtimes must refuse to bind a QP archive unless this separately
/// encoded plan payload is present and validates against the archive structure.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct EmbeddedQpPlan {
    pub profile: EmbeddedQpProfile,
    pub version: u16,
    pub settings: EmbeddedOsqpSettings,
    pub qdldl_plan: EmbeddedQdldlPlan,
}

/// Validates matrix dimensions, row pointers, and index bounds.
impl EmbeddedCscPattern {
    pub fn validate(&self, field: &'static str) -> Result<(), BytecodeError> {
        validate_embedded_csc_pattern(self.nrows, self.ncols, &self.indptr, &self.indices, field)
    }
}

/// Validates embedded OSQP scalar settings.
impl EmbeddedOsqpSettings {
    pub fn validate(&self) -> Result<(), BytecodeError> {
        validate_embedded_osqp_settings(self)
    }
}

/// Validates the symbolic factorization against the KKT sparsity pattern.
impl QdldlSymbolicL {
    pub fn validate(&self, kkt_pattern: &EmbeddedCscPattern) -> Result<(), BytecodeError> {
        validate_qdldl_symbolic_l(self, kkt_pattern)
    }
}

/// Validates one embedded arena region.
impl QpProgramArenaRegion {
    pub fn validate(&self, field: &'static str) -> Result<(), BytecodeError> {
        validate_qp_program_arena_region(self, field)
    }
}

/// Validates the full embedded OSQP arena layout.
impl QpProgramArenaLayout {
    pub fn validate(&self) -> Result<(), BytecodeError> {
        validate_qp_program_arena_layout(self)
    }
}

/// Validates the embedded QDLDL plan for a mapped QP archive.
impl QpProgramQdldlPlan {
    pub fn validate(&self) -> Result<(), BytecodeError> {
        validate_qp_program_qdldl_plan(self)
    }
}

/// ABI version expected by mapped embedded runtimes.
impl QpProgramPlan {
    /// Embedded solver profile expected by mapped embedded runtimes.
    pub const ABI_VERSION: u16 = 1;
    /// Plan encoding version expected by mapped embedded runtimes.
    pub const PROFILE: EmbeddedQpProfile = EmbeddedQpProfile::Osqp063Embedded2Qdldl;
    pub const VERSION: u16 = EMBEDDED_QP_PLAN_VERSION;

    /// Validates the embedded plan header, settings, layout, and QDLDL metadata.
    pub fn validate(&self) -> Result<(), BytecodeError> {
        if self.abi_version != Self::ABI_VERSION {
            return Err(BytecodeError::Decode(format!(
                "unsupported embedded QP plan abi version: expected {}, found {}",
                Self::ABI_VERSION,
                self.abi_version
            )));
        }
        if self.profile != Self::PROFILE {
            return Err(BytecodeError::Decode(format!(
                "unsupported embedded QP plan profile: expected {:?}, found {:?}",
                Self::PROFILE,
                self.profile
            )));
        }
        if self.version != Self::VERSION {
            return Err(BytecodeError::Decode(format!(
                "unsupported embedded QP plan version: expected {}, found {}",
                Self::VERSION,
                self.version
            )));
        }
        self.settings.validate()?;
        self.arena_layout.validate()?;
        self.qdldl_plan.validate()?;
        Ok(())
    }
}

/// Validates the standalone embedded QDLDL payload.
impl EmbeddedQdldlPlan {
    pub fn validate(&self) -> Result<(), BytecodeError> {
        validate_embedded_qp_plan_dimensions_impl(
            self.p_pattern.nrows,
            self.p_pattern.ncols,
            &self.p_pattern.indptr,
            &self.p_pattern.indices,
            self.a_pattern.nrows,
            self.a_pattern.ncols,
            &self.a_pattern.indptr,
            &self.a_pattern.indices,
            self.kkt_pattern.nrows,
            self.kkt_pattern.ncols,
            &self.kkt_pattern.indptr,
            &self.kkt_pattern.indices,
            &self.p_diag_indices,
            &self.kkt_permutation,
            &self.p_to_kkt,
            &self.a_to_kkt,
            &self.rho_to_kkt,
        )
    }
}

/// Embedded solver profile expected by the standalone plan payload.
impl EmbeddedQpPlan {
    /// Standalone embedded plan encoding version.
    pub const PROFILE: EmbeddedQpProfile = EmbeddedQpProfile::Osqp063Embedded2Qdldl;
    pub const VERSION: u16 = EMBEDDED_QP_PLAN_VERSION;

    /// Validates the standalone embedded plan payload.
    pub fn validate(&self) -> Result<(), BytecodeError> {
        if self.profile != Self::PROFILE {
            return Err(BytecodeError::Decode(format!(
                "unsupported embedded QP plan profile: expected {:?}, found {:?}",
                Self::PROFILE,
                self.profile
            )));
        }
        if self.version != Self::VERSION {
            return Err(BytecodeError::Decode(format!(
                "unsupported embedded QP plan version: expected {}, found {}",
                Self::VERSION,
                self.version
            )));
        }
        self.settings.validate()?;
        self.qdldl_plan.validate()?;
        Ok(())
    }
}

/// Fixed-size mapped-bytecode header stored before every archived payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct BytecodeHeader {
    /// Payload alignment in bytes.
    ///
    /// This field is encoded as a little-endian `u16`. The mapped-bytecode
    /// validation step must reject zero, non-power-of-two, and mismatched
    /// values before the payload is accessed.
    pub payload_alignment: u16,
}

impl BytecodeHeader {
    fn write_into(self, bytes: &mut Vec<u8>, magic: &[u8; 8], version: u16) {
        bytes.extend_from_slice(magic);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&self.payload_alignment.to_le_bytes());
        bytes.resize(HEADER_SIZE, 0);
    }
}
fn payload_start_offset(payload_alignment: usize) -> usize {
    let payload_alignment = payload_alignment.max(1);
    let padding = (payload_alignment - HEADER_SIZE % payload_alignment) % payload_alignment;
    HEADER_SIZE + padding
}

fn version_from_header(bytes: &[u8], magic: &[u8; 8]) -> u16 {
    let version_bytes: [u8; 2] = bytes[magic.len()..magic.len() + 2]
        .try_into()
        .expect("header size includes version bytes");
    u16::from_le_bytes(version_bytes)
}

#[cfg(feature = "std")]
fn payload_alignment_from_header(bytes: &[u8], magic: &[u8; 8]) -> Result<usize, BytecodeError> {
    let alignment_bytes: [u8; 2] = bytes[magic.len() + 2..magic.len() + 4]
        .try_into()
        .expect("header size includes payload alignment bytes");
    Ok(u16::from_le_bytes(alignment_bytes) as usize)
}

#[cfg(feature = "std")]
fn decode_archived_with_legacy_fallback<T, F>(
    bytes: &[u8],
    payload_start: usize,
    decode: F,
) -> Result<T, BytecodeError>
where
    F: Fn(&[u8]) -> Result<T, BytecodeError>,
{
    let payload = match bytes.get(payload_start..) {
        Some(payload) => payload,
        None if payload_start != HEADER_SIZE => &bytes[HEADER_SIZE..],
        None => {
            return Err(BytecodeError::Decode(
                "bytecode archive payload too short".to_string(),
            ))
        }
    };

    match decode(payload) {
        Ok(value) => Ok(value),
        Err(error) if payload_start != HEADER_SIZE => decode(&bytes[HEADER_SIZE..]).or(Err(error)),
        Err(error) => Err(error),
    }
}

/// Alignment required for directly mapping an archived [`BytecodeModule`].
pub const ARCHIVED_MODULE_ALIGNMENT: usize = align_of::<ArchivedBytecodeModule>();

/// Fixed-alignment byte wrapper for embedding archived modules in static storage.
#[repr(C, align(16))]
pub struct AlignedModuleBytes<const N: usize> {
    bytes: [u8; N],
}

impl<const N: usize> AlignedModuleBytes<N> {
    /// Wraps already-aligned bytes for direct archived access.
    pub const fn new(bytes: [u8; N]) -> Self {
        Self { bytes }
    }

    /// Returns the underlying byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl<const N: usize> AsRef<[u8]> for AlignedModuleBytes<N> {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// Top-level bytecode module containing ordinary and QP-backed executables.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct BytecodeModule {
    pub executables: Vec<Executable>,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
struct SplitBytecodeModule {
    functions: Vec<SplitProgram>,
    qp_programs: Vec<SplitQpProgram>,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
struct LegacyBytecodeModule {
    functions: Vec<SplitProgram>,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
struct SplitProgram {
    function_id: u16,
    workspace_size: u32,
    required_workspace_size: u32,
    input_specs: Vec<InputSpec>,
    output_specs: Vec<OutputSpec>,
    intermediate_layers: Vec<Layer>,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
struct SplitQpProgram {
    function_id: u16,
    coefficient_function_id: u16,
    required_primal_workspace_size: u32,
    required_tangent_workspace_size: u32,
    input_specs: Vec<InputSpec>,
    output_spec: OutputSpec,
    p_pattern: EmbeddedCscPattern,
    a_pattern: EmbeddedCscPattern,
    coefficient_outputs: QpCoefficientOutputs,
    embedded_plan: QpProgramPlan,
}

/// One owned executable stored in a bytecode module.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub enum Executable {
    Program(Program),
    QpProgram(QpProgram),
}

/// Borrowed view of either a plain program or a QP program in an owned module.
#[derive(Clone, Copy)]
pub enum ExecutableRef<'a> {
    Program(u16, &'a Program),
    QpProgram(u16, &'a QpProgram),
}

/// Borrowed view of either a plain program or a QP program in an archived module.
#[derive(Clone, Copy)]
pub enum ArchivedExecutableRef<'a> {
    Program(u16, &'a ArchivedProgram),
    QpProgram(u16, &'a ArchivedQpProgram),
}

impl BytecodeModule {
    /// Builds a module containing only ordinary programs.
    pub fn new(functions: Vec<Program>) -> Self {
        Self {
            executables: functions.into_iter().map(Executable::Program).collect(),
        }
    }

    /// Builds a module by appending ordinary programs first, then QP programs.
    pub fn with_qp_programs(functions: Vec<Program>, qp_programs: Vec<QpProgram>) -> Self {
        let mut executables = Vec::with_capacity(functions.len() + qp_programs.len());
        executables.extend(functions.into_iter().map(Executable::Program));
        executables.extend(qp_programs.into_iter().map(Executable::QpProgram));
        Self { executables }
    }

    /// Builds a module from an explicit executable ordering.
    pub fn from_executables(executables: Vec<Executable>) -> Self {
        Self { executables }
    }

    /// Iterates all executables with their index-derived ids.
    pub fn executables(&self) -> impl Iterator<Item = ExecutableRef<'_>> {
        self.executables
            .iter()
            .enumerate()
            .map(|(function_id, executable)| match executable {
                Executable::Program(program) => ExecutableRef::Program(function_id as u16, program),
                Executable::QpProgram(program) => {
                    ExecutableRef::QpProgram(function_id as u16, program)
                }
            })
    }

    /// Iterates all ordinary programs with their index-derived ids.
    pub fn functions(&self) -> impl Iterator<Item = (u16, &Program)> {
        self.executables
            .iter()
            .enumerate()
            .filter_map(|(function_id, executable)| match executable {
                Executable::Program(program) => Some((function_id as u16, program)),
                Executable::QpProgram(_) => None,
            })
    }

    /// Iterates all QP-backed programs with their index-derived ids.
    pub fn qp_programs(&self) -> impl Iterator<Item = (u16, &QpProgram)> {
        self.executables
            .iter()
            .enumerate()
            .filter_map(|(function_id, executable)| match executable {
                Executable::Program(_) => None,
                Executable::QpProgram(program) => Some((function_id as u16, program)),
            })
    }

    /// Returns the ordinary program for `function_id`, if present.
    pub fn program(&self, function_id: u16) -> Option<&Program> {
        match self.executables.get(function_id as usize)? {
            Executable::Program(program) => Some(program),
            Executable::QpProgram(_) => None,
        }
    }

    /// Returns the mutable ordinary program for `function_id`, if present.
    pub fn program_mut(&mut self, function_id: u16) -> Option<&mut Program> {
        match self.executables.get_mut(function_id as usize)? {
            Executable::Program(program) => Some(program),
            Executable::QpProgram(_) => None,
        }
    }

    /// Returns the QP program for `function_id`, if present.
    pub fn qp_program(&self, function_id: u16) -> Option<&QpProgram> {
        match self.executables.get(function_id as usize)? {
            Executable::Program(_) => None,
            Executable::QpProgram(program) => Some(program),
        }
    }

    /// Returns the mutable QP program for `function_id`, if present.
    pub fn qp_program_mut(&mut self, function_id: u16) -> Option<&mut QpProgram> {
        match self.executables.get_mut(function_id as usize)? {
            Executable::Program(_) => None,
            Executable::QpProgram(program) => Some(program),
        }
    }

    /// Returns either executable kind for `function_id`.
    pub fn executable(&self, function_id: u16) -> Option<ExecutableRef<'_>> {
        match self.executables.get(function_id as usize)? {
            Executable::Program(program) => Some(ExecutableRef::Program(function_id, program)),
            Executable::QpProgram(program) => Some(ExecutableRef::QpProgram(function_id, program)),
        }
    }

    /// Returns the conventional entry program (`function_id == 0`), if present.
    pub fn entry_program(&self) -> Option<&Program> {
        self.program(0)
    }

    /// Validates module-level structural and cross-reference invariants.
    pub fn validate_semantics(&self) -> Result<(), BytecodeError> {
        validate_bytecode_module_semantics(self)
    }
}

impl ExecutableRef<'_> {
    /// Returns the index-derived id of the referenced executable.
    pub fn function_id(&self) -> u16 {
        match self {
            Self::Program(function_id, _) | Self::QpProgram(function_id, _) => *function_id,
        }
    }
}

/// Plain bytecode program with explicit workspace, inputs, outputs, and layers.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct Program {
    pub workspace_size: u32,
    pub required_workspace_size: u32,
    pub input_specs: Vec<InputSpec>,
    pub output_specs: Vec<OutputSpec>,
    pub intermediate_layers: Vec<Layer>,
}

/// QP-backed bytecode program with a coefficient evaluator and embedded solver plan.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct QpProgram {
    pub coefficient_function_id: u16,
    pub required_primal_workspace_size: u32,
    pub required_tangent_workspace_size: u32,
    pub input_specs: Vec<InputSpec>,
    pub output_spec: OutputSpec,
    pub p_pattern: EmbeddedCscPattern,
    pub a_pattern: EmbeddedCscPattern,
    pub coefficient_outputs: QpCoefficientOutputs,
    pub embedded_plan: QpProgramPlan,
}

impl Program {
    /// Constructs a plain bytecode program from already-lowered components.
    pub fn new(
        workspace_size: u32,
        required_workspace_size: u32,
        input_specs: Vec<InputSpec>,
        output_specs: Vec<OutputSpec>,
        intermediate_layers: Vec<Layer>,
    ) -> Self {
        Self {
            workspace_size,
            required_workspace_size,
            input_specs,
            output_specs,
            intermediate_layers,
        }
    }

    /// Computes the flat input width implied by `input_specs`.
    pub fn checked_flat_input_size(&self) -> Result<u32, BytecodeError> {
        checked_flat_input_specs(&self.input_specs, "program input specs")
    }

    /// Computes the flat output width implied by `output_specs`.
    pub fn checked_flat_output_size(&self) -> Result<u32, BytecodeError> {
        checked_flat_output_specs(&self.output_specs, "program output specs")
    }
}

impl QpProgram {
    /// Constructs a QP-backed bytecode program from already-lowered components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coefficient_function_id: u16,
        required_primal_workspace_size: u32,
        required_tangent_workspace_size: u32,
        input_specs: Vec<InputSpec>,
        output_spec: OutputSpec,
        p_pattern: EmbeddedCscPattern,
        a_pattern: EmbeddedCscPattern,
        coefficient_outputs: QpCoefficientOutputs,
        embedded_plan: QpProgramPlan,
    ) -> Self {
        Self {
            coefficient_function_id,
            required_primal_workspace_size,
            required_tangent_workspace_size,
            input_specs,
            output_spec,
            p_pattern,
            a_pattern,
            coefficient_outputs,
            embedded_plan,
        }
    }

    /// Returns the QP parameter input specs.
    pub fn input_specs(&self) -> &[InputSpec] {
        &self.input_specs
    }

    /// Returns the primal solution output spec.
    pub fn output_spec(&self) -> &OutputSpec {
        &self.output_spec
    }

    /// Computes the flat parameter input width implied by `input_specs`.
    pub fn checked_flat_input_size(&self) -> Result<u32, BytecodeError> {
        checked_flat_input_specs(&self.input_specs, "QP input specs")
    }

    /// Returns the flat primal output width.
    pub fn checked_flat_output_size(&self) -> u32 {
        u32::from(self.output_spec.length)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct InputSpec {
    pub workspace_offset: u32,
    pub length: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct OutputSpec {
    pub workspace_offset: u32,
    pub length: u16,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub enum Layer {
    Bilinear(BilinearLayer),
    Generic(GenericLayer),
    Evaluate(EvaluateLayer),
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct BilinearLayer {
    pub in_offset: u32,
    pub out_offset: u32,
    pub in_length: u16,
    pub out_length: u16,
    pub scratch_offset: u32,
    pub scratch_length: u16,
    pub quadratic: SparseTensor,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct GenericLayer {
    pub in_offset: u32,
    pub out_offset: u32,
    pub in_length: u16,
    pub out_length: u16,
    pub scratch_offset: u32,
    pub scratch_length: u16,
    pub ops: Vec<RowOp>,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct EvaluateLayer {
    pub scratch_offset: u32,
    pub callee_function_id: u16,
    pub input_bindings: Vec<EvaluateInputBinding>,
    pub output_bindings: Vec<EvaluateOutputBinding>,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub enum EvaluateInputBinding {
    WorkspaceSlice { offset: u32, length: u16 },
    ConstantSlice { length: u16, values: Vec<f32> },
}

#[derive(Debug, Clone, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct EvaluateOutputBinding {
    pub destination_offset: u32,
    pub length: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum ScalarOp {
    Identity,
    Sin,
    Cos,
    Tan,
    Exp,
    Sqrt,
    Log,
    Neg,
    Abs,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    IntPow,
    Atan2,
    Equal,
    LessThan,
    LessEqual,
    Case,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct RowOp {
    pub first: u16,
    pub second: u16,
    pub third: u16,
    pub op: ScalarOp,
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct SparseTensor {
    pub shape: (u16, u16, u16),
    pub entries: Vec<SparseEntry>,
}

impl SparseTensor {
    pub fn try_from_row_major_array<const ROW_COUNT: usize, const COLUMN_COUNT: usize>(
        data: &[[f32; ROW_COUNT]; COLUMN_COUNT],
    ) -> Result<Self, BytecodeError> {
        if ROW_COUNT >= u16::MAX as usize || COLUMN_COUNT >= u16::MAX as usize {
            return Err(BytecodeError::Encode("array too large".into()));
        }

        let mut entries = Vec::new();
        for (column_index, column) in data.iter().enumerate() {
            for (row_index, value) in column.iter().enumerate() {
                entries.push(SparseEntry {
                    index: (row_index as u16, column_index as u16, u16::MAX),
                    value: *value,
                });
            }
        }

        Ok(Self {
            shape: (ROW_COUNT as u16, COLUMN_COUNT as u16, 0),
            entries,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct SparseEntry {
    pub index: (u16, u16, u16),
    pub value: f32,
}

/// Errors raised while validating a mapped bytecode header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BytecodeHeaderError {
    #[error("bytecode header too short")]
    TooShort,
    #[error("bytecode magic mismatch")]
    MagicMismatch,
    #[error("unsupported bytecode version: {found}")]
    UnsupportedVersion { found: u16 },
    #[error("invalid bytecode payload alignment: {actual}")]
    InvalidPayloadAlignment { actual: u16 },
    #[error("bytecode payload alignment too small: minimum {minimum}, got {actual}")]
    PayloadAlignmentTooSmall { minimum: usize, actual: usize },
    #[error(
        "bytecode archive payload too short: expected at least {expected} bytes, got {actual}"
    )]
    PayloadTooShort { expected: usize, actual: usize },
    #[error("bytecode archive payload must be {expected}-byte aligned, got remainder {actual}")]
    PayloadMisaligned { expected: usize, actual: usize },
}

/// Errors raised while encoding, decoding, or validating bytecode payloads.
#[derive(Debug, Error)]
pub enum BytecodeError {
    #[error("failed to encode bytecode module: {0}")]
    Encode(String),
    #[error("failed to decode bytecode module: {0}")]
    Decode(String),
    #[error("invalid bytecode header: {0}")]
    Header(#[from] BytecodeHeaderError),
}

mod archived;
mod codec;
mod convert;
mod validate;

pub use self::{
    archived::{archived_embedded_qp_plan, archived_module, archived_qp_program},
    codec::{decode_embedded_qp_plan, encode_embedded_qp_plan, encode_module, encode_qp_program},
};
#[cfg(feature = "std")]
pub use self::codec::{decode_module, decode_qp_program};

#[allow(unused_imports)]
use self::{codec::*, convert::*, validate::*};

#[cfg(test)]
mod tests;
