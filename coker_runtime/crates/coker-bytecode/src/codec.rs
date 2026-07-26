use super::*;
use core::mem::align_of;
use rkyv::{rancor::Error as RkyvError, to_bytes};
#[cfg(feature = "std")]
use rkyv::{access, util::AlignedVec};
pub(crate) fn archive_payload<'a>(
    bytes: &'a [u8],
    magic: &[u8; 8],
    version: u16,
    archive_alignment: usize,
) -> Result<&'a [u8], BytecodeError> {
    let payload_start = validate_mapped_header(bytes, magic, version, archive_alignment)?;
    Ok(&bytes[payload_start..])
}

pub(crate) fn validate_mapped_header(
    bytes: &[u8],
    magic: &[u8; 8],
    version: u16,
    archive_alignment: usize,
) -> Result<usize, BytecodeHeaderError> {
    if bytes.len() < HEADER_SIZE {
        return Err(BytecodeHeaderError::TooShort);
    }
    if bytes[..magic.len()] != *magic {
        return Err(BytecodeHeaderError::MagicMismatch);
    }
    let found_version = version_from_header(bytes, magic);
    if found_version != version {
        return Err(BytecodeHeaderError::UnsupportedVersion {
            found: found_version,
        });
    }

    let alignment_bytes: [u8; 2] = bytes[magic.len() + 2..magic.len() + 4]
        .try_into()
        .expect("header size includes payload alignment bytes");
    let payload_alignment = u16::from_le_bytes(alignment_bytes);
    if payload_alignment == 0 || !payload_alignment.is_power_of_two() {
        return Err(BytecodeHeaderError::InvalidPayloadAlignment {
            actual: payload_alignment,
        });
    }

    let payload_alignment = payload_alignment as usize;
    if payload_alignment < archive_alignment {
        return Err(BytecodeHeaderError::PayloadAlignmentTooSmall {
            minimum: archive_alignment,
            actual: payload_alignment,
        });
    }

    let payload_start = payload_start_offset(payload_alignment);
    if bytes.len() < payload_start {
        return Err(BytecodeHeaderError::PayloadTooShort {
            expected: payload_start,
            actual: bytes.len(),
        });
    }

    let payload = &bytes[payload_start..];
    let actual_alignment = payload.as_ptr() as usize % archive_alignment;
    if actual_alignment != 0 {
        return Err(BytecodeHeaderError::PayloadMisaligned {
            expected: archive_alignment,
            actual: actual_alignment,
        });
    }

    Ok(payload_start)
}

#[cfg(feature = "std")]
pub(crate) fn validate_header(bytes: &[u8]) -> Result<(), BytecodeError> {
    validate_mapped_header(bytes, &MAGIC, VERSION, align_of::<ArchivedBytecodeModule>())?;
    Ok(())
}

/// Serializes a bytecode module with an aligned mapped-archive header.
pub fn encode_module(module: &BytecodeModule) -> Result<Vec<u8>, BytecodeError> {
    let archived_module =
        to_bytes::<RkyvError>(module).map_err(|error| BytecodeError::Encode(error.to_string()))?;
    let payload_offset = payload_start_offset(ARCHIVED_MODULE_ALIGNMENT);
    let mut bytes = Vec::with_capacity(payload_offset + archived_module.len());
    BytecodeHeader {
        payload_alignment: ARCHIVED_MODULE_ALIGNMENT as u16,
    }
    .write_into(&mut bytes, &MAGIC, VERSION);
    bytes.resize(payload_offset, 0);
    bytes.extend_from_slice(archived_module.as_slice());
    Ok(bytes)
}

#[cfg(feature = "std")]
/// Decodes a serialized bytecode module into an owned representation.
pub fn decode_module(bytes: &[u8]) -> Result<BytecodeModule, BytecodeError> {
    validate_header(bytes)?;
    let payload_start = payload_start_offset(payload_alignment_from_header(bytes, &MAGIC)?);
    decode_archived_with_legacy_fallback(bytes, payload_start, |payload| {
        let mut aligned_bytes: AlignedVec<{ ARCHIVED_MODULE_ALIGNMENT }> =
            AlignedVec::with_capacity(payload.len());
        aligned_bytes.extend_from_slice(payload);
        match access::<ArchivedBytecodeModule, RkyvError>(aligned_bytes.as_slice()) {
            Ok(archived) => {
                let module = module_from_archived(archived);
                module.validate_semantics()?;
                Ok(module)
            }
            Err(module_error) => {
                if let Ok(archived) =
                    access::<ArchivedSplitBytecodeModule, RkyvError>(aligned_bytes.as_slice())
                {
                    let module = module_from_split_archived(archived);
                    module.validate_semantics()?;
                    return Ok(module);
                }
                let archived =
                    access::<ArchivedLegacyBytecodeModule, RkyvError>(aligned_bytes.as_slice())
                        .map_err(|_| BytecodeError::Decode(module_error.to_string()))?;
                let module = module_from_legacy_archived(archived);
                module.validate_semantics()?;
                Ok(module)
            }
        }
    })
}

/// Serializes a host QP payload with its aligned mapped-archive header.
pub fn encode_qp_program(program: &QpProgramArchive) -> Result<Vec<u8>, BytecodeError> {
    let archived =
        to_bytes::<RkyvError>(program).map_err(|error| BytecodeError::Encode(error.to_string()))?;
    let payload_offset = payload_start_offset(align_of::<ArchivedQpProgramArchive>());
    let mut bytes = Vec::with_capacity(payload_offset + archived.len());
    BytecodeHeader {
        payload_alignment: align_of::<ArchivedQpProgramArchive>() as u16,
    }
    .write_into(&mut bytes, &QP_MAGIC, VERSION);
    bytes.resize(payload_offset, 0);
    bytes.extend_from_slice(archived.as_slice());
    Ok(bytes)
}

#[cfg(feature = "std")]
/// Decodes a serialized host QP payload into an owned representation.
pub fn decode_qp_program(bytes: &[u8]) -> Result<QpProgramArchive, BytecodeError> {
    let archived = archived_qp_program(bytes)?;
    Ok(QpProgramArchive {
        n: archived.n.into(),
        m: archived.m.into(),
        parameter_lengths: archived
            .parameter_lengths
            .iter()
            .map(|value| value.to_native())
            .collect(),
        decision_input_indices: archived
            .decision_input_indices
            .iter()
            .map(|value| value.to_native())
            .collect(),
        constraint_row_offsets: archived
            .constraint_row_offsets
            .iter()
            .map(|value| value.to_native())
            .collect(),
        p_indptr: archived
            .p_indptr
            .iter()
            .map(|value| value.to_native())
            .collect(),
        p_indices: archived
            .p_indices
            .iter()
            .map(|value| value.to_native())
            .collect(),
        a_indptr: archived
            .a_indptr
            .iter()
            .map(|value| value.to_native())
            .collect(),
        a_indices: archived
            .a_indices
            .iter()
            .map(|value| value.to_native())
            .collect(),
        coefficient_program: module_from_archived(&archived.coefficient_program),
        coefficient_outputs: qp_coefficient_outputs_from_archived(&archived.coefficient_outputs),
        warm_start: archived.warm_start,
    })
}

/// Serializes a standalone embedded QP plan with its aligned mapped-archive header.
pub fn encode_embedded_qp_plan(plan: &EmbeddedQpPlan) -> Result<Vec<u8>, BytecodeError> {
    plan.validate()?;
    let archived =
        to_bytes::<RkyvError>(plan).map_err(|error| BytecodeError::Encode(error.to_string()))?;
    let payload_offset = payload_start_offset(align_of::<ArchivedEmbeddedQpPlan>());
    let mut bytes = Vec::with_capacity(payload_offset + archived.len());
    BytecodeHeader {
        payload_alignment: align_of::<ArchivedEmbeddedQpPlan>() as u16,
    }
    .write_into(&mut bytes, &EMBEDDED_QP_PLAN_MAGIC, EmbeddedQpPlan::VERSION);
    bytes.resize(payload_offset, 0);
    bytes.extend_from_slice(archived.as_slice());
    Ok(bytes)
}

/// Decodes a serialized standalone embedded QP plan into an owned representation.
pub fn decode_embedded_qp_plan(bytes: &[u8]) -> Result<EmbeddedQpPlan, BytecodeError> {
    let archived = archived_embedded_qp_plan(bytes)?;
    Ok(embedded_qp_plan_from_archived(archived))
}
