use serde_json::Value;

use crate::{CompileError, UNUSED_OPERAND};

pub(crate) fn checked_u16(value: u32, field_name: &'static str) -> Result<u16, CompileError> {
    u16::try_from(value).map_err(|_| CompileError::InvalidField {
        field: field_name,
        reason: "expected u16",
    })
}

pub(crate) fn checked_add_u32(
    left: u32,
    right: u32,
    field_name: &'static str,
) -> Result<u32, CompileError> {
    left.checked_add(right).ok_or(CompileError::InvalidField {
        field: field_name,
        reason: "u32 overflow",
    })
}

pub(crate) fn required_field<T>(
    field_value: Option<T>,
    field_name: &'static str,
) -> Result<T, CompileError> {
    field_value.ok_or(CompileError::MissingField { field: field_name })
}

pub(crate) fn string_field<'a>(
    value: &'a Value,
    field_name: &'static str,
    error_field: &'static str,
) -> Result<&'a str, CompileError> {
    object_field(value, field_name)?
        .as_str()
        .ok_or(CompileError::InvalidField {
            field: error_field,
            reason: "expected string",
        })
}

pub(crate) fn object_field<'a>(
    value: &'a Value,
    field_name: &'static str,
) -> Result<&'a Value, CompileError> {
    value
        .get(field_name)
        .ok_or(CompileError::MissingField { field: field_name })
}

pub(crate) fn compile_operand_index(
    value: i32,
    field_name: &'static str,
) -> Result<u16, CompileError> {
    if value < 0 {
        return Ok(UNUSED_OPERAND);
    }

    u16::try_from(value).map_err(|_| CompileError::InvalidField {
        field: field_name,
        reason: "expected non-negative u16 or sentinel",
    })
}
