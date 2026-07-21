use serde_json::Value;

use crate::{CompileError, UNUSED_OPERAND};

pub(crate) fn checked_u8_length(
    value: usize,
    field_name: &'static str,
) -> Result<u8, CompileError> {
    u8::try_from(value).map_err(|_| CompileError::InvalidField {
        field: field_name,
        reason: "expected u8-sized collection",
    })
}

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

pub(crate) fn operator_kind(operator_value: &Value) -> Result<String, CompileError> {
    object_field(operator_value, "kind")?
        .as_str()
        .map(str::to_string)
        .ok_or(CompileError::InvalidField {
            field: "op.kind",
            reason: "expected string",
        })
}

pub(crate) fn operator_name(operator_value: &Value) -> Result<String, CompileError> {
    object_field(operator_value, "value")?
        .as_str()
        .map(str::to_string)
        .ok_or(CompileError::InvalidField {
            field: "op.value",
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
