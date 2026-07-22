use coker_bytecode::{
    BilinearLayer, EvaluateInputBinding, EvaluateOutputBinding, GenericLayer, RowOp, ScalarOp,
    SparseEntry, SparseTensor,
};

use crate::{
    model::{
        ExportedEvaluateInputBinding, ExportedEvaluateOutputBinding, ExportedLayer, ExportedRowOp,
        ExportedSparseEntry, ExportedSparseTensor,
    },
    util::{
        checked_add_u32, checked_u16, compile_operand_index, operator_kind, operator_name,
        required_field,
    },
    CompileError,
};

pub(crate) fn compile_bilinear_layer(
    exported_layer: ExportedLayer,
    required_workspace_size: &mut u32,
) -> Result<BilinearLayer, CompileError> {
    let memory_in = required_field(exported_layer.memory_in, "memory_in")?;
    let memory_out = required_field(exported_layer.memory_out, "memory_out")?;
    let exported_weights = required_field(exported_layer.weights, "weights")?;
    let (scratch_offset, scratch_length) = compile_layer_scratch(
        memory_in.location,
        memory_in.count,
        memory_out.location,
        memory_out.count,
        required_workspace_size,
    )?;

    Ok(BilinearLayer {
        in_offset: memory_in.location,
        out_offset: memory_out.location,
        in_length: checked_u16(memory_in.count, "memory_in.count")?,
        out_length: checked_u16(memory_out.count, "memory_out.count")?,
        scratch_offset,
        scratch_length,
        quadratic: compile_sparse_tensor(exported_weights.quadratic)?,
    })
}

pub(crate) fn compile_generic_layer(
    exported_layer: ExportedLayer,
    required_workspace_size: &mut u32,
) -> Result<GenericLayer, CompileError> {
    let memory_in = required_field(exported_layer.memory_in, "memory_in")?;
    let memory_out = required_field(exported_layer.memory_out, "memory_out")?;

    if exported_layer
        .constants
        .as_ref()
        .is_some_and(|constants| !constants.is_empty())
    {
        return Err(CompileError::NotImplemented(
            "generic constants must be folded into bilinear layers".to_string(),
        ));
    }

    if exported_layer
        .opaque_programs
        .as_ref()
        .is_some_and(|opaque_programs| !opaque_programs.is_empty())
    {
        return Err(CompileError::NotImplemented(
            "function evaluation and opaque programs".to_string(),
        ));
    }

    let ops = required_field(exported_layer.ops, "ops")?
        .into_iter()
        .map(compile_row_op)
        .collect::<Result<Vec<_>, _>>()?;
    let (scratch_offset, scratch_length) = compile_layer_scratch(
        memory_in.location,
        memory_in.count,
        memory_out.location,
        memory_out.count,
        required_workspace_size,
    )?;

    Ok(GenericLayer {
        in_offset: memory_in.location,
        out_offset: memory_out.location,
        in_length: checked_u16(memory_in.count, "memory_in.count")?,
        out_length: checked_u16(memory_out.count, "memory_out.count")?,
        scratch_offset,
        scratch_length,
        ops,
    })
}

fn compile_layer_scratch(
    input_offset: u32,
    input_count: u32,
    output_offset: u32,
    output_count: u32,
    required_workspace_size: &mut u32,
) -> Result<(u32, u16), CompileError> {
    if !ranges_overlap(input_offset, input_count, output_offset, output_count)? {
        return Ok((0, 0));
    }

    let scratch_offset = *required_workspace_size;
    *required_workspace_size = checked_add_u32(scratch_offset, input_count, "scratch_offset")?;
    Ok((scratch_offset, checked_u16(input_count, "memory_in.count")?))
}

fn ranges_overlap(
    first_offset: u32,
    first_count: u32,
    second_offset: u32,
    second_count: u32,
) -> Result<bool, CompileError> {
    let first_end = checked_add_u32(first_offset, first_count, "memory_in.location")?;
    let second_end = checked_add_u32(second_offset, second_count, "memory_out.location")?;
    Ok(first_end > second_offset && second_end > first_offset)
}

pub(crate) fn compile_evaluate_input_binding(
    exported_binding: ExportedEvaluateInputBinding,
    expected_length: u16,
) -> Result<EvaluateInputBinding, CompileError> {
    match exported_binding {
        ExportedEvaluateInputBinding::Workspace { offset, length } => {
            let compiled_length = checked_u16(length, "inputs.length")?;
            if compiled_length != expected_length {
                return Err(CompileError::InvalidField {
                    field: "inputs.length",
                    reason: "evaluate input length does not match callee input",
                });
            }
            Ok(EvaluateInputBinding::WorkspaceSlice {
                offset,
                length: compiled_length,
            })
        }
        ExportedEvaluateInputBinding::Constant { length, values } => {
            let compiled_length = checked_u16(length, "inputs.length")?;
            if compiled_length != expected_length {
                return Err(CompileError::InvalidField {
                    field: "inputs.length",
                    reason: "evaluate input length does not match callee input",
                });
            }
            if values.len() != compiled_length as usize {
                return Err(CompileError::InvalidField {
                    field: "inputs.values",
                    reason: "evaluate constant input length does not match values",
                });
            }
            Ok(EvaluateInputBinding::ConstantSlice {
                length: compiled_length,
                values,
            })
        }
    }
}

pub(crate) fn compile_evaluate_output_binding(
    exported_binding: ExportedEvaluateOutputBinding,
    expected_length: u16,
) -> Result<EvaluateOutputBinding, CompileError> {
    let compiled_length = checked_u16(exported_binding.length, "outputs.length")?;
    if compiled_length != expected_length {
        return Err(CompileError::InvalidField {
            field: "outputs.length",
            reason: "evaluate output length does not match callee output",
        });
    }
    Ok(EvaluateOutputBinding {
        destination_offset: exported_binding.destination_offset,
        length: compiled_length,
    })
}

fn compile_row_op(exported_row_op: ExportedRowOp) -> Result<RowOp, CompileError> {
    let operator_value = exported_row_op.op;
    let operator_kind = operator_kind(&operator_value)?;
    let operator_name = operator_name(&operator_value)?;
    let op = match (operator_kind.as_str(), operator_name.as_str()) {
        ("internal", "identity") => ScalarOp::Identity,
        ("enum", "SIN") => ScalarOp::Sin,
        ("enum", "COS") => ScalarOp::Cos,
        ("enum", "TAN") => ScalarOp::Tan,
        ("enum", "EXP") => ScalarOp::Exp,
        ("enum", "SQRT") => ScalarOp::Sqrt,
        ("enum", "LOG") => ScalarOp::Log,
        ("enum", "NEG") => ScalarOp::Neg,
        ("enum", "ABS") => ScalarOp::Abs,
        ("enum", "ADD") => ScalarOp::Add,
        ("enum", "SUB") => ScalarOp::Sub,
        ("enum", "MUL") => ScalarOp::Mul,
        ("enum", "DIV") => ScalarOp::Div,
        ("enum", "POW") => ScalarOp::Pow,
        ("enum", "INT_POW") => ScalarOp::IntPow,
        ("enum", "ATAN2") | ("enum", "ARCTAN2") => ScalarOp::Atan2,
        ("enum", "EQ") | ("enum", "EQUAL") => ScalarOp::Equal,
        ("enum", "LT") | ("enum", "LESS_THAN") => ScalarOp::LessThan,
        ("enum", "LE") | ("enum", "LESS_EQUAL") => ScalarOp::LessEqual,
        ("enum", "CASE") => ScalarOp::Case,
        unsupported_operation => {
            return Err(CompileError::NotImplemented(format!(
                "operator {unsupported_operation:?}"
            )));
        }
    };

    Ok(RowOp {
        first: compile_operand_index(exported_row_op.first, "first")?,
        second: compile_operand_index(exported_row_op.second, "second")?,
        third: compile_operand_index(exported_row_op.third, "third")?,
        op,
    })
}

fn compile_sparse_tensor(
    exported_tensor: ExportedSparseTensor,
) -> Result<SparseTensor, CompileError> {
    if exported_tensor.shape.len() != 3 {
        return Err(CompileError::InvalidField {
            field: "weights.quadratic.shape",
            reason: "expected rank-3 sparse tensor shape",
        });
    }

    Ok(SparseTensor {
        shape: (
            checked_u16(exported_tensor.shape[0], "weights.quadratic.shape.0")?,
            checked_u16(exported_tensor.shape[1], "weights.quadratic.shape.1")?,
            checked_u16(exported_tensor.shape[2], "weights.quadratic.shape.2")?,
        ),
        entries: exported_tensor
            .entries
            .into_iter()
            .map(compile_sparse_entry)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn compile_sparse_entry(exported_entry: ExportedSparseEntry) -> Result<SparseEntry, CompileError> {
    if exported_entry.index.len() != 3 {
        return Err(CompileError::InvalidField {
            field: "weights.quadratic.entries.index",
            reason: "expected rank-3 sparse tensor index",
        });
    }

    Ok(SparseEntry {
        index: (
            checked_u16(exported_entry.index[0], "weights.quadratic.entries.index.0")?,
            checked_u16(exported_entry.index[1], "weights.quadratic.entries.index.1")?,
            checked_u16(exported_entry.index[2], "weights.quadratic.entries.index.2")?,
        ),
        value: exported_entry.value,
    })
}
