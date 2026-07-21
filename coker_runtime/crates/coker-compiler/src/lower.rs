use coker_bytecode::{
    BilinearLayer, EvaluateInputBinding, EvaluateOutputBinding, GenericLayer, RowOp, ScalarOp,
    SparseEntry, SparseTensor,
};
use serde_json::Value;

use crate::{
    model::{
        ExportedEvaluateInputBinding, ExportedEvaluateOutputBinding, ExportedLayer,
        ExportedRowOp, ExportedSparseEntry, ExportedSparseTensor,
    },
    util::{checked_u16, compile_operand_index, operator_kind, operator_name, required_field},
    CompileError,
};

pub(crate) fn compile_bilinear_layer(
    exported_layer: ExportedLayer,
) -> Result<BilinearLayer, CompileError> {
    let memory_in = required_field(exported_layer.memory_in, "memory_in")?;
    let memory_out = required_field(exported_layer.memory_out, "memory_out")?;
    let exported_weights = required_field(exported_layer.weights, "weights")?;

    Ok(BilinearLayer {
        in_offset: memory_in.location,
        out_offset: memory_out.location,
        in_length: checked_u16(memory_in.count, "memory_in.count")?,
        out_length: checked_u16(memory_out.count, "memory_out.count")?,
        quadratic: compile_sparse_tensor(exported_weights.quadratic)?,
    })
}

pub(crate) fn compile_generic_layer(
    exported_layer: ExportedLayer,
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

    Ok(GenericLayer {
        in_offset: memory_in.location,
        out_offset: memory_out.location,
        in_length: checked_u16(memory_in.count, "memory_in.count")?,
        out_length: checked_u16(memory_out.count, "memory_out.count")?,
        ops,
    })
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
    Ok(RowOp {
        first: compile_operand_index(exported_row_op.first, "first")?,
        second: compile_operand_index(exported_row_op.second, "second")?,
        third: compile_operand_index(exported_row_op.third, "third")?,
        op: compile_scalar_operator(&exported_row_op.op)?,
    })
}

fn compile_scalar_operator(operator_value: &Value) -> Result<ScalarOp, CompileError> {
    match operator_kind(operator_value)?.as_str() {
        "internal" => match operator_name(operator_value)?.as_str() {
            "identity" => Ok(ScalarOp::Identity),
            unsupported_name => Err(CompileError::NotImplemented(format!(
                "internal operator {unsupported_name}"
            ))),
        },
        "enum" => match operator_name(operator_value)?.as_str() {
            "SIN" => Ok(ScalarOp::Sin),
            "COS" => Ok(ScalarOp::Cos),
            "TAN" => Ok(ScalarOp::Tan),
            "EXP" => Ok(ScalarOp::Exp),
            "SQRT" => Ok(ScalarOp::Sqrt),
            "LOG" => Ok(ScalarOp::Log),
            "NEG" => Ok(ScalarOp::Neg),
            "ABS" => Ok(ScalarOp::Abs),
            "ADD" => Ok(ScalarOp::Add),
            "SUB" => Ok(ScalarOp::Sub),
            "MUL" => Ok(ScalarOp::Mul),
            "DIV" => Ok(ScalarOp::Div),
            "PWR" => Ok(ScalarOp::Pow),
            "INT_PWR" => Ok(ScalarOp::IntPow),
            "ARCTAN2" => Ok(ScalarOp::Atan2),
            "EQUAL" => Ok(ScalarOp::Equal),
            "LESS_THAN" => Ok(ScalarOp::LessThan),
            "LESS_EQUAL" => Ok(ScalarOp::LessEqual),
            "CASE" => Ok(ScalarOp::Case),
            unsupported_name => Err(CompileError::NotImplemented(format!(
                "operator {unsupported_name}"
            ))),
        },
        unsupported_kind => Err(CompileError::NotImplemented(format!(
            "operator kind {unsupported_kind}"
        ))),
    }
}

fn compile_sparse_tensor(
    exported_sparse_tensor: ExportedSparseTensor,
) -> Result<SparseTensor, CompileError> {
    Ok(SparseTensor {
        shape: compile_sparse_shape(exported_sparse_tensor.shape)?,
        entries: exported_sparse_tensor
            .entries
            .into_iter()
            .map(compile_sparse_entry)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn compile_sparse_entry(
    exported_sparse_entry: ExportedSparseEntry,
) -> Result<SparseEntry, CompileError> {
    let index = compile_sparse_shape(exported_sparse_entry.index)?;
    Ok(SparseEntry {
        index,
        value: exported_sparse_entry.value,
    })
}

fn compile_sparse_shape(values: Vec<u32>) -> Result<(u16, u16, u16), CompileError> {
    if values.len() != 3 {
        return Err(CompileError::InvalidField {
            field: "shape",
            reason: "expected exactly three entries",
        });
    }

    Ok((
        checked_u16(values[0], "shape[0]")?,
        checked_u16(values[1], "shape[1]")?,
        checked_u16(values[2], "shape[2]")?,
    ))
}
