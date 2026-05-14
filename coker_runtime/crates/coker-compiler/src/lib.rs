use coker_bytecode::{
    encode_program, BilinearLayer, GenericLayer, InputSpec, Layer, OutputSpec,
    Program, RowOp, ScalarOp, SparseEntry, SparseTensor,
};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

const UNUSED_OPERAND: u16 = u16::MAX;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("failed to parse exported graph json: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("missing field {field}")]
    MissingField { field: &'static str },
    #[error("invalid field {field}: {reason}")]
    InvalidField {
        field: &'static str,
        reason: &'static str,
    },
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("failed to encode bytecode: {0}")]
    Encode(#[from] coker_bytecode::BytecodeError),
}

#[derive(Debug, Deserialize)]
struct ExportedProgram {
    workspace: ExportedMemorySpec,
    input_layer: ExportedInputLayer,
    output_layer: ExportedOutputLayer,
    intermediate_layers: Vec<ExportedLayer>,
}

#[derive(Debug, Deserialize)]
struct ExportedInputLayer {
    inputs: Vec<ExportedInputSpec>,
}

#[derive(Debug, Deserialize)]
struct ExportedInputSpec {
    memory: ExportedMemorySpec,
}

#[derive(Debug, Deserialize)]
struct ExportedOutputLayer {
    outputs: Vec<ExportedOutputSpec>,
}

#[derive(Debug, Deserialize)]
struct ExportedOutputSpec {
    memory: ExportedMemorySpec,
}

#[derive(Debug, Deserialize)]
struct ExportedMemorySpec {
    location: u32,
    count: u32,
}

#[derive(Debug, Deserialize)]
struct ExportedLayer {
    kind: String,
    memory_in: Option<ExportedMemorySpec>,
    memory_out: Option<ExportedMemorySpec>,
    weights: Option<ExportedWeights>,
    ops: Option<Vec<ExportedRowOp>>,
    constants: Option<Vec<Value>>,
    opaque_programs: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
struct ExportedWeights {
    quadratic: ExportedSparseTensor,
}

#[derive(Debug, Deserialize)]
struct ExportedSparseTensor {
    shape: Vec<u32>,
    entries: Vec<ExportedSparseEntry>,
}

#[derive(Debug, Deserialize)]
struct ExportedSparseEntry {
    index: Vec<u32>,
    value: f32,
}

#[derive(Debug, Deserialize)]
struct ExportedRowOp {
    op: Value,
    first: i32,
    second: i32,
    third: i32,
}

pub fn compile_exported_json(exported_graph_json: &[u8]) -> Result<Vec<u8>, CompileError> {
    let exported_program: ExportedProgram = serde_json::from_slice(exported_graph_json)?;
    let program = compile_exported_program(exported_program)?;
    encode_program(&program).map_err(CompileError::from)
}

fn compile_exported_program(
    exported_program: ExportedProgram,
) -> Result<Program, CompileError> {
    let input_specs = exported_program
        .input_layer
        .inputs
        .into_iter()
        .map(|input_spec| {
            let memory = input_spec.memory;
            Ok::<_, CompileError>(InputSpec {
                workspace_offset: memory.location,
                length: checked_u16(memory.count, "input.memory.count")?,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;

    let output_specs = exported_program
        .output_layer
        .outputs
        .into_iter()
        .map(|output_spec| {
            let memory = output_spec.memory;
            Ok::<_, CompileError>(OutputSpec {
                workspace_offset: memory.location,
                length: checked_u16(memory.count, "output.memory.count")?,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;

    let intermediate_layers = exported_program
        .intermediate_layers
        .into_iter()
        .map(compile_layer)
        .collect::<Result<Vec<_>, _>>()?;

    let workspace_size = exported_program.workspace.count;
    Ok(Program::new(
        workspace_size,
        workspace_size,
        input_specs,
        output_specs,
        intermediate_layers,
    ))
}

fn compile_layer(exported_layer: ExportedLayer) -> Result<Layer, CompileError> {
    match exported_layer.kind.as_str() {
        "bilinear" => compile_bilinear_layer(exported_layer).map(Layer::Bilinear),
        "generic" => compile_generic_layer(exported_layer).map(Layer::Generic),
        unsupported_kind => Err(CompileError::NotImplemented(format!(
            "layer kind {unsupported_kind}"
        ))),
    }
}

fn compile_bilinear_layer(
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

fn compile_generic_layer(exported_layer: ExportedLayer) -> Result<GenericLayer, CompileError> {
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

fn compile_operand_index(value: i32, field_name: &'static str) -> Result<u16, CompileError> {
    if value < 0 {
        return Ok(UNUSED_OPERAND);
    }

    u16::try_from(value).map_err(|_| CompileError::InvalidField {
        field: field_name,
        reason: "expected non-negative u16 or sentinel",
    })
}

fn checked_u16(value: u32, field_name: &'static str) -> Result<u16, CompileError> {
    u16::try_from(value).map_err(|_| CompileError::InvalidField {
        field: field_name,
        reason: "expected u16",
    })
}


fn required_field<T>(
    field_value: Option<T>,
    field_name: &'static str,
) -> Result<T, CompileError> {
    field_value.ok_or(CompileError::MissingField { field: field_name })
}

fn operator_kind(operator_value: &Value) -> Result<String, CompileError> {
    object_field(operator_value, "kind")?
        .as_str()
        .map(str::to_string)
        .ok_or(CompileError::InvalidField {
            field: "op.kind",
            reason: "expected string",
        })
}

fn operator_name(operator_value: &Value) -> Result<String, CompileError> {
    object_field(operator_value, "value")?
        .as_str()
        .map(str::to_string)
        .ok_or(CompileError::InvalidField {
            field: "op.value",
            reason: "expected string",
        })
}

fn object_field<'a>(
    value: &'a Value,
    field_name: &'static str,
) -> Result<&'a Value, CompileError> {
    value
        .get(field_name)
        .ok_or(CompileError::MissingField { field: field_name })
}

#[cfg(test)]
mod tests {
    use super::*;
    use coker_bytecode::{decode_program, Layer};

    #[test]
    fn compile_exported_json_builds_program_bytecode() {
        let exported_program_json = r#"
        {
            "workspace": {"location": 0, "count": 2},
            "input_layer": {
                "inputs": [
                    {"memory": {"location": 0, "count": 1}}
                ]
            },
            "output_layer": {
                "outputs": [
                    {"memory": {"location": 1, "count": 1}}
                ]
            },
            "intermediate_layers": [
                {
                    "kind": "generic",
                    "memory_in": {"location": 0, "count": 1},
                    "memory_out": {"location": 0, "count": 2},
                    "ops": [
                        {
                            "op": {"kind": "internal", "value": "identity"},
                            "first": 0,
                            "second": -1,
                            "third": -1
                        },
                        {
                            "op": {"kind": "enum", "value": "SIN"},
                            "first": 0,
                            "second": -1,
                            "third": -1
                        }
                    ]
                }
            ]
        }
        "#;

        let program_bytes = compile_exported_json(exported_program_json.as_bytes()).unwrap();
        let program = decode_program(&program_bytes).unwrap();

        assert_eq!(program.workspace_size, 2);
        assert_eq!(program.required_workspace_size, 2);
        assert_eq!(program.input_specs[0].length, 1);
        assert_eq!(program.output_specs[0].length, 1);
        match &program.intermediate_layers[0] {
            Layer::Generic(generic_layer) => {
                assert_eq!(generic_layer.ops.len(), 2);
                assert_eq!(generic_layer.ops[0].second, u16::MAX);
                assert_eq!(generic_layer.ops[1].op, ScalarOp::Sin);
            }
            _ => panic!("expected generic layer"),
        }
    }

    #[test]
    fn compile_exported_json_rejects_opaque_programs() {
        let exported_program_json = r#"
        {
            "workspace": {"location": 0, "count": 1},
            "input_layer": {"inputs": []},
            "output_layer": {"outputs": []},
            "intermediate_layers": [
                {
                    "kind": "generic",
                    "memory_in": {"location": 0, "count": 0},
                    "memory_out": {"location": 0, "count": 1},
                    "ops": [
                        {
                            "op": {"kind": "internal", "value": "identity"},
                            "first": -1,
                            "second": -1,
                            "third": -1
                        }
                    ],
                    "opaque_programs": [{}]
                }
            ]
        }
        "#;

        let error = compile_exported_json(exported_program_json.as_bytes()).unwrap_err();
        assert!(matches!(error, CompileError::NotImplemented(_)));
    }
}
