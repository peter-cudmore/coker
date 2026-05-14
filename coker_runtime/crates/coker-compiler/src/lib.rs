use coker_bytecode::{
    encode_module, BilinearLayer, BytecodeModule, EvaluateInputBinding,
    EvaluateLayer, EvaluateOutputBinding, GenericLayer, InputSpec, Layer,
    OutputSpec, Program, RowOp, ScalarOp, SparseEntry, SparseTensor,
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

#[derive(Debug, Clone, Deserialize)]
struct ExportedModule {
    functions: Vec<ExportedFunction>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedFunction {
    function_id: u32,
    program: ExportedProgram,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedProgram {
    workspace: ExportedMemorySpec,
    input_layer: ExportedInputLayer,
    output_layer: ExportedOutputLayer,
    intermediate_layers: Vec<ExportedLayer>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedInputLayer {
    inputs: Vec<ExportedInputSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedInputSpec {
    memory: ExportedMemorySpec,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedOutputLayer {
    outputs: Vec<ExportedOutputSpec>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedOutputSpec {
    memory: ExportedMemorySpec,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedMemorySpec {
    location: u32,
    count: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedLayer {
    kind: String,
    memory_in: Option<ExportedMemorySpec>,
    memory_out: Option<ExportedMemorySpec>,
    weights: Option<ExportedWeights>,
    ops: Option<Vec<ExportedRowOp>>,
    constants: Option<Vec<Value>>,
    opaque_programs: Option<Vec<Value>>,
    callee_function_id: Option<u32>,
    inputs: Option<Vec<ExportedEvaluateInputBinding>>,
    outputs: Option<Vec<ExportedEvaluateOutputBinding>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedWeights {
    quadratic: ExportedSparseTensor,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedSparseTensor {
    shape: Vec<u32>,
    entries: Vec<ExportedSparseEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedSparseEntry {
    index: Vec<u32>,
    value: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedRowOp {
    op: Value,
    first: i32,
    second: i32,
    third: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
enum ExportedEvaluateInputBinding {
    #[serde(rename = "workspace")]
    Workspace { offset: u32, length: u32 },
    #[serde(rename = "constant")]
    Constant { length: u32, values: Vec<f32> },
}

#[derive(Debug, Clone, Deserialize)]
struct ExportedEvaluateOutputBinding {
    destination_offset: u32,
    length: u32,
}

struct CompileContext {
    exported_programs: Vec<ExportedProgram>,
    compiled_programs: Vec<Option<Program>>,
    visiting: Vec<bool>,
}

pub fn compile_exported_json(exported_graph_json: &[u8]) -> Result<Vec<u8>, CompileError> {
    let exported_module: ExportedModule = serde_json::from_slice(exported_graph_json)?;
    let bytecode_module = compile_exported_module(exported_module)?;
    encode_module(&bytecode_module).map_err(CompileError::from)
}

fn compile_exported_module(
    exported_module: ExportedModule,
) -> Result<BytecodeModule, CompileError> {
    let exported_programs = index_exported_programs(exported_module)?;
    let function_count = exported_programs.len();
    let mut compile_context = CompileContext {
        compiled_programs: vec![None; function_count],
        visiting: vec![false; function_count],
        exported_programs,
    };

    let mut functions = Vec::with_capacity(function_count);
    for function_id in 0..function_count {
        functions.push(compile_context.compile_function(function_id as u16)?);
    }
    Ok(BytecodeModule::new(functions))
}

fn index_exported_programs(
    exported_module: ExportedModule,
) -> Result<Vec<ExportedProgram>, CompileError> {
    if exported_module.functions.is_empty() {
        return Err(CompileError::InvalidField {
            field: "functions",
            reason: "expected at least one function",
        });
    }

    let function_count = exported_module.functions.len();
    let mut indexed_programs = vec![None; function_count];
    for exported_function in exported_module.functions {
        let function_id = checked_u16(exported_function.function_id, "function_id")?;
        let function_index = function_id as usize;
        if function_index >= function_count {
            return Err(CompileError::InvalidField {
                field: "function_id",
                reason: "expected dense ids from 0 to function_count - 1",
            });
        }
        if indexed_programs[function_index].is_some() {
            return Err(CompileError::InvalidField {
                field: "function_id",
                reason: "duplicate function id",
            });
        }
        indexed_programs[function_index] = Some(exported_function.program);
    }

    indexed_programs
        .into_iter()
        .map(|program| {
            program.ok_or(CompileError::InvalidField {
                field: "function_id",
                reason: "expected dense ids from 0 to function_count - 1",
            })
        })
        .collect()
}

impl CompileContext {
    fn compile_function(&mut self, function_id: u16) -> Result<Program, CompileError> {
        let function_index = function_id as usize;
        if let Some(compiled_program) = self.compiled_programs[function_index].clone() {
            return Ok(compiled_program);
        }
        if self.visiting[function_index] {
            return Err(CompileError::NotImplemented(
                "recursive function evaluation".to_string(),
            ));
        }

        self.visiting[function_index] = true;
        let exported_program = self.exported_programs[function_index].clone();
        let compiled_program = self.compile_program(function_id, exported_program)?;
        self.visiting[function_index] = false;
        self.compiled_programs[function_index] = Some(compiled_program.clone());
        Ok(compiled_program)
    }

    fn compile_program(
        &mut self,
        function_id: u16,
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

        let workspace_size = exported_program.workspace.count;
        let mut required_workspace_size = workspace_size;
        let intermediate_layers = exported_program
            .intermediate_layers
            .into_iter()
            .map(|layer| self.compile_layer(layer, &mut required_workspace_size))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Program::new(
            function_id,
            workspace_size,
            required_workspace_size,
            input_specs,
            output_specs,
            intermediate_layers,
        ))
    }

    fn compile_layer(
        &mut self,
        exported_layer: ExportedLayer,
        required_workspace_size: &mut u32,
    ) -> Result<Layer, CompileError> {
        match exported_layer.kind.as_str() {
            "bilinear" => compile_bilinear_layer(exported_layer).map(Layer::Bilinear),
            "generic" => compile_generic_layer(exported_layer).map(Layer::Generic),
            "evaluate" => self
                .compile_evaluate_layer(exported_layer, required_workspace_size)
                .map(Layer::Evaluate),
            unsupported_kind => Err(CompileError::NotImplemented(format!(
                "layer kind {unsupported_kind}"
            ))),
        }
    }

    fn compile_evaluate_layer(
        &mut self,
        exported_layer: ExportedLayer,
        required_workspace_size: &mut u32,
    ) -> Result<EvaluateLayer, CompileError> {
        let callee_function_id = checked_u16(
            required_field(
                exported_layer.callee_function_id,
                "callee_function_id",
            )?,
            "callee_function_id",
        )?;
        let callee_program = self.compile_function(callee_function_id)?;

        let exported_inputs = required_field(exported_layer.inputs, "inputs")?;
        let exported_outputs = required_field(exported_layer.outputs, "outputs")?;
        if exported_inputs.len() != callee_program.input_specs.len() {
            return Err(CompileError::InvalidField {
                field: "inputs",
                reason: "evaluate input binding count does not match callee inputs",
            });
        }
        if exported_outputs.len() != callee_program.output_specs.len() {
            return Err(CompileError::InvalidField {
                field: "outputs",
                reason: "evaluate output binding count does not match callee outputs",
            });
        }

        let input_bindings = exported_inputs
            .into_iter()
            .zip(callee_program.input_specs.iter())
            .map(|(binding, input_spec)| {
                compile_evaluate_input_binding(binding, input_spec.length)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_bindings = exported_outputs
            .into_iter()
            .zip(callee_program.output_specs.iter())
            .map(|(binding, output_spec)| {
                compile_evaluate_output_binding(binding, output_spec.length)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let scratch_offset = *required_workspace_size;
        *required_workspace_size = checked_add_u32(
            scratch_offset,
            callee_program.required_workspace_size,
            "scratch_offset",
        )?;

        Ok(EvaluateLayer {
            scratch_offset,
            callee_function_id,
            input_count: checked_u8_length(input_bindings.len(), "inputs")?,
            output_count: checked_u8_length(output_bindings.len(), "outputs")?,
            input_bindings,
            output_bindings,
        })
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

fn compile_generic_layer(
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

fn compile_evaluate_input_binding(
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

fn compile_evaluate_output_binding(
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

fn compile_operand_index(
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

fn checked_u8_length(
    value: usize,
    field_name: &'static str,
) -> Result<u8, CompileError> {
    u8::try_from(value).map_err(|_| CompileError::InvalidField {
        field: field_name,
        reason: "expected u8-sized collection",
    })
}
fn checked_u16(value: u32, field_name: &'static str) -> Result<u16, CompileError> {
    u16::try_from(value).map_err(|_| CompileError::InvalidField {
        field: field_name,
        reason: "expected u16",
    })
}

fn checked_add_u32(
    left: u32,
    right: u32,
    field_name: &'static str,
) -> Result<u32, CompileError> {
    left.checked_add(right).ok_or(CompileError::InvalidField {
        field: field_name,
        reason: "u32 overflow",
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
    use coker_bytecode::{decode_module, Layer};

    #[test]
    fn compile_exported_json_builds_module_bytecode() {
        let exported_module_json = r#"
        {
            "functions": [
                {
                    "function_id": 0,
                    "program": {
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
                }
            ]
        }
        "#;

        let module_bytes = compile_exported_json(exported_module_json.as_bytes()).unwrap();
        let module = decode_module(&module_bytes).unwrap();

        assert_eq!(module.functions.len(), 1);
        let program = &module.functions[0];
        assert_eq!(program.function_id, 0);
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
    fn compile_exported_json_builds_evaluate_layer() {
        let exported_module_json = r#"
        {
            "functions": [
                {
                    "function_id": 0,
                    "program": {
                        "workspace": {"location": 0, "count": 1},
                        "input_layer": {"inputs": []},
                        "output_layer": {
                            "outputs": [
                                {"memory": {"location": 0, "count": 1}}
                            ]
                        },
                        "intermediate_layers": [
                            {
                                "kind": "evaluate",
                                "memory_in": {"location": 0, "count": 0},
                                "memory_out": {"location": 0, "count": 1},
                                "callee_function_id": 1,
                                "inputs": [
                                    {"kind": "constant", "length": 1, "values": [2.0]}
                                ],
                                "outputs": [
                                    {"destination_offset": 0, "length": 1}
                                ]
                            }
                        ]
                    }
                },
                {
                    "function_id": 1,
                    "program": {
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
                }
            ]
        }
        "#;

        let module_bytes = compile_exported_json(exported_module_json.as_bytes()).unwrap();
        let module = decode_module(&module_bytes).unwrap();
        let program = &module.functions[0];
        assert_eq!(program.required_workspace_size, 3);
        match &program.intermediate_layers[0] {
            Layer::Evaluate(evaluate_layer) => {
                assert_eq!(evaluate_layer.callee_function_id, 1);
                assert_eq!(evaluate_layer.scratch_offset, 1);
            }
            _ => panic!("expected evaluate layer"),
        }
    }

    #[test]
    fn compile_exported_json_rejects_opaque_programs() {
        let exported_module_json = r#"
        {
            "functions": [
                {
                    "function_id": 0,
                    "program": {
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
                }
            ]
        }
        "#;

        let error = compile_exported_json(exported_module_json.as_bytes()).unwrap_err();
        assert!(matches!(error, CompileError::NotImplemented(_)));
    }
}
