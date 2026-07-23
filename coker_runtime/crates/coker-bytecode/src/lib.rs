#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::mem::align_of;
use rkyv::{
    access, rancor::Error as RkyvError, to_bytes, util::AlignedVec, Archive, Deserialize, Serialize,
};
use thiserror::Error;

const MAGIC: [u8; 8] = *b"COKERB03";
const VERSION: u16 = 3;
const HEADER_SIZE: usize = 16;

pub const ARCHIVED_MODULE_ALIGNMENT: usize = align_of::<ArchivedBytecodeModule>();

#[repr(C, align(16))]
pub struct AlignedModuleBytes<const N: usize> {
    bytes: [u8; N],
}

impl<const N: usize> AlignedModuleBytes<N> {
    pub const fn new(bytes: [u8; N]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl<const N: usize> AsRef<[u8]> for AlignedModuleBytes<N> {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct BytecodeModule {
    pub functions: Vec<Program>,
}

impl BytecodeModule {
    pub fn new(functions: Vec<Program>) -> Self {
        Self { functions }
    }
}

#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct Program {
    pub function_id: u16,
    pub workspace_size: u32,
    pub required_workspace_size: u32,
    pub input_specs: Vec<InputSpec>,
    pub output_specs: Vec<OutputSpec>,
    pub intermediate_layers: Vec<Layer>,
}

impl Program {
    pub fn new(
        function_id: u16,
        workspace_size: u32,
        required_workspace_size: u32,
        input_specs: Vec<InputSpec>,
        output_specs: Vec<OutputSpec>,
        intermediate_layers: Vec<Layer>,
    ) -> Self {
        Self {
            function_id,
            workspace_size,
            required_workspace_size,
            input_specs,
            output_specs,
            intermediate_layers,
        }
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
        for column_index in 0..COLUMN_COUNT {
            for row_index in 0..ROW_COUNT {
                entries.push(SparseEntry {
                    index: (row_index as u16, column_index as u16, u16::MAX),
                    value: data[column_index][row_index],
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

#[derive(Debug, Error)]
pub enum BytecodeError {
    #[error("failed to encode bytecode module: {0}")]
    Encode(String),
    #[error("failed to decode bytecode module: {0}")]
    Decode(String),
}

pub fn archived_module(bytes: &[u8]) -> Result<&ArchivedBytecodeModule, BytecodeError> {
    let payload = archive_payload(bytes)?;
    if payload.as_ptr() as usize % ARCHIVED_MODULE_ALIGNMENT != 0 {
        return Err(BytecodeError::Decode(format!(
            "bytecode archive payload must be {}-byte aligned",
            ARCHIVED_MODULE_ALIGNMENT
        )));
    }
    access::<ArchivedBytecodeModule, RkyvError>(payload)
        .map_err(|error| BytecodeError::Decode(error.to_string()))
}

fn archive_payload(bytes: &[u8]) -> Result<&[u8], BytecodeError> {
    validate_header(bytes)?;
    Ok(&bytes[HEADER_SIZE..])
}

fn validate_header(bytes: &[u8]) -> Result<(), BytecodeError> {
    if bytes.len() < HEADER_SIZE {
        return Err(BytecodeError::Decode(
            "bytecode header too short".to_string(),
        ));
    }
    if bytes[..MAGIC.len()] != MAGIC {
        return Err(BytecodeError::Decode("bytecode magic mismatch".to_string()));
    }

    let version_bytes: [u8; 2] = bytes[MAGIC.len()..MAGIC.len() + 2]
        .try_into()
        .expect("header size includes version bytes");
    if u16::from_le_bytes(version_bytes) != VERSION {
        return Err(BytecodeError::Decode(
            "unsupported bytecode version".to_string(),
        ));
    }

    Ok(())
}

pub fn encode_module(module: &BytecodeModule) -> Result<Vec<u8>, BytecodeError> {
    let archived_module =
        to_bytes::<RkyvError>(module).map_err(|error| BytecodeError::Encode(error.to_string()))?;
    let mut bytes = Vec::with_capacity(HEADER_SIZE + archived_module.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.resize(HEADER_SIZE, 0);
    bytes.extend_from_slice(archived_module.as_slice());
    Ok(bytes)
}

pub fn decode_module(bytes: &[u8]) -> Result<BytecodeModule, BytecodeError> {
    let payload = archive_payload(bytes)?;
    let mut aligned_bytes: AlignedVec<16> = AlignedVec::with_capacity(payload.len());
    aligned_bytes.extend_from_slice(payload);

    let archived = access::<ArchivedBytecodeModule, RkyvError>(aligned_bytes.as_slice())
        .map_err(|error| BytecodeError::Decode(error.to_string()))?;
    Ok(module_from_archived(archived))
}

fn module_from_archived(module: &ArchivedBytecodeModule) -> BytecodeModule {
    BytecodeModule::new(module.functions.iter().map(program_from_archived).collect())
}

fn program_from_archived(program: &ArchivedProgram) -> Program {
    Program::new(
        program.function_id.into(),
        program.workspace_size.into(),
        program.required_workspace_size.into(),
        program
            .input_specs
            .iter()
            .map(input_spec_from_archived)
            .collect(),
        program
            .output_specs
            .iter()
            .map(output_spec_from_archived)
            .collect(),
        program
            .intermediate_layers
            .iter()
            .map(layer_from_archived)
            .collect(),
    )
}

fn input_spec_from_archived(input_spec: &ArchivedInputSpec) -> InputSpec {
    InputSpec {
        workspace_offset: input_spec.workspace_offset.into(),
        length: input_spec.length.into(),
    }
}

fn output_spec_from_archived(output_spec: &ArchivedOutputSpec) -> OutputSpec {
    OutputSpec {
        workspace_offset: output_spec.workspace_offset.into(),
        length: output_spec.length.into(),
    }
}

fn layer_from_archived(layer: &ArchivedLayer) -> Layer {
    match layer {
        ArchivedLayer::Bilinear(bilinear_layer) => {
            Layer::Bilinear(bilinear_layer_from_archived(bilinear_layer))
        }
        ArchivedLayer::Generic(generic_layer) => {
            Layer::Generic(generic_layer_from_archived(generic_layer))
        }
        ArchivedLayer::Evaluate(evaluate_layer) => {
            Layer::Evaluate(evaluate_layer_from_archived(evaluate_layer))
        }
    }
}

fn bilinear_layer_from_archived(bilinear_layer: &ArchivedBilinearLayer) -> BilinearLayer {
    BilinearLayer {
        in_offset: bilinear_layer.in_offset.into(),
        out_offset: bilinear_layer.out_offset.into(),
        in_length: bilinear_layer.in_length.into(),
        out_length: bilinear_layer.out_length.into(),
        scratch_offset: bilinear_layer.scratch_offset.into(),
        scratch_length: bilinear_layer.scratch_length.into(),
        quadratic: sparse_tensor_from_archived(&bilinear_layer.quadratic),
    }
}

fn generic_layer_from_archived(generic_layer: &ArchivedGenericLayer) -> GenericLayer {
    GenericLayer {
        in_offset: generic_layer.in_offset.into(),
        out_offset: generic_layer.out_offset.into(),
        in_length: generic_layer.in_length.into(),
        out_length: generic_layer.out_length.into(),
        scratch_offset: generic_layer.scratch_offset.into(),
        scratch_length: generic_layer.scratch_length.into(),
        ops: generic_layer.ops.iter().map(row_op_from_archived).collect(),
    }
}

fn evaluate_layer_from_archived(evaluate_layer: &ArchivedEvaluateLayer) -> EvaluateLayer {
    EvaluateLayer {
        scratch_offset: evaluate_layer.scratch_offset.into(),
        callee_function_id: evaluate_layer.callee_function_id.into(),
        input_bindings: evaluate_layer
            .input_bindings
            .iter()
            .map(evaluate_input_binding_from_archived)
            .collect(),
        output_bindings: evaluate_layer
            .output_bindings
            .iter()
            .map(evaluate_output_binding_from_archived)
            .collect(),
    }
}

fn evaluate_input_binding_from_archived(
    binding: &ArchivedEvaluateInputBinding,
) -> EvaluateInputBinding {
    match binding {
        ArchivedEvaluateInputBinding::WorkspaceSlice { offset, length } => {
            EvaluateInputBinding::WorkspaceSlice {
                offset: (*offset).into(),
                length: (*length).into(),
            }
        }
        ArchivedEvaluateInputBinding::ConstantSlice { length, values } => {
            EvaluateInputBinding::ConstantSlice {
                length: (*length).into(),
                values: values.iter().map(|value| (*value).into()).collect(),
            }
        }
    }
}

fn evaluate_output_binding_from_archived(
    binding: &ArchivedEvaluateOutputBinding,
) -> EvaluateOutputBinding {
    EvaluateOutputBinding {
        destination_offset: binding.destination_offset.into(),
        length: binding.length.into(),
    }
}

fn row_op_from_archived(row_op: &ArchivedRowOp) -> RowOp {
    RowOp {
        first: row_op.first.into(),
        second: row_op.second.into(),
        third: row_op.third.into(),
        op: scalar_op_from_archived(&row_op.op),
    }
}

fn scalar_op_from_archived(scalar_op: &ArchivedScalarOp) -> ScalarOp {
    match scalar_op {
        ArchivedScalarOp::Identity => ScalarOp::Identity,
        ArchivedScalarOp::Sin => ScalarOp::Sin,
        ArchivedScalarOp::Cos => ScalarOp::Cos,
        ArchivedScalarOp::Tan => ScalarOp::Tan,
        ArchivedScalarOp::Exp => ScalarOp::Exp,
        ArchivedScalarOp::Sqrt => ScalarOp::Sqrt,
        ArchivedScalarOp::Log => ScalarOp::Log,
        ArchivedScalarOp::Neg => ScalarOp::Neg,
        ArchivedScalarOp::Abs => ScalarOp::Abs,
        ArchivedScalarOp::Add => ScalarOp::Add,
        ArchivedScalarOp::Sub => ScalarOp::Sub,
        ArchivedScalarOp::Mul => ScalarOp::Mul,
        ArchivedScalarOp::Div => ScalarOp::Div,
        ArchivedScalarOp::Pow => ScalarOp::Pow,
        ArchivedScalarOp::IntPow => ScalarOp::IntPow,
        ArchivedScalarOp::Atan2 => ScalarOp::Atan2,
        ArchivedScalarOp::Equal => ScalarOp::Equal,
        ArchivedScalarOp::LessThan => ScalarOp::LessThan,
        ArchivedScalarOp::LessEqual => ScalarOp::LessEqual,
        ArchivedScalarOp::Case => ScalarOp::Case,
    }
}

fn sparse_tensor_from_archived(tensor: &ArchivedSparseTensor) -> SparseTensor {
    SparseTensor {
        shape: (
            tensor.shape.0.into(),
            tensor.shape.1.into(),
            tensor.shape.2.into(),
        ),
        entries: tensor
            .entries
            .iter()
            .map(sparse_entry_from_archived)
            .collect(),
    }
}

fn sparse_entry_from_archived(entry: &ArchivedSparseEntry) -> SparseEntry {
    SparseEntry {
        index: (
            entry.index.0.into(),
            entry.index.1.into(),
            entry.index.2.into(),
        ),
        value: entry.value.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::align_of;

    #[test]
    fn encode_decode_round_trip_preserves_module() {
        let tensor_data: [[f32; 4]; 2] = [[1.0, 2.0, 3.0, 0.0], [0.0, 4.0, 0.0, 5.0]];
        let tensor = SparseTensor::try_from_row_major_array(&tensor_data).unwrap();
        let entry_program = Program::new(
            0,
            6,
            9,
            vec![InputSpec {
                workspace_offset: 0,
                length: 3,
            }],
            vec![OutputSpec {
                workspace_offset: 4,
                length: 2,
            }],
            vec![
                Layer::Bilinear(BilinearLayer {
                    in_offset: 0,
                    out_offset: 4,
                    in_length: 3,
                    out_length: 2,
                    scratch_offset: 0,
                    scratch_length: 0,
                    quadratic: tensor,
                }),
                Layer::Evaluate(EvaluateLayer {
                    scratch_offset: 6,
                    callee_function_id: 1,
                    input_bindings: vec![EvaluateInputBinding::WorkspaceSlice {
                        offset: 4,
                        length: 2,
                    }],
                    output_bindings: vec![EvaluateOutputBinding {
                        destination_offset: 4,
                        length: 2,
                    }],
                }),
            ],
        );
        let callee_program = Program::new(
            1,
            2,
            2,
            vec![InputSpec {
                workspace_offset: 0,
                length: 2,
            }],
            vec![OutputSpec {
                workspace_offset: 0,
                length: 2,
            }],
            vec![],
        );
        let module = BytecodeModule::new(vec![entry_program, callee_program]);

        let encoded_module = encode_module(&module).unwrap();
        let decoded_module = decode_module(&encoded_module).unwrap();
        assert_eq!(decoded_module, module);
    }

    #[test]
    fn archived_module_accepts_aligned_bytes_without_copy() {
        let module = BytecodeModule::new(vec![Program::new(0, 0, 0, vec![], vec![], vec![])]);
        let encoded = encode_module(&module).unwrap();
        let mut aligned = AlignedVec::<16>::with_capacity(encoded.len());
        aligned.extend_from_slice(&encoded);
        let archived = archived_module(aligned.as_slice()).unwrap();
        assert_eq!(archived.functions.len(), 1);
        assert_eq!(archived.functions[0].function_id.to_native(), 0);
    }

    #[test]
    fn archived_module_rejects_misaligned_payload() {
        let module = BytecodeModule::new(vec![Program::new(0, 0, 0, vec![], vec![], vec![])]);
        let encoded = encode_module(&module).unwrap();
        let mut misaligned = vec![0u8];
        misaligned.extend_from_slice(&encoded);
        let error = match archived_module(&misaligned[1..]) {
            Ok(_) => panic!("expected misaligned archive access to fail"),
            Err(error) => error,
        };
        assert!(matches!(error, BytecodeError::Decode(_)));
        assert!(error.to_string().contains("aligned"));
    }
}
