#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{boxed::Box, string::{String, ToString}, vec::Vec};
use binrw::{binrw, io::Cursor, BinReaderExt, BinWriterExt};
use thiserror::Error;

#[binrw]
#[brw(little, magic = b"COKERBC\0")]
#[derive(Debug, Clone, PartialEq)]
pub struct BytecodeModule {
    #[bw(try_calc(u16::try_from(functions.len())))]
    pub function_count: u16,
    #[br(count = function_count)]
    pub functions: Vec<Program>,
}

impl BytecodeModule {
    pub fn new(functions: Vec<Program>) -> Self {
        Self { functions }
    }
}

#[binrw]
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub function_id: u16,
    #[bw(try_calc(u8::try_from(input_specs.len())))]
    pub input_space_count: u8,
    #[bw(try_calc(u8::try_from(output_specs.len())))]
    pub output_space_count: u8,
    #[bw(try_calc(u8::try_from(intermediate_layers.len())))]
    pub intermediate_layer_count: u8,
    pub workspace_size: u32,
    pub required_workspace_size: u32,
    #[br(count = input_space_count)]
    pub input_specs: Vec<InputSpec>,
    #[br(count = output_space_count)]
    pub output_specs: Vec<OutputSpec>,
    #[br(count = intermediate_layer_count)]
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

#[binrw]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputSpec {
    pub workspace_offset: u32,
    pub length: u16,
}

#[binrw]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputSpec {
    pub workspace_offset: u32,
    pub length: u16,
}

#[binrw]
#[brw(little)]
#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
    #[brw(magic = 0u8)]
    Bilinear(BilinearLayer),
    #[brw(magic = 1u8)]
    Generic(GenericLayer),
    #[brw(magic = 2u8)]
    Evaluate(EvaluateLayer),
}

#[binrw]
#[derive(Debug, Clone, PartialEq)]
pub struct BilinearLayer {
    pub in_offset: u32,
    pub out_offset: u32,
    pub in_length: u16,
    pub out_length: u16,
    pub quadratic: SparseTensor,
}

#[binrw]
#[brw(little)]
#[bw(assert(*out_length == ops.len() as u16))]
#[derive(Debug, Clone, PartialEq)]
pub struct GenericLayer {
    pub in_offset: u32,
    pub out_offset: u32,
    pub in_length: u16,
    pub out_length: u16,
    #[br(count = out_length)]
    pub ops: Vec<RowOp>,
}

#[binrw]
#[brw(little)]
#[bw(assert(*input_count == input_bindings.len() as u8))]
#[bw(assert(*output_count == output_bindings.len() as u8))]
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluateLayer {
    pub scratch_offset: u32,
    pub callee_function_id: u16,
    pub input_count: u8,
    pub output_count: u8,
    #[br(count = input_count)]
    pub input_bindings: Vec<EvaluateInputBinding>,
    #[br(count = output_count)]
    pub output_bindings: Vec<EvaluateOutputBinding>,
}

#[binrw]
#[brw(little)]
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluateInputBinding {
    #[brw(magic = 0u8)]
    WorkspaceSlice { offset: u32, length: u16 },
    #[brw(magic = 1u8)]
    ConstantSlice {
        length: u16,
        #[br(count = length)]
        values: Vec<f32>,
    },
}

#[binrw]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluateOutputBinding {
    pub destination_offset: u32,
    pub length: u16,
}

#[binrw]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[brw(little, repr = u8)]
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

#[binrw]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowOp {
    pub first: u16,
    pub second: u16,
    pub third: u16,
    pub op: ScalarOp,
}

#[binrw]
#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensor {
    pub shape: (u16, u16, u16),
    #[bw(try_calc(u32::try_from(entries.len())))]
    pub entry_count: u32,
    #[br(count = entry_count)]
    pub entries: Vec<SparseEntry>,
}

impl SparseTensor {
    pub fn try_from_row_major_array<
        const ROW_COUNT: usize,
        const COLUMN_COUNT: usize,
    >(
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

#[binrw]
#[derive(Debug, Clone, PartialEq)]
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

pub fn encode_module(module: &BytecodeModule) -> Result<Vec<u8>, BytecodeError> {
    let mut stream = Cursor::new(Vec::new());
    stream
        .write_le(module)
        .map_err(|error| BytecodeError::Encode(error.to_string()))?;
    Ok(stream.into_inner())
}

pub fn decode_module(bytes: &[u8]) -> Result<BytecodeModule, BytecodeError> {
    let mut cursor = Cursor::new(bytes);
    cursor
        .read_le()
        .map_err(|error| BytecodeError::Decode(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    quadratic: tensor,
                }),
                Layer::Evaluate(EvaluateLayer {
                    scratch_offset: 6,
                    callee_function_id: 1,
                    input_count: 1,
                    output_count: 1,
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
}
