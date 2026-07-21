use coker_bytecode::Program;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedModule {
    pub(crate) functions: Vec<ExportedFunction>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedFunction {
    pub(crate) function_id: u32,
    pub(crate) program: ExportedProgram,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedProgram {
    pub(crate) workspace: ExportedMemorySpec,
    pub(crate) input_layer: ExportedInputLayer,
    pub(crate) output_layer: ExportedOutputLayer,
    pub(crate) intermediate_layers: Vec<ExportedLayer>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedInputLayer {
    pub(crate) inputs: Vec<ExportedInputSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedInputSpec {
    pub(crate) memory: ExportedMemorySpec,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedOutputLayer {
    pub(crate) outputs: Vec<ExportedOutputSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedOutputSpec {
    pub(crate) memory: ExportedMemorySpec,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedMemorySpec {
    pub(crate) location: u32,
    pub(crate) count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedLayer {
    pub(crate) kind: String,
    pub(crate) memory_in: Option<ExportedMemorySpec>,
    pub(crate) memory_out: Option<ExportedMemorySpec>,
    pub(crate) weights: Option<ExportedWeights>,
    pub(crate) ops: Option<Vec<ExportedRowOp>>,
    pub(crate) constants: Option<Vec<Value>>,
    pub(crate) opaque_programs: Option<Vec<Value>>,
    pub(crate) callee_function_id: Option<u32>,
    pub(crate) inputs: Option<Vec<ExportedEvaluateInputBinding>>,
    pub(crate) outputs: Option<Vec<ExportedEvaluateOutputBinding>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedWeights {
    pub(crate) quadratic: ExportedSparseTensor,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedSparseTensor {
    pub(crate) shape: Vec<u32>,
    pub(crate) entries: Vec<ExportedSparseEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedSparseEntry {
    pub(crate) index: Vec<u32>,
    pub(crate) value: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedRowOp {
    pub(crate) op: Value,
    pub(crate) first: i32,
    pub(crate) second: i32,
    pub(crate) third: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
pub(crate) enum ExportedEvaluateInputBinding {
    #[serde(rename = "workspace")]
    Workspace { offset: u32, length: u32 },
    #[serde(rename = "constant")]
    Constant { length: u32, values: Vec<f32> },
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ExportedEvaluateOutputBinding {
    pub(crate) destination_offset: u32,
    pub(crate) length: u32,
}

pub(crate) struct CompileContext {
    pub(crate) exported_programs: Vec<ExportedProgram>,
    pub(crate) compiled_programs: Vec<Option<Program>>,
    pub(crate) visiting: Vec<bool>,
}
