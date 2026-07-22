use coker_compiler::{compile_exported_json, CompileError};
use coker_runtime::{program_info, validate_module, Module, ModuleBuilder, ProgramInfo};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyclass(name = "RuntimeProgram")]
struct PyRuntimeProgram {
    module: Module,
    output_lengths: Vec<usize>,
}

#[pymethods]
impl PyRuntimeProgram {
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        program_info_dict(py, &self.module.info())
    }

    fn execute(&mut self, inputs: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let execution_inputs = self
            .module
            .validate_inputs(&input_slices)
            .map_err(runtime_error)?;
        let mut outputs = vec![0.0; self.output_lengths.iter().sum()];
        let execution_outputs = self
            .module
            .validate_outputs(&mut outputs)
            .map_err(runtime_error)?;
        self.module.execute(execution_inputs, execution_outputs);
        Ok(outputs)
    }

    fn push_forward(
        &mut self,
        inputs: Vec<Vec<f32>>,
        tangents: Vec<Vec<f32>>,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let tangent_slices: Vec<&[f32]> = tangents.iter().map(|input| input.as_slice()).collect();
        let push_forward_inputs = self
            .module
            .validate_push_forward_inputs(&input_slices, &tangent_slices)
            .map_err(runtime_error)?;
        let output_length: usize = self.output_lengths.iter().sum();
        let mut outputs = vec![0.0; output_length];
        let mut tangent_outputs = vec![0.0; output_length];
        let push_forward_outputs = self
            .module
            .validate_push_forward_outputs(&mut outputs, &mut tangent_outputs)
            .map_err(runtime_error)?;
        self.module
            .push_forward(push_forward_inputs, push_forward_outputs);
        Ok((outputs, tangent_outputs))
    }
}

#[pyfunction]
fn compile_exported_graph<'py>(
    py: Python<'py>,
    exported_graph_json: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let module_bytes =
        compile_exported_json(exported_graph_json).map_err(compile_error_to_python)?;
    Ok(PyBytes::new(py, &module_bytes))
}

#[pyfunction]
fn load_program(program: &[u8]) -> PyResult<PyRuntimeProgram> {
    let module = ModuleBuilder::new_from_bytes(program)
        .and_then(ModuleBuilder::build)
        .map_err(runtime_error)?;
    let output_lengths = output_lengths(&module.info());
    Ok(PyRuntimeProgram {
        module,
        output_lengths,
    })
}

#[pyfunction]
fn validate_compiled_program(program: &[u8]) -> PyResult<bool> {
    validate_module(program)
        .map(|_| true)
        .map_err(runtime_error)
}

#[pyfunction]
fn program_info_py<'py>(py: Python<'py>, program: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let info = program_info(program).map_err(runtime_error)?;
    program_info_dict(py, &info)
}

#[pyfunction]
fn execute_program(program: &[u8], inputs: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let mut module = ModuleBuilder::new_from_bytes(program)
        .and_then(ModuleBuilder::build)
        .map_err(runtime_error)?;
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let execution_inputs = module
        .validate_inputs(&input_slices)
        .map_err(runtime_error)?;
    let output_length: usize = module
        .info()
        .output_specs
        .iter()
        .map(|output_spec| output_spec.length as usize)
        .sum();
    let mut outputs = vec![0.0; output_length];
    let execution_outputs = module
        .validate_outputs(&mut outputs)
        .map_err(runtime_error)?;
    module.execute(execution_inputs, execution_outputs);
    Ok(outputs)
}

#[pyfunction]
fn push_forward_program(
    program: &[u8],
    inputs: Vec<Vec<f32>>,
    tangents: Vec<Vec<f32>>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let mut module = ModuleBuilder::new_from_bytes(program)
        .and_then(ModuleBuilder::build)
        .map_err(runtime_error)?;
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let tangent_slices: Vec<&[f32]> = tangents.iter().map(|input| input.as_slice()).collect();
    let push_forward_inputs = module
        .validate_push_forward_inputs(&input_slices, &tangent_slices)
        .map_err(runtime_error)?;
    let output_length: usize = module
        .info()
        .output_specs
        .iter()
        .map(|output_spec| output_spec.length as usize)
        .sum();
    let mut outputs = vec![0.0; output_length];
    let mut tangent_outputs = vec![0.0; output_length];
    let push_forward_outputs = module
        .validate_push_forward_outputs(&mut outputs, &mut tangent_outputs)
        .map_err(runtime_error)?;
    module.push_forward(push_forward_inputs, push_forward_outputs);
    Ok((outputs, tangent_outputs))
}

fn output_lengths(info: &ProgramInfo) -> Vec<usize> {
    info.output_specs
        .iter()
        .map(|output_spec| output_spec.length as usize)
        .collect()
}

fn program_info_dict<'py>(py: Python<'py>, info: &ProgramInfo) -> PyResult<Bound<'py, PyDict>> {
    let info_dict = PyDict::new(py);
    info_dict.set_item("workspace_size", info.workspace_size)?;
    info_dict.set_item("required_workspace_size", info.required_workspace_size)?;
    let input_specs = info
        .input_specs
        .iter()
        .map(|input_spec| input_spec.length as usize)
        .collect::<Vec<_>>();
    let output_specs = info
        .output_specs
        .iter()
        .map(|output_spec| output_spec.length as usize)
        .collect::<Vec<_>>();
    info_dict.set_item("input_specs", input_specs)?;
    info_dict.set_item("output_specs", output_specs)?;
    Ok(info_dict)
}

fn compile_error_to_python(error: CompileError) -> PyErr {
    match error {
        CompileError::NotImplemented(message) => PyNotImplementedError::new_err(message),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn runtime_error(error: impl core::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
#[pyo3(name = "_coker_runtime")]
fn coker_python(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRuntimeProgram>()?;
    module.add_function(wrap_pyfunction!(compile_exported_graph, module)?)?;
    module.add_function(wrap_pyfunction!(load_program, module)?)?;
    module.add_function(wrap_pyfunction!(validate_compiled_program, module)?)?;
    module.add_function(wrap_pyfunction!(program_info_py, module)?)?;
    module.add_function(wrap_pyfunction!(execute_program, module)?)?;
    module.add_function(wrap_pyfunction!(push_forward_program, module)?)?;
    Ok(())
}
