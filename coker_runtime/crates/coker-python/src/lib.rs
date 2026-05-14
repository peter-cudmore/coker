use coker_bytecode::Program;
use coker_compiler::{compile_exported_json, CompileError};
use coker_runtime::{
    execute, program_info, program_info_from_program, push_forward, validate_program,
    ProgramInfo,
};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyclass(name = "RuntimeProgram")]
struct PyRuntimeProgram {
    program: Program,
}

#[pymethods]
impl PyRuntimeProgram {
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        program_info_dict(py, &program_info_from_program(&self.program))
    }

    fn execute(&self, inputs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        let mut workspace = vec![0.0; self.program.required_workspace_size as usize];
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        execute(&self.program, &input_slices, &mut workspace).map_err(runtime_error)
    }

    fn push_forward(
        &self,
        inputs: Vec<Vec<f32>>,
        tangents: Vec<Vec<f32>>,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let mut workspace = vec![0.0; self.program.required_workspace_size as usize];
        let mut tangent_workspace =
            vec![0.0; self.program.required_workspace_size as usize];
        let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
        let tangent_slices: Vec<&[f32]> =
            tangents.iter().map(|input| input.as_slice()).collect();
        push_forward(
            &self.program,
            &input_slices,
            &tangent_slices,
            &mut workspace,
            &mut tangent_workspace,
        )
        .map_err(runtime_error)
    }
}

#[pyfunction]
fn compile_exported_graph<'py>(
    py: Python<'py>,
    exported_graph_json: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let program_bytes = compile_exported_json(exported_graph_json)
        .map_err(compile_error_to_python)?;
    Ok(PyBytes::new(py, &program_bytes))
}

#[pyfunction]
fn load_program(program: &[u8]) -> PyResult<PyRuntimeProgram> {
    let parsed_program = validate_program(program).map_err(runtime_error)?;
    Ok(PyRuntimeProgram {
        program: parsed_program,
    })
}

#[pyfunction]
fn validate_compiled_program(program: &[u8]) -> PyResult<bool> {
    validate_program(program).map(|_| true).map_err(runtime_error)
}

#[pyfunction]
fn program_info_py<'py>(
    py: Python<'py>,
    program: &[u8],
) -> PyResult<Bound<'py, PyDict>> {
    let info = program_info(program).map_err(runtime_error)?;
    program_info_dict(py, &info)
}

#[pyfunction]
fn execute_program(program: &[u8], inputs: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    let parsed_program = validate_program(program).map_err(runtime_error)?;
    let mut workspace = vec![0.0; parsed_program.required_workspace_size as usize];
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    execute(&parsed_program, &input_slices, &mut workspace).map_err(runtime_error)
}

#[pyfunction]
fn push_forward_program(
    program: &[u8],
    inputs: Vec<Vec<f32>>,
    tangents: Vec<Vec<f32>>,
) -> PyResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let parsed_program = validate_program(program).map_err(runtime_error)?;
    let mut workspace = vec![0.0; parsed_program.required_workspace_size as usize];
    let mut tangent_workspace =
        vec![0.0; parsed_program.required_workspace_size as usize];
    let input_slices: Vec<&[f32]> = inputs.iter().map(|input| input.as_slice()).collect();
    let tangent_slices: Vec<&[f32]> =
        tangents.iter().map(|input| input.as_slice()).collect();
    push_forward(
        &parsed_program,
        &input_slices,
        &tangent_slices,
        &mut workspace,
        &mut tangent_workspace,
    )
    .map_err(runtime_error)
}

fn program_info_dict<'py>(
    py: Python<'py>,
    info: &ProgramInfo,
) -> PyResult<Bound<'py, PyDict>> {
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
